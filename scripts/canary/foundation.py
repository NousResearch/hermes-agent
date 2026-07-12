#!/usr/bin/env python3
"""Digest-bound phase 1 foundation for the isolated Cloud Muncho canary.

Phase 1 creates only the runtime identity and dedicated PostgreSQL foundation.
It deliberately cannot create credentials, firewall rules, or a VM.  The VM is
owned by the later host phase, whose preflight requires the exact network phase
to be live first.
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


PROJECT = "adventico-ai-platform"
REGION = "europe-west3"
ZONE = "europe-west3-a"
NETWORK = "muncho-canary-vpc"
SUBNET = "muncho-canary-europe-west3"
SUBNET_CIDR = "10.90.0.0/24"
PRIVATE_SERVICE_RANGE_NAME = "muncho-canary-sql-range"
PRIVATE_SERVICE_RANGE_CIDR = "10.91.0.0/24"
SERVICE_ACCOUNT_NAME = "muncho-canary-v2-runtime"
SQL_INSTANCE = "muncho-canary-pg18-v2"
DATABASE = "muncho_canary_brain"
FORBIDDEN_CANARY_SECRET_NAMES = (
    "muncho-canary-db-password",
    "muncho-canary-discord-bot-token",
)
PREFLIGHT_MAX_AGE_SECONDS = 900

_NAME = re.compile(r"^[a-z][a-z0-9-]{0,61}[a-z0-9]$")
_PROJECT = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FoundationSpec:
    project: str = PROJECT
    region: str = REGION
    zone: str = ZONE
    network: str = NETWORK
    subnet: str = SUBNET
    subnet_cidr: str = SUBNET_CIDR
    private_service_range_name: str = PRIVATE_SERVICE_RANGE_NAME
    private_service_range_cidr: str = PRIVATE_SERVICE_RANGE_CIDR
    service_account_name: str = SERVICE_ACCOUNT_NAME
    sql_instance: str = SQL_INSTANCE
    database: str = DATABASE

    @property
    def service_account_email(self) -> str:
        return f"{self.service_account_name}@{self.project}.iam.gserviceaccount.com"

    def validate(self) -> None:
        if not _PROJECT.fullmatch(self.project):
            raise ValueError("invalid project")
        for label, value in (
            ("network", self.network),
            ("subnet", self.subnet),
            ("private_service_range_name", self.private_service_range_name),
            ("service_account_name", self.service_account_name),
            ("sql_instance", self.sql_instance),
        ):
            if not _NAME.fullmatch(value):
                raise ValueError(f"invalid {label}")
        if not re.fullmatch(r"^[a-z][a-z0-9_]{2,62}$", self.database):
            raise ValueError("invalid database")
        if self.subnet_cidr != SUBNET_CIDR:
            raise ValueError("subnet CIDR is production-pinned")
        if self.private_service_range_cidr != PRIVATE_SERVICE_RANGE_CIDR:
            raise ValueError("private service CIDR is production-pinned")
        if not re.fullmatch(r"^[a-z]+-[a-z]+[0-9]$", self.region):
            raise ValueError("invalid region")
        if not re.fullmatch(rf"^{re.escape(self.region)}-[a-z]$", self.zone):
            raise ValueError("zone must belong to region")


@dataclass(frozen=True)
class PlanStep:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class FoundationPlan:
    schema: str
    spec: FoundationSpec
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
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def report(self) -> dict[str, object]:
        return {**self.payload(), "plan_sha256": self.sha256}


def build_plan(spec: FoundationSpec = FoundationSpec()) -> FoundationPlan:
    spec.validate()
    project = spec.project
    service_account = spec.service_account_email
    return FoundationPlan(
        schema="muncho-isolated-canary-foundation-plan.v2",
        spec=spec,
        architecture={
            "phase": 1,
            "creates_vm": False,
            "creates_network_rules": False,
            "creates_secret_manager_resources": False,
            "isolated_cloud_sql_instance_required": True,
            "dedicated_vpc_required": True,
            "production_vpc_peering_forbidden": True,
            "allowed_vpc_peerings": ["servicenetworking-googleapis-com"],
            "sql_runtime_iam_required": False,
            "runtime_service_account_allowed_project_roles": [
                "roles/logging.logWriter",
                "roles/monitoring.metricWriter",
            ],
            "credential_source": "owner_provisioned_outside_shared_project_secret_manager",
            "next_phase": "network_boundary",
        },
        steps=(
            PlanStep(
                "create_isolated_vpc",
                (
                    "gcloud",
                    "compute",
                    "networks",
                    "create",
                    spec.network,
                    f"--project={project}",
                    "--subnet-mode=custom",
                    "--bgp-routing-mode=regional",
                    "--mtu=1460",
                    "--quiet",
                ),
            ),
            PlanStep(
                "create_isolated_subnet",
                (
                    "gcloud",
                    "compute",
                    "networks",
                    "subnets",
                    "create",
                    spec.subnet,
                    f"--project={project}",
                    f"--network={spec.network}",
                    f"--region={spec.region}",
                    f"--range={spec.subnet_cidr}",
                    "--stack-type=IPV4_ONLY",
                    "--quiet",
                ),
            ),
            PlanStep(
                "reserve_private_service_range",
                (
                    "gcloud",
                    "compute",
                    "addresses",
                    "create",
                    spec.private_service_range_name,
                    f"--project={project}",
                    "--global",
                    "--purpose=VPC_PEERING",
                    f"--addresses={spec.private_service_range_cidr.split('/')[0]}",
                    f"--prefix-length={spec.private_service_range_cidr.split('/')[1]}",
                    f"--network={spec.network}",
                    "--quiet",
                ),
            ),
            PlanStep(
                "connect_private_service_networking",
                (
                    "gcloud",
                    "services",
                    "vpc-peerings",
                    "connect",
                    f"--project={project}",
                    f"--network={spec.network}",
                    "--service=servicenetworking.googleapis.com",
                    f"--ranges={spec.private_service_range_name}",
                    "--quiet",
                ),
            ),
            PlanStep(
                "create_runtime_service_account",
                (
                    "gcloud",
                    "iam",
                    "service-accounts",
                    "create",
                    spec.service_account_name,
                    f"--project={project}",
                    "--display-name=Muncho isolated canary runtime",
                    "--quiet",
                ),
            ),
            PlanStep(
                "grant_logging_writer",
                (
                    "gcloud",
                    "projects",
                    "add-iam-policy-binding",
                    project,
                    f"--member=serviceAccount:{service_account}",
                    "--role=roles/logging.logWriter",
                    "--condition=None",
                    "--quiet",
                ),
            ),
            PlanStep(
                "grant_monitoring_writer",
                (
                    "gcloud",
                    "projects",
                    "add-iam-policy-binding",
                    project,
                    f"--member=serviceAccount:{service_account}",
                    "--role=roles/monitoring.metricWriter",
                    "--condition=None",
                    "--quiet",
                ),
            ),
            PlanStep(
                "create_isolated_postgres",
                (
                    "gcloud",
                    "sql",
                    "instances",
                    "create",
                    spec.sql_instance,
                    f"--project={project}",
                    "--database-version=POSTGRES_18",
                    "--edition=ENTERPRISE",
                    "--tier=db-f1-micro",
                    f"--zone={spec.zone}",
                    "--availability-type=zonal",
                    f"--network=projects/{project}/global/networks/{spec.network}",
                    "--no-assign-ip",
                    "--ssl-mode=ENCRYPTED_ONLY",
                    "--server-ca-mode=GOOGLE_MANAGED_INTERNAL_CA",
                    "--storage-size=10",
                    "--storage-type=SSD",
                    "--no-storage-auto-increase",
                    "--no-backup",
                    "--deletion-protection",
                    "--maintenance-release-channel=stable",
                    "--database-flags=cloudsql.iam_authentication=off",
                    "--quiet",
                ),
            ),
            PlanStep(
                "create_canonical_database",
                (
                    "gcloud",
                    "sql",
                    "databases",
                    "create",
                    spec.database,
                    f"--instance={spec.sql_instance}",
                    f"--project={project}",
                    "--charset=UTF8",
                    "--collation=en_US.UTF8",
                    "--quiet",
                ),
            ),
        ),
    )


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), check=False, capture_output=True, text=True, timeout=3600
    )


def execute_plan(
    plan: FoundationPlan,
    *,
    approved_plan_sha256: str,
    preflight: Mapping[str, object],
    runner: Runner = _default_runner,
    now_unix: int | None = None,
) -> dict[str, object]:
    if not _SHA256.fullmatch(approved_plan_sha256):
        raise ValueError("approval digest must be lowercase SHA-256")
    if approved_plan_sha256 != plan.sha256:
        raise RuntimeError("approved plan digest does not match exact plan")
    if preflight.get("schema") != "muncho-isolated-canary-foundation-preflight.v2":
        raise RuntimeError("preflight schema mismatch")
    if preflight.get("ok") is not True or preflight.get("plan_sha256") != plan.sha256:
        raise RuntimeError("preflight did not pass for exact plan")
    raw_satisfied = preflight.get("satisfied_steps")
    if not isinstance(raw_satisfied, list) or any(
        not isinstance(item, str) for item in raw_satisfied
    ):
        raise RuntimeError("preflight satisfied-step inventory is invalid")
    plan_step_names = {step.name for step in plan.steps}
    satisfied = set(raw_satisfied)
    if not satisfied <= plan_step_names:
        raise RuntimeError("preflight contains an unknown satisfied step")
    collected_at = preflight.get("collected_at_unix")
    now = int(time.time()) if now_unix is None else now_unix
    if (
        type(collected_at) is not int
        or now < collected_at
        or now - collected_at > PREFLIGHT_MAX_AGE_SECONDS
    ):
        raise RuntimeError("preflight is stale")

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
                "schema": "muncho-isolated-canary-foundation-receipt.v2",
                "ok": False,
                "plan_sha256": plan.sha256,
                "receipts": receipts,
            }
    return {
        "schema": "muncho-isolated-canary-foundation-receipt.v2",
        "ok": True,
        "requires_post_apply_attestation": True,
        "plan_sha256": plan.sha256,
        "receipts": receipts,
    }


def _load_preflight(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("preflight artifact must be a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "apply"), nargs="?", default="plan")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--preflight", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = build_plan(FoundationSpec(project=args.project))
    if args.command == "plan":
        print(json.dumps(plan.report(), indent=2, sort_keys=True))
        return 0
    if not args.approved_plan_sha256 or args.preflight is None:
        raise SystemExit("apply requires --approved-plan-sha256 and --preflight")
    receipt = execute_plan(
        plan,
        approved_plan_sha256=args.approved_plan_sha256,
        preflight=_load_preflight(args.preflight),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
