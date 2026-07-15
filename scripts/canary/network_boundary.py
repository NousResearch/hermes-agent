#!/usr/bin/env python3
"""Digest-bound phase 2 network boundary for the isolated Muncho canary."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
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
    PlanStep,
    build_plan as build_foundation_plan,
)


IAP_SSH_RULE = "muncho-canary-v2-allow-iap-ssh"
ALLOW_RULE = "muncho-canary-v2-allow-sql-egress"
DENY_RULE = "muncho-canary-v2-deny-private-egress"
IAP_SOURCE_RANGE = "35.235.240.0/20"
_SHA256_LENGTH = 64
_RFC1918 = tuple(
    ipaddress.ip_network(value)
    for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


@dataclass(frozen=True)
class NetworkBoundarySpec:
    sql_private_ip: str
    project: str = PROJECT
    network: str = NETWORK
    service_account_name: str = SERVICE_ACCOUNT_NAME

    @property
    def service_account_email(self) -> str:
        return f"{self.service_account_name}@{self.project}.iam.gserviceaccount.com"

    def validate(self) -> None:
        try:
            address = ipaddress.ip_address(self.sql_private_ip)
        except ValueError as exc:
            raise ValueError("SQL private IP is invalid") from exc
        if address.version != 4 or not any(address in network for network in _RFC1918):
            raise ValueError("SQL address must be RFC1918 IPv4")


@dataclass(frozen=True)
class NetworkBoundaryPlan:
    schema: str
    spec: NetworkBoundarySpec
    steps: tuple[PlanStep, ...]

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "spec": asdict(self.spec),
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


def build_plan(spec: NetworkBoundarySpec) -> NetworkBoundaryPlan:
    spec.validate()
    target = spec.service_account_email
    common = (
        f"--project={spec.project}",
        f"--network={spec.network}",
        "--direction=EGRESS",
        f"--target-service-accounts={target}",
        "--enable-logging",
        "--quiet",
    )
    return NetworkBoundaryPlan(
        schema="muncho-isolated-canary-network-plan.v1",
        spec=spec,
        steps=(
            PlanStep(
                "create_canary_iap_ssh_ingress",
                (
                    "gcloud",
                    "compute",
                    "firewall-rules",
                    "create",
                    IAP_SSH_RULE,
                    f"--project={spec.project}",
                    f"--network={spec.network}",
                    "--direction=INGRESS",
                    "--priority=1000",
                    "--action=ALLOW",
                    "--rules=tcp:22",
                    f"--source-ranges={IAP_SOURCE_RANGE}",
                    "--target-tags=iap-ssh",
                    "--enable-logging",
                    "--quiet",
                ),
            ),
            PlanStep(
                "create_canary_sql_egress_allow",
                (
                    "gcloud",
                    "compute",
                    "firewall-rules",
                    "create",
                    ALLOW_RULE,
                    "--priority=800",
                    "--action=ALLOW",
                    "--rules=tcp:5432",
                    f"--destination-ranges={spec.sql_private_ip}/32",
                    *common,
                ),
            ),
            PlanStep(
                "create_canary_private_egress_deny",
                (
                    "gcloud",
                    "compute",
                    "firewall-rules",
                    "create",
                    DENY_RULE,
                    "--priority=900",
                    "--action=DENY",
                    "--rules=all",
                    "--destination-ranges=10.0.0.0/8,172.16.0.0/12,192.168.0.0/16",
                    *common,
                ),
            ),
        ),
    )


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), check=False, capture_output=True, text=True, timeout=300
    )


def execute_plan(
    plan: NetworkBoundaryPlan,
    *,
    approved_plan_sha256: str,
    preflight: Mapping[str, object],
    runner: Runner = _runner,
    now_unix: int | None = None,
) -> dict[str, object]:
    if (
        len(approved_plan_sha256) != _SHA256_LENGTH
        or approved_plan_sha256 != plan.sha256
    ):
        raise RuntimeError("approved network plan digest mismatch")
    if preflight.get("schema") != "muncho-isolated-canary-network-preflight.v1":
        raise RuntimeError("network preflight schema mismatch")
    if preflight.get("ok") is not True or preflight.get("plan_sha256") != plan.sha256:
        raise RuntimeError("network preflight did not pass for exact plan")
    if preflight.get("foundation_plan_sha256") != build_foundation_plan().sha256:
        raise RuntimeError("network preflight is not bound to the exact foundation")
    if preflight.get("sql_private_ip") != plan.spec.sql_private_ip:
        raise RuntimeError("network preflight SQL endpoint mismatch")
    collected_at = preflight.get("collected_at_unix")
    now = int(time.time()) if now_unix is None else now_unix
    if (
        type(collected_at) is not int
        or now < collected_at
        or now - collected_at > PREFLIGHT_MAX_AGE_SECONDS
    ):
        raise RuntimeError("network preflight is stale")
    raw_satisfied = preflight.get("satisfied_steps")
    if not isinstance(raw_satisfied, list) or any(
        not isinstance(item, str) for item in raw_satisfied
    ):
        raise RuntimeError("network preflight satisfied-step inventory is invalid")
    allowed_steps = {step.name for step in plan.steps}
    satisfied = set(raw_satisfied)
    if not satisfied <= allowed_steps:
        raise RuntimeError("network preflight contains unknown satisfied step")

    receipts = []
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
                "schema": "muncho-isolated-canary-network-receipt.v1",
                "ok": False,
                "plan_sha256": plan.sha256,
                "receipts": receipts,
            }
    return {
        "schema": "muncho-isolated-canary-network-receipt.v1",
        "ok": True,
        "requires_post_apply_attestation": True,
        "plan_sha256": plan.sha256,
        "receipts": receipts,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "apply"), nargs="?", default="plan")
    parser.add_argument("--sql-private-ip", required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--approved-plan-sha256")
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = build_plan(NetworkBoundarySpec(sql_private_ip=args.sql_private_ip))
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
