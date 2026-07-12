from __future__ import annotations

import subprocess

import pytest

from scripts.canary.foundation import build_plan as build_foundation_plan
from scripts.canary.host import HostSpec, build_plan, execute_plan
from scripts.canary.host_preflight import evaluate
from scripts.canary.network_boundary import (
    NetworkBoundarySpec,
    build_plan as build_network_plan,
)


COLLECTED_AT = 1_800_000_000
SQL_IP = "10.91.0.3"
NETWORK_PLAN = build_network_plan(NetworkBoundarySpec(sql_private_ip=SQL_IP))
NETWORK_CHECK_NAMES = [
    "foundation.complete_exact",
    "foundation.preflight_fresh",
    "sql.exact_private_ready",
    "identity.target_service_account_exact",
    "firewall.iap_absent_or_exact",
    "firewall.allow_absent_or_exact",
    "firewall.deny_absent_or_exact",
    "firewall.effective_surface_exact",
]


def _network_report():
    return {
        "schema": "muncho-isolated-canary-network-preflight.v1",
        "ok": True,
        "collected_at_unix": COLLECTED_AT,
        "sql_private_ip": SQL_IP,
        "plan_sha256": NETWORK_PLAN.sha256,
        "satisfied_steps": [step.name for step in NETWORK_PLAN.steps],
        "checks": [
            {"name": name, "passed": True, "detail": "exact"}
            for name in NETWORK_CHECK_NAMES
        ],
    }


def _evidence():
    return {
        "collected_at_unix": COLLECTED_AT,
        "network_report": _network_report(),
        "instances": [],
        "image": {
            "name": "debian-12-bookworm-v20260609",
            "status": "READY",
        },
    }


def _exact_vm():
    target = (
        "muncho-canary-v2-runtime@"
        "adventico-ai-platform.iam.gserviceaccount.com"
    )
    return {
        "name": "muncho-canary-v2-01",
        "status": "RUNNING",
        "zone": "projects/p/zones/europe-west3-a",
        "machineType": "projects/p/zones/europe-west3-a/machineTypes/e2-medium",
        "canIpForward": False,
        "deletionProtection": False,
        "tags": {"items": ["iap-ssh"]},
        "metadata": {
            "items": [
                {"key": "disable-legacy-endpoints", "value": "TRUE"},
                {"key": "enable-oslogin", "value": "TRUE"},
            ]
        },
        "serviceAccounts": [
            {
                "email": target,
                "scopes": [
                    "https://www.googleapis.com/auth/logging.write",
                    "https://www.googleapis.com/auth/monitoring.write",
                ],
            }
        ],
        "networkInterfaces": [
            {
                "network": (
                    "projects/p/global/networks/muncho-canary-vpc"
                ),
                "subnetwork": (
                    "projects/p/regions/europe-west3/subnetworks/"
                    "muncho-canary-europe-west3"
                ),
                "stackType": "IPV4_ONLY",
                "networkIP": "10.90.0.2",
                "accessConfigs": [
                    {
                        "type": "ONE_TO_ONE_NAT",
                        "networkTier": "PREMIUM",
                        "natIP": "203.0.113.10",
                    }
                ],
            }
        ],
        "disks": [{"boot": True, "autoDelete": True}],
        "scheduling": {
            "automaticRestart": True,
            "onHostMaintenance": "MIGRATE",
            "preemptible": False,
            "provisioningModel": "STANDARD",
        },
        "shieldedInstanceConfig": {
            "enableSecureBoot": True,
            "enableVtpm": True,
            "enableIntegrityMonitoring": True,
        },
    }


def _exact_disk():
    return {
        "name": "muncho-canary-v2-01",
        "sizeGb": "20",
        "type": "projects/p/zones/europe-west3-a/diskTypes/pd-balanced",
        "sourceImage": (
            "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
            "global/images/debian-12-bookworm-v20260609"
        ),
    }


def _host_plan():
    return build_plan(
        HostSpec(
            sql_private_ip=SQL_IP,
            network_plan_sha256=NETWORK_PLAN.sha256,
        )
    )


def test_vm_create_exists_only_in_the_single_phase_three_step():
    foundation = build_foundation_plan()
    host = _host_plan()

    assert len(host.steps) == 1
    assert host.steps[0].name == "create_isolated_canary_vm"
    assert host.steps[0].argv[:4] == (
        "gcloud",
        "compute",
        "instances",
        "create",
    )
    assert not any(
        step.argv[:4] == ("gcloud", "compute", "instances", "create")
        for step in foundation.steps
    )
    assert not any(
        step.argv[:4] == ("gcloud", "compute", "instances", "create")
        for step in NETWORK_PLAN.steps
    )
    rendered = str(host.report())
    assert "--network=muncho-canary-vpc" in rendered
    assert "--subnet=muncho-canary-europe-west3" in rendered
    assert "--service-account=muncho-canary-v2-runtime@" in rendered
    assert "--scopes=logging-write,monitoring-write" in rendered


def test_absent_vm_is_provisionable_only_after_complete_network_attestation():
    report = evaluate(_evidence())

    assert report["ok"] is True
    assert report["plan_sha256"] == _host_plan().sha256
    assert report["network_plan_sha256"] == NETWORK_PLAN.sha256
    assert report["satisfied_steps"] == []


def test_incomplete_network_rule_inventory_blocks_vm_creation():
    evidence = _evidence()
    evidence["network_report"]["satisfied_steps"].pop()

    report = evaluate(evidence)

    assert report["ok"] is False
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert failed == {"network.complete_exact"}


def test_missing_effective_firewall_check_blocks_vm_creation():
    evidence = _evidence()
    evidence["network_report"]["checks"].pop()

    report = evaluate(evidence)

    assert report["ok"] is False
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert failed == {"network.complete_exact"}


def test_exact_existing_vm_is_verified_and_satisfied():
    evidence = _evidence()
    evidence["instances"] = [{"name": "muncho-canary-v2-01"}]
    evidence["planned_vm"] = _exact_vm()
    evidence["planned_vm_disk"] = _exact_disk()

    report = evaluate(evidence)

    assert report["ok"] is True
    assert report["satisfied_steps"] == ["create_isolated_canary_vm"]


def test_vm_on_production_network_fails_closed():
    evidence = _evidence()
    evidence["instances"] = [{"name": "muncho-canary-v2-01"}]
    evidence["planned_vm"] = _exact_vm()
    evidence["planned_vm"]["networkInterfaces"][0]["network"] = (
        "projects/p/global/networks/ai-platform-vpc"
    )
    evidence["planned_vm_disk"] = _exact_disk()

    report = evaluate(evidence)

    assert report["ok"] is False
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert failed == {"resource.vm_absent_or_exact_running"}


def test_wrong_network_digest_is_rejected_before_host_runner():
    plan = _host_plan()
    called = []
    preflight = evaluate(_evidence())
    preflight["network_plan_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="network plan"):
        execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            runner=lambda argv: called.append(argv),
            now_unix=COLLECTED_AT,
        )

    assert called == []


def test_host_apply_runs_only_after_fresh_exact_preflight():
    plan = _host_plan()
    called = []

    def runner(argv):
        called.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    receipt = execute_plan(
        plan,
        approved_plan_sha256=plan.sha256,
        preflight=evaluate(_evidence()),
        runner=runner,
        now_unix=COLLECTED_AT,
    )

    assert receipt["ok"] is True
    assert receipt["requires_post_apply_attestation"] is True
    assert called == [plan.steps[0].argv]
