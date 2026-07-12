from __future__ import annotations

import pytest

from scripts.canary.foundation import build_plan as build_foundation_plan
from scripts.canary.network_boundary import (
    ALLOW_RULE,
    DENY_RULE,
    IAP_SSH_RULE,
    NetworkBoundarySpec,
    build_plan,
)
from scripts.canary.network_preflight import evaluate


COLLECTED_AT = 1_800_000_000
SQL_IP = "10.91.0.3"


def _foundation_report():
    plan = build_foundation_plan()
    return {
        "schema": "muncho-isolated-canary-foundation-preflight.v2",
        "ok": True,
        "collected_at_unix": COLLECTED_AT,
        "plan_sha256": plan.sha256,
        "satisfied_steps": [step.name for step in plan.steps],
    }


def _iap_rule():
    return {
        "name": IAP_SSH_RULE,
        "network": (
            "https://www.googleapis.com/compute/v1/projects/"
            "adventico-ai-platform/global/networks/muncho-canary-vpc"
        ),
        "direction": "INGRESS",
        "priority": 1000,
        "disabled": False,
        "sourceRanges": ["35.235.240.0/20"],
        "targetTags": ["iap-ssh"],
        "allowed": [{"IPProtocol": "tcp", "ports": ["22"]}],
        "logConfig": {"enable": True},
    }


def _egress_rules(target):
    common = {
        "network": (
            "https://www.googleapis.com/compute/v1/projects/"
            "adventico-ai-platform/global/networks/muncho-canary-vpc"
        ),
        "direction": "EGRESS",
        "disabled": False,
        "targetServiceAccounts": [target],
        "logConfig": {"enable": True},
    }
    return [
        {
            **common,
            "name": ALLOW_RULE,
            "priority": 800,
            "destinationRanges": [f"{SQL_IP}/32"],
            "allowed": [{"IPProtocol": "tcp", "ports": ["5432"]}],
        },
        {
            **common,
            "name": DENY_RULE,
            "priority": 900,
            "destinationRanges": [
                "10.0.0.0/8",
                "172.16.0.0/12",
                "192.168.0.0/16",
            ],
            "denied": [{"IPProtocol": "all"}],
        },
    ]


def _evidence():
    target = (
        "muncho-canary-v2-runtime@"
        "adventico-ai-platform.iam.gserviceaccount.com"
    )
    return {
        "collected_at_unix": COLLECTED_AT,
        "foundation_report": _foundation_report(),
        "sql": {
            "name": "muncho-canary-pg18-v2",
            "state": "RUNNABLE",
            "ipAddresses": [{"type": "PRIVATE", "ipAddress": SQL_IP}],
            "settings": {
                "ipConfiguration": {
                    "ipv4Enabled": False,
                    "privateNetwork": (
                        "projects/adventico-ai-platform/global/networks/"
                        "muncho-canary-vpc"
                    ),
                }
            },
        },
        "service_account": {"email": target, "disabled": False},
        "firewalls": [],
        "effective_firewalls": {"firewalls": []},
    }


def _install_exact_rules(evidence):
    rules = [_iap_rule(), *_egress_rules(evidence["service_account"]["email"])]
    evidence["firewalls"].extend(rules)
    evidence["effective_firewalls"]["firewalls"].extend(rules)


def test_network_plan_is_derived_from_exact_sql_private_ip_and_orders_iap_first():
    plan = build_plan(NetworkBoundarySpec(sql_private_ip=SQL_IP))
    rendered = str(plan.report())

    assert [step.name for step in plan.steps] == [
        "create_canary_iap_ssh_ingress",
        "create_canary_sql_egress_allow",
        "create_canary_private_egress_deny",
    ]
    assert IAP_SSH_RULE in rendered
    assert f"--destination-ranges={SQL_IP}/32" in rendered
    assert "--priority=800" in rendered
    assert "--rules=tcp:5432" in rendered
    assert "--priority=900" in rendered
    assert "--rules=all" in rendered
    assert "--enable-logging" in rendered
    assert "--target-service-accounts=muncho-canary-v2-runtime@" in rendered


def test_absent_rules_pass_as_provisionable_but_are_not_satisfied():
    report = evaluate(_evidence())

    assert report["ok"] is True
    assert report["satisfied_steps"] == []
    assert report["foundation_plan_sha256"] == build_foundation_plan().sha256


def test_exact_existing_rules_are_verified_in_raw_and_effective_views():
    evidence = _evidence()
    _install_exact_rules(evidence)

    report = evaluate(evidence)

    assert report["ok"] is True
    assert report["satisfied_steps"] == [
        "create_canary_iap_ssh_ingress",
        "create_canary_sql_egress_allow",
        "create_canary_private_egress_deny",
    ]


def test_rule_drift_fails_closed():
    evidence = _evidence()
    drifted = _egress_rules(evidence["service_account"]["email"])[0]
    drifted["priority"] = 801
    evidence["firewalls"].append(drifted)
    evidence["effective_firewalls"]["firewalls"].append(drifted)

    report = evaluate(evidence)

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "firewall.allow_absent_or_exact" in failed
    assert "firewall.effective_surface_exact" in failed


def test_network_plan_rejects_non_rfc1918_destination():
    with pytest.raises(ValueError, match="RFC1918"):
        build_plan(NetworkBoundarySpec(sql_private_ip="203.0.113.9"))


def test_broad_no_target_ingress_allow_fails_effective_surface_closed():
    evidence = _evidence()
    broad = {
        "name": "broad-canary-ingress",
        "network": (
            "projects/adventico-ai-platform/global/networks/muncho-canary-vpc"
        ),
        "direction": "INGRESS",
        "priority": 1100,
        "disabled": False,
        "sourceRanges": ["0.0.0.0/0"],
        "allowed": [{"IPProtocol": "tcp", "ports": ["443"]}],
    }
    evidence["firewalls"].append(broad)
    evidence["effective_firewalls"]["firewalls"].append(broad)

    report = evaluate(evidence)

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert failed == {"firewall.effective_surface_exact"}


def test_hierarchical_policy_action_without_network_field_fails_closed():
    evidence = _evidence()
    evidence["effective_firewalls"]["firewalls"].append(
        {
            "name": "org-wide-ingress",
            "direction": "INGRESS",
            "priority": 100,
            "disabled": False,
            "action": "allow",
            "ipRanges": ["0.0.0.0/0"],
        }
    )

    report = evaluate(evidence)

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert failed == {"firewall.effective_surface_exact"}


def test_rule_on_an_unrelated_network_is_ignored():
    evidence = _evidence()
    unrelated = {
        "name": "broad-prod-ingress",
        "network": "projects/p/global/networks/ai-platform-vpc",
        "direction": "INGRESS",
        "priority": 100,
        "disabled": False,
        "allowed": [{"IPProtocol": "all"}],
    }
    evidence["firewalls"].append(unrelated)
    evidence["effective_firewalls"]["firewalls"].append(unrelated)

    report = evaluate(evidence)

    assert report["ok"] is True


def test_incomplete_foundation_blocks_network_phase():
    evidence = _evidence()
    evidence["foundation_report"]["satisfied_steps"].pop()

    report = evaluate(evidence)

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert failed == {"foundation.complete_exact"}
