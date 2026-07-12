#!/usr/bin/env python3
"""Read-only exact attestation for phase 2 of the Muncho canary."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from scripts.canary.foundation import (
    NETWORK,
    PREFLIGHT_MAX_AGE_SECONDS,
    PROJECT,
    SERVICE_ACCOUNT_NAME,
    SQL_INSTANCE,
    build_plan as build_foundation_plan,
)
from scripts.canary.foundation_preflight import (
    collect as collect_foundation,
    evaluate as evaluate_foundation,
)
from scripts.canary.network_boundary import (
    ALLOW_RULE,
    DENY_RULE,
    IAP_SOURCE_RANGE,
    IAP_SSH_RULE,
    NetworkBoundarySpec,
    build_plan,
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _firewall_exact(
    raw: object,
    *,
    name: str,
    priority: int,
    target: str,
    destinations: set[str],
    allowed: list[dict[str, object]] | None = None,
    denied: list[dict[str, object]] | None = None,
) -> bool:
    if not isinstance(raw, Mapping):
        return False
    log_config = raw.get("logConfig")
    return bool(
        raw.get("name") == name
        and raw.get("direction") == "EGRESS"
        and raw.get("priority") == priority
        and raw.get("disabled") is False
        and str(raw.get("network") or "").endswith(f"/networks/{NETWORK}")
        and set(raw.get("targetServiceAccounts") or []) == {target}
        and not (raw.get("targetTags") or [])
        and set(raw.get("destinationRanges") or []) == destinations
        and (raw.get("allowed") or None) == allowed
        and (raw.get("denied") or None) == denied
        and isinstance(log_config, Mapping)
        and log_config.get("enable") is True
    )


def _iap_firewall_exact(raw: object) -> bool:
    if not isinstance(raw, Mapping):
        return False
    log_config = raw.get("logConfig")
    return bool(
        raw.get("name") == IAP_SSH_RULE
        and raw.get("direction") == "INGRESS"
        and raw.get("priority") == 1000
        and raw.get("disabled") is False
        and str(raw.get("network") or "").endswith(f"/networks/{NETWORK}")
        and set(raw.get("sourceRanges") or []) == {IAP_SOURCE_RANGE}
        and not (raw.get("sourceTags") or [])
        and not (raw.get("sourceServiceAccounts") or [])
        and set(raw.get("targetTags") or []) == {"iap-ssh"}
        and not (raw.get("targetServiceAccounts") or [])
        and raw.get("allowed") == [{"IPProtocol": "tcp", "ports": ["22"]}]
        and not (raw.get("denied") or [])
        and isinstance(log_config, Mapping)
        and log_config.get("enable") is True
    )


def _as_values(raw: object) -> set[str]:
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {str(item) for item in raw if item}
    return set()


def _targets_canary(raw: Mapping[str, object], target: str) -> bool:
    tags = _as_values(raw.get("targetTags")) | _as_values(raw.get("target_tags"))
    accounts = _as_values(raw.get("targetServiceAccounts")) | _as_values(
        raw.get("target_svc_acct")
    )
    resources = _as_values(raw.get("targetResources"))
    has_explicit_target = bool(tags or accounts or resources)
    return bool(
        not has_explicit_target
        or "iap-ssh" in tags
        or target in accounts
        or any(value == target or value.endswith(f"/{target}") for value in resources)
    )


def _effective_rules(raw: object) -> list[Mapping[str, object]] | None:
    if isinstance(raw, Mapping):
        raw = raw.get("firewalls")
    if not isinstance(raw, list):
        return None
    if not all(isinstance(item, Mapping) for item in raw):
        return None
    return list(raw)


def _effective_surface(
    raw: object,
    *,
    target: str,
    sql_ip: str,
) -> tuple[bool, set[str]]:
    """Reject every applicable explicit rule except the three exact plan rules.

    ``get-effective-firewalls`` includes VPC, network-policy, folder, and
    organization policy rules.  Rules for another explicit target are harmless
    to this VM; unscoped rules apply to it and therefore fail closed.
    """

    rules = _effective_rules(raw)
    if rules is None:
        return False, set()
    accepted: set[str] = set()
    for rule in rules:
        network = rule.get("network")
        if isinstance(network, str) and not network.endswith(f"/networks/{NETWORK}"):
            continue
        if rule.get("disabled") is True or not _targets_canary(rule, target):
            continue
        name = str(rule.get("name") or "")
        if name == IAP_SSH_RULE and _iap_firewall_exact(rule):
            accepted.add(name)
            continue
        if name == ALLOW_RULE and _firewall_exact(
            rule,
            name=ALLOW_RULE,
            priority=800,
            target=target,
            destinations={f"{sql_ip}/32"},
            allowed=[{"IPProtocol": "tcp", "ports": ["5432"]}],
        ):
            accepted.add(name)
            continue
        if name == DENY_RULE and _firewall_exact(
            rule,
            name=DENY_RULE,
            priority=900,
            target=target,
            destinations={"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"},
            denied=[{"IPProtocol": "all"}],
        ):
            accepted.add(name)
            continue
        # Ignore rules that cannot affect packet flow, but fail closed for any
        # applicable allow, deny, or hierarchical-policy action.
        if (
            rule.get("direction") in {"INGRESS", "EGRESS"}
            and (rule.get("allowed") or rule.get("denied") or rule.get("action"))
        ):
            return False, accepted
    return True, accepted


def _foundation_complete(
    report: object, *, collected_at_unix: object
) -> tuple[bool, bool, str | None]:
    plan = build_foundation_plan()
    expected_steps = [step.name for step in plan.steps]
    if not isinstance(report, Mapping):
        return False, False, None
    complete = bool(
        report.get("schema") == "muncho-isolated-canary-foundation-preflight.v2"
        and report.get("ok") is True
        and report.get("plan_sha256") == plan.sha256
        and report.get("satisfied_steps") == expected_steps
    )
    foundation_collected = report.get("collected_at_unix")
    fresh = bool(
        type(collected_at_unix) is int
        and type(foundation_collected) is int
        and 0 <= collected_at_unix - foundation_collected <= PREFLIGHT_MAX_AGE_SECONDS
    )
    return complete, fresh, report.get("plan_sha256")


def evaluate(evidence: Mapping[str, object]) -> dict[str, object]:
    sql = evidence.get("sql")
    service_account = evidence.get("service_account")
    firewalls = evidence.get("firewalls")
    sql_settings = sql.get("settings") if isinstance(sql, Mapping) else None
    ip_config = (
        sql_settings.get("ipConfiguration")
        if isinstance(sql_settings, Mapping)
        else None
    )
    addresses = sql.get("ipAddresses") if isinstance(sql, Mapping) else None
    private_addresses = [
        str(item.get("ipAddress"))
        for item in (addresses if isinstance(addresses, list) else [])
        if isinstance(item, Mapping)
        and item.get("type") == "PRIVATE"
        and item.get("ipAddress")
    ]
    sql_ip = private_addresses[0] if len(private_addresses) == 1 else ""
    spec = NetworkBoundarySpec(sql_private_ip=sql_ip or "0.0.0.0")
    target = spec.service_account_email
    firewall_items = firewalls if isinstance(firewalls, list) else []
    firewall_map = {
        str(item.get("name")): item
        for item in firewall_items
        if isinstance(item, Mapping)
        and item.get("name")
        and str(item.get("network") or "").endswith(f"/networks/{NETWORK}")
    }
    iap_present = IAP_SSH_RULE in firewall_map
    allow_present = ALLOW_RULE in firewall_map
    deny_present = DENY_RULE in firewall_map
    iap_exact = _iap_firewall_exact(firewall_map.get(IAP_SSH_RULE))
    allow_exact = bool(
        sql_ip
        and _firewall_exact(
            firewall_map.get(ALLOW_RULE),
            name=ALLOW_RULE,
            priority=800,
            target=target,
            destinations={f"{sql_ip}/32"},
            allowed=[{"IPProtocol": "tcp", "ports": ["5432"]}],
        )
    )
    deny_exact = _firewall_exact(
        firewall_map.get(DENY_RULE),
        name=DENY_RULE,
        priority=900,
        target=target,
        destinations={"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"},
        denied=[{"IPProtocol": "all"}],
    )
    effective_safe, effective_names = _effective_surface(
        evidence.get("effective_firewalls"), target=target, sql_ip=sql_ip
    )
    exact_names = {
        name
        for name, exact in (
            (IAP_SSH_RULE, iap_exact),
            (ALLOW_RULE, allow_exact),
            (DENY_RULE, deny_exact),
        )
        if exact
    }
    foundation_complete, foundation_fresh, foundation_sha = _foundation_complete(
        evidence.get("foundation_report"),
        collected_at_unix=evidence.get("collected_at_unix"),
    )
    checks = [
        Check(
            "foundation.complete_exact",
            foundation_complete,
            "all phase-1 resources must be live and exactly attested",
        ),
        Check(
            "foundation.preflight_fresh",
            foundation_fresh,
            "the nested phase-1 attestation must be fresh",
        ),
        Check(
            "sql.exact_private_ready",
            isinstance(sql, Mapping)
            and sql.get("name") == SQL_INSTANCE
            and sql.get("state") == "RUNNABLE"
            and len(private_addresses) == 1
            and isinstance(ip_config, Mapping)
            and ip_config.get("ipv4Enabled") is False
            and str(ip_config.get("privateNetwork") or "").endswith(
                f"/networks/{NETWORK}"
            ),
            "one RUNNABLE private-only canary SQL endpoint is required",
        ),
        Check(
            "identity.target_service_account_exact",
            isinstance(service_account, Mapping)
            and service_account.get("email") == target
            and service_account.get("disabled") is not True,
            "firewall target service account must exist and be enabled",
        ),
        Check(
            "firewall.iap_absent_or_exact",
            not iap_present or iap_exact,
            "IAP ingress must be absent or exact tcp:22 with logging",
        ),
        Check(
            "firewall.allow_absent_or_exact",
            not allow_present or allow_exact,
            "priority 800 allow must target only SQL /32 tcp:5432 with logging",
        ),
        Check(
            "firewall.deny_absent_or_exact",
            not deny_present or deny_exact,
            "priority 900 deny must cover all RFC1918 targets with logging",
        ),
        Check(
            "firewall.effective_surface_exact",
            effective_safe and effective_names == exact_names,
            "effective VPC and hierarchical policy surface must contain no other applicable rule",
        ),
    ]
    plan = build_plan(spec) if sql_ip else None
    satisfied_steps: list[str] = []
    if iap_exact:
        satisfied_steps.append("create_canary_iap_ssh_ingress")
    if allow_exact:
        satisfied_steps.append("create_canary_sql_egress_allow")
    if deny_exact:
        satisfied_steps.append("create_canary_private_egress_deny")
    return {
        "schema": "muncho-isolated-canary-network-preflight.v1",
        "ok": bool(plan) and all(check.passed for check in checks),
        "collected_at_unix": evidence.get("collected_at_unix"),
        "foundation_plan_sha256": foundation_sha,
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
    service_account = f"{SERVICE_ACCOUNT_NAME}@{PROJECT}.iam.gserviceaccount.com"
    foundation_report = evaluate_foundation(collect_foundation())
    return {
        "collected_at_unix": int(time.time()),
        "foundation_report": foundation_report,
        "sql": _run_json(
            (
                "gcloud",
                "sql",
                "instances",
                "describe",
                SQL_INSTANCE,
                project_flag,
                "--format=json",
            )
        ),
        "service_account": _run_json(
            (
                "gcloud",
                "iam",
                "service-accounts",
                "describe",
                service_account,
                project_flag,
                "--format=json",
            )
        ),
        "firewalls": _run_json(
            (
                "gcloud",
                "compute",
                "firewall-rules",
                "list",
                project_flag,
                "--format=json",
            )
        ),
        "effective_firewalls": _run_json(
            (
                "gcloud",
                "compute",
                "networks",
                "get-effective-firewalls",
                NETWORK,
                project_flag,
                "--format=json",
            )
        ),
    }


def main() -> int:
    report = evaluate(collect())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
