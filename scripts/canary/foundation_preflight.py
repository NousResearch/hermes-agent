#!/usr/bin/env python3
"""Read-only exact attestation for phase 1 of the Muncho canary."""

from __future__ import annotations

import argparse
import ipaddress
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

from scripts.canary.foundation import (
    FORBIDDEN_CANARY_SECRET_NAMES,
    FoundationSpec,
    build_plan,
)


REQUIRED_SERVICES = frozenset(
    {
        "compute.googleapis.com",
        "iam.googleapis.com",
        "logging.googleapis.com",
        "monitoring.googleapis.com",
        "servicenetworking.googleapis.com",
        "sqladmin.googleapis.com",
    }
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _names(items: object, field: str = "name") -> set[str]:
    if not isinstance(items, list):
        return set()
    names: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        value: object = item
        for part in field.split("."):
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        if value:
            names.add(str(value))
    return names


def _ends_with(value: object, suffix: str) -> bool:
    return isinstance(value, str) and value.endswith(suffix)


def _network_exact(evidence: Mapping[str, object], spec: FoundationSpec) -> bool:
    network = evidence.get("planned_network")
    if not isinstance(network, Mapping):
        return False
    routing = network.get("routingConfig")
    subnetworks = network.get("subnetworks") or []
    peerings = network.get("peerings") or []
    if (
        not isinstance(routing, Mapping)
        or not isinstance(subnetworks, list)
        or not isinstance(peerings, list)
    ):
        return False
    subnets_exact = len(subnetworks) <= 1 and all(
        _ends_with(item, f"/subnetworks/{spec.subnet}") for item in subnetworks
    )
    peerings_exact = len(peerings) <= 1
    if peerings:
        peering = peerings[0]
        peerings_exact = bool(
            len(peerings) == 1
            and isinstance(peering, Mapping)
            and peering.get("name") == "servicenetworking-googleapis-com"
            and _ends_with(peering.get("network"), "/networks/servicenetworking")
            and peering.get("state") == "ACTIVE"
            and peering.get("exchangeSubnetRoutes") is True
            and peering.get("importCustomRoutes") is False
            and peering.get("exportCustomRoutes") is False
            and peering.get("importSubnetRoutesWithPublicIp") is False
            and peering.get("exportSubnetRoutesWithPublicIp") is False
        )
    return bool(
        network.get("name") == spec.network
        and network.get("autoCreateSubnetworks") is False
        and network.get("mtu") in {None, 1460}
        and routing.get("routingMode") == "REGIONAL"
        and subnets_exact
        and peerings_exact
    )


def _subnet_exact(evidence: Mapping[str, object], spec: FoundationSpec) -> bool:
    subnet = evidence.get("planned_subnet")
    return bool(
        isinstance(subnet, Mapping)
        and subnet.get("name") == spec.subnet
        and subnet.get("ipCidrRange") == spec.subnet_cidr
        and _ends_with(subnet.get("network"), f"/networks/{spec.network}")
        and _ends_with(subnet.get("region"), f"/regions/{spec.region}")
        and subnet.get("privateIpGoogleAccess") in {None, False}
        and subnet.get("purpose") in {None, "PRIVATE"}
        and subnet.get("stackType") in {None, "IPV4_ONLY"}
        and not (subnet.get("secondaryIpRanges") or [])
    )


def _private_range_exact(
    evidence: Mapping[str, object], spec: FoundationSpec
) -> bool:
    address = evidence.get("planned_private_service_range")
    expected_address, expected_prefix = spec.private_service_range_cidr.split("/")
    return bool(
        isinstance(address, Mapping)
        and address.get("name") == spec.private_service_range_name
        and address.get("addressType") == "INTERNAL"
        and address.get("purpose") == "VPC_PEERING"
        and address.get("status") == "RESERVED"
        and address.get("address") == expected_address
        and str(address.get("prefixLength")) == expected_prefix
        and _ends_with(address.get("network"), f"/networks/{spec.network}")
    )


def _service_connection_exact(
    evidence: Mapping[str, object], spec: FoundationSpec
) -> bool:
    connections = evidence.get("planned_service_networking")
    if not isinstance(connections, list) or len(connections) != 1:
        return False
    connection = connections[0]
    return bool(
        isinstance(connection, Mapping)
        and connection.get("peering") == "servicenetworking-googleapis-com"
        and _ends_with(connection.get("network"), f"/networks/{spec.network}")
        and connection.get("service")
        == "services/servicenetworking.googleapis.com"
        and connection.get("reservedPeeringRanges")
        == [spec.private_service_range_name]
    )


def _routes_safe(
    evidence: Mapping[str, object],
    spec: FoundationSpec,
    *,
    network_present: bool,
    subnet_present: bool,
    service_connection_present: bool,
) -> bool:
    routes = evidence.get("planned_routes")
    if not isinstance(routes, list):
        return not network_present
    observed: set[str] = set()
    for route in routes:
        if not isinstance(route, Mapping) or not _ends_with(
            route.get("network"), f"/networks/{spec.network}"
        ):
            return False
        destination = route.get("destRange")
        if (
            destination == "0.0.0.0/0"
            and route.get("priority") == 1000
            and _ends_with(
                route.get("nextHopGateway"), "/gateways/default-internet-gateway"
            )
            and not (route.get("tags") or [])
        ):
            observed.add("default")
            continue
        if (
            destination == spec.subnet_cidr
            and route.get("priority") == 0
            and _ends_with(route.get("nextHopNetwork"), f"/networks/{spec.network}")
            and not (route.get("tags") or [])
        ):
            observed.add("subnet")
            continue
        if (
            destination == spec.private_service_range_cidr
            and route.get("priority") == 0
            and route.get("nextHopPeering") == "servicenetworking-googleapis-com"
            and not (route.get("tags") or [])
        ):
            observed.add("service_networking")
            continue
        return False
    required: set[str] = set()
    if network_present:
        required.add("default")
    if subnet_present:
        required.add("subnet")
    if service_connection_present:
        required.add("service_networking")
    return required <= observed


def _service_account_roles(
    evidence: Mapping[str, object], service_account_email: str
) -> tuple[set[str], bool]:
    policy = evidence.get("project_iam_policy")
    if not isinstance(policy, Mapping) or not isinstance(policy.get("bindings"), list):
        return set(), False
    member = f"serviceAccount:{service_account_email}"
    roles: set[str] = set()
    exact = True
    for binding in policy["bindings"]:
        if not isinstance(binding, Mapping):
            exact = False
            continue
        members = binding.get("members")
        if not isinstance(members, list) or member not in members:
            continue
        role = binding.get("role")
        if not isinstance(role, str):
            exact = False
            continue
        roles.add(role)
        if binding.get("condition"):
            exact = False
    return roles, exact


def _sql_exact(evidence: Mapping[str, object], spec: FoundationSpec) -> bool:
    instance = evidence.get("planned_sql")
    if not isinstance(instance, Mapping):
        return False
    settings = instance.get("settings")
    if not isinstance(settings, Mapping):
        return False
    ip_config = settings.get("ipConfiguration")
    backup = settings.get("backupConfiguration")
    if not isinstance(ip_config, Mapping) or not isinstance(backup, Mapping):
        return False
    addresses = instance.get("ipAddresses")
    private_ip = ""
    if (
        isinstance(addresses, list)
        and len(addresses) == 1
        and isinstance(addresses[0], Mapping)
        and addresses[0].get("type") == "PRIVATE"
    ):
        private_ip = str(addresses[0].get("ipAddress") or "")
    try:
        private_ip_in_range = ipaddress.ip_address(private_ip) in ipaddress.ip_network(
            spec.private_service_range_cidr
        )
    except ValueError:
        private_ip_in_range = False
    flags = settings.get("databaseFlags") or []
    flag_map = {
        str(item.get("name")): str(item.get("value")).casefold()
        for item in flags
        if isinstance(item, Mapping)
    }
    return bool(
        instance.get("name") == spec.sql_instance
        and instance.get("state") == "RUNNABLE"
        and instance.get("databaseVersion") == "POSTGRES_18"
        and instance.get("region") == spec.region
        and instance.get("gceZone") == spec.zone
        and settings.get("tier") == "db-f1-micro"
        and settings.get("edition") == "ENTERPRISE"
        and settings.get("availabilityType") == "ZONAL"
        and settings.get("activationPolicy") == "ALWAYS"
        and str(settings.get("dataDiskSizeGb")) == "10"
        and settings.get("dataDiskType") == "PD_SSD"
        and settings.get("storageAutoResize") is False
        and settings.get("deletionProtectionEnabled") is True
        and backup.get("enabled") is False
        and backup.get("pointInTimeRecoveryEnabled") in {None, False}
        and private_ip_in_range
        and ip_config.get("ipv4Enabled") is False
        and _ends_with(ip_config.get("privateNetwork"), f"/networks/{spec.network}")
        and ip_config.get("sslMode") == "ENCRYPTED_ONLY"
        and ip_config.get("serverCaMode") == "GOOGLE_MANAGED_INTERNAL_CA"
        and flag_map in ({}, {"cloudsql.iam_authentication": "off"})
    )


def _database_exact(evidence: Mapping[str, object], spec: FoundationSpec) -> bool:
    databases = evidence.get("planned_sql_databases")
    if not isinstance(databases, list):
        return False
    matches = [
        item
        for item in databases
        if isinstance(item, Mapping) and item.get("name") == spec.database
    ]
    return bool(
        len(matches) == 1
        and matches[0].get("charset") == "UTF8"
        and matches[0].get("collation") == "en_US.UTF8"
    )


def _sql_private_ip(evidence: Mapping[str, object]) -> str:
    sql = evidence.get("planned_sql")
    addresses = sql.get("ipAddresses") if isinstance(sql, Mapping) else None
    if not isinstance(addresses, list):
        return ""
    private = [
        str(item.get("ipAddress"))
        for item in addresses
        if isinstance(item, Mapping)
        and item.get("type") == "PRIVATE"
        and item.get("ipAddress")
    ]
    return private[0] if len(private) == 1 else ""


def evaluate(
    evidence: Mapping[str, object], spec: FoundationSpec = FoundationSpec()
) -> dict[str, object]:
    spec.validate()
    service_accounts = _names(evidence.get("service_accounts"), "email")
    sql_instances = _names(evidence.get("sql_instances"))
    enabled_services = _names(evidence.get("enabled_services"), "config.name")
    secret_names = {
        name.rsplit("/", 1)[-1] for name in _names(evidence.get("secrets"))
    }

    network = evidence.get("planned_network")
    subnet = evidence.get("planned_subnet")
    private_range = evidence.get("planned_private_service_range")
    service_connection = evidence.get("planned_service_networking")
    network_present = isinstance(network, Mapping)
    subnet_present = isinstance(subnet, Mapping)
    private_range_present = isinstance(private_range, Mapping)
    service_connection_present = isinstance(service_connection, list) and bool(
        service_connection
    )
    network_exact = _network_exact(evidence, spec) if network_present else False
    subnet_exact = _subnet_exact(evidence, spec) if subnet_present else False
    private_range_exact = (
        _private_range_exact(evidence, spec) if private_range_present else False
    )
    service_connection_exact = (
        _service_connection_exact(evidence, spec)
        if service_connection_present
        else False
    )
    network_subnets = network.get("subnetworks") if network_present else []
    network_peerings = network.get("peerings") if network_present else []
    inventory_consistent = bool(
        (not network_present or isinstance(network_subnets, list))
        and (not network_present or isinstance(network_peerings, list))
        and (not subnet_present or network_present)
        and (not private_range_present or network_present)
        and (not service_connection_present or private_range_present)
        and (not service_connection_present or subnet_present)
        and (not network_present or len(network_subnets or []) == int(subnet_present))
        and (
            not network_present
            or len(network_peerings or []) == int(service_connection_present)
        )
    )
    routes_exact = _routes_safe(
        evidence,
        spec,
        network_present=network_present,
        subnet_present=subnet_present,
        service_connection_present=service_connection_present,
    )

    service_account_present = spec.service_account_email in service_accounts
    allowed_roles = {"roles/logging.logWriter", "roles/monitoring.metricWriter"}
    roles, bindings_exact = _service_account_roles(evidence, spec.service_account_email)
    raw_accounts = evidence.get("service_accounts")
    account_items = raw_accounts if isinstance(raw_accounts, list) else []
    records = [
        item
        for item in account_items
        if isinstance(item, Mapping) and item.get("email") == spec.service_account_email
    ]
    raw_keys = evidence.get("planned_service_account_keys")
    key_items = raw_keys if isinstance(raw_keys, list) else []
    service_account_exact = bool(
        service_account_present
        and len(records) == 1
        and records[0].get("disabled") is not True
        and roles <= allowed_roles
        and bindings_exact
        and not any(
            isinstance(item, Mapping) and item.get("keyType") == "USER_MANAGED"
            for item in key_items
        )
    )
    sql_present = spec.sql_instance in sql_instances
    sql_exact = _sql_exact(evidence, spec) if sql_present else False
    raw_databases = evidence.get("planned_sql_databases")
    database_items = raw_databases if isinstance(raw_databases, list) else []
    database_present = any(
        isinstance(item, Mapping) and item.get("name") == spec.database
        for item in database_items
    )
    database_exact = _database_exact(evidence, spec) if database_present else False
    forbidden_secrets_absent = not (
        set(FORBIDDEN_CANARY_SECRET_NAMES) & secret_names
    )
    dependencies_exact = bool(
        inventory_consistent
        and (not sql_present or service_connection_exact)
        and (not database_present or sql_exact)
        and (service_account_present or not roles)
    )

    checks = [
        Check(
            "identity.active_account",
            bool(evidence.get("active_account")),
            "active gcloud identity",
        ),
        Check(
            "project.exact",
            evidence.get("project") == spec.project,
            "exact target project",
        ),
        Check(
            "apis.foundation_enabled",
            REQUIRED_SERVICES <= enabled_services,
            "phase 1 APIs are enabled without a Secret Manager requirement",
        ),
        Check(
            "resource.network_absent_or_exact",
            not network_present or network_exact,
            "dedicated VPC must be absent or match the complete exact shape",
        ),
        Check(
            "resource.subnet_absent_or_exact",
            not subnet_present or subnet_exact,
            "dedicated regional subnet must be absent or exact",
        ),
        Check(
            "resource.private_service_range_absent_or_exact",
            not private_range_present or private_range_exact,
            "private service range must be absent or exact",
        ),
        Check(
            "resource.service_networking_absent_or_exact",
            not service_connection_present or service_connection_exact,
            "only the exact Service Networking connection is permitted",
        ),
        Check(
            "resource.network_routes_exact",
            routes_exact,
            "only generated default, subnet, and Service Networking routes are allowed",
        ),
        Check(
            "resource.foundation_dependency_order_exact",
            dependencies_exact,
            "partial resources must preserve the exact phase dependency order",
        ),
        Check(
            "resource.service_account_absent_or_exact",
            (not service_account_present and not roles) or service_account_exact,
            "runtime identity must be absent or exact with no user-managed keys",
        ),
        Check(
            "resource.service_account_roles_allowed",
            roles <= allowed_roles,
            "runtime identity may hold only logging and monitoring writer roles",
        ),
        Check(
            "resource.sql_absent_or_exact_ready",
            not sql_present or sql_exact,
            "dedicated Cloud SQL must be absent or exact and RUNNABLE",
        ),
        Check(
            "resource.database_absent_or_exact",
            not database_present or database_exact,
            "canonical database must be absent or exact",
        ),
        Check(
            "resource.canary_secret_names_absent",
            forbidden_secrets_absent,
            "canary credentials must not exist in shared-project Secret Manager",
        ),
    ]
    plan = build_plan(spec)
    satisfied_steps: list[str] = []
    if network_exact:
        satisfied_steps.append("create_isolated_vpc")
    if subnet_exact:
        satisfied_steps.append("create_isolated_subnet")
    if private_range_exact:
        satisfied_steps.append("reserve_private_service_range")
    if service_connection_exact:
        satisfied_steps.append("connect_private_service_networking")
    if service_account_exact:
        satisfied_steps.append("create_runtime_service_account")
    if service_account_exact and "roles/logging.logWriter" in roles:
        satisfied_steps.append("grant_logging_writer")
    if service_account_exact and "roles/monitoring.metricWriter" in roles:
        satisfied_steps.append("grant_monitoring_writer")
    if sql_exact:
        satisfied_steps.append("create_isolated_postgres")
    if database_exact:
        satisfied_steps.append("create_canonical_database")
    return {
        "schema": "muncho-isolated-canary-foundation-preflight.v2",
        "ok": all(check.passed for check in checks),
        "collected_at_unix": evidence.get("collected_at_unix"),
        "plan_sha256": plan.sha256,
        "satisfied_steps": satisfied_steps,
        "spec": asdict(spec),
        "checks": [asdict(check) for check in checks],
        "evidence_summary": {
            "active_account_present": bool(evidence.get("active_account")),
            "project": evidence.get("project"),
            "sql_instance_count": len(sql_instances),
            "service_account_count": len(service_accounts),
            "dedicated_network_present": network_present,
            "dedicated_subnet_present": subnet_present,
            "private_service_range_present": private_range_present,
            "service_networking_connection_present": service_connection_present,
            "sql_private_ip": _sql_private_ip(evidence),
            "forbidden_canary_secret_names_present": sorted(
                set(FORBIDDEN_CANARY_SECRET_NAMES) & secret_names
            ),
        },
    }


CommandRunner = Callable[[Sequence[str]], object]


def _run_json(argv: Sequence[str]) -> object:
    completed = subprocess.run(
        list(argv), check=False, capture_output=True, text=True, timeout=60
    )
    if completed.returncode != 0:
        raise RuntimeError(f"read-only gcloud command failed: {argv[1:3]}")
    return json.loads(completed.stdout or "null")


def collect(
    spec: FoundationSpec = FoundationSpec(),
    *,
    run_json: CommandRunner = _run_json,
) -> dict[str, object]:
    """Collect only read-only list/describe/IAM evidence."""

    project_flag = f"--project={spec.project}"
    active = run_json(
        ("gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=json")
    )
    service_accounts = run_json(
        ("gcloud", "iam", "service-accounts", "list", project_flag, "--format=json")
    )
    sql_instances = run_json(
        ("gcloud", "sql", "instances", "list", project_flag, "--format=json")
    )
    networks = run_json(
        ("gcloud", "compute", "networks", "list", project_flag, "--format=json")
    )
    subnets = run_json(
        (
            "gcloud",
            "compute",
            "networks",
            "subnets",
            "list",
            project_flag,
            "--format=json",
        )
    )
    addresses = run_json(
        (
            "gcloud",
            "compute",
            "addresses",
            "list",
            "--global",
            project_flag,
            "--format=json",
        )
    )
    service_account_names = _names(service_accounts, "email")
    sql_instance_names = _names(sql_instances)
    network_names = _names(networks)
    subnet_names = _names(subnets)
    address_names = _names(addresses)
    evidence: dict[str, object] = {
        "collected_at_unix": int(time.time()),
        "active_account": sorted(_names(active, "account"))[0] if active else "",
        "project": spec.project,
        "enabled_services": run_json(
            ("gcloud", "services", "list", "--enabled", project_flag, "--format=json")
        ),
        "service_accounts": service_accounts,
        "sql_instances": sql_instances,
        # This is an absence audit only. The plan never creates or consumes a
        # canary Secret Manager resource.
        "secrets": run_json(
            ("gcloud", "secrets", "list", project_flag, "--format=json")
        ),
        "networks": networks,
        "subnets": subnets,
        "addresses": addresses,
        "planned_routes": [],
        "project_iam_policy": run_json(
            ("gcloud", "projects", "get-iam-policy", spec.project, "--format=json")
        ),
    }
    if spec.network in network_names:
        evidence["planned_network"] = run_json(
            (
                "gcloud",
                "compute",
                "networks",
                "describe",
                spec.network,
                project_flag,
                "--format=json",
            )
        )
        all_routes = run_json(
            ("gcloud", "compute", "routes", "list", project_flag, "--format=json")
        )
        evidence["planned_routes"] = [
            item
            for item in (all_routes if isinstance(all_routes, list) else [])
            if isinstance(item, Mapping)
            and _ends_with(item.get("network"), f"/networks/{spec.network}")
        ]
        evidence["planned_service_networking"] = run_json(
            (
                "gcloud",
                "services",
                "vpc-peerings",
                "list",
                f"--network={spec.network}",
                "--service=servicenetworking.googleapis.com",
                project_flag,
                "--format=json",
            )
        )
    if spec.subnet in subnet_names:
        evidence["planned_subnet"] = run_json(
            (
                "gcloud",
                "compute",
                "networks",
                "subnets",
                "describe",
                spec.subnet,
                f"--region={spec.region}",
                project_flag,
                "--format=json",
            )
        )
    if spec.private_service_range_name in address_names:
        evidence["planned_private_service_range"] = next(
            item
            for item in (addresses if isinstance(addresses, list) else [])
            if isinstance(item, Mapping)
            and item.get("name") == spec.private_service_range_name
        )
    if spec.service_account_email in service_account_names:
        evidence["planned_service_account_keys"] = run_json(
            (
                "gcloud",
                "iam",
                "service-accounts",
                "keys",
                "list",
                f"--iam-account={spec.service_account_email}",
                project_flag,
                "--format=json",
            )
        )
    if spec.sql_instance in sql_instance_names:
        planned_sql = run_json(
            (
                "gcloud",
                "sql",
                "instances",
                "describe",
                spec.sql_instance,
                project_flag,
                "--format=json",
            )
        )
        evidence["planned_sql"] = planned_sql
        if isinstance(planned_sql, Mapping) and planned_sql.get("state") == "RUNNABLE":
            evidence["planned_sql_databases"] = run_json(
                (
                    "gcloud",
                    "sql",
                    "databases",
                    "list",
                    f"--instance={spec.sql_instance}",
                    project_flag,
                    "--format=json",
                )
            )
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=FoundationSpec().project)
    return parser


def main() -> int:
    args = _parser().parse_args()
    spec = FoundationSpec(project=args.project)
    report = evaluate(collect(spec), spec)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
