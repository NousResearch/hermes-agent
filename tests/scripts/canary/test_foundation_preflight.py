from __future__ import annotations

import pytest

from scripts.canary.foundation import FoundationSpec, build_plan
from scripts.canary.foundation_preflight import REQUIRED_SERVICES, collect, evaluate


def _evidence():
    return {
        "collected_at_unix": 1_800_000_000,
        "active_account": "operator@example.invalid",
        "project": "adventico-ai-platform",
        "enabled_services": [{"config": {"name": name}} for name in REQUIRED_SERVICES],
        "service_accounts": [],
        "sql_instances": [{"name": "ai-platform-postgres"}],
        "secrets": [{"name": "projects/p/secrets/ai-platform-db-password"}],
        "planned_routes": [],
        "project_iam_policy": {"bindings": []},
    }


def _exact_foundation():
    evidence = _evidence()
    spec = FoundationSpec()
    evidence.update(
        {
            "planned_network": {
                "name": spec.network,
                "autoCreateSubnetworks": False,
                "mtu": 1460,
                "routingConfig": {"routingMode": "REGIONAL"},
                "subnetworks": [
                    f"projects/p/regions/{spec.region}/subnetworks/{spec.subnet}"
                ],
                "peerings": [
                    {
                        "name": "servicenetworking-googleapis-com",
                        "network": "projects/123/global/networks/servicenetworking",
                        "state": "ACTIVE",
                        "exchangeSubnetRoutes": True,
                        "importCustomRoutes": False,
                        "exportCustomRoutes": False,
                        "importSubnetRoutesWithPublicIp": False,
                        "exportSubnetRoutesWithPublicIp": False,
                    }
                ],
            },
            "planned_subnet": {
                "name": spec.subnet,
                "ipCidrRange": spec.subnet_cidr,
                "network": f"projects/p/global/networks/{spec.network}",
                "region": f"projects/p/regions/{spec.region}",
                "privateIpGoogleAccess": False,
                "purpose": "PRIVATE",
                "stackType": "IPV4_ONLY",
                "secondaryIpRanges": [],
            },
            "planned_private_service_range": {
                "name": spec.private_service_range_name,
                "addressType": "INTERNAL",
                "purpose": "VPC_PEERING",
                "status": "RESERVED",
                "address": "10.91.0.0",
                "prefixLength": 24,
                "network": f"projects/p/global/networks/{spec.network}",
            },
            "planned_service_networking": [
                {
                    "peering": "servicenetworking-googleapis-com",
                    "network": f"projects/p/global/networks/{spec.network}",
                    "service": "services/servicenetworking.googleapis.com",
                    "reservedPeeringRanges": [spec.private_service_range_name],
                }
            ],
            "planned_routes": [
                {
                    "network": f"projects/p/global/networks/{spec.network}",
                    "destRange": "0.0.0.0/0",
                    "priority": 1000,
                    "nextHopGateway": (
                        "projects/p/global/gateways/default-internet-gateway"
                    ),
                },
                {
                    "network": f"projects/p/global/networks/{spec.network}",
                    "destRange": spec.subnet_cidr,
                    "priority": 0,
                    "nextHopNetwork": f"projects/p/global/networks/{spec.network}",
                },
                {
                    "network": f"projects/p/global/networks/{spec.network}",
                    "destRange": spec.private_service_range_cidr,
                    "priority": 0,
                    "nextHopPeering": "servicenetworking-googleapis-com",
                },
            ],
            "service_accounts": [
                {"email": spec.service_account_email, "disabled": False}
            ],
            "planned_service_account_keys": [{"keyType": "SYSTEM_MANAGED"}],
            "project_iam_policy": {
                "bindings": [
                    {
                        "role": "roles/logging.logWriter",
                        "members": [f"serviceAccount:{spec.service_account_email}"],
                    },
                    {
                        "role": "roles/monitoring.metricWriter",
                        "members": [f"serviceAccount:{spec.service_account_email}"],
                    },
                ]
            },
            "sql_instances": [
                {"name": "ai-platform-postgres"},
                {"name": spec.sql_instance},
            ],
            "planned_sql": {
                "name": spec.sql_instance,
                "state": "RUNNABLE",
                "databaseVersion": "POSTGRES_18",
                "region": spec.region,
                "gceZone": spec.zone,
                "ipAddresses": [{"type": "PRIVATE", "ipAddress": "10.91.0.3"}],
                "settings": {
                    "tier": "db-f1-micro",
                    "edition": "ENTERPRISE",
                    "availabilityType": "ZONAL",
                    "activationPolicy": "ALWAYS",
                    "dataDiskSizeGb": "10",
                    "dataDiskType": "PD_SSD",
                    "storageAutoResize": False,
                    "deletionProtectionEnabled": True,
                    "backupConfiguration": {
                        "enabled": False,
                        "pointInTimeRecoveryEnabled": False,
                    },
                    "ipConfiguration": {
                        "ipv4Enabled": False,
                        "privateNetwork": f"projects/p/global/networks/{spec.network}",
                        "sslMode": "ENCRYPTED_ONLY",
                        "serverCaMode": "GOOGLE_MANAGED_INTERNAL_CA",
                    },
                    "databaseFlags": [
                        {"name": "cloudsql.iam_authentication", "value": "off"}
                    ],
                },
            },
            "planned_sql_databases": [
                {
                    "name": spec.database,
                    "charset": "UTF8",
                    "collation": "en_US.UTF8",
                }
            ],
        }
    )
    return evidence


def test_absent_v2_resources_are_provisionable_and_bind_exact_plan():
    report = evaluate(_evidence())

    assert report["ok"] is True
    assert report["plan_sha256"] == build_plan().sha256
    assert report["satisfied_steps"] == []
    assert all(check["passed"] for check in report["checks"])


def test_existing_production_resources_do_not_count_as_v2_resources():
    report = evaluate(_evidence())

    assert report["ok"] is True
    assert report["evidence_summary"]["sql_instance_count"] == 1
    assert report["evidence_summary"]["dedicated_network_present"] is False


def test_exact_existing_foundation_satisfies_every_phase_one_step():
    report = evaluate(_exact_foundation())

    assert report["ok"] is True
    assert report["satisfied_steps"] == [step.name for step in build_plan().steps]
    assert report["evidence_summary"]["sql_private_ip"] == "10.91.0.3"


@pytest.mark.parametrize(
    "secret_name",
    ["muncho-canary-db-password", "muncho-canary-discord-bot-token"],
)
def test_any_forbidden_canary_secret_resource_fails_closed(secret_name):
    evidence = _evidence()
    evidence["secrets"].append({"name": f"projects/p/secrets/{secret_name}"})

    report = evaluate(evidence)

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert failed == {"resource.canary_secret_names_absent"}


def test_user_managed_runtime_service_account_key_fails_closed():
    evidence = _exact_foundation()
    evidence["planned_service_account_keys"] = [{"keyType": "USER_MANAGED"}]

    report = evaluate(evidence)

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "resource.service_account_absent_or_exact" in failed


def test_production_vpc_peering_or_route_fails_closed():
    evidence = _exact_foundation()
    evidence["planned_network"]["peerings"].append(
        {
            "name": "prod-peering",
            "network": "projects/p/global/networks/ai-platform-vpc",
            "state": "ACTIVE",
        }
    )
    evidence["planned_routes"].append(
        {
            "network": "projects/p/global/networks/muncho-canary-vpc",
            "destRange": "10.80.0.0/24",
            "priority": 1000,
            "nextHopPeering": "prod-peering",
        }
    )

    report = evaluate(evidence)

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "resource.network_absent_or_exact" in failed
    assert "resource.network_routes_exact" in failed
    assert "resource.foundation_dependency_order_exact" in failed


def test_custom_route_without_peering_also_fails_closed():
    evidence = _exact_foundation()
    evidence["planned_routes"].append(
        {
            "network": "projects/p/global/networks/muncho-canary-vpc",
            "destRange": "203.0.113.0/24",
            "priority": 100,
            "nextHopIp": "10.90.0.9",
        }
    )

    report = evaluate(evidence)

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert failed == {"resource.network_routes_exact"}


def test_sql_private_ip_outside_reserved_range_fails_closed():
    evidence = _exact_foundation()
    evidence["planned_sql"]["ipAddresses"][0]["ipAddress"] = "10.80.0.5"

    report = evaluate(evidence)

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "resource.sql_absent_or_exact_ready" in failed
    assert "resource.database_absent_or_exact" not in failed


def test_secret_manager_is_not_a_phase_one_required_service():
    assert "secretmanager.googleapis.com" not in REQUIRED_SERVICES


def test_collector_lists_before_describing_and_never_mutates_absent_resources():
    calls = []

    def run_json(argv):
        calls.append(tuple(argv))
        if argv[1:3] == ("auth", "list"):
            return [{"account": "operator@example.invalid"}]
        if argv[1:3] == ("services", "list"):
            return [{"config": {"name": name}} for name in REQUIRED_SERVICES]
        if argv[1:3] == ("projects", "get-iam-policy"):
            return {"bindings": []}
        return []

    report = evaluate(collect(run_json=run_json))

    assert report["ok"] is True
    assert not any("describe" in call for call in calls)
    assert not any(
        token in {"create", "delete", "patch", "update", "add-iam-policy-binding"}
        for call in calls
        for token in call
    )
