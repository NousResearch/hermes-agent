from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_network_evidence_author as author
from scripts.canary import owner_gate_owner_reauth as reauth
from scripts.canary import owner_gate_trust_author as release_author
from scripts.canary import trusted_signer_author as signer_author


RELEASE = "a" * 40
NOW = 2_000_000_000
PREFIX = "https://www.googleapis.com/compute/v1/"


@pytest.fixture(autouse=True)
def _isolated_authority_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority-parent"
    parent.mkdir(mode=0o700)
    key_directory = parent / "owner-gate-release-authority"
    monkeypatch.setattr(release_author, "AUTHORITY_PARENT", parent)
    monkeypatch.setattr(release_author, "KEY_DIRECTORY", key_directory)
    monkeypatch.setattr(
        release_author,
        "MANIFEST_DIRECTORY",
        key_directory / "manifests",
    )
    monkeypatch.setattr(
        signer_author,
        "OBSERVATION_ROOT",
        key_directory / "observation-signers",
    )
    release_author.initialize_keypair()
    signer_author.initialize_observation_keys(
        RELEASE,
        mode="network-bootstrap",
    )


def _network() -> str:
    return (
        f"{PREFIX}projects/{foundation.PROJECT}/global/networks/"
        f"{foundation.NETWORK_NAME}"
    )


def _raw() -> dict[tuple[str, ...], Any]:
    commands = author.inventory_commands()
    network = _network()
    subnet_prefix = f"{PREFIX}projects/{foundation.PROJECT}/regions/"
    values: dict[tuple[str, ...], Any] = {
        commands["instance"]: {
            "id": foundation.PRODUCTION_SOURCE_VM_ID,
            "name": foundation.PRODUCTION_SOURCE_VM,
            "selfLink": (
                f"{PREFIX}projects/{foundation.PROJECT}/zones/"
                f"{foundation.ZONE}/instances/"
                f"{foundation.PRODUCTION_SOURCE_VM}"
            ),
            "networkInterfaces": [{
                "network": network,
                "subnetwork": (
                    f"{subnet_prefix}{foundation.REGION}/subnetworks/"
                    f"{foundation.PRODUCTION_SUBNET_NAME}"
                ),
                "networkIP": "10.80.0.2",
            }],
            "serviceAccounts": [{
                "email": foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT,
                "scopes": ["scope-not-authority-here"],
            }],
        },
        commands["subnets"]: [
            {
                "name": foundation.PRODUCTION_SUBNET_NAME,
                "selfLink": (
                    f"{subnet_prefix}{foundation.REGION}/subnetworks/"
                    f"{foundation.PRODUCTION_SUBNET_NAME}"
                ),
                "network": network,
                "ipCidrRange": "10.80.0.0/24",
                "privateIpGoogleAccess": False,
                "secondaryIpRanges": [],
                "logConfig": {"flowSampling": 0.5},
            },
            {
                "name": "canary",
                "selfLink": f"{subnet_prefix}{foundation.REGION}/subnetworks/canary",
                "network": network,
                "ipCidrRange": "10.80.1.0/28",
                "privateIpGoogleAccess": True,
                "secondaryIpRanges": [],
            },
            {
                "name": "other-network",
                "selfLink": f"{subnet_prefix}{foundation.REGION}/subnetworks/other",
                "network": f"{PREFIX}projects/{foundation.PROJECT}/global/networks/other",
                "ipCidrRange": "10.80.3.0/28",
                "privateIpGoogleAccess": False,
            },
        ],
        commands["routes"]: [
            {
                "name": "default",
                "network": network,
                "destRange": "0.0.0.0/0",
            },
            {
                "name": "public-service",
                "network": network,
                "destRange": "203.0.113.0/24",
            },
            {
                "name": "peer-subnet",
                "network": network,
                "destRange": "10.80.2.0/28",
                "nextHopPeering": "peer-a",
            },
        ],
        commands["peerings"]: [{
            "name": "peer-a",
            "network": foundation.NETWORK_NAME,
            "peerNetwork": "projects/peer/global/networks/peer",
            "state": "ACTIVE",
        }],
        commands["private_service_ranges"]: [
            {
                "name": "managed-services",
                "network": network,
                "purpose": "VPC_PEERING",
                "addressType": "INTERNAL",
                "address": "10.71.208.0",
                "prefixLength": 24,
                "status": "RESERVED",
            },
            {
                "name": "foreign-managed-services",
                "network": (
                    f"{PREFIX}projects/{foundation.PROJECT}/global/networks/"
                    "muncho-canary-vpc"
                ),
                "purpose": "VPC_PEERING",
                "addressType": "INTERNAL",
                "address": "10.80.3.0",
                "prefixLength": 28,
                "status": "RESERVED",
            },
        ],
        commands["regions"]: [
            {"name": "europe-west3", "status": "UP"},
            {"name": "us-central1", "status": "UP"},
        ],
        commands["iap_firewall"]: {
            "name": "allow-iap-ssh",
            "direction": "INGRESS",
            "disabled": False,
            "sourceRanges": [foundation.IAP_SOURCE_RANGE],
            "targetTags": [foundation.IAP_NETWORK_TAG],
            "allowed": [{"IPProtocol": "tcp", "ports": ["22"]}],
        },
        commands["network_connectivity_service"]: [],
        commands["policy_based_routes"]: [],
        author.connector_command("europe-west3"): [{
            "name": "connector-eu",
            "network": foundation.NETWORK_NAME,
            "ipCidrRange": "10.72.0.0/28",
            "state": "READY",
        }],
        author.connector_command("us-central1"): [],
    }
    return values


def _exact_owner_subnet() -> dict[str, Any]:
    return {
        "allowSubnetCidrRoutesOverlap": False,
        "creationTimestamp": "2026-07-18T10:55:39.616-07:00",
        "fingerprint": "FzPFdmwu4-E=",
        "gatewayAddress": "10.80.3.1",
        "id": "7031348902426444020",
        "ipCidrRange": foundation.OWNER_GATE_SUBNET_CIDR,
        "kind": "compute#subnetwork",
        "name": foundation.OWNER_GATE_SUBNET_NAME,
        "network": _network(),
        "privateIpGoogleAccess": True,
        "privateIpv6GoogleAccess": "DISABLE_GOOGLE_ACCESS",
        "purpose": "PRIVATE",
        "region": (
            f"{PREFIX}projects/{foundation.PROJECT}/regions/"
            f"{foundation.REGION}"
        ),
        "selfLink": (
            f"{PREFIX}projects/{foundation.PROJECT}/regions/"
            f"{foundation.REGION}/subnetworks/"
            f"{foundation.OWNER_GATE_SUBNET_NAME}"
        ),
        "stackType": "IPV4_ONLY",
    }


def _exact_owner_subnet_route() -> dict[str, Any]:
    name = "default-route-r-a067554b8415d325"
    return {
        "creationTimestamp": "2026-07-18T10:55:43.221-07:00",
        "description": (
            "Default local route to the subnetwork "
            f"{foundation.OWNER_GATE_SUBNET_CIDR}."
        ),
        "destRange": foundation.OWNER_GATE_SUBNET_CIDR,
        "id": "2065908379405385968",
        "kind": "compute#route",
        "name": name,
        "network": _network(),
        "nextHopNetwork": _network(),
        "priority": 0,
        "selfLink": (
            f"{PREFIX}projects/{foundation.PROJECT}/global/routes/{name}"
        ),
    }


def _with_exact_owner_subnet(
    values: dict[tuple[str, ...], Any],
) -> None:
    commands = author.inventory_commands()
    values[commands["subnets"]].append(_exact_owner_subnet())
    values[commands["routes"]].append(_exact_owner_subnet_route())


def _runner(values: dict[tuple[str, ...], Any], calls: list[tuple[str, ...]]):
    def run(argv: Sequence[str]) -> Any:
        logical = tuple(argv)
        calls.append(logical)
        return copy.deepcopy(values[logical])

    return run


def test_collect_and_author_uses_complete_read_only_inventory_and_signs() -> None:
    values = _raw()
    calls: list[tuple[str, ...]] = []

    result = author._collect_and_author_with_runner(
        release_revision=RELEASE,
        run_json=_runner(values, calls),
        collected_at_unix=NOW,
    )

    public_path = signer_author._public_path(RELEASE, "network")
    expected_key_id = hashlib.sha256(public_path.read_bytes()).hexdigest()
    assert result["collector_public_key_id"] == expected_key_id
    assert result["source_internal_ip"] == "10.80.0.2"
    assert result["reserved_network_ranges"] == [
        "10.71.208.0/24",
        "10.72.0.0/28",
        "10.80.0.0/24",
        "10.80.1.0/28",
        "10.80.2.0/28",
    ]
    assert set(result["range_inventory_receipts"]) == {
        "aggregate_subnets",
        "routes",
        "peerings",
        "private_service_ranges",
        "serverless_connectors",
        "network_connectivity_service",
        "policy_based_routes",
    }
    assert result["network_connectivity_api_disabled"] is True
    assert result["preexisting_owner_gate_subnet_identity"] is None
    assert result["preexisting_owner_gate_subnet_route_identity"] is None
    assert result["range_inventory_receipts"]["aggregate_subnets"] == (
        foundation.sha256_json(
            author._canonical_inventory(
                values[author.inventory_commands()["subnets"]]
            )
        )
    )
    assert result["range_inventory_receipts"]["private_service_ranges"] == (
        foundation.sha256_json(
            author._canonical_inventory(
                values[author.inventory_commands()["private_service_ranges"]]
            )
        )
    )
    assert set(calls) == {
        *author.inventory_commands().values(),
        author.connector_command("europe-west3"),
        author.connector_command("us-central1"),
    }
    assert len(calls) == len(set(calls))


def test_collect_rejects_invalid_route_destination() -> None:
    values = _raw()
    values[author.inventory_commands()["routes"]].append({
        "name": "invalid",
        "network": _network(),
        "destRange": "10.80.0.1/24",
    })

    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_route_invalid",
    ):
        author._collect_and_author_with_runner(
            release_revision=RELEASE,
            run_json=_runner(values, []),
            collected_at_unix=NOW,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_inventory_rejects_non_finite_float(value: float) -> None:
    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_inventory_invalid",
    ):
        author._canonical_inventory({"value": value})


@pytest.mark.parametrize("mutation", ["network_suffix", "host_bits"])
def test_collect_rejects_malformed_target_private_service_range(
    mutation: str,
) -> None:
    values = _raw()
    address = values[author.inventory_commands()["private_service_ranges"]][0]
    if mutation == "network_suffix":
        address["network"] += "/extra"
    else:
        address["address"] = "10.71.208.1"

    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_private_service_ranges_invalid",
    ):
        author._collect_and_author_with_runner(
            release_revision=RELEASE,
            run_json=_runner(values, []),
            collected_at_unix=NOW,
        )


def test_collect_rejects_owner_gate_subnet_overlap() -> None:
    values = _raw()
    values[author.inventory_commands()["subnets"]].append({
        "name": "collision",
        "selfLink": (
            f"{PREFIX}projects/{foundation.PROJECT}/regions/"
            f"{foundation.REGION}/subnetworks/collision"
        ),
        "network": _network(),
        "ipCidrRange": foundation.OWNER_GATE_SUBNET_CIDR,
        "privateIpGoogleAccess": True,
        "secondaryIpRanges": [],
    })

    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_owner_subnet_overlap",
    ):
        author._collect_and_author_with_runner(
            release_revision=RELEASE,
            run_json=_runner(values, []),
            collected_at_unix=NOW,
        )


def test_collect_accepts_exact_preexisting_owner_subnet_and_local_route() -> None:
    values = _raw()
    _with_exact_owner_subnet(values)

    result = author._collect_and_author_with_runner(
        release_revision=RELEASE,
        run_json=_runner(values, []),
        collected_at_unix=NOW,
    )

    assert foundation.OWNER_GATE_SUBNET_CIDR not in (
        result["reserved_network_ranges"]
    )
    commands = author.inventory_commands()
    assert result["range_inventory_receipts"]["aggregate_subnets"] == (
        foundation.sha256_json(
            author._canonical_inventory(values[commands["subnets"]])
        )
    )
    assert result["range_inventory_receipts"]["routes"] == (
        foundation.sha256_json(
            author._canonical_inventory(values[commands["routes"]])
        )
    )
    assert result["preexisting_owner_gate_subnet_identity"] == {
        "resource_type": "compute_subnetwork",
        "kind": "compute#subnetwork",
        "name": foundation.OWNER_GATE_SUBNET_NAME,
        "self_link": (
            f"projects/{foundation.PROJECT}/regions/{foundation.REGION}/"
            f"subnetworks/{foundation.OWNER_GATE_SUBNET_NAME}"
        ),
        "numeric_id": "7031348902426444020",
        "fingerprint": "FzPFdmwu4-E=",
        "creation_timestamp": "2026-07-18T10:55:39.616-07:00",
        "network_self_link": (
            f"projects/{foundation.PROJECT}/global/networks/"
            f"{foundation.NETWORK_NAME}"
        ),
        "region_self_link": (
            f"projects/{foundation.PROJECT}/regions/{foundation.REGION}"
        ),
        "ip_cidr_range": foundation.OWNER_GATE_SUBNET_CIDR,
        "private_ip_google_access": True,
        "stack_type": "IPV4_ONLY",
        "purpose": "PRIVATE",
        "secondary_ip_ranges": [],
        "allow_subnet_cidr_routes_overlap": False,
        "gateway_address": "10.80.3.1",
        "private_ipv6_google_access": "DISABLE_GOOGLE_ACCESS",
    }
    route_name = "default-route-r-a067554b8415d325"
    assert result["preexisting_owner_gate_subnet_route_identity"] == {
        "resource_type": "compute_route",
        "kind": "compute#route",
        "name": route_name,
        "self_link": (
            f"projects/{foundation.PROJECT}/global/routes/{route_name}"
        ),
        "numeric_id": "2065908379405385968",
        "creation_timestamp": "2026-07-18T10:55:43.221-07:00",
        "network_self_link": (
            f"projects/{foundation.PROJECT}/global/networks/"
            f"{foundation.NETWORK_NAME}"
        ),
        "destination_range": foundation.OWNER_GATE_SUBNET_CIDR,
        "next_hop_network_self_link": (
            f"projects/{foundation.PROJECT}/global/networks/"
            f"{foundation.NETWORK_NAME}"
        ),
        "priority": 0,
        "description": (
            "Default local route to the subnetwork "
            f"{foundation.OWNER_GATE_SUBNET_CIDR}."
        ),
        "route_type": "SUBNET",
        "tags": [],
    }


def test_collect_rejects_enabled_network_connectivity_api() -> None:
    values = _raw()
    values[author.inventory_commands()["network_connectivity_service"]] = [{
        "config": {
            "name": "networkconnectivity.googleapis.com",
            "title": "Network Connectivity API",
        },
        "state": "ENABLED",
    }]

    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_policy_based_routes_not_disabled",
    ):
        author._collect_and_author_with_runner(
            release_revision=RELEASE,
            run_json=_runner(values, []),
            collected_at_unix=NOW,
        )


def test_collect_rejects_policy_based_route_asset() -> None:
    values = _raw()
    values[author.inventory_commands()["policy_based_routes"]] = [{
        "assetType": "networkconnectivity.googleapis.com/PolicyBasedRoute",
        "name": (
            "//networkconnectivity.googleapis.com/projects/"
            f"{foundation.PROJECT}/locations/global/policyBasedRoutes/redirect"
        ),
        "resource": {"data": {"priority": 100}},
    }]

    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_policy_based_routes_present",
    ):
        author._collect_and_author_with_runner(
            release_revision=RELEASE,
            run_json=_runner(values, []),
            collected_at_unix=NOW,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "compute#network"),
        ("name", "owner-subnet-name-drift"),
        ("selfLink", f"{PREFIX}projects/other/regions/europe-west3/subnetworks/drift"),
        ("network", f"{PREFIX}projects/{foundation.PROJECT}/global/networks/other"),
        ("region", f"{PREFIX}projects/{foundation.PROJECT}/regions/europe-west2"),
        ("ipCidrRange", "10.80.4.0/28"),
        ("id", "0"),
        ("fingerprint", ""),
        ("privateIpGoogleAccess", False),
        ("stackType", "IPV4_IPV6"),
        ("purpose", "REGIONAL_MANAGED_PROXY"),
        ("secondaryIpRanges", [{"ipCidrRange": "10.80.5.0/28"}]),
        ("allowSubnetCidrRoutesOverlap", True),
        ("gatewayAddress", "10.80.3.2"),
        ("privateIpv6GoogleAccess", "ENABLE_OUTBOUND_VM_ACCESS_TO_GOOGLE"),
    ],
)
def test_collect_rejects_preexisting_owner_subnet_identity_drift(
    field: str,
    value: Any,
) -> None:
    values = _raw()
    subnet = _exact_owner_subnet()
    subnet[field] = value
    commands = author.inventory_commands()
    values[commands["subnets"]].append(subnet)
    values[commands["routes"]].append(_exact_owner_subnet_route())

    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_owner_subnet_invalid",
    ):
        author._collect_and_author_with_runner(
            release_revision=RELEASE,
            run_json=_runner(values, []),
            collected_at_unix=NOW,
        )


def test_collect_rejects_duplicate_exact_owner_subnets() -> None:
    values = _raw()
    commands = author.inventory_commands()
    values[commands["subnets"]].extend([
        _exact_owner_subnet(),
        _exact_owner_subnet(),
    ])
    values[commands["routes"]].append(_exact_owner_subnet_route())

    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_owner_subnet_invalid",
    ):
        author._collect_and_author_with_runner(
            release_revision=RELEASE,
            run_json=_runner(values, []),
            collected_at_unix=NOW,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "compute#forwardingRule"),
        ("name", "custom-owner-route"),
        ("selfLink", f"{PREFIX}projects/{foundation.PROJECT}/global/routes/drift"),
        ("id", "0"),
        ("network", f"{PREFIX}projects/{foundation.PROJECT}/global/networks/other"),
        ("nextHopNetwork", f"{PREFIX}projects/{foundation.PROJECT}/global/networks/other"),
        ("priority", 1),
        ("description", "custom route"),
        ("routeType", "STATIC"),
        ("tags", ["owner"]),
        ("nextHopPeering", "peer-a"),
    ],
)
def test_collect_rejects_preexisting_owner_subnet_route_drift(
    field: str,
    value: Any,
) -> None:
    values = _raw()
    subnet_route = _exact_owner_subnet_route()
    subnet_route[field] = value
    commands = author.inventory_commands()
    values[commands["subnets"]].append(_exact_owner_subnet())
    values[commands["routes"]].append(subnet_route)

    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_owner_subnet_route_invalid",
    ):
        author._collect_and_author_with_runner(
            release_revision=RELEASE,
            run_json=_runner(values, []),
            collected_at_unix=NOW,
        )


@pytest.mark.parametrize("route_count", [0, 2])
def test_collect_rejects_missing_or_duplicate_owner_subnet_route(
    route_count: int,
) -> None:
    values = _raw()
    commands = author.inventory_commands()
    values[commands["subnets"]].append(_exact_owner_subnet())
    values[commands["routes"]].extend(
        _exact_owner_subnet_route() for _ in range(route_count)
    )

    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_owner_subnet_route_invalid",
    ):
        author._collect_and_author_with_runner(
            release_revision=RELEASE,
            run_json=_runner(values, []),
            collected_at_unix=NOW,
        )


def test_collect_exact_owner_subnet_does_not_hide_other_overlap() -> None:
    values = _raw()
    _with_exact_owner_subnet(values)
    commands = author.inventory_commands()
    values[commands["private_service_ranges"]].append({
        "name": "owner-range-collision",
        "network": _network(),
        "purpose": "VPC_PEERING",
        "addressType": "INTERNAL",
        "address": "10.80.3.0",
        "prefixLength": 28,
        "status": "RESERVED",
    })

    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_owner_subnet_overlap",
    ):
        author._collect_and_author_with_runner(
            release_revision=RELEASE,
            run_json=_runner(values, []),
            collected_at_unix=NOW,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda firewall: firewall.update({"sourceRanges": ["0.0.0.0/0"]}),
        lambda firewall: firewall.update({"targetTags": ["all-vms"]}),
        lambda firewall: firewall.update({"disabled": True}),
        lambda firewall: firewall.update(
            {"allowed": [{"IPProtocol": "tcp", "ports": ["22", "80"]}]}
        ),
    ],
)
def test_collect_rejects_iap_firewall_drift(mutation) -> None:
    values = _raw()
    firewall = values[author.inventory_commands()["iap_firewall"]]
    mutation(firewall)

    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_iap_firewall_invalid",
    ):
        author._collect_and_author_with_runner(
            release_revision=RELEASE,
            run_json=_runner(values, []),
            collected_at_unix=NOW,
        )


def test_collect_rejects_network_signing_key_drift() -> None:
    public_path = signer_author._public_path(RELEASE, "network")
    public_path.chmod(0o600)
    public_path.write_bytes(b"Z" * 32)
    public_path.chmod(0o444)

    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_signing_key_invalid",
    ):
        author._collect_and_author_with_runner(
            release_revision=RELEASE,
            run_json=_runner(_raw(), []),
            collected_at_unix=NOW,
        )


def test_collect_rejects_duplicate_or_invalid_region_inventory() -> None:
    values = _raw()
    values[author.inventory_commands()["regions"]] = [
        {"name": "../../escape", "status": "UP"}
    ]
    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_regions_invalid",
    ):
        author._collect_and_author_with_runner(
            release_revision=RELEASE,
            run_json=_runner(values, []),
            collected_at_unix=NOW,
        )


class _Executable:
    def __init__(self, prefix: tuple[str, ...]) -> None:
        self.prefix = prefix

    def trusted_command_prefix(self) -> tuple[str, ...]:
        return self.prefix


class _Configuration:
    account = reauth.OWNER_ACCOUNT

    def __init__(self, root: Path) -> None:
        self.root = root

    def assert_stable(self) -> None:
        return None

    def environment_values(self) -> dict[str, str]:
        return {
            "HOME": str(self.root),
            "CLOUDSDK_CONFIG": str(self.root / ".config" / "gcloud"),
        }


def _runtime(tmp_path: Path) -> tuple[_Executable, _Configuration]:
    python = tmp_path / "python" / "bin" / "python3"
    module = (
        tmp_path
        / f"google-cloud-sdk-{reauth.launcher._GCLOUD_SDK_VERSION}"
        / "lib"
        / "gcloud.py"
    )
    python.parent.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    python.write_bytes(b"#!/bin/sh\nexit 0\n")
    module.write_bytes(b"# sealed gcloud module\n")
    os.chmod(python, 0o700)
    os.chmod(module, 0o600)
    return (
        _Executable((
            str(python),
            *reauth.launcher._GCLOUD_PYTHON_ISOLATION_ARGS,
            str(module),
        )),
        _Configuration(tmp_path),
    )


def _sealed_identity(prefix: tuple[str, ...]) -> dict:
    unsigned = {
        "schema": "muncho-owner-sealed-gcloud-runtime-identity.v1",
        "release_sha": RELEASE,
        "command_prefix_sha256": foundation.sha256_json(list(prefix)),
        "sdk_tree_entries": 10,
        "sdk_tree_bytes": 100,
        "sdk_tree_sha256": "1" * 64,
        "sdk_publication_tree_entries": 10,
        "sdk_publication_tree_bytes": 100,
        "sdk_publication_tree_sha256": "2" * 64,
        "sdk_publication_intent_sha256": "3" * 64,
        "python_version": "3.11.2",
        "python_tree_entries": 4,
        "python_tree_bytes": 40,
        "python_tree_sha256": "4" * 64,
        "owner_support_tree_entries": 5,
        "owner_support_tree_bytes": 50,
        "owner_support_tree_sha256": "5" * 64,
        "owner_support_manifest_sha256": "6" * 64,
        "owner_support_source_tree_oid": "b" * 40,
        "bootstrap_receipt_file_sha256": "7" * 64,
    }
    return {**unsigned, "identity_sha256": foundation.sha256_json(unsigned)}


class _RuntimeRunner:
    def __init__(
        self,
        prefix: tuple[str, ...],
        values: Mapping[tuple[str, ...], Any],
    ) -> None:
        self.prefix = prefix
        self.values = values
        self.calls: list[tuple[tuple[str, ...], dict[str, str], float]] = []

    def run_capture(self, argv, *, env, timeout_seconds):
        command = tuple(argv)
        self.calls.append((command, dict(env), timeout_seconds))
        logical = (
            "gcloud",
            *command[len(self.prefix) : -3],
        )
        return author._CapturedJson(
            0,
            json.dumps(self.values[logical]).encode("ascii"),
            b"",
        )


def test_runtime_collection_uses_only_fixed_prefix_and_closed_environment(
    tmp_path: Path,
) -> None:
    executable, configuration = _runtime(tmp_path)
    runner = _RuntimeRunner(executable.prefix, _raw())

    author._collect_and_author_with_runtime(
        release_revision=RELEASE,
        collected_at_unix=NOW,
        gcloud_executable=executable,
        gcloud_configuration=configuration,
        sealed_runtime_snapshot=lambda: _sealed_identity(executable.prefix),
        runner=runner,
    )

    assert runner.calls
    for argv, environment, timeout in runner.calls:
        assert argv[: len(executable.prefix)] == executable.prefix
        assert argv[-3:] == (
            f"--account={reauth.OWNER_ACCOUNT}",
            f"--configuration={reauth.GCLOUD_CONFIGURATION}",
            "--quiet",
        )
        assert environment["CLOUDSDK_CORE_DISABLE_PROMPTS"] == "1"
        assert environment["CLOUDSDK_CORE_LOG_HTTP_REDACT_TOKEN"] == "1"
        assert not {
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "PYTHONPATH",
            "PYTHONHOME",
            "REQUESTS_CA_BUNDLE",
            "SSL_CERT_FILE",
        } & set(environment)
        assert timeout == author._CAPTURE_TIMEOUT_SECONDS


def test_public_collection_rejects_structural_runtime_fakes(tmp_path: Path) -> None:
    executable, configuration = _runtime(tmp_path)
    with pytest.raises(
        author.OwnerGateNetworkEvidenceAuthorError,
        match="owner_gate_network_runtime_invalid",
    ):
        author.collect_and_author(
            release_revision=RELEASE,
            collected_at_unix=NOW,
            gcloud_executable=executable,  # type: ignore[arg-type]
            gcloud_configuration=configuration,  # type: ignore[arg-type]
        )
