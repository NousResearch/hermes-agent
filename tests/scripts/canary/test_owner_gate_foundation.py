from __future__ import annotations

import base64
import copy
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import owner_gate_foundation as gate


NOW = 2_000_000_000
REVISION = "a" * 40
IMAGE = "projects/debian-cloud/global/images/debian-12-bookworm-v20260609"
NETWORK_KEY = Ed25519PrivateKey.generate()
NETWORK_KEY_ID = hashlib.sha256(
    NETWORK_KEY.public_key().public_bytes_raw()
).hexdigest()


def _preexisting_owner_gate_identities() -> tuple[dict, dict]:
    network = f"projects/{gate.PROJECT}/global/networks/{gate.NETWORK_NAME}"
    subnet = {
        "resource_type": "compute_subnetwork",
        "kind": "compute#subnetwork",
        "name": gate.OWNER_GATE_SUBNET_NAME,
        "self_link": (
            f"projects/{gate.PROJECT}/regions/{gate.REGION}/subnetworks/"
            f"{gate.OWNER_GATE_SUBNET_NAME}"
        ),
        "numeric_id": "7031348902426444020",
        "fingerprint": "FzPFdmwu4-E=",
        "creation_timestamp": "2026-07-18T10:55:39.616-07:00",
        "network_self_link": network,
        "region_self_link": (
            f"projects/{gate.PROJECT}/regions/{gate.REGION}"
        ),
        "ip_cidr_range": gate.OWNER_GATE_SUBNET_CIDR,
        "private_ip_google_access": True,
        "stack_type": "IPV4_ONLY",
        "purpose": "PRIVATE",
        "secondary_ip_ranges": [],
        "allow_subnet_cidr_routes_overlap": False,
        "gateway_address": "10.80.3.1",
        "private_ipv6_google_access": "DISABLE_GOOGLE_ACCESS",
    }
    route_name = "default-route-r-a067554b8415d325"
    route = {
        "resource_type": "compute_route",
        "kind": "compute#route",
        "name": route_name,
        "self_link": (
            f"projects/{gate.PROJECT}/global/routes/{route_name}"
        ),
        "numeric_id": "2065908379405385968",
        "creation_timestamp": "2026-07-18T10:55:43.221-07:00",
        "network_self_link": network,
        "destination_range": gate.OWNER_GATE_SUBNET_CIDR,
        "next_hop_network_self_link": network,
        "priority": 0,
        "description": (
            "Default local route to the subnetwork "
            f"{gate.OWNER_GATE_SUBNET_CIDR}."
        ),
        "route_type": "SUBNET",
        "tags": [],
    }
    return subnet, route


def _signed_network_evidence(
    key: Ed25519PrivateKey,
    *,
    collected_at: int = NOW - 1,
    preexisting_owner_subnet: bool = False,
) -> dict:
    body = {
        "schema": gate.NETWORK_EVIDENCE_SCHEMA,
        "collected_at_unix": collected_at,
        "project": gate.PROJECT,
        "zone": gate.ZONE,
        "source_instance": gate.PRODUCTION_SOURCE_VM,
        "source_instance_id": gate.PRODUCTION_SOURCE_VM_ID,
        "source_instance_self_link": (
            f"projects/{gate.PROJECT}/zones/{gate.ZONE}/instances/"
            f"{gate.PRODUCTION_SOURCE_VM}"
        ),
        "source_service_account": gate.PRODUCTION_SOURCE_SERVICE_ACCOUNT,
        "network_self_link": (
            f"projects/{gate.PROJECT}/global/networks/{gate.NETWORK_NAME}"
        ),
        "subnetwork_self_link": (
            f"projects/{gate.PROJECT}/regions/{gate.REGION}/subnetworks/"
            f"{gate.PRODUCTION_SUBNET_NAME}"
        ),
        "source_internal_ip": "10.80.0.2",
        "subnetwork_cidr": "10.80.0.0/24",
        "reserved_network_ranges": [
            "10.80.0.0/24",
            "10.80.1.0/28",
            "10.80.2.0/28",
            "10.71.208.0/24",
        ],
        "preexisting_owner_gate_subnet_identity": None,
        "preexisting_owner_gate_subnet_route_identity": None,
        "range_inventory_receipts": {
            "aggregate_subnets": "1" * 64,
            "routes": "2" * 64,
            "peerings": "3" * 64,
            "private_service_ranges": "4" * 64,
            "serverless_connectors": "5" * 64,
            "network_connectivity_service": gate.sha256_json([]),
            "policy_based_routes": gate.sha256_json([]),
        },
        "network_connectivity_api_disabled": True,
        "private_google_access": False,
        "iap_firewall_rule": "allow-iap-ssh",
        "iap_source_range": gate.IAP_SOURCE_RANGE,
    }
    if preexisting_owner_subnet:
        subnet, route = _preexisting_owner_gate_identities()
        body["preexisting_owner_gate_subnet_identity"] = subnet
        body["preexisting_owner_gate_subnet_route_identity"] = route
    evidence_sha256 = gate.sha256_json(body)
    public_key_id = hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()
    signed = {
        **body,
        "evidence_sha256": evidence_sha256,
        "collector_public_key_id": public_key_id,
    }
    signature = key.sign(
        gate.NETWORK_EVIDENCE_SIGNATURE_DOMAIN
        + gate.canonical_json_bytes(signed)
    )
    return {
        **signed,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode("ascii"),
    }


def _plan(
    *,
    preexisting_owner_subnet: bool = False,
) -> gate.OwnerGateFoundationPlan:
    key = NETWORK_KEY
    key_id = NETWORK_KEY_ID
    evidence = gate.ProductionNetworkEvidence.from_mapping(
        _signed_network_evidence(
            key,
            preexisting_owner_subnet=preexisting_owner_subnet,
        ),
        public_key=key.public_key(),
        expected_public_key_id=key_id,
        now_unix=NOW,
    )
    return gate.build_plan(
        spec=gate.OwnerGateSpec(
            release_revision=REVISION,
            boot_image_self_link=IMAGE,
            package_inventory_sha256="5" * 64,
            interpreter_sha256="6" * 64,
            network_collector_public_key_id=key_id,
            organization_id="123456789012",
            ancestry_evidence_sha256="9" * 64,
            cloud_collector_public_key_id="7" * 64,
            host_collector_public_key_id="8" * 64,
        ),
        network_evidence=evidence,
        network_collector_public_key=key.public_key(),
        now_unix=NOW,
    )


def test_plan_is_private_fixed_ip_and_dedicated_subnet() -> None:
    plan = _plan()
    report = plan.report()
    joined = "\n".join(
        " ".join(step["argv"]) for step in report["foundation_steps"]
    )
    assert "--no-address" in joined
    assert f"--private-network-ip={gate.OWNER_GATE_PRIVATE_IP}" in joined
    assert f"--range={gate.OWNER_GATE_SUBNET_CIDR}" in joined
    assert "--enable-private-ip-google-access" in joined
    assert gate.PRODUCTION_SUBNET_NAME not in " ".join(
        next(
            step["argv"]
            for step in report["foundation_steps"]
            if step["name"] == "create_dedicated_private_owner_gate_subnet"
        )
    )
    assert report["architecture"]["same_production_vpc"] is True
    assert report["architecture"]["same_production_subnet"] is False


def test_plan_binds_preexisting_network_identity_and_disabled_api() -> None:
    subnet, route = _preexisting_owner_gate_identities()
    architecture = _plan(preexisting_owner_subnet=True).report()[
        "architecture"
    ]

    assert architecture["network_evidence_collected_at_unix"] == NOW - 1
    assert architecture["preexisting_owner_gate_subnet_identity"] == subnet
    assert (
        architecture["preexisting_owner_gate_subnet_route_identity"]
        == route
    )
    assert architecture["network_connectivity_api_disabled"] is True


def test_bootstrap_has_no_mutation_binding() -> None:
    plan = _plan().report()
    foundation_text = repr(plan["foundation_steps"])
    deferred_text = repr(plan["deferred_mutation_iam_steps"])
    assert "set-iam-binding-cas" in foundation_text
    assert "add-iam-policy-binding" not in foundation_text
    assert gate.MUTATION_ROLE_ID not in "".join(
        repr(step)
        for step in plan["foundation_steps"]
        if step["name"].startswith("bind_")
    )
    assert gate.READ_ONLY_IAM_ROLE_ID in foundation_text
    assert "add-iam-policy-binding" in deferred_text
    assert gate.MUTATION_ROLE_ID in deferred_text
    assert plan["architecture"]["mutation_iam_enabled_during_bootstrap"] is False
    assert plan["post_binding_validation"]["executor_activation_seal_must_be_absent"] is True


def test_iam_is_exact_disk_and_instance_without_operation_permission() -> None:
    plan = _plan().report()
    assert gate.EXECUTION_PERMISSIONS == (
        "compute.disks.get",
        "compute.disks.resize",
        "compute.instances.get",
        "compute.instances.start",
        "compute.instances.stop",
    )
    assert "zoneOperations" not in repr(plan)
    assert gate.TARGET_DISK in gate.TARGET_DISK_SELF_LINK
    assert "persistent-disk-0" not in gate.TARGET_DISK_SELF_LINK
    assert plan["architecture"]["target_disk_id"] == gate.TARGET_DISK_ID
    assert plan["architecture"]["target_instance_id"] == gate.TARGET_INSTANCE_ID


def test_no_owner_editor_key_or_runtime_shell_surface() -> None:
    plan = _plan().report()
    text = repr(plan)
    assert "roles/owner" not in repr(plan["foundation_steps"])
    assert "roles/editor" not in repr(plan["foundation_steps"])
    assert "service-account key" not in text.lower()
    assert plan["architecture"]["service_account_keys_allowed"] is False
    assert plan["architecture"]["local_gcloud_runtime_fallback"] is False
    assert plan["architecture"]["generic_ssh_runtime_transport"] is False


def test_network_evidence_rejects_overlap() -> None:
    key = Ed25519PrivateKey.generate()
    raw = _signed_network_evidence(key)
    body = {
        k: v
        for k, v in raw.items()
        if k not in {
            "evidence_sha256",
            "collector_public_key_id",
            "signature_ed25519_b64url",
        }
    }
    body["reserved_network_ranges"].append(gate.OWNER_GATE_SUBNET_CIDR)
    body_hash = gate.sha256_json(body)
    key_id = hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()
    signed = {**body, "evidence_sha256": body_hash, "collector_public_key_id": key_id}
    signed["signature_ed25519_b64url"] = base64.urlsafe_b64encode(
        key.sign(
            gate.NETWORK_EVIDENCE_SIGNATURE_DOMAIN
            + gate.canonical_json_bytes(signed)
        )
    ).rstrip(b"=").decode("ascii")
    with pytest.raises(gate.OwnerGateFoundationError):
        gate.ProductionNetworkEvidence.from_mapping(
            signed,
            public_key=key.public_key(),
            expected_public_key_id=key_id,
            now_unix=NOW,
        )


def test_network_evidence_uses_v2_attestation_domain() -> None:
    assert gate.NETWORK_EVIDENCE_SCHEMA == (
        "muncho-owner-gate-production-network-evidence.v2"
    )
    assert gate.NETWORK_EVIDENCE_SIGNATURE_DOMAIN == (
        b"muncho-owner-gate/production-network-evidence/v2\x00"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "subnet_without_route",
        "route_without_subnet",
        "subnet_identity_drift",
        "route_identity_drift",
        "network_connectivity_api_enabled",
    ],
)
def test_network_evidence_rejects_unbound_preexisting_network_state(
    mutation: str,
) -> None:
    key = Ed25519PrivateKey.generate()
    subnet, route = _preexisting_owner_gate_identities()
    raw = _signed_network_evidence(key)
    body = {
        name: value
        for name, value in raw.items()
        if name not in {
            "evidence_sha256",
            "collector_public_key_id",
            "signature_ed25519_b64url",
        }
    }
    if mutation == "subnet_without_route":
        body["preexisting_owner_gate_subnet_identity"] = subnet
    elif mutation == "route_without_subnet":
        body["preexisting_owner_gate_subnet_route_identity"] = route
    elif mutation == "subnet_identity_drift":
        subnet["numeric_id"] = "0"
        body["preexisting_owner_gate_subnet_identity"] = subnet
        body["preexisting_owner_gate_subnet_route_identity"] = route
    elif mutation == "route_identity_drift":
        route["next_hop_network_self_link"] = (
            f"projects/{gate.PROJECT}/global/networks/other"
        )
        body["preexisting_owner_gate_subnet_identity"] = subnet
        body["preexisting_owner_gate_subnet_route_identity"] = route
    else:
        body["network_connectivity_api_disabled"] = False
    evidence_sha256 = gate.sha256_json(body)
    key_id = hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()
    signed = {
        **body,
        "evidence_sha256": evidence_sha256,
        "collector_public_key_id": key_id,
    }
    signed["signature_ed25519_b64url"] = base64.urlsafe_b64encode(
        key.sign(
            gate.NETWORK_EVIDENCE_SIGNATURE_DOMAIN
            + gate.canonical_json_bytes(signed)
        )
    ).rstrip(b"=").decode("ascii")

    with pytest.raises(
        gate.OwnerGateFoundationError,
        match="owner_gate_network_evidence_invalid",
    ):
        gate.ProductionNetworkEvidence.from_mapping(
            signed,
            public_key=key.public_key(),
            expected_public_key_id=key_id,
            now_unix=NOW,
        )


@pytest.mark.parametrize("offset", [gate.PREFLIGHT_MAX_AGE_SECONDS + 1, -1])
def test_network_evidence_rejects_stale_or_future(offset: int) -> None:
    key = Ed25519PrivateKey.generate()
    key_id = hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()
    collected = NOW - offset if offset > 0 else NOW + 1
    with pytest.raises(gate.OwnerGateFoundationError):
        gate.ProductionNetworkEvidence.from_mapping(
            _signed_network_evidence(key, collected_at=collected),
            public_key=key.public_key(),
            expected_public_key_id=key_id,
            now_unix=NOW,
        )


def test_attacker_key_cannot_replace_pinned_network_collector() -> None:
    trusted = Ed25519PrivateKey.generate()
    attacker = Ed25519PrivateKey.generate()
    trusted_id = hashlib.sha256(trusted.public_key().public_bytes_raw()).hexdigest()
    with pytest.raises(gate.OwnerGateFoundationError):
        gate.ProductionNetworkEvidence.from_mapping(
            _signed_network_evidence(attacker),
            public_key=attacker.public_key(),
            expected_public_key_id=trusted_id,
            now_unix=NOW,
        )


def _noncanonical_tail(value: str) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    index = alphabet.index(value[-1])
    assert index % 16 == 0
    return value[:-1] + alphabet[index + 1]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value + "=",
        lambda value: value[:-1] + "!",
        lambda value: value[:-1],
        _noncanonical_tail,
    ],
)
def test_network_evidence_rejects_noncanonical_ed25519_encoding(mutate) -> None:
    key = Ed25519PrivateKey.generate()
    key_id = hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()
    raw = _signed_network_evidence(key)
    raw["signature_ed25519_b64url"] = mutate(
        raw["signature_ed25519_b64url"]
    )

    with pytest.raises(
        gate.OwnerGateFoundationError,
        match="owner_gate_network_evidence_signature_invalid",
    ):
        gate.ProductionNetworkEvidence.from_mapping(
            raw,
            public_key=key.public_key(),
            expected_public_key_id=key_id,
            now_unix=NOW,
        )


def test_rollback_is_bounded_and_removes_iam_first() -> None:
    rollback = _plan().report()["rollback_steps"]
    names = [step["name"] for step in rollback]
    assert names[0] == "remove_exact_mutation_binding_if_present"
    assert names[-1] == "delete_dedicated_service_account_if_created"
    assert all("upstream" not in name and "v1" not in name for name in names)
