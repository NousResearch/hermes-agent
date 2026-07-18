from __future__ import annotations

import base64
import copy
import hashlib
import json
import os

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import owner_gate_foundation as gate
from scripts.canary import owner_gate_foundation_apply as foundation_apply
from scripts.canary import owner_gate_owner_reauth as reauth
from scripts.canary import owner_gate_pre_foundation as pre
from scripts.canary import owner_gate_project_ancestry as ancestry
from scripts.canary import owner_gate_trust as trust


NOW = 2_000_000_000
REVISION = "a" * 40
TREE_OID = "b" * 40
IMAGE = "projects/debian-cloud/global/images/debian-12-bookworm-v20260609"
IMAGE_NUMERIC_ID = "1234567890123456789"
INTERPRETER_SHA256 = "6" * 64
ORGANIZATION_ID = "123456789012"
PROJECT_NUMBER = "234567890123"
FOLDER_ID = "345678901234"
RELEASE_KEY = Ed25519PrivateKey.generate()
RELEASE_KEY_ID = hashlib.sha256(
    RELEASE_KEY.public_key().public_bytes_raw()
).hexdigest()
NETWORK_KEY = Ed25519PrivateKey.generate()
NETWORK_KEY_ID = hashlib.sha256(
    NETWORK_KEY.public_key().public_bytes_raw()
).hexdigest()


def _preexisting_owner_gate_identities() -> tuple[dict, dict]:
    network = f"projects/{gate.PROJECT}/global/networks/{gate.NETWORK_NAME}"
    region = f"projects/{gate.PROJECT}/regions/{gate.REGION}"
    subnet = {
        "resource_type": "compute_subnetwork",
        "kind": "compute#subnetwork",
        "name": gate.OWNER_GATE_SUBNET_NAME,
        "self_link": (
            f"{region}/subnetworks/{gate.OWNER_GATE_SUBNET_NAME}"
        ),
        "numeric_id": "2222222222222222222",
        "fingerprint": "fingerprint-subnet-1",
        "creation_timestamp": "2026-07-18T10:55:39.616-07:00",
        "network_self_link": network,
        "region_self_link": region,
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


@pytest.fixture(autouse=True)
def _pin_release_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        RELEASE_KEY_ID,
    )


def _signed_network_evidence(
    key: Ed25519PrivateKey = NETWORK_KEY,
    *,
    collected_at_unix: int = NOW - 1,
    preexisting_owner_subnet: bool = False,
) -> dict:
    body = {
        "schema": gate.NETWORK_EVIDENCE_SCHEMA,
        "collected_at_unix": collected_at_unix,
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
    key_id = hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()
    signed = {
        **body,
        "evidence_sha256": evidence_sha256,
        "collector_public_key_id": key_id,
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


def _owner_reauth_receipt(*, expires_at_unix: int = NOW + 300) -> dict:
    body = {
        "schema": reauth.RECEIPT_SCHEMA,
        "purpose": reauth.RECEIPT_PURPOSE,
        "trusted_runtime_identity": {
            "release_revision": REVISION,
            "sealed_runtime_identity_sha256": "0" * 64,
            "command_prefix_sha256": "1" * 64,
            "python_executable_sha256": "2" * 64,
            "gcloud_module_sha256": "3" * 64,
            "sdk_root": (
                "/sealed/google-cloud-sdk-"
                f"{reauth.launcher._GCLOUD_SDK_VERSION}"
            ),
            "sdk_python_config_identity_sha256": "4" * 64,
            "closed_environment_sha256": "5" * 64,
            "configuration": reauth.GCLOUD_CONFIGURATION,
            "account": reauth.OWNER_ACCOUNT,
            "project": gate.PROJECT,
            "zone": gate.ZONE,
        },
        "interactive_reauthentication": {
            "method": "gcloud_auth_login_force_interactive",
            "started_at_unix": NOW - 2,
            "completed_at_unix": NOW - 1,
            "command_sha256": "6" * 64,
            "interactive_tty_verified": True,
            "access_token_requested": False,
            "credential_material_captured": False,
        },
        "authenticated_probe": {
            "command_sha256": "7" * 64,
            "output_sha256": "8" * 64,
            "project_id": gate.PROJECT,
            "project_number": PROJECT_NUMBER,
        },
        "issued_at_unix": NOW,
        "expires_at_unix": expires_at_unix,
        "signer_key_id": RELEASE_KEY_ID,
    }
    return dict(
        reauth._sign_owner_reauth_receipt(body, private_key=RELEASE_KEY)
    )


def _signed_ancestry_raw(
    *,
    owner_expires_at_unix: int = NOW + 300,
    key: Ed25519PrivateKey = NETWORK_KEY,
) -> bytes:
    reauth_receipt = _owner_reauth_receipt(
        expires_at_unix=owner_expires_at_unix
    )
    chain = [
        {
            "resource_type": "project",
            "resource_name": f"projects/{PROJECT_NUMBER}",
            "numeric_id": PROJECT_NUMBER,
            "display_name": "Adventico AI Platform",
            "state": "ACTIVE",
            "etag": "etag-project-1",
            "parent_resource_name": f"folders/{FOLDER_ID}",
        },
        {
            "resource_type": "folder",
            "resource_name": f"folders/{FOLDER_ID}",
            "numeric_id": FOLDER_ID,
            "display_name": "Adventico Production",
            "state": "ACTIVE",
            "etag": "etag-folder-1",
            "parent_resource_name": f"organizations/{ORGANIZATION_ID}",
        },
        {
            "resource_type": "organization",
            "resource_name": f"organizations/{ORGANIZATION_ID}",
            "numeric_id": ORGANIZATION_ID,
            "display_name": "Adventico",
            "state": "ACTIVE",
            "etag": "etag-organization-1",
            "parent_resource_name": None,
        },
    ]
    chain_sha = gate.sha256_json(chain)
    consistency_token = gate.sha256_json([
        {
            "resource_name": item["resource_name"],
            "state": item["state"],
            "etag": item["etag"],
            "parent_resource_name": item["parent_resource_name"],
        }
        for item in chain
    ])
    key_id = hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()
    body = {
        "schema": ancestry.EVIDENCE_SCHEMA,
        "purpose": ancestry.EVIDENCE_PURPOSE,
        "release_revision": REVISION,
        "project_id": gate.PROJECT,
        "project_number": PROJECT_NUMBER,
        "project_resource_name": f"projects/{PROJECT_NUMBER}",
        "organization_id": ORGANIZATION_ID,
        "organization_resource_name": f"organizations/{ORGANIZATION_ID}",
        "ordered_chain": chain,
        "stable_chain_sha256": chain_sha,
        "stable_reads": [
            {
                "ordinal": ordinal,
                "chain_sha256": chain_sha,
                "provider_consistency_token_sha256": consistency_token,
            }
            for ordinal in (1, 2)
        ],
        "collected_at_unix": NOW,
        "expires_at_unix": owner_expires_at_unix,
        "owner_reauthentication_receipt_sha256": reauth_receipt[
            "owner_reauthentication_receipt_sha256"
        ],
        "collector_public_key_id": key_id,
    }
    signed_payload = {**body, "evidence_sha256": gate.sha256_json(body)}
    result = {
        **signed_payload,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(
            key.sign(
                ancestry.SIGNATURE_DOMAIN
                + gate.canonical_json_bytes(signed_payload)
            )
        )
        .rstrip(b"=")
        .decode("ascii"),
    }
    return gate.canonical_json_bytes(result)


def _evidence(
    key: Ed25519PrivateKey = NETWORK_KEY,
    *,
    collected_at_unix: int = NOW - 1,
    preexisting_owner_subnet: bool = False,
) -> gate.ProductionNetworkEvidence:
    key_id = hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()
    return gate.ProductionNetworkEvidence.from_mapping(
        _signed_network_evidence(
            key,
            collected_at_unix=collected_at_unix,
            preexisting_owner_subnet=preexisting_owner_subnet,
        ),
        public_key=key.public_key(),
        expected_public_key_id=key_id,
        now_unix=NOW,
    )


def _plan(
    *,
    tree_oid: str = TREE_OID,
    image_numeric_id: str = IMAGE_NUMERIC_ID,
    interpreter_sha256: str = INTERPRETER_SHA256,
    evidence: gate.ProductionNetworkEvidence | None = None,
    ancestry_raw: bytes | None = None,
) -> gate.OwnerGateFoundationPlan:
    evidence = _evidence() if evidence is None else evidence
    ancestry_raw = (
        _signed_ancestry_raw() if ancestry_raw is None else ancestry_raw
    )
    return gate.build_plan(
        spec=gate.OwnerGateSpec(
            release_revision=REVISION,
            source_tree_oid=tree_oid,
            boot_image_self_link=IMAGE,
            boot_image_numeric_id=image_numeric_id,
            interpreter_sha256=interpreter_sha256,
            network_collector_public_key_id=NETWORK_KEY_ID,
            organization_id=ORGANIZATION_ID,
            ancestry_evidence_sha256=hashlib.sha256(
                ancestry_raw
            ).hexdigest(),
        ),
        network_evidence=evidence,
        network_collector_public_key=NETWORK_KEY.public_key(),
        now_unix=NOW,
    )


def _authority(
    *,
    plan: gate.OwnerGateFoundationPlan | None = None,
    evidence: gate.ProductionNetworkEvidence | None = None,
    expires_at_unix: int = NOW + 300,
    owner_expires_at_unix: int = NOW + 300,
) -> tuple[dict, gate.OwnerGateFoundationPlan, gate.ProductionNetworkEvidence]:
    evidence = _evidence() if evidence is None else evidence
    ancestry_raw = _signed_ancestry_raw(
        owner_expires_at_unix=owner_expires_at_unix
    )
    plan = (
        _plan(evidence=evidence, ancestry_raw=ancestry_raw)
        if plan is None
        else plan
    )
    body = pre.build_authority_body(
        plan=plan,
        network_evidence=evidence,
        network_collector_public_key=NETWORK_KEY.public_key(),
        project_ancestry_evidence_raw=ancestry_raw,
        project_ancestry_collector_public_key=NETWORK_KEY.public_key(),
        owner_reauthentication_receipt=_owner_reauth_receipt(
            expires_at_unix=owner_expires_at_unix
        ),
        owner_reauthentication_public_key=RELEASE_KEY.public_key(),
        issued_at_unix=NOW,
        expires_at_unix=expires_at_unix,
        signer_key_id=RELEASE_KEY_ID,
    )
    return (
        dict(
            pre.sign_pre_foundation_authority(
                body,
                private_key=RELEASE_KEY,
                owner_reauthentication_receipt=_owner_reauth_receipt(
                    expires_at_unix=owner_expires_at_unix
                ),
                project_ancestry_evidence_raw=ancestry_raw,
                project_ancestry_collector_public_key=(
                    NETWORK_KEY.public_key()
                ),
            )
        ),
        plan,
        evidence,
    )


def _resource_identity(
    step_name: str,
    plan: gate.OwnerGateFoundationPlan,
) -> dict:
    spec = plan.spec
    provider = "https://www.googleapis.com/compute/v1/"
    network = provider + (
        f"projects/{gate.PROJECT}/global/networks/{gate.NETWORK_NAME}"
    )
    subnet = provider + (
        f"projects/{gate.PROJECT}/regions/{gate.REGION}/subnetworks/"
        f"{gate.OWNER_GATE_SUBNET_NAME}"
    )
    if step_name == "create_dedicated_service_account":
        return {
            "resource_type": "iam_service_account",
            "resource_name": (
                f"projects/{gate.PROJECT}/serviceAccounts/"
                f"{spec.service_account_email}"
            ),
            "email": spec.service_account_email,
            "unique_id": "111111111111111111111",
            "etag": "etag-SA-1",
            "disabled": False,
            "user_managed_key_count": 0,
            "user_managed_keys": [],
        }
    role = {
        "create_narrow_iam_observation_reader_role": (
            "project_custom_role",
            spec.read_only_iam_role,
            gate.PROJECT_READ_ROLE_TITLE,
            gate.PROJECT_READ_ROLE_DESCRIPTION,
            list(gate.READ_ONLY_IAM_PERMISSIONS),
        ),
        "create_narrow_storage_executor_role": (
            "project_custom_role",
            spec.custom_role,
            gate.MUTATION_ROLE_TITLE,
            gate.MUTATION_ROLE_DESCRIPTION,
            list(gate.MUTATION_PERMISSIONS),
        ),
        "create_narrow_organization_iam_observation_reader_role": (
            "organization_custom_role",
            spec.ancestor_read_only_iam_role,
            gate.ANCESTOR_READ_ROLE_TITLE,
            gate.ANCESTOR_READ_ROLE_DESCRIPTION,
            list(gate.DIRECT_IAM_ANCESTOR_PERMISSIONS),
        ),
    }.get(step_name)
    if role is not None:
        kind, name, title, description, permissions = role
        return {
            "resource_type": kind,
            "name": name,
            "etag": "etag-role-1",
            "title": title,
            "description": description,
            "stage": "GA",
            "included_permissions": permissions,
            "deleted": False,
        }
    binding = {
        "bind_narrow_iam_observation_reader_to_owner_gate_service_account": (
            "project_iam_binding",
            f"projects/{gate.PROJECT}",
            spec.read_only_iam_role,
        ),
        "bind_narrow_organization_iam_observation_reader_to_owner_gate_service_account": (
            "organization_iam_binding",
            spec.organization_resource,
            spec.ancestor_read_only_iam_role,
        ),
    }.get(step_name)
    if binding is not None:
        kind, resource_name, bound_role = binding
        return {
            "resource_type": kind,
            "resource_name": resource_name,
            "role": bound_role,
            "member": f"serviceAccount:{spec.service_account_email}",
            "condition": None,
            "policy_etag": "etag-policy-1",
            "policy_version": 3,
            "matching_binding_count": 1,
            "matching_member_occurrences": 1,
            "binding_members": [
                f"serviceAccount:{spec.service_account_email}"
            ],
        }
    if step_name == "create_dedicated_private_owner_gate_subnet":
        subnet_identity, route_identity = _preexisting_owner_gate_identities()
        subnet_identity = dict(subnet_identity)
        for field in (
            "self_link",
            "network_self_link",
            "region_self_link",
        ):
            subnet_identity[field] = provider + subnet_identity[field]
        route_identity = dict(route_identity)
        for field in (
            "self_link",
            "network_self_link",
            "next_hop_network_self_link",
        ):
            route_identity[field] = provider + route_identity[field]
        return {
            **subnet_identity,
            "local_route_identity": route_identity,
        }
    if step_name == "create_private_owner_gate_vm":
        return {
            "resource_type": "compute_instance",
            "name": gate.VM_NAME,
            "self_link": provider
            + (
                f"projects/{gate.PROJECT}/zones/{gate.ZONE}/instances/"
                f"{gate.VM_NAME}"
            ),
            "numeric_id": "3333333333333333333",
            "metadata_fingerprint": "fingerprint-vm-1",
            "machine_type": provider
            + (
                f"projects/{gate.PROJECT}/zones/{gate.ZONE}/machineTypes/"
                f"{gate.MACHINE_TYPE}"
            ),
            "network_self_link": network,
            "network_numeric_id": "6666666666666666666",
            "internal_ip": gate.OWNER_GATE_PRIVATE_IP,
            "subnetwork_self_link": subnet,
            "subnetwork_numeric_id": "2222222222222222222",
            "network_stack_type": "IPV4_ONLY",
            "access_configs": [],
            "service_account_email": spec.service_account_email,
            "deletion_protection": False,
            "boot_image_numeric_id": IMAGE_NUMERIC_ID,
            "boot_image_self_link": provider + IMAGE,
            "boot_image_architecture": "X86_64",
            "boot_image_license_self_links": [
                provider
                + "projects/debian-cloud/global/licenses/debian-12-bookworm"
            ],
            "tags": sorted([
                gate.IAP_NETWORK_TAG,
                gate.OWNER_GATE_NETWORK_TAG,
            ]),
            "metadata": {
                "block-project-ssh-keys": "TRUE",
                "enable-oslogin": "TRUE",
                "serial-port-enable": "FALSE",
            },
            "shielded_instance_config": {
                "enable_integrity_monitoring": True,
                "enable_secure_boot": True,
                "enable_vtpm": True,
            },
            "oauth_scopes": sorted(gate.OWNER_GATE_OAUTH_SCOPES),
            "can_ip_forward": False,
            "maintenance_policy": "MIGRATE",
            "provisioning_model": "STANDARD",
            "automatic_restart": True,
            "preemptible": False,
            "instance_termination_action": "DELETE",
            "network_interface_count": 1,
            "alias_ip_ranges": [],
            "creation_timestamp": "2026-07-17T00:00:00.000-07:00",
            "labels": {},
            "resource_policies": [],
            "min_cpu_platform": "Automatic",
            "confidential_instance_config": {
                "enable_confidential_compute": False
            },
            "boot_disk_name": gate.VM_NAME,
            "boot_disk_self_link": provider
            + (
                f"projects/{gate.PROJECT}/zones/{gate.ZONE}/disks/"
                f"{gate.VM_NAME}"
            ),
            "boot_disk_numeric_id": "5555555555555555555",
            "boot_disk_size_gb": gate.BOOT_DISK_SIZE_GB,
            "boot_disk_type_self_link": provider
            + (
                f"projects/{gate.PROJECT}/zones/{gate.ZONE}/diskTypes/"
                f"{gate.BOOT_DISK_TYPE}"
            ),
            "boot_disk_auto_delete": True,
            "boot_disk_boot": True,
            "boot_disk_mode": "READ_WRITE",
            "boot_disk_interface": "SCSI",
            "boot_disk_attachment_type": "PERSISTENT",
            "boot_disk_attachment_index": 0,
        }
    assert step_name == "allow_private_web_upstream_from_current_caddy_host"
    name = "muncho-owner-gate-web-from-production"
    return {
        "resource_type": "compute_firewall",
        "name": name,
        "self_link": provider
        + f"projects/{gate.PROJECT}/global/firewalls/{name}",
        "numeric_id": "4444444444444444444",
        "creation_timestamp": "2026-07-18T10:57:39.516-07:00",
        "network_self_link": network,
        "direction": "INGRESS",
        "priority": 700,
        "disabled": False,
        "action": "ALLOW",
        "allowed": [{
            "ip_protocol": "tcp",
            "ports": [str(gate.WEB_LISTEN_PORT)],
        }],
        "denied": [],
        "source_ranges": [],
        "destination_ranges": [],
        "source_tags": [],
        "target_tags": [],
        "source_service_accounts": [
            gate.PRODUCTION_SOURCE_SERVICE_ACCOUNT
        ],
        "target_service_accounts": [spec.service_account_email],
        "log_config": {"enable": True},
    }


def _step_receipts(plan: gate.OwnerGateFoundationPlan) -> list[dict]:
    return [
        {
            "step_name": step.name,
            "argv_sha256": gate.sha256_json(list(step.argv)),
            "disposition": "created",
            "operation_receipt_sha256": hashlib.sha256(
                f"operation:{step.name}".encode("ascii")
            ).hexdigest(),
            "postcondition_receipt_sha256": hashlib.sha256(
                f"postcondition:{step.name}".encode("ascii")
            ).hexdigest(),
            "resource_identity": _resource_identity(step.name, plan),
        }
        for index, step in enumerate(plan.foundation_steps)
    ]


def _apply_receipt(
    authority: dict,
    plan: gate.OwnerGateFoundationPlan,
) -> dict:
    execution = foundation_apply._ProviderExecutionResult(
        step_receipts=tuple(_step_receipts(plan)),
        started_at_unix=NOW + 10,
        completed_at_unix=NOW + 20,
    )
    return dict(pre._sign_foundation_apply_execution(
        execution,
        private_key=RELEASE_KEY,
        authority=authority,
        owner_reauthentication_receipt=_owner_reauth_receipt(),
        project_ancestry_evidence_raw=_signed_ancestry_raw(),
        project_ancestry_collector_public_key=NETWORK_KEY.public_key(),
        plan=plan,
    ))


def _apply_body(
    authority: dict,
    plan: gate.OwnerGateFoundationPlan,
    *,
    steps: list[dict] | None = None,
) -> dict:
    execution = foundation_apply._ProviderExecutionResult(
        step_receipts=tuple(_step_receipts(plan) if steps is None else steps),
        started_at_unix=NOW + 10,
        completed_at_unix=NOW + 20,
    )
    return dict(pre._build_apply_receipt_body_from_execution(
        execution=execution,
        authority=authority,
        owner_reauthentication_receipt=_owner_reauth_receipt(),
        owner_reauthentication_public_key=RELEASE_KEY.public_key(),
        project_ancestry_evidence_raw=_signed_ancestry_raw(),
        project_ancestry_collector_public_key=NETWORK_KEY.public_key(),
        plan=plan,
    ))


def _b64url(signature: bytes) -> str:
    return base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")


def _ancestry_validation_kwargs() -> dict:
    return {
        "project_ancestry_evidence_raw": _signed_ancestry_raw(),
        "project_ancestry_collector_public_key": NETWORK_KEY.public_key(),
    }


def test_authority_binds_exact_cycle_breaking_inputs_without_final_package() -> None:
    authority, plan, evidence = _authority()

    checked = pre.validate_pre_foundation_authority(
        authority,
        public_key=RELEASE_KEY.public_key(),
        owner_reauthentication_receipt=_owner_reauth_receipt(),
        now_unix=NOW + 1,
        expected_plan=plan,
        network_evidence=evidence,
        network_collector_public_key=NETWORK_KEY.public_key(),
        **_ancestry_validation_kwargs(),
    )
    projection = pre.inert_plan_projection(plan)

    assert checked["foundation_source_revision"] == REVISION
    assert checked["foundation_source_tree_oid"] == TREE_OID
    assert checked["inert_plan_sha256"] == pre.inert_plan_sha256(plan)
    assert checked["network_evidence_sha256"] == evidence.evidence_sha256
    assert checked["network_collector_public_key_id"] == NETWORK_KEY_ID
    assert checked["interpreter_image"] == {
        "project": "debian-cloud",
        "image_name": IMAGE.rsplit("/", 1)[-1],
        "image_numeric_id": IMAGE_NUMERIC_ID,
        "image_self_link": "https://www.googleapis.com/compute/v1/" + IMAGE,
        "python_version": "3.11.2",
        "interpreter_sha256": INTERPRETER_SHA256,
    }
    assert checked["owner_reauthentication"] == {
        "account": pre.OWNER_ACCOUNT,
        "receipt_sha256": _owner_reauth_receipt()[
            "owner_reauthentication_receipt_sha256"
        ],
        "expires_at_unix": NOW + 300,
    }
    assert checked["organization_id"] == ORGANIZATION_ID
    assert checked["ancestry_evidence_sha256"] == hashlib.sha256(
        _signed_ancestry_raw()
    ).hexdigest()
    assert checked["project_number"] == PROJECT_NUMBER
    assert checked["ancestry_collector_public_key_id"] == NETWORK_KEY_ID
    assert checked["mutation_iam_binding_authorized"] is False
    assert checked["package_deployment_authorized"] is False
    assert checked["service_start_authorized"] is False
    assert checked["final_package_inventory_present"] is False
    assert projection["deferred_private_api_connectivity_steps"] == []
    assert projection["deferred_mutation_iam_steps_authorized"] is False
    assert "package_inventory_sha256" not in projection["spec"]
    assert "cloud_collector_public_key_id" not in projection["spec"]
    assert "host_collector_public_key_id" not in projection["spec"]
    assert "owner_gate_vm_numeric_id" not in repr(checked)
    assert "service_account_unique_id" not in repr(checked)


def test_authority_requires_fixed_pin_and_separate_signature_domain() -> None:
    authority, plan, _ = _authority()
    attacker = Ed25519PrivateKey.generate()

    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_pre_foundation_signer_not_pinned",
    ):
        pre.validate_pre_foundation_authority(
            authority,
            public_key=attacker.public_key(),
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            now_unix=NOW,
            expected_plan=plan,
            **_ancestry_validation_kwargs(),
        )

    swapped = dict(authority)
    signed_payload = {
        key: value
        for key, value in swapped.items()
        if key != "signature_ed25519_b64url"
    }
    swapped["signature_ed25519_b64url"] = _b64url(
        RELEASE_KEY.sign(
            pre.APPLY_RECEIPT_SIGNATURE_DOMAIN
            + gate.canonical_json_bytes(signed_payload)
        )
    )
    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_pre_foundation_signature_invalid",
    ):
        pre.validate_pre_foundation_authority(
            swapped,
            public_key=RELEASE_KEY.public_key(),
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            now_unix=NOW,
            expected_plan=plan,
            **_ancestry_validation_kwargs(),
        )


@pytest.mark.parametrize(
    "field",
    [
        "mutation_iam_binding_authorized",
        "package_deployment_authorized",
        "service_start_authorized",
        "final_package_inventory_present",
    ],
)
def test_authority_cannot_sign_any_unsafe_permission(field: str) -> None:
    authority, plan, evidence = _authority()
    body = {
        key: value
        for key, value in authority.items()
        if key
        not in {
            "pre_foundation_authority_sha256",
            "signature_ed25519_b64url",
        }
    }
    body[field] = True

    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_pre_foundation_authority_invalid",
    ):
        pre.sign_pre_foundation_authority(
            body,
            private_key=RELEASE_KEY,
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            **_ancestry_validation_kwargs(),
        )

    assert evidence.evidence_sha256 == plan.network_evidence_sha256


def test_authority_rejects_plan_drift_and_extra_final_package_field() -> None:
    authority, plan, _ = _authority()
    drifted = _plan(interpreter_sha256="9" * 64)

    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_pre_foundation_plan_mismatch",
    ):
        pre.validate_pre_foundation_authority(
            authority,
            public_key=RELEASE_KEY.public_key(),
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            now_unix=NOW,
            expected_plan=drifted,
            **_ancestry_validation_kwargs(),
        )

    injected = dict(authority)
    injected["package_inventory_sha256"] = "0" * 64
    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_pre_foundation_authority_invalid",
    ):
        pre.validate_pre_foundation_authority(
            injected,
            public_key=RELEASE_KEY.public_key(),
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            now_unix=NOW,
            expected_plan=plan,
            **_ancestry_validation_kwargs(),
        )


def test_authority_rejects_wrong_network_evidence_or_collector_key() -> None:
    authority, plan, _ = _authority()
    different_evidence = _evidence(collected_at_unix=NOW - 2)
    attacker = Ed25519PrivateKey.generate()

    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_pre_foundation_network_evidence_mismatch",
    ):
        pre.validate_pre_foundation_authority(
            authority,
            public_key=RELEASE_KEY.public_key(),
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            now_unix=NOW,
            expected_plan=plan,
            network_evidence=different_evidence,
            network_collector_public_key=NETWORK_KEY.public_key(),
            **_ancestry_validation_kwargs(),
        )

    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_pre_foundation_network_evidence_invalid",
    ):
        pre.validate_pre_foundation_authority(
            authority,
            public_key=RELEASE_KEY.public_key(),
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            now_unix=NOW,
            expected_plan=plan,
            network_evidence=_evidence(),
            network_collector_public_key=attacker.public_key(),
            **_ancestry_validation_kwargs(),
        )


def test_authority_rejects_expiry_bad_owner_expiry_and_bad_image_id() -> None:
    authority, plan, evidence = _authority()
    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_pre_foundation_authority_expired",
    ):
        pre.validate_pre_foundation_authority(
            authority,
            public_key=RELEASE_KEY.public_key(),
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            now_unix=NOW + 301,
            expected_plan=plan,
            **_ancestry_validation_kwargs(),
        )

    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_pre_foundation_authority_invalid",
    ):
        _authority(
            owner_expires_at_unix=NOW + 299,
        )

    with pytest.raises(
        gate.OwnerGateFoundationError,
        match="owner_gate_spec_invalid",
    ):
        _plan(image_numeric_id="123")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value + "=",
        lambda value: value[:-1],
        lambda value: value[:-1] + "!",
    ],
)
def test_authority_rejects_noncanonical_signature_and_json(mutate) -> None:
    authority, plan, _ = _authority()
    authority["signature_ed25519_b64url"] = mutate(
        authority["signature_ed25519_b64url"]
    )
    with pytest.raises(pre.OwnerGatePreFoundationError):
        pre.validate_pre_foundation_authority(
            authority,
            public_key=RELEASE_KEY.public_key(),
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            now_unix=NOW,
            expected_plan=plan,
            **_ancestry_validation_kwargs(),
        )

    valid, _, _ = _authority(plan=plan)
    pretty = json.dumps(valid, indent=2, sort_keys=True).encode("utf-8")
    with pytest.raises(pre.OwnerGatePreFoundationError):
        pre.decode_canonical_authority(
            pretty,
            public_key=RELEASE_KEY.public_key(),
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            now_unix=NOW,
            expected_plan=plan,
            **_ancestry_validation_kwargs(),
        )


def test_apply_receipt_is_signed_exact_complete_and_inert() -> None:
    authority, plan, _ = _authority()
    receipt = _apply_receipt(authority, plan)

    checked = pre.validate_foundation_apply_receipt(
        receipt,
        public_key=RELEASE_KEY.public_key(),
        authority=authority,
        owner_reauthentication_receipt=_owner_reauth_receipt(),
        **_ancestry_validation_kwargs(),
        plan=plan,
        now_unix=NOW + 21,
    )

    assert checked["pre_foundation_authority_sha256"] == authority[
        "pre_foundation_authority_sha256"
    ]
    assert checked["inert_plan_sha256"] == pre.inert_plan_sha256(plan)
    assert [item["step_name"] for item in checked["applied_steps"]] == [
        step.name for step in plan.foundation_steps
    ]
    assert checked["partial_unknown_state"] is False
    assert checked["mutation_iam_binding_created"] is False
    assert checked["package_deployed"] is False
    assert checked["service_started"] is False


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("partial_unknown_state", True),
        ("mutation_iam_binding_created", True),
        ("package_deployed", True),
        ("service_started", True),
    ],
)
def test_apply_receipt_cannot_sign_unsafe_or_partial_state(
    field: str,
    replacement: bool,
) -> None:
    authority, plan, _ = _authority()
    body = _apply_body(authority, plan)
    body[field] = replacement

    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_foundation_apply_receipt_invalid",
    ):
        pre._sign_foundation_apply_receipt_body(
            body,
            private_key=RELEASE_KEY,
            authority=authority,
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            **_ancestry_validation_kwargs(),
            plan=plan,
        )


def test_apply_receipt_requires_exact_ordered_step_receipts() -> None:
    authority, plan, _ = _authority()
    steps = _step_receipts(plan)

    for invalid in (steps[:-1], list(reversed(steps))):
        with pytest.raises(
            pre.OwnerGatePreFoundationError,
            match="owner_gate_foundation_apply_steps_invalid",
        ):
            _apply_body(authority, plan, steps=invalid)

    drifted = copy.deepcopy(steps)
    drifted[0]["argv_sha256"] = "0" * 64
    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_foundation_apply_steps_invalid",
    ):
        _apply_body(authority, plan, steps=drifted)


def test_apply_identity_accepts_fresh_subnet_receipt_when_author_saw_none() -> None:
    authority, plan, evidence = _authority()
    assert evidence.preexisting_owner_gate_subnet_identity is None
    assert evidence.preexisting_owner_gate_subnet_route_identity is None

    body = _apply_body(authority, plan)

    assert body["partial_unknown_state"] is False


def test_apply_identity_accepts_exact_preexisting_signed_network_receipt() -> None:
    evidence = _evidence(preexisting_owner_subnet=True)
    authority, plan, _ = _authority(evidence=evidence)
    expected_subnet, expected_route = _preexisting_owner_gate_identities()

    body = _apply_body(authority, plan)

    assert plan.architecture["preexisting_owner_gate_subnet_identity"] == (
        expected_subnet
    )
    assert plan.architecture[
        "preexisting_owner_gate_subnet_route_identity"
    ] == expected_route
    assert body["partial_unknown_state"] is False


@pytest.mark.parametrize("identity", ["subnet", "route"])
def test_apply_identity_rejects_preexisting_signed_network_drift(
    identity: str,
) -> None:
    evidence = _evidence(preexisting_owner_subnet=True)
    authority, plan, _ = _authority(evidence=evidence)
    steps = _step_receipts(plan)
    target = next(
        item
        for item in steps
        if item["step_name"]
        == "create_dedicated_private_owner_gate_subnet"
    )
    if identity == "subnet":
        target["resource_identity"]["numeric_id"] = (
            "9999999999999999999"
        )
    else:
        target["resource_identity"]["local_route_identity"][
            "numeric_id"
        ] = "9999999999999999999"

    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_foundation_apply_identity_invalid",
    ):
        _apply_body(authority, plan, steps=steps)


@pytest.mark.parametrize(
    ("step_name", "field"),
    [
        ("create_dedicated_service_account", "disabled"),
        ("create_dedicated_service_account", "user_managed_keys"),
        ("create_narrow_iam_observation_reader_role", "deleted"),
        (
            "bind_narrow_iam_observation_reader_to_owner_gate_service_account",
            "matching_binding_count",
        ),
        (
            "bind_narrow_iam_observation_reader_to_owner_gate_service_account",
            "binding_members",
        ),
        ("create_dedicated_private_owner_gate_subnet", "kind"),
        (
            "create_dedicated_private_owner_gate_subnet",
            "creation_timestamp",
        ),
        ("create_dedicated_private_owner_gate_subnet", "region_self_link"),
        ("create_dedicated_private_owner_gate_subnet", "stack_type"),
        ("create_dedicated_private_owner_gate_subnet", "purpose"),
        (
            "create_dedicated_private_owner_gate_subnet",
            "secondary_ip_ranges",
        ),
        (
            "create_dedicated_private_owner_gate_subnet",
            "allow_subnet_cidr_routes_overlap",
        ),
        (
            "create_dedicated_private_owner_gate_subnet",
            "gateway_address",
        ),
        (
            "create_dedicated_private_owner_gate_subnet",
            "private_ipv6_google_access",
        ),
        (
            "create_dedicated_private_owner_gate_subnet",
            "local_route_identity",
        ),
        ("create_private_owner_gate_vm", "network_self_link"),
        ("create_private_owner_gate_vm", "network_numeric_id"),
        ("create_private_owner_gate_vm", "subnetwork_numeric_id"),
        ("create_private_owner_gate_vm", "network_stack_type"),
        ("create_private_owner_gate_vm", "access_configs"),
        ("create_private_owner_gate_vm", "deletion_protection"),
        ("create_private_owner_gate_vm", "boot_image_self_link"),
        ("create_private_owner_gate_vm", "boot_image_architecture"),
        ("create_private_owner_gate_vm", "boot_image_license_self_links"),
        ("create_private_owner_gate_vm", "tags"),
        ("create_private_owner_gate_vm", "metadata"),
        ("create_private_owner_gate_vm", "shielded_instance_config"),
        ("create_private_owner_gate_vm", "oauth_scopes"),
        ("create_private_owner_gate_vm", "can_ip_forward"),
        ("create_private_owner_gate_vm", "automatic_restart"),
        ("create_private_owner_gate_vm", "preemptible"),
        ("create_private_owner_gate_vm", "instance_termination_action"),
        ("create_private_owner_gate_vm", "creation_timestamp"),
        ("create_private_owner_gate_vm", "labels"),
        ("create_private_owner_gate_vm", "resource_policies"),
        ("create_private_owner_gate_vm", "min_cpu_platform"),
        ("create_private_owner_gate_vm", "confidential_instance_config"),
        ("create_private_owner_gate_vm", "boot_disk_self_link"),
        ("create_private_owner_gate_vm", "boot_disk_numeric_id"),
        ("create_private_owner_gate_vm", "boot_disk_size_gb"),
        ("create_private_owner_gate_vm", "boot_disk_type_self_link"),
        ("create_private_owner_gate_vm", "boot_disk_auto_delete"),
        ("create_private_owner_gate_vm", "boot_disk_boot"),
        ("create_private_owner_gate_vm", "boot_disk_mode"),
        ("create_private_owner_gate_vm", "boot_disk_interface"),
        ("create_private_owner_gate_vm", "boot_disk_attachment_type"),
        ("create_private_owner_gate_vm", "boot_disk_attachment_index"),
        (
            "allow_private_web_upstream_from_current_caddy_host",
            "disabled",
        ),
        (
            "allow_private_web_upstream_from_current_caddy_host",
            "creation_timestamp",
        ),
        ("allow_private_web_upstream_from_current_caddy_host", "action"),
        ("allow_private_web_upstream_from_current_caddy_host", "allowed"),
        ("allow_private_web_upstream_from_current_caddy_host", "denied"),
        (
            "allow_private_web_upstream_from_current_caddy_host",
            "source_ranges",
        ),
        (
            "allow_private_web_upstream_from_current_caddy_host",
            "destination_ranges",
        ),
        ("allow_private_web_upstream_from_current_caddy_host", "source_tags"),
        ("allow_private_web_upstream_from_current_caddy_host", "target_tags"),
    ],
)
def test_apply_identity_rejects_omitted_exact_projection_field(
    step_name: str,
    field: str,
) -> None:
    authority, plan, _ = _authority()
    steps = _step_receipts(plan)
    target = next(item for item in steps if item["step_name"] == step_name)
    target["resource_identity"].pop(field)

    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_foundation_apply_identity_invalid",
    ):
        _apply_body(authority, plan, steps=steps)


@pytest.mark.parametrize(
    "field",
    [
        "kind",
        "self_link",
        "numeric_id",
        "creation_timestamp",
        "network_self_link",
        "destination_range",
        "next_hop_network_self_link",
        "priority",
        "description",
        "route_type",
        "tags",
    ],
)
def test_apply_identity_rejects_omitted_local_subnet_route_field(
    field: str,
) -> None:
    authority, plan, _ = _authority()
    steps = _step_receipts(plan)
    target = next(
        item
        for item in steps
        if item["step_name"]
        == "create_dedicated_private_owner_gate_subnet"
    )
    target["resource_identity"]["local_route_identity"].pop(field)

    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_foundation_apply_identity_invalid",
    ):
        _apply_body(authority, plan, steps=steps)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("allowed", [{"ip_protocol": "tcp", "ports": ["8080", "22"]}]),
        ("denied", [{"ip_protocol": "udp", "ports": ["53"]}]),
        ("source_ranges", ["0.0.0.0/0"]),
        ("destination_ranges", ["10.80.3.0/28"]),
        ("source_tags", ["any-source"]),
        ("target_tags", [gate.OWNER_GATE_NETWORK_TAG]),
    ],
)
def test_apply_identity_rejects_extra_firewall_rule_surface(
    field: str,
    value: object,
) -> None:
    authority, plan, _ = _authority()
    steps = _step_receipts(plan)
    target = next(
        item
        for item in steps
        if item["step_name"]
        == "allow_private_web_upstream_from_current_caddy_host"
    )
    target["resource_identity"][field] = value

    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_foundation_apply_identity_invalid",
    ):
        _apply_body(authority, plan, steps=steps)


def test_apply_receipt_signature_domain_cannot_be_reused() -> None:
    authority, plan, _ = _authority()
    receipt = _apply_receipt(authority, plan)
    signed_payload = {
        key: value
        for key, value in receipt.items()
        if key != "signature_ed25519_b64url"
    }
    receipt["signature_ed25519_b64url"] = _b64url(
        RELEASE_KEY.sign(
            pre.AUTHORITY_SIGNATURE_DOMAIN
            + gate.canonical_json_bytes(signed_payload)
        )
    )

    with pytest.raises(
        pre.OwnerGatePreFoundationError,
        match="owner_gate_foundation_apply_signature_invalid",
    ):
        pre.validate_foundation_apply_receipt(
            receipt,
            public_key=RELEASE_KEY.public_key(),
            authority=authority,
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            **_ancestry_validation_kwargs(),
            plan=plan,
        )


def test_opaque_foundation_chains_require_canonical_signed_lineage() -> None:
    authority, plan, _ = _authority()
    owner_raw = gate.canonical_json_bytes(_owner_reauth_receipt())
    network_raw = gate.canonical_json_bytes(_signed_network_evidence())
    ancestry_raw = _signed_ancestry_raw()
    authority_raw = gate.canonical_json_bytes(authority)

    foundation_a = foundation_apply.decode_validated_foundation_a_chain(
        pre_foundation_authority_raw=authority_raw,
        owner_reauthentication_receipt_raw=owner_raw,
        network_evidence_raw=network_raw,
        project_ancestry_evidence_raw=ancestry_raw,
        release_public_key=RELEASE_KEY.public_key(),
        network_collector_public_key=NETWORK_KEY.public_key(),
        project_ancestry_collector_public_key=NETWORK_KEY.public_key(),
        now_unix=NOW + 1,
    )
    receipt = _apply_receipt(authority, plan)
    apply_chain = (
        foundation_apply._decode_validated_foundation_apply_chain(
            foundation_a=foundation_a,
            apply_receipt_raw=gate.canonical_json_bytes(receipt),
            now_unix=NOW + 21,
        )
    )

    assert type(foundation_a) is foundation_apply.ValidatedFoundationAChain
    assert (
        type(apply_chain)
        is foundation_apply.ValidatedFoundationApplyChain
    )
    assert apply_chain.foundation_source_revision == REVISION
    assert apply_chain.foundation_source_tree_oid == TREE_OID
    assert apply_chain.foundation_apply_receipt_sha256 == receipt[
        "foundation_apply_receipt_sha256"
    ]
    assert apply_chain.owner_gate_vm_identity["boot_disk_numeric_id"]
    assert apply_chain.owner_gate_vm_identity["network_numeric_id"]
    assert apply_chain.service_account_identity["unique_id"]
    assert apply_chain.subnet_identity["numeric_id"]

    with pytest.raises(foundation_apply.OwnerGateFoundationApplyError):
        foundation_apply.ValidatedFoundationAChain()  # type: ignore[call-arg]
    with pytest.raises(foundation_apply.OwnerGateFoundationApplyError):
        foundation_apply.decode_validated_foundation_a_chain(
            pre_foundation_authority_raw=b" " + authority_raw,
            owner_reauthentication_receipt_raw=owner_raw,
            network_evidence_raw=network_raw,
            project_ancestry_evidence_raw=ancestry_raw,
            release_public_key=RELEASE_KEY.public_key(),
            network_collector_public_key=NETWORK_KEY.public_key(),
            project_ancestry_collector_public_key=(
                NETWORK_KEY.public_key()
            ),
            now_unix=NOW + 1,
        )


def test_foundation_main_requires_authority_and_emits_only_inert_plan(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    authority, _, evidence = _authority()
    public_key_path = tmp_path / "release-author.pub"
    authority_path = tmp_path / "pre-foundation-authority.json"
    reauth_path = tmp_path / "owner-reauth-receipt.json"
    network_key_path = tmp_path / "network-collector.pem"
    evidence_path = tmp_path / "network-evidence.json"
    ancestry_path = tmp_path / "project-ancestry-evidence.json"
    public_key_path.write_bytes(RELEASE_KEY.public_key().public_bytes_raw())
    authority_path.write_bytes(gate.canonical_json_bytes(authority))
    reauth_path.write_bytes(
        gate.canonical_json_bytes(_owner_reauth_receipt())
    )
    network_key_path.write_bytes(
        NETWORK_KEY.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    evidence_path.write_bytes(gate.canonical_json_bytes(evidence.__dict__))
    ancestry_path.write_bytes(_signed_ancestry_raw())
    os.chmod(public_key_path, 0o444)
    os.chmod(authority_path, 0o444)
    os.chmod(reauth_path, 0o444)
    os.chmod(ancestry_path, 0o444)
    monkeypatch.setattr(gate.time, "time", lambda: float(NOW + 1))

    assert gate.main([
        "--pre-foundation-authority",
        str(authority_path),
        "--owner-reauth-receipt",
        str(reauth_path),
        "--release-trust-public-key",
        str(public_key_path),
        "--network-collector-public-key",
        str(network_key_path),
        "--network-evidence",
        str(evidence_path),
        "--project-ancestry-evidence",
        str(ancestry_path),
        "--project-ancestry-collector-public-key",
        str(network_key_path),
    ]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["pre_foundation_authority_sha256"] == authority[
        "pre_foundation_authority_sha256"
    ]
    assert report["inert_plan_sha256"] == authority["inert_plan_sha256"]
    assert report["mutation_iam_binding_authorized"] is False
    assert report["package_deployment_authorized"] is False
    assert report["service_start_authorized"] is False
    assert report["architecture"]["pre_foundation_only"] is True
