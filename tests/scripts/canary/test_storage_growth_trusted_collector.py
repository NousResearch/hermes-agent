from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping, cast

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import owner_gate_foundation as owner_gate_foundation
from scripts.canary import storage_growth_contract as contract
from scripts.canary import storage_growth_evidence as evidence
from scripts.canary import storage_growth_trusted_collector as collector
from tests.scripts.canary.test_host_storage_growth import (
    _disk,
    _instance,
    _pending_report,
    _source_report,
    _target_report,
)


NOW = 1_800_000_001
RELEASE = "a" * 40
TRANSACTION = "b" * 64
PRIOR_HEAD = protocol.GENESIS_JOURNAL_HEAD_SHA256
PROJECT_NUMBER = contract.PROJECT_NUMBER
ANCESTOR_CHAIN = ("folders/123456789", "organizations/987654321")
TARGET_SERVICE_ACCOUNT_UNIQUE_ID = "123456789012345678901"
RUNTIME_INSTANCE_ID = "1111111111111111111"
RUNTIME_SERVICE_ACCOUNT_UNIQUE_ID = "222222222222222222222"
ACTIVATION_SEAL_SHA256 = "5" * 64
OWNER_REAUTH_RECEIPT_SHA256 = "8" * 64
EXTERNAL_SENSITIVE_ROLE = "roles/resourcemanager.organizationAdmin"


def _external_sensitive_role_resource() -> dict[str, Any]:
    return {
        "name": EXTERNAL_SENSITIVE_ROLE,
        "title": "Organization Administrator",
        "description": "Organization-wide administrative authority",
        "includedPermissions": [
            "resourcemanager.organizations.setIamPolicy"
        ],
        "stage": "GA",
        "deleted": False,
        "etag": "external-sensitive-role-etag",
    }


def _external_gcp_admin_trust_root(
    ancestor_chain: tuple[str, ...] = ANCESTOR_CHAIN,
) -> dict[str, Any]:
    runtime_permissions = {
        contract.RUNTIME_ROLES[0]: [
            item
            for item in contract.RUNTIME_PERMISSIONS
            if item.startswith("logging.")
        ],
        contract.RUNTIME_ROLES[1]: [
            item
            for item in contract.RUNTIME_PERMISSIONS
            if item.startswith("monitoring.")
        ],
        contract.RUNTIME_ROLES[2]: ["cloudsql.instances.get"],
    }
    runtime_member = f"serviceAccount:{contract.RUNTIME_SERVICE_ACCOUNT}"
    bindings = [
        {
            "resource": f"projects/{contract.PROJECT}",
            "role": role,
            "members": [runtime_member],
            "condition": None,
        }
        for role in contract.RUNTIME_ROLES
    ] + [{
        "resource": ancestor_chain[-1],
        "role": EXTERNAL_SENSITIVE_ROLE,
        "members": ["user:owner@example.test"],
        "condition": None,
    }]
    bindings.sort(key=protocol.canonical_json_bytes)
    definitions = [
        {
            "name": role,
            "title": role.rsplit("/", 1)[-1],
            "description": "",
            "included_permissions": sorted(runtime_permissions[role]),
            "stage": "GA",
            "deleted": False,
            "etag": f"role-{index}-etag",
        }
        for index, role in enumerate(contract.RUNTIME_ROLES)
    ] + [{
        "name": EXTERNAL_SENSITIVE_ROLE,
        "title": "Organization Administrator",
        "description": "Organization-wide administrative authority",
        "included_permissions": [
            "resourcemanager.organizations.setIamPolicy"
        ],
        "stage": "GA",
        "deleted": False,
        "etag": "external-sensitive-role-etag",
    }]
    definitions.sort(key=lambda item: item["name"])
    policy_resources = [f"projects/{contract.PROJECT}", *ancestor_chain]
    policy_etags = [
        "project-policy-etag",
        *("folder-policy-etag" for _ in ancestor_chain[:-1]),
        "organization-policy-etag",
    ]
    return {
        "inventory_complete": True,
        "structural_partition_complete": True,
        "passkey_protects_against_external_gcp_admins": False,
        "passkey_protects_against_pinned_external_roots": False,
        "google_provider_control_plane_outside_passkey": True,
        "collected_under_owner_reauthentication_receipt_sha256": (
            OWNER_REAUTH_RECEIPT_SHA256
        ),
        "resource_policy_generations": [
            {
                "resource": resource,
                "version": 3,
                "etag": etag,
                "audit_configs": [],
            }
            for resource, etag in zip(
                policy_resources,
                policy_etags,
                strict=True,
            )
        ],
        "allowed_residual_bindings": bindings,
        "allowed_residual_role_definitions": definitions,
    }


def _sensitive_role_resource(
    name: str,
    permissions: list[str],
    *,
    title: str = "Pinned sensitive role",
    description: str = "Owner-reauthored external authority",
) -> dict[str, Any]:
    return {
        "name": name,
        "title": title,
        "description": description,
        "includedPermissions": sorted(permissions),
        "stage": "GA",
        "deleted": False,
        "etag": "pinned-sensitive-role-etag",
    }


def _root_with_sensitive_binding(
    *,
    resource: str,
    role_resource: Mapping[str, Any],
    members: list[str],
    condition: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    value = copy.deepcopy(_external_gcp_admin_trust_root())
    value["allowed_residual_bindings"].append({
        "resource": resource,
        "role": role_resource["name"],
        "members": sorted(members),
        "condition": condition,
    })
    value["allowed_residual_bindings"].sort(
        key=protocol.canonical_json_bytes
    )
    value["allowed_residual_role_definitions"].append({
        "name": role_resource["name"],
        "title": role_resource["title"],
        "description": role_resource.get("description", ""),
        "included_permissions": sorted(role_resource["includedPermissions"]),
        "stage": role_resource["stage"],
        "deleted": False,
        "etag": role_resource.get("etag"),
    })
    value["allowed_residual_role_definitions"].sort(
        key=lambda item: item["name"]
    )
    return value


def _request(
    checkpoint: str = "source",
    *,
    prior_head: str = PRIOR_HEAD,
    attempt_sequence: int = 1,
    issued_at: int = NOW,
) -> Mapping[str, Any]:
    canonical_state = {
        "source": "await_source",
        "post_resize": "await_post_resize",
        "post_stop": "await_post_stop",
        "post_start": "await_post_start",
    }[checkpoint]
    binding = evidence.observation_request_binding_sha256(
        transaction_id=TRANSACTION,
        checkpoint=checkpoint,
        prior_event_head_sha256=prior_head,
        release_sha=RELEASE,
        plan_sha256=contract.canonical_plan_sha256(),
    )
    collection_context = {
        "schema": "muncho-storage-growth-collection-context.v1",
        "transaction_id": TRANSACTION,
        "checkpoint": checkpoint,
        "prior_event_head_sha256": prior_head,
        "release_sha": RELEASE,
        "plan_sha256": contract.canonical_plan_sha256(),
    }
    attempt_identity = {
        "schema": "muncho-storage-growth-collection-attempt-id.v1",
        "context_sha256": protocol.sha256_json(collection_context),
        "context_sequence": attempt_sequence,
        "issued_at_unix": issued_at,
    }
    unsigned = {
        "schema": "muncho-storage-growth-observation-request.v1",
        "transaction_id": TRANSACTION,
        "checkpoint": checkpoint,
        "canonical_state": canonical_state,
        "prior_event_head_sha256": prior_head,
        "request_binding_sha256": binding,
        "observation_nonce_sha256": evidence.observation_nonce_sha256(
            request_binding_sha256=binding,
            transaction_id=TRANSACTION,
            checkpoint=checkpoint,
        ),
        "collection_attempt_id": protocol.sha256_json(attempt_identity),
        "collection_attempt_sequence": attempt_sequence,
        "collection_attempt_issued_at_unix": issued_at,
        "collection_attempt_expires_at_unix": (
            issued_at + evidence.OBSERVATION_BUNDLE_TTL_SECONDS
        ),
        "release_sha": RELEASE,
        "plan_sha256": contract.canonical_plan_sha256(),
    }
    return {
        **unsigned,
        "observation_request_sha256": protocol.sha256_json(unsigned),
    }


def _terminal_request() -> Mapping[str, Any]:
    unsigned = {
        "schema": "muncho-storage-growth-observation-request.v1",
        "transaction_id": TRANSACTION,
        "checkpoint": None,
        "canonical_state": "terminal",
        "prior_event_head_sha256": "9" * 64,
        "request_binding_sha256": None,
        "observation_nonce_sha256": None,
        "collection_attempt_id": None,
        "collection_attempt_sequence": None,
        "collection_attempt_issued_at_unix": None,
        "collection_attempt_expires_at_unix": None,
        "release_sha": RELEASE,
        "plan_sha256": contract.canonical_plan_sha256(),
    }
    return {
        **unsigned,
        "observation_request_sha256": protocol.sha256_json(unsigned),
    }


def _config(
    tmp_path: Path, role: str, private_key: Ed25519PrivateKey
) -> Mapping[str, Any]:
    root = tmp_path / role
    root.mkdir(mode=0o700, parents=True)
    root.chmod(0o700)
    key = root / "attestation.key"
    key.write_bytes(private_key.private_bytes_raw())
    key.chmod(0o400)
    replay = root / "replay"
    replay.mkdir(mode=0o700)
    replay.chmod(0o700)
    key_state = key.stat()
    replay_state = replay.stat()
    return {
        "schema": (
            collector.HOST_CONFIG_SCHEMA
            if role == "host"
            else collector.CLOUD_CONFIG_SCHEMA
        ),
        "role": role,
        "private_key_path": str(key),
        "private_key_uid": key_state.st_uid,
        "private_key_gid": key_state.st_gid,
        "private_key_mode": "0400",
        "public_key_id": protocol.sha256_bytes(
            private_key.public_key().public_bytes_raw()
        ),
        "replay_directory": str(replay),
        "replay_directory_uid": replay_state.st_uid,
        "replay_directory_gid": replay_state.st_gid,
        "replay_directory_mode": "0700",
    }


def _host_projection(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "schema": "muncho-storage-growth-host-recollection.v1",
        **{
            name: report[name]
            for name in (
                "boot_id_sha256",
                "root_source",
                "root_filesystem",
                "root_mountpoint",
                "root_size_bytes",
                "root_available_bytes",
                "service_states",
                "service_states_sha256",
                "canonical_receipt_source",
                "current_stopped_release_sha",
                "current_host_receipt_file_sha256",
                "current_host_receipt_sha256",
                "current_stopped_release_receipt_file_sha256",
                "current_stopped_release_receipt_sha256",
            )
        },
    }


def _cloud_projection(report: Mapping[str, Any]) -> Mapping[str, Any]:
    iam = report["external_iam_receipt"]
    return {
        "schema": "muncho-storage-growth-cloud-recollection.v1",
        "project": report["project"],
        "zone": report["zone"],
        "vm_name": report["vm_name"],
        "vm_instance_id": report["vm_instance_id"],
        "instance_status": report["instance_status"],
        "disk_name": report["disk_name"],
        "disk_id": report["disk_id"],
        "disk_size_gb": report["disk_size_gb"],
        "boot_device_name": report["boot_device_name"],
        "runtime_service_account": iam["service_account"],
        "runtime_scopes": iam["scopes"],
        "external_iam_receipt_sha256": report[
            "external_iam_receipt_sha256"
        ],
        "external_iam_policy_sha256": report[
            "external_iam_policy_sha256"
        ],
    }


def _iam_raw_inputs() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    member = f"serviceAccount:{contract.RUNTIME_SERVICE_ACCOUNT}"
    collector_member = (
        f"serviceAccount:{collector.OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT}"
    )
    ancestor_role = collector._owner_gate_ancestor_read_role(
        ANCESTOR_CHAIN
    )
    project = {
        "name": f"projects/{PROJECT_NUMBER}",
        "projectId": contract.PROJECT,
        "parent": ANCESTOR_CHAIN[0],
        "state": "ACTIVE",
        "etag": "project-etag",
    }
    ancestors = [
        {
            "name": ANCESTOR_CHAIN[0],
            "parent": ANCESTOR_CHAIN[1],
            "state": "ACTIVE",
            "etag": "folder-etag",
        },
        {
            "name": ANCESTOR_CHAIN[1],
            "state": "ACTIVE",
            "etag": "organization-etag",
        },
    ]
    policies = [
        {
            "version": 3,
            "etag": "project-policy-etag",
            "bindings": [
                {"role": role, "members": [member]}
                for role in contract.RUNTIME_ROLES
            ]
            + [
                {
                    "role": collector.OWNER_GATE_PROJECT_READ_ROLE,
                    "members": [collector_member],
                },
                {
                    "role": collector.OWNER_GATE_MUTATION_ROLE,
                    "members": [collector_member],
                    "condition": dict(
                        collector.OWNER_GATE_MUTATION_CONDITION
                    ),
                },
            ],
        },
        {"version": 3, "etag": "folder-policy-etag", "bindings": []},
        {
            "version": 3,
            "etag": "organization-policy-etag",
            "bindings": [
                {
                    "role": ancestor_role,
                    "members": [collector_member],
                },
                {
                    "role": EXTERNAL_SENSITIVE_ROLE,
                    "members": ["user:owner@example.test"],
                },
            ],
        },
    ]
    role_permissions = {
        contract.RUNTIME_ROLES[0]: [
            item for item in contract.RUNTIME_PERMISSIONS
            if item.startswith("logging.")
        ],
        contract.RUNTIME_ROLES[1]: [
            item for item in contract.RUNTIME_PERMISSIONS
            if item.startswith("monitoring.")
        ],
        contract.RUNTIME_ROLES[2]: ["cloudsql.instances.get"],
    }
    roles = [
        {
            "name": role,
            "title": role.rsplit("/", 1)[-1],
            "stage": "GA",
            "deleted": False,
            "etag": f"role-{index}-etag",
            "includedPermissions": role_permissions[role],
        }
        for index, role in enumerate(contract.RUNTIME_ROLES)
    ]
    account = {
        "name": (
            f"projects/{contract.PROJECT}/serviceAccounts/"
            f"{contract.RUNTIME_SERVICE_ACCOUNT}"
        ),
        "projectId": contract.PROJECT,
        "uniqueId": TARGET_SERVICE_ACCOUNT_UNIQUE_ID,
        "email": contract.RUNTIME_SERVICE_ACCOUNT,
        "displayName": "Muncho canary runtime",
        "oauth2ClientId": "987654321012345678901",
        "disabled": False,
    }
    return project, ancestors, policies, roles, account


def _collector_role_raw_inputs(
    ancestor_chain: tuple[str, ...] = ANCESTOR_CHAIN,
) -> list[dict[str, Any]]:
    contracts = collector._owner_gate_collector_role_contracts(
        ancestor_chain
    )
    return [
        {
            "name": role_contract["name"],
            "title": role_contract["title"],
            "description": role_contract["description"],
            "stage": "GA",
            "deleted": False,
            "etag": f"collector-role-{index}-etag",
            "includedPermissions": list(
                role_contract["included_permissions"]
            ),
        }
        for index, role_contract in enumerate(contracts)
    ]


def _collector_raw_inputs() -> tuple[
    dict[str, Any], dict[str, Any]
]:
    instance = {
        "id": RUNTIME_INSTANCE_ID,
        "name": collector.OWNER_GATE_VM_NAME,
        "status": "RUNNING",
        "zone": (
            f"https://compute.googleapis.com/compute/v1/projects/"
            f"{contract.PROJECT}/zones/{contract.ZONE}"
        ),
        "serviceAccounts": [{
            "email": collector.OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT,
            "scopes": list(collector.OWNER_GATE_METADATA_SCOPES),
        }],
    }
    account = {
        "name": (
            f"projects/{contract.PROJECT}/serviceAccounts/"
            f"{collector.OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT}"
        ),
        "projectId": contract.PROJECT,
        "uniqueId": RUNTIME_SERVICE_ACCOUNT_UNIQUE_ID,
        "email": collector.OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT,
        "oauth2ClientId": "333333333333333333333",
        "disabled": False,
    }
    return instance, account


def _trusted_iam_projection(
    report: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    now_unix: int = NOW,
) -> Mapping[str, Any]:
    project, ancestors, policies, roles, account = _iam_raw_inputs()
    return _build_trusted_iam_projection_from_raw(
        report,
        request,
        project=project,
        ancestors=ancestors,
        policies=policies,
        roles=roles,
        account=account,
        now_unix=now_unix,
    )


def _build_trusted_iam_projection_from_raw(
    report: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    project: dict[str, Any],
    ancestors: list[dict[str, Any]],
    policies: list[dict[str, Any]],
    roles: list[dict[str, Any]],
    account: dict[str, Any],
    ancestor_chain: tuple[str, ...] = ANCESTOR_CHAIN,
    now_unix: int = NOW,
    collector_instance: Mapping[str, Any] | None = None,
    collector_account: Mapping[str, Any] | None = None,
    collector_service_account_policy: Mapping[str, Any] | None = None,
    collector_user_managed_keys: Mapping[str, Any] | None = None,
    metadata_instance_id: str = RUNTIME_INSTANCE_ID,
    metadata_service_account_email: str = (
        collector.OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT
    ),
    metadata_scopes: tuple[str, ...] = (
        collector.OWNER_GATE_METADATA_SCOPES
    ),
    collector_roles: list[dict[str, Any]] | None = None,
    additional_policy_roles: list[dict[str, Any]] | None = None,
    external_gcp_admin_trust_root: Mapping[str, Any] | None = None,
    expected_mutation_binding_present: bool = True,
    activation_seal_sha256: str | None = ACTIVATION_SEAL_SHA256,
) -> Mapping[str, Any]:
    default_instance, default_account = _collector_raw_inputs()
    if collector_instance is None:
        collector_instance = default_instance
    if collector_account is None:
        collector_account = default_account
    if collector_service_account_policy is None:
        collector_service_account_policy = {
            "version": 1,
            "etag": "collector-service-account-policy-etag",
            "bindings": [],
        }
    if collector_user_managed_keys is None:
        collector_user_managed_keys = {"keys": []}
    if additional_policy_roles is None:
        additional_policy_roles = [_external_sensitive_role_resource()]
    if external_gcp_admin_trust_root is None:
        external_gcp_admin_trust_root = _external_gcp_admin_trust_root(
            ancestor_chain
        )
    return collector.build_trusted_iam_projection(
        observation_request=request,
        candidate_observation=report,
        expected_project_number=PROJECT_NUMBER,
        expected_ancestor_chain=ancestor_chain,
        expected_runtime_instance_numeric_id=RUNTIME_INSTANCE_ID,
        expected_runtime_service_account_email=(
            collector.OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT
        ),
        expected_runtime_service_account_unique_id=(
            RUNTIME_SERVICE_ACCOUNT_UNIQUE_ID
        ),
        expected_target_service_account_unique_id=(
            TARGET_SERVICE_ACCOUNT_UNIQUE_ID
        ),
        metadata_instance_id=metadata_instance_id,
        metadata_service_account_email=metadata_service_account_email,
        metadata_scopes=metadata_scopes,
        collector_instance_resource=collector_instance,
        collector_service_account_resource=collector_account,
        collector_service_account_policy=collector_service_account_policy,
        collector_user_managed_keys=collector_user_managed_keys,
        project_resource=project,
        ancestor_resources=ancestors,
        resource_policies=policies,
        role_resources=roles,
        collector_role_resources=(
            collector_roles
            if collector_roles is not None
            else _collector_role_raw_inputs(ancestor_chain)
        ),
        additional_policy_role_resources=additional_policy_roles,
        service_account_resource=account,
        expected_external_gcp_admin_trust_root=(
            external_gcp_admin_trust_root
        ),
        expected_mutation_binding_present=(
            expected_mutation_binding_present
        ),
        activation_seal_sha256=activation_seal_sha256,
        now_unix=now_unix,
    )


def _host_frame(
    request: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    now_unix: int = NOW,
) -> Mapping[str, Any]:
    return collector.build_attestation_request(
        request,
        report,
        role="host",
        trusted_iam_projection=_trusted_iam_projection(
            report, request, now_unix=now_unix
        ),
        now_unix=now_unix,
    )


class HostReader:
    def __init__(self, report: Mapping[str, Any]) -> None:
        self.projection = _host_projection(report)
        self.calls = 0

    def collect(self) -> Mapping[str, Any]:
        self.calls += 1
        return copy.deepcopy(self.projection)


class CloudReader:
    def __init__(self, report: Mapping[str, Any]) -> None:
        self.projection = _cloud_projection(report)
        self.calls = 0

    def collect(
        self,
        observation: Mapping[str, Any],
        observation_request: Mapping[str, Any],
        now_unix: int,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        self.calls += 1
        return (
            copy.deepcopy(self.projection),
            _trusted_iam_projection(
                observation, observation_request, now_unix=now_unix
            ),
        )


def _signed_pair(
    tmp_path: Path,
    report: Mapping[str, Any],
    request: Mapping[str, Any],
) -> tuple[
    Ed25519PrivateKey,
    Ed25519PrivateKey,
    dict[str, Any],
    dict[str, Any],
]:
    return _signed_pair_at(tmp_path, report, request, now_unix=NOW)


def _signed_pair_at(
    tmp_path: Path,
    report: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    now_unix: int,
) -> tuple[
    Ed25519PrivateKey,
    Ed25519PrivateKey,
    dict[str, Any],
    dict[str, Any],
]:
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    cloud_frame = collector.build_attestation_request(
        request, report, role="cloud", now_unix=now_unix
    )
    cloud = collector.run_cloud_attestor(
        cloud_frame,
        config=_config(tmp_path, "cloud", cloud_key),
        facts_reader=CloudReader(report),
        now_unix=now_unix,
    )
    host_frame = collector.build_attestation_request(
        request,
        report,
        role="host",
        trusted_iam_projection=cloud["trusted_iam_projection"],
        now_unix=now_unix,
    )
    host = collector.run_host_attestor(
        host_frame,
        config=_config(tmp_path, "host", host_key),
        facts_reader=HostReader(report),
        now_unix=now_unix,
    )
    return cloud_key, host_key, dict(cloud), dict(host)


def test_dual_signers_recollect_same_core_and_combine_to_existing_bundle(
    tmp_path: Path,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    request = _request()
    cloud_key, host_key, cloud, host = _signed_pair(
        tmp_path, report, request
    )
    bundle = collector.combine_trusted_attestations(
        observation_request=request,
        refreshed_observation_request=copy.deepcopy(request),
        candidate_observation=report,
        cloud_response=cloud,
        cloud_public_key=cloud_key.public_key(),
        host_response=host,
        host_public_key=host_key.public_key(),
        now_unix=NOW,
    )
    assert cloud["bundle_core_sha256"] == host["bundle_core_sha256"]
    verified = evidence.validate_attested_observation(
        bundle,
        cloud_public_key=cloud_key.public_key(),
        cloud_public_key_id=cloud["attestation"]["public_key_id"],
        host_public_key=host_key.public_key(),
        host_public_key_id=host["attestation"]["public_key_id"],
        now_unix=NOW,
        allowed_states=frozenset({"source_ready"}),
        expected_transaction_id=TRANSACTION,
        expected_checkpoint="source",
        expected_request_binding_sha256=request[
            "request_binding_sha256"
        ],
        expected_prior_event_head_sha256=PRIOR_HEAD,
        expected_observation_request_sha256=request[
            "observation_request_sha256"
        ],
        expected_collection_attempt_id=request[
            "collection_attempt_id"
        ],
        expected_collection_attempt_sequence=request[
            "collection_attempt_sequence"
        ],
        expected_collection_attempt_issued_at_unix=request[
            "collection_attempt_issued_at_unix"
        ],
        expected_collection_attempt_expires_at_unix=request[
            "collection_attempt_expires_at_unix"
        ],
    )
    assert verified == report


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("extra_direct_role", "iam_binding_drift"),
        ("missing_direct_role", "iam_binding_drift"),
        ("conditional_direct_role", "iam_binding_drift"),
        ("inherited_direct_member", "iam_inherited_binding_drift"),
        ("wrong_service_account_unique_id", "iam_service_account_invalid"),
        ("wrong_project_parent", "iam_hierarchy_invalid"),
        ("wrong_project_id", "iam_hierarchy_invalid"),
        ("extra_role_permission", "iam_permission_drift"),
    ],
)
def test_direct_iam_allow_grant_drift_is_fail_closed(
    case: str,
    error: str,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    request = _request()
    project, ancestors, policies, roles, account = _iam_raw_inputs()
    member = f"serviceAccount:{contract.RUNTIME_SERVICE_ACCOUNT}"
    if case == "extra_direct_role":
        policies[0]["bindings"].append({
            "role": "roles/viewer", "members": [member]
        })
    elif case == "missing_direct_role":
        del policies[0]["bindings"][len(contract.RUNTIME_ROLES) - 1]
    elif case == "conditional_direct_role":
        policies[0]["bindings"][0]["condition"] = {
            "title": "sometimes",
            "expression": "request.time < timestamp('2030-01-01T00:00:00Z')",
        }
    elif case == "inherited_direct_member":
        policies[1]["bindings"].append({
            "role": contract.RUNTIME_ROLES[0], "members": [member]
        })
    elif case == "wrong_service_account_unique_id":
        account["uniqueId"] = "123456789012345678902"
    elif case == "wrong_project_parent":
        project["parent"] = ANCESTOR_CHAIN[-1]
    elif case == "wrong_project_id":
        project["projectId"] = "attacker-project"
    elif case == "extra_role_permission":
        roles[0]["includedPermissions"].append(
            "resourcemanager.projects.get"
        )
    with pytest.raises(collector.TrustedObservationError, match=error):
        _build_trusted_iam_projection_from_raw(
            report,
            request,
            project=project,
            ancestors=ancestors,
            policies=policies,
            roles=roles,
            account=account,
        )


@pytest.mark.parametrize(
    "member",
    [
        "allUsers",
        "allAuthenticatedUsers",
        "deleted:user:former@example.test?uid=123456789012345678901",
        "projectOwner:adventico-ai-platform",
        (
            "principal://iam.googleapis.com/locations/global/"
            "workforcePools/example/subject/123"
        ),
        (
            "principalSet://iam.googleapis.com/locations/global/"
            "workforcePools/example/group/ops"
        ),
    ],
)
def test_direct_iam_policy_members_are_opaque_exact_pinned_values(
    member: str,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    request = _request()
    project, ancestors, policies, roles, account = _iam_raw_inputs()
    role = _sensitive_role_resource(
        "roles/viewer",
        ["resourcemanager.projects.get"],
        title="Viewer",
        description="Opaque policy member fixture",
    )
    policies[1]["bindings"].append({
        "role": role["name"],
        "members": [member],
    })
    expected_root = _root_with_sensitive_binding(
        resource=ANCESTOR_CHAIN[0],
        role_resource=role,
        members=[member],
    )

    projection = _build_trusted_iam_projection_from_raw(
        report,
        request,
        project=project,
        ancestors=ancestors,
        policies=policies,
        roles=roles,
        account=account,
        additional_policy_roles=[
            _external_sensitive_role_resource(),
            role,
        ],
        external_gcp_admin_trust_root=expected_root,
    )

    assert {
        "resource": ANCESTOR_CHAIN[0],
        "role": role["name"],
        "members": [member],
        "condition": None,
    } in projection["residual_external_bindings"]


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("missing_project_read", "collector_binding_drift"),
        ("missing_ancestor_read", "collector_binding_drift"),
        ("missing_active_mutation", "collector_binding_drift"),
        ("mutation_title_drift", "collector_binding_drift"),
        ("mutation_description_drift", "collector_binding_drift"),
        ("mutation_expression_drift", "collector_binding_drift"),
        ("extra_collector_grant", "collector_binding_drift"),
        ("collector_role_permission_drift", "collector_permission_drift"),
        ("collector_role_title_drift", "collector_permission_drift"),
        ("collector_role_stage_drift", "collector_permission_drift"),
        ("service_account_impersonator", "collector_impersonation_invalid"),
        (
            "service_account_conditional_impersonator",
            "collector_impersonation_invalid",
        ),
        ("user_managed_key", "collector_user_keys_invalid"),
    ],
)
def test_owner_gate_collector_authority_drift_is_fail_closed(
    case: str,
    error: str,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    request = _request()
    project, ancestors, policies, roles, account = _iam_raw_inputs()
    collector_roles = _collector_role_raw_inputs()
    collector_policy: Mapping[str, Any] = {
        "version": 1,
        "etag": "collector-service-account-policy-etag",
        "bindings": [],
    }
    collector_keys: Mapping[str, Any] = {"keys": []}
    project_bindings = policies[0]["bindings"]
    if case == "missing_project_read":
        project_bindings[:] = [
            item for item in project_bindings
            if item["role"] != collector.OWNER_GATE_PROJECT_READ_ROLE
        ]
    elif case == "missing_ancestor_read":
        policies[-1]["bindings"] = [
            item for item in policies[-1]["bindings"]
            if item["role"] != collector._owner_gate_ancestor_read_role(
                ANCESTOR_CHAIN
            )
        ]
    elif case == "missing_active_mutation":
        project_bindings[:] = [
            item for item in project_bindings
            if item["role"] != collector.OWNER_GATE_MUTATION_ROLE
        ]
    elif case.startswith("mutation_"):
        mutation = next(
            item for item in project_bindings
            if item["role"] == collector.OWNER_GATE_MUTATION_ROLE
        )
        field = case.removeprefix("mutation_").removesuffix("_drift")
        mutation["condition"][field] = f"wrong-{field}"
    elif case == "extra_collector_grant":
        project_bindings.append({
            "role": contract.RUNTIME_ROLES[0],
            "members": [
                "serviceAccount:"
                + collector.OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT
            ],
        })
    elif case == "collector_role_permission_drift":
        collector_roles[0]["includedPermissions"].append(
            "resourcemanager.projects.delete"
        )
    elif case == "collector_role_title_drift":
        collector_roles[0]["title"] = "Wrong title"
    elif case == "collector_role_stage_drift":
        collector_roles[0]["stage"] = "BETA"
    elif case in {
        "service_account_impersonator",
        "service_account_conditional_impersonator",
    }:
        binding: dict[str, Any] = {
            "role": "roles/iam.serviceAccountTokenCreator",
            "members": ["user:attacker@example.com"],
        }
        if case == "service_account_conditional_impersonator":
            binding["condition"] = {
                "title": "temporary",
                "expression": "request.time < timestamp('2030-01-01T00:00:00Z')",
            }
        collector_policy = {
            "version": 3,
            "etag": "collector-service-account-policy-etag",
            "bindings": [binding],
        }
    elif case == "user_managed_key":
        collector_keys = {"keys": [{
            "name": "projects/example/serviceAccounts/sa/keys/key-id",
            "keyType": "USER_MANAGED",
        }]}
    with pytest.raises(collector.TrustedObservationError, match=error):
        _build_trusted_iam_projection_from_raw(
            report,
            request,
            project=project,
            ancestors=ancestors,
            policies=policies,
            roles=roles,
            account=account,
            collector_roles=collector_roles,
            collector_service_account_policy=collector_policy,
            collector_user_managed_keys=collector_keys,
        )


@pytest.mark.parametrize(
    ("resource_index", "role", "permissions"),
    [
        (
            1,
            "roles/iam.serviceAccountTokenCreator",
            ["iam.serviceAccounts.getAccessToken"],
        ),
        (
            2,
            "organizations/987654321/roles/externalDelegator",
            ["iam.serviceAccounts.actAs"],
        ),
        (
            0,
            "projects/adventico-ai-platform/roles/externalDiskOperator",
            ["compute.disks.resize"],
        ),
    ],
)
def test_unpinned_project_folder_or_org_sensitive_grant_fails_closed(
    resource_index: int,
    role: str,
    permissions: list[str],
) -> None:
    report = _source_report(collected_at=NOW - 1)
    project, ancestors, policies, roles, account = _iam_raw_inputs()
    policies[resource_index]["bindings"].append({
        "role": role,
        "members": ["user:untrusted@example.test"],
    })
    with pytest.raises(
        collector.TrustedObservationError,
        match="external_gcp_admin_trust_root_drift",
    ):
        _build_trusted_iam_projection_from_raw(
            report,
            _request(),
            project=project,
            ancestors=ancestors,
            policies=policies,
            roles=roles,
            account=account,
            additional_policy_roles=[
                _external_sensitive_role_resource(),
                _sensitive_role_resource(role, permissions),
            ],
        )


def test_exact_owner_reauthored_sensitive_root_is_signed_and_accepted() -> None:
    report = _source_report(collected_at=NOW - 1)
    project, ancestors, policies, roles, account = _iam_raw_inputs()
    role = _sensitive_role_resource(
        "projects/adventico-ai-platform/roles/externalDiskOperator",
        ["compute.disks.get", "compute.disks.resize"],
    )
    binding = {
        "role": role["name"],
        "members": ["group:platform-admins@example.test"],
    }
    policies[0]["bindings"].append(binding)
    expected_root = _root_with_sensitive_binding(
        resource=f"projects/{contract.PROJECT}",
        role_resource=role,
        members=binding["members"],
    )
    projection = _build_trusted_iam_projection_from_raw(
        report,
        _request(),
        project=project,
        ancestors=ancestors,
        policies=policies,
        roles=roles,
        account=account,
        additional_policy_roles=[
            _external_sensitive_role_resource(),
            role,
        ],
        external_gcp_admin_trust_root=expected_root,
    )
    assert projection["residual_external_bindings"] == expected_root[
        "allowed_residual_bindings"
    ]
    assert projection["external_gcp_admin_trust_root"] == expected_root
    assert projection["external_gcp_admin_trust_root"][
        "passkey_protects_against_pinned_external_roots"
    ] is False
    assert projection["external_gcp_admin_trust_root"][
        "google_provider_control_plane_outside_passkey"
    ] is True


def test_projection_consumer_requires_the_exact_signed_external_root_pin() -> None:
    report = _source_report(collected_at=NOW - 1)
    request = _request()
    projection = _trusted_iam_projection(report, request)
    wrong_root = copy.deepcopy(_external_gcp_admin_trust_root())
    wrong_root[
        "collected_under_owner_reauthentication_receipt_sha256"
    ] = "9" * 64
    with pytest.raises(
        collector.TrustedObservationError,
        match="external_gcp_admin_trust_root_drift",
    ):
        collector.validate_trusted_iam_projection(
            projection,
            observation_request=request,
            candidate_observation=report,
            now_unix=NOW,
            expected_external_gcp_admin_trust_root=wrong_root,
        )


def test_direct_mutation_bypass_outside_exact_condition_fails_closed() -> None:
    report = _source_report(collected_at=NOW - 1)
    project, ancestors, policies, roles, account = _iam_raw_inputs()
    policies[0]["bindings"].append({
        "role": collector.OWNER_GATE_MUTATION_ROLE,
        "members": ["user:direct-bypass@example.test"],
    })
    with pytest.raises(
        collector.TrustedObservationError,
        match="external_gcp_admin_trust_root_drift",
    ):
        _build_trusted_iam_projection_from_raw(
            report,
            _request(),
            project=project,
            ancestors=ancestors,
            policies=policies,
            roles=roles,
            account=account,
        )


@pytest.mark.parametrize(
    "drift",
    ["member", "condition", "resource", "definition"],
)
def test_signed_sensitive_root_member_condition_resource_or_definition_drift_rejected(
    drift: str,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    project, ancestors, policies, roles, account = _iam_raw_inputs()
    role = _sensitive_role_resource(
        "projects/adventico-ai-platform/roles/externalDiskOperator",
        ["compute.disks.resize"],
    )
    policies[0]["bindings"].append({
        "role": role["name"],
        "members": ["user:operator@example.test"],
    })
    expected_root = _root_with_sensitive_binding(
        resource=f"projects/{contract.PROJECT}",
        role_resource=role,
        members=["user:operator@example.test"],
    )
    external_binding = next(
        item
        for item in expected_root["allowed_residual_bindings"]
        if item["role"] == role["name"]
    )
    external_definition = next(
        item
        for item in expected_root["allowed_residual_role_definitions"]
        if item["name"] == role["name"]
    )
    if drift == "member":
        external_binding["members"] = ["user:other@example.test"]
    elif drift == "condition":
        external_binding["condition"] = {
            "title": "temporary",
            "description": "Owner-pinned temporary access",
            "expression": "request.time < timestamp('2030-01-01T00:00:00Z')",
        }
    elif drift == "resource":
        external_binding["resource"] = ANCESTOR_CHAIN[0]
    else:
        external_definition["included_permissions"] = [
            "compute.disks.get",
            "compute.disks.resize",
        ]
    expected_root["allowed_residual_bindings"].sort(
        key=protocol.canonical_json_bytes
    )
    with pytest.raises(
        collector.TrustedObservationError,
        match="external_gcp_admin_trust_root_drift",
    ):
        _build_trusted_iam_projection_from_raw(
            report,
            _request(),
            project=project,
            ancestors=ancestors,
            policies=policies,
            roles=roles,
            account=account,
            additional_policy_roles=[
                _external_sensitive_role_resource(),
                role,
            ],
            external_gcp_admin_trust_root=expected_root,
        )


def test_policy_role_inventory_is_bounded_and_role_paths_are_exact() -> None:
    report = _source_report(collected_at=NOW - 1)
    project, ancestors, policies, roles, account = _iam_raw_inputs()
    policies[1]["bindings"].append({
        "role": "projects/other-project/roles/escape",
        "members": ["user:operator@example.test"],
    })
    with pytest.raises(
        collector.TrustedObservationError,
        match="policy_role_inventory_invalid",
    ):
        _build_trusted_iam_projection_from_raw(
            report,
            _request(),
            project=project,
            ancestors=ancestors,
            policies=policies,
            roles=roles,
            account=account,
        )

    project, ancestors, policies, roles, account = _iam_raw_inputs()
    policies[1]["bindings"].extend(
        {
            "role": f"roles/test.sensitiveInventory{index}",
            "members": ["user:operator@example.test"],
        }
        for index in range(collector.MAX_POLICY_ROLE_DEFINITIONS + 1)
    )
    with pytest.raises(
        collector.TrustedObservationError,
        match="policy_role_inventory_invalid",
    ):
        _build_trusted_iam_projection_from_raw(
            report,
            _request(),
            project=project,
            ancestors=ancestors,
            policies=policies,
            roles=roles,
            account=account,
        )


def test_mutation_binding_presence_is_correlated_to_activation_seal() -> None:
    report = _source_report(collected_at=NOW - 1)
    request = _request()
    project, ancestors, policies, roles, account = _iam_raw_inputs()
    policies[0]["bindings"] = [
        item for item in policies[0]["bindings"]
        if item["role"] != collector.OWNER_GATE_MUTATION_ROLE
    ]
    inactive = _build_trusted_iam_projection_from_raw(
        report,
        request,
        project=project,
        ancestors=ancestors,
        policies=policies,
        roles=roles,
        account=account,
        expected_mutation_binding_present=False,
        activation_seal_sha256=None,
    )
    authority = inactive["collector_service_account_authority"]
    assert authority["mutation_binding_present"] is False
    assert authority["mutation_condition"] is None
    assert authority["activation_seal_sha256"] is None
    with pytest.raises(
        collector.TrustedObservationError,
        match="collector_authority_invalid",
    ):
        collector.validate_trusted_iam_projection(
            inactive,
            observation_request=request,
            candidate_observation=report,
            now_unix=NOW,
            expected_project_number=PROJECT_NUMBER,
            expected_ancestor_chain=ANCESTOR_CHAIN,
            expected_runtime_instance_numeric_id=RUNTIME_INSTANCE_ID,
            expected_runtime_service_account_email=(
                collector.OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT
            ),
            expected_runtime_service_account_unique_id=(
                RUNTIME_SERVICE_ACCOUNT_UNIQUE_ID
            ),
            expected_target_service_account_unique_id=(
                TARGET_SERVICE_ACCOUNT_UNIQUE_ID
            ),
            expected_mutation_binding_present=True,
            expected_activation_seal_sha256=ACTIVATION_SEAL_SHA256,
        )


@pytest.mark.parametrize(
    ("surface", "error"),
    [
        ("project", "hierarchy_invalid"),
        ("policy", "policy_invalid"),
        ("target_role", "role_invalid"),
        ("collector_role", "collector_role_invalid"),
        ("collector_instance", "collector_identity_invalid"),
        ("collector_account", "collector_identity_invalid"),
        ("collector_policy", "collector_impersonation_invalid"),
        ("collector_keys", "collector_user_keys_invalid"),
        ("target_account", "service_account_invalid"),
    ],
)
def test_malformed_direct_api_json_fails_with_stable_boundary_error(
    surface: str,
    error: str,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    request = _request()
    project, ancestors, policies, roles, account = _iam_raw_inputs()
    collector_instance, collector_account = _collector_raw_inputs()
    collector_roles = _collector_role_raw_inputs()
    inputs: dict[str, Any] = {
        "project": project,
        "policies": policies,
        "roles": roles,
        "account": account,
        "collector_instance": collector_instance,
        "collector_account": collector_account,
        "collector_roles": collector_roles,
        "collector_policy": {
            "version": 1,
            "etag": "collector-service-account-policy-etag",
            "bindings": [],
        },
        "collector_keys": {"keys": []},
    }
    if surface == "project":
        inputs["project"] = []
    elif surface == "policy":
        cast(Any, inputs["policies"])[0] = []
    elif surface == "target_role":
        cast(Any, inputs["roles"])[0] = []
    elif surface == "collector_role":
        cast(Any, inputs["collector_roles"])[0] = []
    elif surface == "collector_instance":
        inputs["collector_instance"] = []
    elif surface == "collector_account":
        inputs["collector_account"] = []
    elif surface == "collector_policy":
        inputs["collector_policy"] = []
    elif surface == "collector_keys":
        inputs["collector_keys"] = []
    elif surface == "target_account":
        inputs["account"] = []
    with pytest.raises(collector.TrustedObservationError, match=error):
        _build_trusted_iam_projection_from_raw(
            report,
            request,
            project=cast(Any, inputs["project"]),
            ancestors=ancestors,
            policies=cast(Any, inputs["policies"]),
            roles=cast(Any, inputs["roles"]),
            account=cast(Any, inputs["account"]),
            collector_instance=cast(Any, inputs["collector_instance"]),
            collector_account=cast(Any, inputs["collector_account"]),
            collector_service_account_policy=cast(
                Any, inputs["collector_policy"]
            ),
            collector_user_managed_keys=cast(
                Any, inputs["collector_keys"]
            ),
            collector_roles=cast(Any, inputs["collector_roles"]),
        )


def test_direct_iam_hierarchy_accepts_org_only_and_multiple_folders() -> None:
    report = _source_report(collected_at=NOW - 1)
    request = _request()
    project, ancestors, policies, roles, account = _iam_raw_inputs()
    project["parent"] = ANCESTOR_CHAIN[-1]
    org_only = _build_trusted_iam_projection_from_raw(
        report,
        request,
        project=project,
        ancestors=[ancestors[-1]],
        policies=[policies[0], policies[-1]],
        roles=roles,
        account=account,
        ancestor_chain=(ANCESTOR_CHAIN[-1],),
    )
    assert org_only["resource_ancestor_chain"] == [ANCESTOR_CHAIN[-1]]

    project, ancestors, policies, roles, account = _iam_raw_inputs()
    second_folder = "folders/222222222"
    project["parent"] = second_folder
    extra_folder = {
        "name": second_folder,
        "parent": ANCESTOR_CHAIN[0],
        "state": "ACTIVE",
        "etag": "second-folder-etag",
    }
    extra_policy = {
        "version": 3,
        "etag": "second-folder-policy-etag",
        "bindings": [],
    }
    multi_chain = (second_folder, *ANCESTOR_CHAIN)
    expected_root = copy.deepcopy(
        _external_gcp_admin_trust_root(multi_chain)
    )
    expected_root["resource_policy_generations"][1]["etag"] = (
        "second-folder-policy-etag"
    )
    multi = _build_trusted_iam_projection_from_raw(
        report,
        request,
        project=project,
        ancestors=[extra_folder, *ancestors],
        policies=[policies[0], extra_policy, *policies[1:]],
        roles=roles,
        account=account,
        ancestor_chain=multi_chain,
        external_gcp_admin_trust_root=expected_root,
    )
    assert multi["resource_ancestor_chain"] == list(multi_chain)


def test_projection_omission_extra_field_and_cross_attempt_substitution_fail() -> None:
    report = _source_report(collected_at=NOW - 1)
    first = _request()
    second = _request(attempt_sequence=2, issued_at=NOW + 1)
    projection = dict(_trusted_iam_projection(report, first))
    projection.pop("project_policy_etag")
    with pytest.raises(
        collector.TrustedObservationError,
        match="iam_projection_fields_invalid",
    ):
        collector.build_attestation_request(
            first,
            report,
            role="host",
            trusted_iam_projection=projection,
            now_unix=NOW,
        )
    projection = dict(_trusted_iam_projection(report, first))
    projection["unexpected"] = True
    with pytest.raises(
        collector.TrustedObservationError,
        match="iam_projection_fields_invalid",
    ):
        collector.build_attestation_request(
            first,
            report,
            role="host",
            trusted_iam_projection=projection,
            now_unix=NOW,
        )
    with pytest.raises(
        collector.TrustedObservationError,
        match="iam_projection_invalid",
    ):
        collector.build_attestation_request(
            second,
            report,
            role="host",
            trusted_iam_projection=_trusted_iam_projection(report, first),
            now_unix=NOW + 1,
        )


def test_bundle_expiry_is_capped_by_trusted_iam_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collector, "TRUSTED_IAM_TTL_SECONDS", 60)
    report = _source_report(collected_at=NOW - 1)
    request = _request()
    cloud_key, host_key, cloud, host = _signed_pair(
        tmp_path, report, request
    )
    bundle = collector.combine_trusted_attestations(
        observation_request=request,
        refreshed_observation_request=request,
        candidate_observation=report,
        cloud_response=cloud,
        cloud_public_key=cloud_key.public_key(),
        host_response=host,
        host_public_key=host_key.public_key(),
        now_unix=NOW,
    )
    assert bundle["expires_at_unix"] == cloud[
        "trusted_iam_projection"
    ]["expires_at_unix"]


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("checkpoint", "post_resize", "request_invalid"),
        ("canonical_state", "await_post_resize", "request_invalid"),
        ("request_binding_sha256", "f" * 64, "request_invalid"),
        ("observation_nonce_sha256", "e" * 64, "request_invalid"),
        ("observation_request_sha256", "d" * 64, "request_invalid"),
    ],
)
def test_request_is_self_digested_and_stage_cannot_be_inferred(
    field: str, replacement: str, error: str
) -> None:
    request = dict(_request())
    request[field] = replacement
    with pytest.raises(collector.TrustedObservationError, match=error):
        collector.validate_observation_request(request)


def test_terminal_request_is_exact_short_circuit_and_never_reaches_signers(
    tmp_path: Path,
) -> None:
    terminal = _terminal_request()
    assert collector.classify_observation_request(terminal)[
        "canonical_state"
    ] == "terminal"
    with pytest.raises(
        collector.TrustedObservationError, match="request_terminal"
    ):
        collector.validate_observation_request(terminal)
    with pytest.raises(
        collector.TrustedObservationError, match="request_terminal"
    ):
        collector.build_attestation_request(
            terminal,
            _source_report(collected_at=NOW - 1),
            role="cloud",
            now_unix=NOW,
        )
    broken = dict(terminal)
    broken["request_binding_sha256"] = "1" * 64
    unsigned = dict(broken)
    unsigned.pop("observation_request_sha256")
    broken["observation_request_sha256"] = protocol.sha256_json(unsigned)
    with pytest.raises(
        collector.TrustedObservationError,
        match="terminal_request_invalid",
    ):
        collector.classify_observation_request(broken)


def test_independent_recollection_mismatch_fails_before_receipt(
    tmp_path: Path,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    key = Ed25519PrivateKey.generate()
    config = _config(tmp_path, "host", key)
    reader = HostReader(report)
    reader.projection["root_available_bytes"] += 1
    frame = _host_frame(_request(), report)
    with pytest.raises(
        collector.TrustedObservationError, match="host_candidate_mismatch"
    ):
        collector.run_host_attestor(
            frame,
            config=config,
            facts_reader=reader,
            now_unix=NOW,
        )
    assert list(Path(config["replay_directory"]).glob("host-*.json")) == []


def test_replay_returns_byte_identical_receipt_after_fresh_recollection(
    tmp_path: Path,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    key = Ed25519PrivateKey.generate()
    config = _config(tmp_path, "cloud", key)
    reader = CloudReader(report)
    frame = collector.build_attestation_request(
        _request(), report, role="cloud", now_unix=NOW
    )
    first = collector.run_cloud_attestor(
        frame, config=config, facts_reader=reader, now_unix=NOW
    )
    second = collector.run_cloud_attestor(
        frame, config=config, facts_reader=reader, now_unix=NOW
    )
    assert protocol.canonical_json_bytes(first) == protocol.canonical_json_bytes(
        second
    )
    assert reader.calls == 2
    receipts = list(Path(config["replay_directory"]).glob("cloud-*.json"))
    assert len(receipts) == 1
    assert receipts[0].read_bytes() == protocol.canonical_json_bytes(first)


def test_expired_collection_attempt_rotates_replay_namespace_and_signs_fresh(
    tmp_path: Path,
) -> None:
    key = Ed25519PrivateKey.generate()
    config = _config(tmp_path, "cloud", key)
    first_request = _request(issued_at=NOW)
    first_report = _source_report(collected_at=NOW)
    first = collector.run_cloud_attestor(
        collector.build_attestation_request(
            first_request,
            first_report,
            role="cloud",
            now_unix=NOW,
        ),
        config=config,
        facts_reader=CloudReader(first_report),
        now_unix=NOW,
    )
    rotated_at = NOW + evidence.OBSERVATION_BUNDLE_TTL_SECONDS
    second_request = _request(
        attempt_sequence=2,
        issued_at=rotated_at,
    )
    second_report = _source_report(collected_at=rotated_at)
    second = collector.run_cloud_attestor(
        collector.build_attestation_request(
            second_request,
            second_report,
            role="cloud",
            now_unix=rotated_at,
        ),
        config=config,
        facts_reader=CloudReader(second_report),
        now_unix=rotated_at,
    )
    assert first["observation_request_sha256"] != second[
        "observation_request_sha256"
    ]
    assert len(list(Path(config["replay_directory"]).glob("cloud-*.json"))) == 2


def test_cross_attempt_response_rewrap_cannot_mix_old_cloud_and_new_host(
    tmp_path: Path,
) -> None:
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    first_request = _request(issued_at=NOW)
    late = NOW + evidence.OBSERVATION_BUNDLE_TTL_SECONDS - 1
    report = _source_report(collected_at=late)
    first_cloud_frame = collector.build_attestation_request(
        first_request, report, role="cloud", now_unix=late
    )
    first_cloud = collector.run_cloud_attestor(
        first_cloud_frame,
        config=_config(tmp_path, "cloud", cloud_key),
        facts_reader=CloudReader(report),
        now_unix=late,
    )
    rotated_at = NOW + evidence.OBSERVATION_BUNDLE_TTL_SECONDS
    second_request = _request(
        attempt_sequence=2,
        issued_at=rotated_at,
    )
    second_projection = _trusted_iam_projection(
        report, second_request, now_unix=rotated_at
    )
    second_host_frame = collector.build_attestation_request(
        second_request,
        report,
        role="host",
        trusted_iam_projection=second_projection,
        now_unix=rotated_at,
    )
    second_host = collector.run_host_attestor(
        second_host_frame,
        config=_config(tmp_path, "host", host_key),
        facts_reader=HostReader(report),
        now_unix=rotated_at,
    )
    second_cloud_frame = collector.build_attestation_request(
        second_request, report, role="cloud", now_unix=rotated_at
    )
    second_core = collector._bundle_core(
        second_request, report, second_projection
    )
    rewrapped = dict(first_cloud)
    rewrapped.update({
        "attestation_request_sha256": second_cloud_frame["frame_sha256"],
        "observation_request_sha256": second_request[
            "observation_request_sha256"
        ],
        "trusted_iam_projection": second_projection,
        "trusted_iam_projection_sha256": protocol.sha256_json(
            second_projection
        ),
        "bundle_core_sha256": protocol.sha256_json(second_core),
    })
    unsigned = dict(rewrapped)
    unsigned.pop("response_sha256")
    rewrapped["response_sha256"] = protocol.sha256_json(unsigned)
    with pytest.raises(
        collector.TrustedObservationError, match="signature_invalid"
    ):
        collector.combine_trusted_attestations(
            observation_request=second_request,
            refreshed_observation_request=second_request,
            candidate_observation=report,
            cloud_response=rewrapped,
            cloud_public_key=cloud_key.public_key(),
            host_response=second_host,
            host_public_key=host_key.public_key(),
            now_unix=rotated_at,
        )


def test_bundle_lifetime_is_capped_by_collection_attempt_expiry(
    tmp_path: Path,
) -> None:
    request = _request(issued_at=NOW)
    observed = NOW + evidence.OBSERVATION_BUNDLE_TTL_SECONDS - 1
    report = _source_report(collected_at=observed)
    cloud_key, host_key, cloud, host = _signed_pair_at(
        tmp_path, report, request, now_unix=observed
    )
    bundle = collector.combine_trusted_attestations(
        observation_request=request,
        refreshed_observation_request=request,
        candidate_observation=report,
        cloud_response=cloud,
        cloud_public_key=cloud_key.public_key(),
        host_response=host,
        host_public_key=host_key.public_key(),
        now_unix=observed,
    )
    assert bundle["expires_at_unix"] == request[
        "collection_attempt_expires_at_unix"
    ]
    with pytest.raises(
        evidence.StorageGrowthEvidenceError, match="binding_invalid"
    ):
        evidence.validate_attested_observation(
            bundle,
            cloud_public_key=cloud_key.public_key(),
            cloud_public_key_id=cloud["attestation"]["public_key_id"],
            host_public_key=host_key.public_key(),
            host_public_key_id=host["attestation"]["public_key_id"],
            now_unix=bundle["expires_at_unix"],
            allowed_states=frozenset({"source_ready"}),
            expected_transaction_id=TRANSACTION,
            expected_checkpoint="source",
            expected_request_binding_sha256=request[
                "request_binding_sha256"
            ],
            expected_prior_event_head_sha256=PRIOR_HEAD,
            expected_observation_request_sha256=request[
                "observation_request_sha256"
            ],
            expected_collection_attempt_id=request[
                "collection_attempt_id"
            ],
            expected_collection_attempt_sequence=request[
                "collection_attempt_sequence"
            ],
            expected_collection_attempt_issued_at_unix=request[
                "collection_attempt_issued_at_unix"
            ],
            expected_collection_attempt_expires_at_unix=request[
                "collection_attempt_expires_at_unix"
            ],
        )


def test_private_key_mode_symlink_and_public_id_are_fail_closed(
    tmp_path: Path,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    key = Ed25519PrivateKey.generate()
    config = dict(_config(tmp_path, "host", key))
    frame = _host_frame(_request(), report)
    key_path = Path(config["private_key_path"])
    key_path.chmod(0o440)
    with pytest.raises(
        collector.TrustedObservationError, match="file_metadata_invalid"
    ):
        collector.run_host_attestor(
            frame,
            config=config,
            facts_reader=HostReader(report),
            now_unix=NOW,
        )
    key_path.chmod(0o400)
    config["public_key_id"] = "f" * 64
    with pytest.raises(
        collector.TrustedObservationError,
        match="private_key_identity_invalid",
    ):
        collector.run_host_attestor(
            frame,
            config=config,
            facts_reader=HostReader(report),
            now_unix=NOW,
        )
    replacement = key_path.with_suffix(".real")
    key_path.rename(replacement)
    key_path.symlink_to(replacement)
    config["public_key_id"] = protocol.sha256_bytes(
        key.public_key().public_bytes_raw()
    )
    with pytest.raises(
        collector.TrustedObservationError, match="file_unavailable"
    ):
        collector.run_host_attestor(
            frame,
            config=config,
            facts_reader=HostReader(report),
            now_unix=NOW,
        )


def test_post_stop_exact_terminated_snapshot_requires_cloud_only(
    tmp_path: Path,
) -> None:
    report = _pending_report(collected_at=NOW - 1, status="TERMINATED")
    request = _request("post_stop", prior_head="c" * 64)
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    assert not collector.host_attestation_required(
        request, report, now_unix=NOW
    )
    with pytest.raises(
        collector.TrustedObservationError,
        match="host_attestation_forbidden",
    ):
        collector.build_attestation_request(
            request,
            report,
            role="host",
            trusted_iam_projection=_trusted_iam_projection(
                report, request
            ),
            now_unix=NOW,
        )
    cloud_frame = collector.build_attestation_request(
        request, report, role="cloud", now_unix=NOW
    )
    cloud = collector.run_cloud_attestor(
        cloud_frame,
        config=_config(tmp_path, "cloud", cloud_key),
        facts_reader=CloudReader(report),
        now_unix=NOW,
    )
    bundle = collector.combine_trusted_attestations(
        observation_request=request,
        refreshed_observation_request=request,
        candidate_observation=report,
        cloud_response=cloud,
        cloud_public_key=cloud_key.public_key(),
        host_response=None,
        host_public_key=host_key.public_key(),
        now_unix=NOW,
    )
    assert bundle["host_attestation"] is None


def test_refreshed_ledger_request_must_be_byte_exact(
    tmp_path: Path,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    request = _request()
    cloud_key, host_key, cloud, host = _signed_pair(
        tmp_path, report, request
    )
    refreshed = _request(prior_head="d" * 64)
    with pytest.raises(
        collector.TrustedObservationError, match="request_changed"
    ):
        collector.combine_trusted_attestations(
            observation_request=request,
            refreshed_observation_request=refreshed,
            candidate_observation=report,
            cloud_response=cloud,
            cloud_public_key=cloud_key.public_key(),
            host_response=host,
            host_public_key=host_key.public_key(),
            now_unix=NOW,
        )


def test_one_signer_on_a_different_core_and_wrong_key_are_rejected(
    tmp_path: Path,
) -> None:
    source = _source_report(collected_at=NOW - 1)
    target = _target_report(collected_at=NOW - 1)
    request = _request()
    cloud_key, host_key, cloud, host = _signed_pair(
        tmp_path / "source", source, request
    )
    other_request = _request("post_start", prior_head="8" * 64)
    other_cloud_key, _other_host_key, other_cloud, _other_host = _signed_pair(
        tmp_path / "target", target, other_request
    )
    with pytest.raises(
        collector.TrustedObservationError,
        match="iam_projection_invalid|attestation_response_invalid",
    ):
        collector.combine_trusted_attestations(
            observation_request=request,
            refreshed_observation_request=request,
            candidate_observation=source,
            cloud_response=other_cloud,
            cloud_public_key=other_cloud_key.public_key(),
            host_response=host,
            host_public_key=host_key.public_key(),
            now_unix=NOW,
        )


@pytest.mark.parametrize("field", ["role", "public_key_id"])
def test_role_or_key_id_substitution_is_rejected(
    tmp_path: Path, field: str
) -> None:
    report = _source_report(collected_at=NOW - 1)
    request = _request()
    cloud_key, host_key, cloud, host = _signed_pair(
        tmp_path, report, request
    )
    changed = copy.deepcopy(cloud)
    if field == "role":
        changed["role"] = "host"
    else:
        changed["attestation"]["public_key_id"] = "0" * 64
    unsigned = dict(changed)
    unsigned.pop("response_sha256")
    changed["response_sha256"] = protocol.sha256_json(unsigned)
    with pytest.raises(
        collector.TrustedObservationError,
        match="attestation_response_invalid",
    ):
        collector.combine_trusted_attestations(
            observation_request=request,
            refreshed_observation_request=request,
            candidate_observation=report,
            cloud_response=changed,
            cloud_public_key=cloud_key.public_key(),
            host_response=host,
            host_public_key=host_key.public_key(),
            now_unix=NOW,
        )
    with pytest.raises(
        collector.TrustedObservationError,
        match="attestation_response_invalid",
    ):
        collector.combine_trusted_attestations(
            observation_request=request,
            refreshed_observation_request=request,
            candidate_observation=report,
            cloud_response=cloud,
            cloud_public_key=Ed25519PrivateKey.generate().public_key(),
            host_response=host,
            host_public_key=host_key.public_key(),
            now_unix=NOW,
        )


def test_host_null_is_forbidden_outside_exact_post_stop_snapshot(
    tmp_path: Path,
) -> None:
    report = _pending_report(collected_at=NOW - 1)
    request = _request("post_resize", prior_head="7" * 64)
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    frame = collector.build_attestation_request(
        request, report, role="cloud", now_unix=NOW
    )
    cloud = collector.run_cloud_attestor(
        frame,
        config=_config(tmp_path, "cloud", cloud_key),
        facts_reader=CloudReader(report),
        now_unix=NOW,
    )
    with pytest.raises(
        collector.TrustedObservationError,
        match="host_attestor_unavailable",
    ):
        collector.combine_trusted_attestations(
            observation_request=request,
            refreshed_observation_request=request,
            candidate_observation=report,
            cloud_response=cloud,
            cloud_public_key=cloud_key.public_key(),
            host_response=None,
            host_public_key=host_key.public_key(),
            now_unix=NOW,
        )


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b"{}\n{}\n",
        b'{"a": 1}\n',
        b'{"a":1,"a":2}\n',
        b"x" * (collector.MAX_FRAME_BYTES + 1) + b"\n",
    ],
)
def test_stdin_is_one_bounded_canonical_newline_frame(raw: bytes) -> None:
    with pytest.raises(
        collector.TrustedObservationError, match="stdin_frame_invalid"
    ):
        collector._read_canonical_line(io.BytesIO(raw))


def test_fixed_once_entrypoint_emits_one_canonical_response_line(
    tmp_path: Path,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    request = _request()
    key = Ed25519PrivateKey.generate()
    config = _config(tmp_path, "host", key)
    frame = _host_frame(request, report)
    stdin = io.BytesIO(protocol.canonical_json_bytes(frame) + b"\n")
    stdout = io.BytesIO()
    assert collector.serve_host_attestor_once(
        stdin=stdin,
        stdout=stdout,
        config=config,
        facts_reader=HostReader(report),
        now_unix=NOW,
    ) == 0
    raw = stdout.getvalue()
    assert raw.endswith(b"\n") and b"\n" not in raw[:-1]
    response = protocol.decode_canonical_json(raw[:-1])
    assert response["role"] == "host"
    assert "private" not in raw.decode("utf-8")


def test_cloud_reader_enforces_exact_numeric_resources_and_iam() -> None:
    report = _source_report(collected_at=NOW - 1)
    projection = collector._cloud_recollection(
        _instance(), _disk(40), observation=report
    )
    assert projection == _cloud_projection(report)
    attacker = _instance()
    attacker["id"] = "9153645328899914618"
    with pytest.raises(
        collector.TrustedObservationError, match="cloud_instance_invalid"
    ):
        collector._cloud_recollection(
            attacker, _disk(40), observation=report
        )


def test_candidate_expiry_blocks_signing_even_with_existing_replay(
    tmp_path: Path,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    key = Ed25519PrivateKey.generate()
    config = _config(tmp_path, "cloud", key)
    frame = collector.build_attestation_request(
        _request(), report, role="cloud", now_unix=NOW
    )
    collector.run_cloud_attestor(
        frame,
        config=config,
        facts_reader=CloudReader(report),
        now_unix=NOW,
    )
    with pytest.raises(
        collector.TrustedObservationError, match="candidate_expired"
    ):
        collector.run_cloud_attestor(
            frame,
            config=config,
            facts_reader=CloudReader(report),
            now_unix=NOW - 1 + evidence.OBSERVATION_BUNDLE_TTL_SECONDS,
        )


def test_partial_receipt_crash_is_not_visible_and_retry_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _source_report(collected_at=NOW - 1)
    key = Ed25519PrivateKey.generate()
    config = _config(tmp_path, "cloud", key)
    frame = collector.build_attestation_request(
        _request(), report, role="cloud", now_unix=NOW
    )
    real_rename = collector.os.rename
    calls = 0

    def crash_once(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected crash before publish")
        real_rename(*args, **kwargs)

    monkeypatch.setattr(collector.os, "rename", crash_once)
    with pytest.raises(
        collector.TrustedObservationError, match="receipt_write_failed"
    ):
        collector.run_cloud_attestor(
            frame,
            config=config,
            facts_reader=CloudReader(report),
            now_unix=NOW,
        )
    replay = Path(config["replay_directory"])
    assert list(replay.glob("cloud-*.json")) == []
    response = collector.run_cloud_attestor(
        frame,
        config=config,
        facts_reader=CloudReader(report),
        now_unix=NOW,
    )
    assert len(list(replay.glob("cloud-*.json"))) == 1
    assert protocol.decode_canonical_json(
        list(replay.glob("cloud-*.json"))[0].read_bytes()
    ) == response


class _HttpResponse:
    def __init__(
        self,
        body: Mapping[str, Any] | bytes | str,
        *,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status = status
        self._raw = (
            body
            if isinstance(body, bytes)
            else body.encode("utf-8")
            if isinstance(body, str)
            else json.dumps(body, separators=(",", ":")).encode("utf-8")
        )
        default_type = (
            "application/text; charset=UTF-8"
            if isinstance(body, (bytes, str))
            else "application/json; charset=UTF-8"
        )
        self._headers = {
            "Content-Length": str(len(self._raw)),
            "Content-Type": default_type,
            **dict(headers or {}),
        }
        self._offset = 0

    def getheader(self, name: str) -> str | None:
        return self._headers.get(name)

    def read(self, amount: int) -> bytes:
        raw = self._raw[self._offset : self._offset + amount]
        self._offset += len(raw)
        return raw


def test_cloud_http_body_bound_accepts_large_role_and_rejects_oversize(
) -> None:
    current_large_role = b"x" * (600 * 1024)
    chunked_response = _HttpResponse(current_large_role)
    chunked_response._headers.pop("Content-Length")
    assert (
        collector._bounded_http_body(chunked_response)
        == current_large_role
    )

    with pytest.raises(
        collector.TrustedObservationError,
        match="^trusted_observation_cloud_http_invalid$",
    ):
        collector._bounded_http_body(
            _HttpResponse(b"x" * (collector.MAX_HTTP_BODY_BYTES + 1))
        )


class _HttpConnection:
    def __init__(self, responses: list[_HttpResponse]) -> None:
        self.responses = responses
        self.requests: list[
            tuple[str, str, bytes | None, Mapping[str, str]]
        ] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: Mapping[str, str],
    ) -> None:
        self.requests.append((method, path, body, dict(headers)))

    def getresponse(self) -> _HttpResponse:
        return self.responses.pop(0)

    def close(self) -> None:
        pass


def test_compute_reader_uses_metadata_header_exact_gets_and_ignores_proxy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    metadata = _HttpConnection([
        _HttpResponse(
            RUNTIME_INSTANCE_ID,
            headers={"Metadata-Flavor": "Google"},
        ),
        _HttpResponse(
            collector.OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT,
            headers={"Metadata-Flavor": "Google"},
        ),
        _HttpResponse(
            "\n".join(collector.OWNER_GATE_METADATA_SCOPES) + "\n",
            headers={"Metadata-Flavor": "Google"},
        ),
        _HttpResponse(
            {"access_token": "not-printed", "expires_in": 300, "token_type": "Bearer"},
            headers={"Metadata-Flavor": "Google"},
        )
    ])
    compute_connections = [
        _HttpConnection([_HttpResponse(_instance())]),
        _HttpConnection([_HttpResponse(_disk(40))]),
        _HttpConnection([_HttpResponse(_collector_raw_inputs()[0])]),
    ]
    project, ancestors, policies, roles, account = _iam_raw_inputs()
    resource_manager_connections = [
        *[_HttpConnection([_HttpResponse(item)]) for item in [project, *ancestors]],
        *[_HttpConnection([_HttpResponse(item)]) for item in policies],
    ]
    external_role_call = _HttpConnection([
        _HttpResponse(_external_sensitive_role_resource())
    ])
    iam_calls = [
        *[_HttpConnection([_HttpResponse(item)]) for item in roles],
        *[
            _HttpConnection([_HttpResponse(item)])
            for item in _collector_role_raw_inputs()
        ],
        external_role_call,
        _HttpConnection([_HttpResponse(account)]),
        _HttpConnection([_HttpResponse(_collector_raw_inputs()[1])]),
        _HttpConnection([_HttpResponse({
            "version": 1,
            "etag": "collector-service-account-policy-etag",
            "bindings": [],
        })]),
        _HttpConnection([_HttpResponse({"keys": []})]),
    ]
    iam_connections = list(iam_calls)
    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.invalid:3128")
    reader = collector.FixedComputeFactsReader(
        expected_project_number=PROJECT_NUMBER,
        expected_ancestor_chain=ANCESTOR_CHAIN,
        expected_runtime_instance_numeric_id=RUNTIME_INSTANCE_ID,
        expected_runtime_service_account_email=(
            collector.OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT
        ),
        expected_runtime_service_account_unique_id=(
            RUNTIME_SERVICE_ACCOUNT_UNIQUE_ID
        ),
        expected_target_service_account_unique_id=(
            TARGET_SERVICE_ACCOUNT_UNIQUE_ID
        ),
        expected_metadata_scopes=collector.OWNER_GATE_METADATA_SCOPES,
        expected_external_gcp_admin_trust_root=(
            _external_gcp_admin_trust_root()
        ),
        expected_mutation_binding_present=True,
        activation_seal_sha256=ACTIVATION_SEAL_SHA256,
        metadata_connection_factory=lambda: metadata,
        compute_connection_factory=lambda: compute_connections.pop(0),
        resource_manager_connection_factory=(
            lambda: resource_manager_connections.pop(0)
        ),
        iam_connection_factory=lambda: iam_connections.pop(0),
    )
    recollection, projection = reader.collect(report, _request(), NOW)
    assert recollection == _cloud_projection(report)
    assert projection["authority_scope"] == (
        collector.TRUSTED_IAM_AUTHORITY_SCOPE
    )
    assert projection["project_policy_version"] == 3
    assert projection["project_policy_etag"] == "project-policy-etag"
    assert projection["collector_service_account_authority"][
        "service_account_iam_policy"
    ]["version"] == 1
    assert projection["collector_service_account_authority"][
        "service_account_iam_policy"
    ]["etag"] == "collector-service-account-policy-etag"
    assert metadata.requests == [
        (
            "GET",
            collector.METADATA_INSTANCE_ID_PATH,
            None,
            {
                "Metadata-Flavor": "Google",
                "Accept": collector.METADATA_TEXT_MEDIA_TYPE,
                "Connection": "close",
            },
        ),
        (
            "GET",
            collector.METADATA_SERVICE_ACCOUNT_EMAIL_PATH,
            None,
            {
                "Metadata-Flavor": "Google",
                "Accept": collector.METADATA_TEXT_MEDIA_TYPE,
                "Connection": "close",
            },
        ),
        (
            "GET",
            collector.METADATA_SCOPES_PATH,
            None,
            {
                "Metadata-Flavor": "Google",
                "Accept": collector.METADATA_TEXT_MEDIA_TYPE,
                "Connection": "close",
            },
        ),
        (
            "GET",
            collector.METADATA_TOKEN_PATH,
            None,
            {
                "Metadata-Flavor": "Google",
                "Accept": "application/json",
                "Connection": "close",
            },
        )
    ]
    # Factories, not urllib or an ambient proxy-aware client, received both
    # direct fixed paths. The bearer value is never present in the result.
    assert compute_connections == []
    assert resource_manager_connections == []
    assert iam_connections == []
    assert iam_calls[-2].requests == [
        (
            "POST",
            collector.COLLECTOR_SERVICE_ACCOUNT_IAM_POLICY_PATH,
            None,
            {
                "Authorization": "Bearer not-printed",
                "Accept": "application/json",
                "Connection": "close",
            },
        )
    ]
    assert external_role_call.requests == [
        (
            "GET",
            f"/v1/{EXTERNAL_SENSITIVE_ROLE}",
            None,
            {
                "Authorization": "Bearer not-printed",
                "Accept": "application/json",
                "Connection": "close",
            },
        )
    ]
    assert iam_calls[-1].requests == [
        (
            "GET",
            collector.COLLECTOR_SERVICE_ACCOUNT_USER_KEYS_PATH,
            None,
            {
                "Authorization": "Bearer not-printed",
                "Accept": "application/json",
                "Connection": "close",
            },
        )
    ]


@pytest.mark.parametrize(
    ("metadata_headers", "compute_headers", "error"),
    [
        ({}, {}, "metadata_token_invalid"),
        ({"Metadata-Flavor": "Google", "Location": "http://evil"}, {}, "metadata_token_invalid"),
        ({"Metadata-Flavor": "Google"}, {"Location": "https://evil"}, "cloud_resource_unavailable"),
    ],
)
def test_metadata_and_compute_redirect_or_header_drift_fail_closed(
    metadata_headers: Mapping[str, str],
    compute_headers: Mapping[str, str],
    error: str,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    metadata = _HttpConnection([
        _HttpResponse(
            RUNTIME_INSTANCE_ID,
            headers=metadata_headers,
        ),
        _HttpResponse(
            collector.OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT,
            headers={"Metadata-Flavor": "Google"},
        ),
        _HttpResponse(
            "\n".join(collector.OWNER_GATE_METADATA_SCOPES) + "\n",
            headers={"Metadata-Flavor": "Google"},
        ),
        _HttpResponse(
            {"access_token": "opaque", "expires_in": 300, "token_type": "Bearer"},
            headers={"Metadata-Flavor": "Google"},
        ),
    ])
    compute = [
        _HttpConnection([_HttpResponse(_instance(), headers=compute_headers)]),
        _HttpConnection([_HttpResponse(_disk(40))]),
        _HttpConnection([_HttpResponse(_collector_raw_inputs()[0])]),
    ]
    reader = collector.FixedComputeFactsReader(
        expected_project_number=PROJECT_NUMBER,
        expected_ancestor_chain=ANCESTOR_CHAIN,
        expected_runtime_instance_numeric_id=RUNTIME_INSTANCE_ID,
        expected_runtime_service_account_email=(
            collector.OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT
        ),
        expected_runtime_service_account_unique_id=(
            RUNTIME_SERVICE_ACCOUNT_UNIQUE_ID
        ),
        expected_target_service_account_unique_id=(
            TARGET_SERVICE_ACCOUNT_UNIQUE_ID
        ),
        expected_metadata_scopes=collector.OWNER_GATE_METADATA_SCOPES,
        expected_external_gcp_admin_trust_root=(
            _external_gcp_admin_trust_root()
        ),
        expected_mutation_binding_present=True,
        activation_seal_sha256=ACTIVATION_SEAL_SHA256,
        metadata_connection_factory=lambda: metadata,
        compute_connection_factory=lambda: compute.pop(0),
        resource_manager_connection_factory=lambda: (_ for _ in ()).throw(
            AssertionError("resource manager must not be reached")
        ),
        iam_connection_factory=lambda: (_ for _ in ()).throw(
            AssertionError("IAM must not be reached")
        ),
    )
    with pytest.raises(collector.TrustedObservationError, match=error):
        reader.collect(report, _request(), NOW)


@pytest.mark.parametrize("media_type", ["text/html", "application/json"])
def test_metadata_text_response_media_type_drift_fails_closed(
    media_type: str,
) -> None:
    report = _source_report(collected_at=NOW - 1)
    metadata = _HttpConnection([
        _HttpResponse(
            RUNTIME_INSTANCE_ID,
            headers={
                "Metadata-Flavor": "Google",
                "Content-Type": media_type,
            },
        ),
    ])
    reader = collector.FixedComputeFactsReader(
        expected_project_number=PROJECT_NUMBER,
        expected_ancestor_chain=ANCESTOR_CHAIN,
        expected_runtime_instance_numeric_id=RUNTIME_INSTANCE_ID,
        expected_runtime_service_account_email=(
            collector.OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT
        ),
        expected_runtime_service_account_unique_id=(
            RUNTIME_SERVICE_ACCOUNT_UNIQUE_ID
        ),
        expected_target_service_account_unique_id=(
            TARGET_SERVICE_ACCOUNT_UNIQUE_ID
        ),
        expected_metadata_scopes=collector.OWNER_GATE_METADATA_SCOPES,
        expected_external_gcp_admin_trust_root=(
            _external_gcp_admin_trust_root()
        ),
        expected_mutation_binding_present=True,
        activation_seal_sha256=ACTIVATION_SEAL_SHA256,
        metadata_connection_factory=lambda: metadata,
        compute_connection_factory=lambda: (_ for _ in ()).throw(
            AssertionError("Compute must not be reached")
        ),
        resource_manager_connection_factory=lambda: (_ for _ in ()).throw(
            AssertionError("resource manager must not be reached")
        ),
        iam_connection_factory=lambda: (_ for _ in ()).throw(
            AssertionError("IAM must not be reached")
        ),
    )
    with pytest.raises(
        collector.TrustedObservationError,
        match="metadata_token_invalid",
    ):
        reader.collect(report, _request(), NOW)


def _test_ca_bundle_bytes() -> bytes:
    candidates = (
        collector.ssl.get_default_verify_paths().cafile,
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/ssl/cert.pem",
    )
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            payload = Path(candidate).read_bytes()
            if b"-----BEGIN CERTIFICATE-----" in payload:
                return payload
    pytest.fail("system test CA bundle unavailable")


def _prepare_fixed_test_ca(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: bytes | None = None,
) -> Path:
    for name in collector._FORBIDDEN_TLS_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "ca-certificates.crt"
    path.write_bytes(_test_ca_bundle_bytes() if payload is None else payload)
    path.chmod(0o400)
    monkeypatch.setattr(collector, "FIXED_DEBIAN_CA_BUNDLE_PATH", path)
    monkeypatch.setattr(collector, "_TRUSTED_CA_OWNER_UID", os.geteuid())
    monkeypatch.setattr(
        collector,
        "_fixed_ca_parent_identities",
        lambda _path: (("/fixed/root-owned/parent", (1,) * 9),),
    )
    return path


def test_fixed_debian_tls_context_reads_descriptor_bytes_into_cadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _test_ca_bundle_bytes()
    _prepare_fixed_test_ca(tmp_path, monkeypatch, payload=payload)
    captured: dict[str, Any] = {}

    class FakeContext:
        keylog_filename = None

        def __init__(self, protocol: Any) -> None:
            captured["protocol"] = protocol

        def load_verify_locations(self, **kwargs: Any) -> None:
            captured["load"] = kwargs

    monkeypatch.setattr(collector.ssl, "SSLContext", FakeContext)

    context = collector.fixed_debian_tls_context()

    assert isinstance(context, FakeContext)
    assert captured["protocol"] == collector.ssl.PROTOCOL_TLS_CLIENT
    assert "cafile" not in captured["load"]
    assert captured["load"] == {"cadata": payload.decode("ascii")}
    assert context.minimum_version == collector.ssl.TLSVersion.TLSv1_2
    assert context.verify_mode == collector.ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_fixed_debian_tls_context_builds_real_verified_no_keylog_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_fixed_test_ca(tmp_path, monkeypatch)

    context = collector.fixed_debian_tls_context()

    assert isinstance(context, collector.ssl.SSLContext)
    assert context.verify_mode == collector.ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert context.keylog_filename is None


@pytest.mark.parametrize(
    "name",
    [
        "HTTPS_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "SSLKEYLOGFILE",
        "OPENSSL_CONF",
        "OPENSSL_MODULES",
    ],
)
def test_fixed_debian_tls_context_rejects_ambient_network_influence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    _prepare_fixed_test_ca(tmp_path, monkeypatch)
    monkeypatch.setenv(name, "/tmp/attacker-controlled")

    with pytest.raises(
        collector.TrustedObservationError,
        match="trusted_observation_cloud_tls_invalid",
    ):
        collector.fixed_debian_tls_context()


def test_fixed_debian_tls_context_rejects_symlink_writable_wrong_owner_and_oversize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = _prepare_fixed_test_ca(tmp_path, monkeypatch)
    link = tmp_path / "linked-ca.crt"
    link.symlink_to(real)
    monkeypatch.setattr(collector, "FIXED_DEBIAN_CA_BUNDLE_PATH", link)
    with pytest.raises(
        collector.TrustedObservationError,
        match="trusted_observation_cloud_tls_invalid",
    ):
        collector.fixed_debian_tls_context()

    monkeypatch.setattr(collector, "FIXED_DEBIAN_CA_BUNDLE_PATH", real)
    real.chmod(0o620)
    with pytest.raises(
        collector.TrustedObservationError,
        match="trusted_observation_cloud_tls_invalid",
    ):
        collector.fixed_debian_tls_context()

    real.chmod(0o400)
    monkeypatch.setattr(collector, "_TRUSTED_CA_OWNER_UID", os.geteuid() + 1)
    with pytest.raises(
        collector.TrustedObservationError,
        match="trusted_observation_cloud_tls_invalid",
    ):
        collector.fixed_debian_tls_context()

    monkeypatch.setattr(collector, "_TRUSTED_CA_OWNER_UID", os.geteuid())
    real.chmod(0o600)
    with real.open("wb") as stream:
        stream.truncate(collector.MAX_TRUSTED_CA_BUNDLE_BYTES + 1)
    real.chmod(0o400)
    with pytest.raises(
        collector.TrustedObservationError,
        match="trusted_observation_cloud_tls_invalid",
    ):
        collector.fixed_debian_tls_context()


def test_fixed_debian_tls_context_rejects_path_replacement_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _prepare_fixed_test_ca(tmp_path, monkeypatch)
    replacement = tmp_path / "replacement-ca.crt"
    replacement.write_bytes(_test_ca_bundle_bytes())
    replacement.chmod(0o400)
    real_read = os.read
    replaced = False

    def replacing_read(descriptor: int, maximum: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, maximum)
        if chunk and not replaced:
            os.replace(replacement, path)
            replaced = True
        return chunk

    monkeypatch.setattr(collector.os, "read", replacing_read)

    with pytest.raises(
        collector.TrustedObservationError,
        match="trusted_observation_cloud_tls_changed",
    ):
        collector.fixed_debian_tls_context()


def test_wrong_checkpoint_state_is_rejected() -> None:
    target = _target_report(collected_at=NOW - 1)
    with pytest.raises(
        collector.TrustedObservationError, match="candidate_invalid"
    ):
        collector.build_attestation_request(
            _request("post_stop", prior_head="e" * 64),
            target,
            role="cloud",
            now_unix=NOW,
        )
