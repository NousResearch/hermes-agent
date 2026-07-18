from __future__ import annotations

import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import direct_iam_identity_author as direct_iam_author
from scripts.canary import owner_gate_cloud_observation_author as author
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_preflight as preflight
from scripts.canary import owner_gate_project_ancestry as project_ancestry
from tests.scripts.canary.test_owner_gate_foundation import (
    IMAGE,
    NOW,
    REVISION,
    _signed_network_evidence,
)


ORGANIZATION = "123456789012"
FOLDER = "234567890123"
PROJECT_NUMBER = "39589465056"
OWNER_GATE_VM_ID = "1234567890123456789"
OWNER_GATE_SERVICE_ACCOUNT_ID = "112233445566778899001"
OWNER_GATE_SUBNET_ID = "2233445566778899001"
OWNER_GATE_BOOT_DISK_ID = "3344556677889900112"
OWNER_GATE_FIREWALL_ID = "4455667788990011223"


def _key_id(key: Ed25519PrivateKey) -> str:
    return hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()


def _context(
    *, with_folder: bool = False
) -> tuple[
    foundation.OwnerGateFoundationPlan,
    project_ancestry.ProjectAncestryEvidence,
    Ed25519PrivateKey,
]:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    project_parent = (
        f"folders/{FOLDER}" if with_folder else f"organizations/{ORGANIZATION}"
    )
    ordered_chain = [
        {
            "resource_type": "project",
            "resource_name": f"projects/{PROJECT_NUMBER}",
            "numeric_id": PROJECT_NUMBER,
            "display_name": "Adventico AI Platform",
            "state": "ACTIVE",
            "etag": "project-etag",
            "parent_resource_name": project_parent,
        },
        *(
            [
                {
                    "resource_type": "folder",
                    "resource_name": f"folders/{FOLDER}",
                    "numeric_id": FOLDER,
                    "display_name": "Production",
                    "state": "ACTIVE",
                    "etag": "folder-etag",
                    "parent_resource_name": f"organizations/{ORGANIZATION}",
                }
            ]
            if with_folder
            else []
        ),
        {
            "resource_type": "organization",
            "resource_name": f"organizations/{ORGANIZATION}",
            "numeric_id": ORGANIZATION,
            "display_name": "adventico.com",
            "state": "ACTIVE",
            "etag": "organization-etag",
            "parent_resource_name": None,
        },
    ]
    ancestry_value = {
        "organization_id": ORGANIZATION,
        "project_number": PROJECT_NUMBER,
        "ordered_chain": ordered_chain,
        "stable_chain_sha256": foundation.sha256_json(ordered_chain),
    }
    ancestry_raw = foundation.canonical_json_bytes(ancestry_value)
    ancestry = project_ancestry.ProjectAncestryEvidence(
        ancestry_value,
        ancestry_raw,
    )
    network = foundation.ProductionNetworkEvidence.from_mapping(
        _signed_network_evidence(network_key),
        public_key=network_key.public_key(),
        expected_public_key_id=_key_id(network_key),
        now_unix=NOW,
    )
    plan = foundation.build_plan(
        spec=foundation.OwnerGateSpec(
            release_revision=REVISION,
            boot_image_self_link=IMAGE,
            package_inventory_sha256="5" * 64,
            interpreter_sha256="6" * 64,
            network_collector_public_key_id=_key_id(network_key),
            organization_id=ORGANIZATION,
            ancestry_evidence_sha256=hashlib.sha256(ancestry_raw).hexdigest(),
            cloud_collector_public_key_id=_key_id(cloud_key),
            host_collector_public_key_id="7" * 64,
        ),
        network_evidence=network,
        network_collector_public_key=network_key.public_key(),
        now_unix=NOW,
    )
    return plan, ancestry, cloud_key


def _verified_probe(phase: str) -> author._VerifiedAttachedSaProbe:
    return author._VerifiedAttachedSaProbe._create(
        phase=phase,
        permission_probe=preflight.expected_effective_permission_probe(
            phase == "post_iam"
        ),
        host_observation_report_sha256="a" * 64,
        host_observation_binding_sha256="d" * 64,
        attached_sa_permission_probe_report_sha256="b" * 64,
        terminal_receipt_sha256="c" * 64,
        cloud_signer_provisioning_receipt_sha256="e" * 64,
        cloud_signer_readiness_sha256="f" * 64,
        host_signer_provisioning_receipt_sha256="1" * 64,
        host_signer_readiness_sha256="2" * 64,
    )


def _identities(
    plan: foundation.OwnerGateFoundationPlan,
    ancestry: project_ancestry.ProjectAncestryEvidence,
) -> author._FoundationIdentities:
    return author._FoundationIdentities(
        ancestry_evidence=ancestry,
        owner_gate_vm={
            "name": foundation.VM_NAME,
            "numeric_id": OWNER_GATE_VM_ID,
            "subnetwork_numeric_id": OWNER_GATE_SUBNET_ID,
            "boot_disk_name": foundation.VM_NAME,
            "boot_disk_numeric_id": OWNER_GATE_BOOT_DISK_ID,
            "boot_disk_size_gb": foundation.BOOT_DISK_SIZE_GB,
            "boot_disk_boot": True,
            "boot_disk_auto_delete": True,
            "boot_disk_mode": "READ_WRITE",
            "boot_disk_interface": "SCSI",
            "boot_disk_attachment_type": "PERSISTENT",
            "boot_disk_attachment_index": 0,
        },
        service_account={
            "email": plan.spec.service_account_email,
            "unique_id": OWNER_GATE_SERVICE_ACCOUNT_ID,
        },
        subnet={
            "name": foundation.OWNER_GATE_SUBNET_NAME,
            "numeric_id": OWNER_GATE_SUBNET_ID,
        },
        private_web_firewall={
            "name": "muncho-owner-gate-web-from-production",
            "numeric_id": OWNER_GATE_FIREWALL_ID,
        },
        pre_foundation_plan=plan,
        network_evidence=None,  # type: ignore[arg-type]
        network_collector_public_key=None,  # type: ignore[arg-type]
        inert_plan_sha256="8" * 64,
        foundation_source_revision=REVISION,
        foundation_source_tree_oid="9" * 40,
        pre_foundation_authority_sha256="a" * 64,
        foundation_apply_receipt_sha256="b" * 64,
    )


def _role(
    name: str,
    title: str,
    description: str,
    permissions: tuple[str, ...],
) -> dict:
    return {
        "name": name,
        "title": title,
        "description": description,
        "includedPermissions": list(permissions),
        "stage": "GA",
        "deleted": False,
        "etag": "role-etag",
    }


def _raw(
    plan: foundation.OwnerGateFoundationPlan,
    ancestry: project_ancestry.ProjectAncestryEvidence,
    *,
    phase: str,
) -> dict[str, object]:
    requests = author.request_inventory(
        plan=plan,
        ancestry_evidence=ancestry,
        phase=phase,
        _connector_regions=(foundation.REGION,),
    )
    raw: dict[str, object] = {}

    def put(method: str, url: str, value: object) -> None:
        key = f"{method} {url}"
        assert key in {author._request_key(item) for item in requests}
        raw[key] = value

    project = foundation.PROJECT
    zone = foundation.ZONE
    region = foundation.REGION
    compute = "/compute/v1"
    compute_resource = f"https://www.googleapis.com/compute/v1/projects/{project}"
    network = f"{compute_resource}/global/networks/{foundation.NETWORK_NAME}"
    production_subnet = (
        f"{compute_resource}/regions/{region}/subnetworks/"
        f"{foundation.PRODUCTION_SUBNET_NAME}"
    )
    owner_subnet = (
        f"{compute_resource}/regions/{region}/subnetworks/"
        f"{foundation.OWNER_GATE_SUBNET_NAME}"
    )
    owner_sa = f"{foundation.SERVICE_ACCOUNT_NAME}@{project}.iam.gserviceaccount.com"
    member = f"serviceAccount:{owner_sa}"

    put(
        "GET",
        author._url(f"{compute}/projects/{project}"),
        {
            "name": project,
            "id": PROJECT_NUMBER,
            "commonInstanceMetadata": {
                "items": [
                    {"key": "block-project-ssh-keys", "value": "true"},
                    {"key": "enable-oslogin", "value": "true"},
                ]
            },
        },
    )
    put(
        "GET",
        author._url(
            f"{compute}/projects/{project}/zones/{zone}/instances/"
            f"{foundation.PRODUCTION_SOURCE_VM}"
        ),
        {
            "name": foundation.PRODUCTION_SOURCE_VM,
            "id": foundation.PRODUCTION_SOURCE_VM_ID,
            "selfLink": (
                f"{compute_resource}/zones/{zone}/instances/"
                f"{foundation.PRODUCTION_SOURCE_VM}"
            ),
            "networkInterfaces": [
                {
                    "network": network,
                    "subnetwork": production_subnet,
                    "networkIP": "10.80.0.2",
                }
            ],
            "serviceAccounts": [
                {
                    "email": foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT,
                    "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
                }
            ],
        },
    )
    owner_instance = {
        "name": foundation.VM_NAME,
        "id": OWNER_GATE_VM_ID,
        "status": "RUNNING",
        "selfLink": f"{compute_resource}/zones/{zone}/instances/{foundation.VM_NAME}",
        "machineType": (
            f"{compute_resource}/zones/{zone}/machineTypes/{foundation.MACHINE_TYPE}"
        ),
        "networkInterfaces": [
            {
                "network": network,
                "subnetwork": owner_subnet,
                "networkIP": foundation.OWNER_GATE_PRIVATE_IP,
                "accessConfigs": [],
            }
        ],
        "serviceAccounts": [
            {
                "email": owner_sa,
                "scopes": list(foundation.OWNER_GATE_OAUTH_SCOPES),
            }
        ],
        "tags": {
            "items": [foundation.IAP_NETWORK_TAG, foundation.OWNER_GATE_NETWORK_TAG]
        },
        "shieldedInstanceConfig": {
            "enableSecureBoot": True,
            "enableVtpm": True,
            "enableIntegrityMonitoring": True,
        },
        "metadata": {"items": [{"key": "serial-port-enable", "value": "false"}]},
        "disks": [
            {
                "source": (
                    f"{compute_resource}/zones/{zone}/disks/{foundation.VM_NAME}"
                ),
                "deviceName": foundation.VM_NAME,
                "boot": True,
                "autoDelete": True,
                "mode": "READ_WRITE",
                "interface": "SCSI",
                "type": "PERSISTENT",
                "index": 0,
            }
        ],
    }
    put(
        "GET",
        author._url(
            f"{compute}/projects/{project}/zones/{zone}/disks/{foundation.VM_NAME}"
        ),
        {
            "name": foundation.VM_NAME,
            "id": OWNER_GATE_BOOT_DISK_ID,
            "selfLink": (f"{compute_resource}/zones/{zone}/disks/{foundation.VM_NAME}"),
            "sizeGb": str(foundation.BOOT_DISK_SIZE_GB),
            "type": (
                f"{compute_resource}/zones/{zone}/diskTypes/{foundation.BOOT_DISK_TYPE}"
            ),
        },
    )
    put(
        "GET",
        author._url(
            f"{compute}/projects/{project}/zones/{zone}/instances/{foundation.VM_NAME}"
        ),
        owner_instance,
    )
    put(
        "GET",
        author._url(f"{compute}/projects/{project}/aggregated/subnetworks"),
        {
            "items": {
                f"regions/{region}": {
                    "subnetworks": [
                        {
                            "name": foundation.PRODUCTION_SUBNET_NAME,
                            "selfLink": production_subnet,
                            "network": network,
                            "ipCidrRange": "10.80.0.0/24",
                        },
                        {
                            "name": foundation.OWNER_GATE_SUBNET_NAME,
                            "selfLink": owner_subnet,
                            "network": network,
                            "ipCidrRange": foundation.OWNER_GATE_SUBNET_CIDR,
                        },
                    ]
                }
            }
        },
    )
    put(
        "GET",
        author._url(
            f"{compute}/projects/{project}/regions/{region}/subnetworks/"
            f"{foundation.OWNER_GATE_SUBNET_NAME}"
        ),
        {
            "name": foundation.OWNER_GATE_SUBNET_NAME,
            "id": OWNER_GATE_SUBNET_ID,
            "selfLink": owner_subnet,
            "network": network,
            "ipCidrRange": foundation.OWNER_GATE_SUBNET_CIDR,
            "privateIpGoogleAccess": True,
            "stackType": "IPV4_ONLY",
            "purpose": "PRIVATE",
            "secondaryIpRanges": [],
        },
    )
    put(
        "GET",
        author._url(
            f"{compute}/projects/{project}/global/networks/{foundation.NETWORK_NAME}"
        ),
        {"name": foundation.NETWORK_NAME, "selfLink": network, "peerings": []},
    )
    put(
        "GET",
        author._url(f"{compute}/projects/{project}/global/routes"),
        {
            "items": [
                {
                    "name": "default-private",
                    "network": network,
                    "destRange": "10.80.0.0/24",
                },
                {
                    "name": "owner-subnet-route",
                    "network": network,
                    "destRange": foundation.OWNER_GATE_SUBNET_CIDR,
                    "nextHopNetwork": network,
                    "routeType": "SUBNET",
                    "priority": 0,
                },
            ]
        },
    )
    put(
        "GET",
        author._url(f"{compute}/projects/{project}/global/addresses"),
        {"items": []},
    )
    put(
        "GET",
        author._url(
            f"/v1/projects/{project}/locations?pageSize=100",
            host="vpcaccess.googleapis.com",
        ),
        {
            "locations": [
                {
                    "name": f"projects/{project}/locations/{region}",
                    "locationId": region,
                }
            ]
        },
    )
    put(
        "GET",
        author._url(
            f"/v1/projects/{project}/locations/{region}/connectors?pageSize=100",
            host="vpcaccess.googleapis.com",
        ),
        {"connectors": []},
    )
    put(
        "GET",
        author._url(
            f"/v3/projects/{PROJECT_NUMBER}",
            host="cloudresourcemanager.googleapis.com",
        ),
        {
            "name": f"projects/{PROJECT_NUMBER}",
            "projectId": project,
            "displayName": "Adventico AI Platform",
            "state": "ACTIVE",
            "etag": "project-etag",
            "parent": ancestry.ordered_chain[0]["parent_resource_name"],
        },
    )
    for node in ancestry.ordered_chain[1:-1]:
        put(
            "GET",
            author._url(
                f"/v3/{node['resource_name']}",
                host="cloudresourcemanager.googleapis.com",
            ),
            {
                "name": node["resource_name"],
                "displayName": node["display_name"],
                "state": node["state"],
                "etag": node["etag"],
                "parent": node["parent_resource_name"],
            },
        )
    put(
        "GET",
        author._url(
            f"/v3/organizations/{ORGANIZATION}",
            host="cloudresourcemanager.googleapis.com",
        ),
        {
            "name": f"organizations/{ORGANIZATION}",
            "displayName": "adventico.com",
            "state": "ACTIVE",
            "etag": "organization-etag",
        },
    )

    iap = {
        "name": "allow-iap-ssh",
        "network": network,
        "direction": "INGRESS",
        "disabled": False,
        "sourceRanges": [foundation.IAP_SOURCE_RANGE],
        "targetTags": [foundation.IAP_NETWORK_TAG],
        "sourceServiceAccounts": [],
        "sourceTags": [],
        "targetServiceAccounts": [],
        "destinationRanges": [],
        "allowed": [{"IPProtocol": "tcp", "ports": ["22"]}],
    }
    private_web = {
        "name": "muncho-owner-gate-web-from-production",
        "id": OWNER_GATE_FIREWALL_ID,
        "selfLink": (
            f"{compute_resource}/global/firewalls/muncho-owner-gate-web-from-production"
        ),
        "network": network,
        "direction": "INGRESS",
        "disabled": False,
        "priority": 700,
        "sourceRanges": [],
        "sourceServiceAccounts": [foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT],
        "sourceTags": [],
        "targetTags": [],
        "targetServiceAccounts": [owner_sa],
        "destinationRanges": [],
        "allowed": [
            {
                "IPProtocol": "tcp",
                "ports": [str(foundation.WEB_LISTEN_PORT)],
            }
        ],
        "logConfig": {"enable": True},
    }
    put(
        "GET",
        author._url(f"{compute}/projects/{project}/global/firewalls/allow-iap-ssh"),
        iap,
    )
    put(
        "GET",
        author._url(
            f"{compute}/projects/{project}/global/firewalls/"
            "muncho-owner-gate-web-from-production"
        ),
        private_web,
    )
    put(
        "GET",
        author._url(f"{compute}/projects/{project}/global/firewalls"),
        {"items": [iap, private_web]},
    )
    put(
        "GET",
        author._url(
            f"{compute}/projects/{project}/zones/{zone}/instances/"
            f"{foundation.VM_NAME}/getEffectiveFirewalls"
        ),
        {"firewalls": [iap, private_web]},
    )

    target_disk_link = f"{compute_resource}/zones/{zone}/disks/{foundation.TARGET_DISK}"
    put(
        "GET",
        author._url(
            f"{compute}/projects/{project}/zones/{zone}/instances/"
            f"{foundation.TARGET_INSTANCE}"
        ),
        {
            "name": foundation.TARGET_INSTANCE,
            "id": foundation.TARGET_INSTANCE_ID,
            "selfLink": (
                f"{compute_resource}/zones/{zone}/instances/"
                f"{foundation.TARGET_INSTANCE}"
            ),
            "disks": [
                {
                    "source": target_disk_link,
                    "deviceName": foundation.TARGET_BOOT_DEVICE,
                    "boot": True,
                }
            ],
        },
    )
    put(
        "GET",
        author._url(
            f"{compute}/projects/{project}/zones/{zone}/disks/{foundation.TARGET_DISK}"
        ),
        {
            "name": foundation.TARGET_DISK,
            "id": foundation.TARGET_DISK_ID,
            "selfLink": target_disk_link,
        },
    )

    sa_path = (
        f"/v1/projects/{project}/serviceAccounts/"
        f"{foundation.SERVICE_ACCOUNT_NAME}%40{project}.iam.gserviceaccount.com"
    )
    put(
        "GET",
        author._url(sa_path, host="iam.googleapis.com"),
        {
            "name": f"projects/{project}/serviceAccounts/{owner_sa}",
            "projectId": project,
            "email": owner_sa,
            "uniqueId": OWNER_GATE_SERVICE_ACCOUNT_ID,
            "disabled": False,
        },
    )
    put(
        "GET",
        author._url(
            f"{sa_path}/keys?keyTypes=USER_MANAGED",
            host="iam.googleapis.com",
        ),
        {"keys": []},
    )
    put(
        "POST",
        author._url(
            f"{sa_path}:getIamPolicy?options.requestedPolicyVersion=3",
            host="iam.googleapis.com",
        ),
        {"version": 3, "bindings": [], "etag": "sa-policy-etag"},
    )

    project_read = plan.spec.read_only_iam_role
    mutation = plan.spec.custom_role
    ancestor_read = plan.spec.ancestor_read_only_iam_role
    put(
        "GET",
        author._url(f"/v1/{project_read}", host="iam.googleapis.com"),
        _role(
            project_read,
            foundation.PROJECT_READ_ROLE_TITLE,
            foundation.PROJECT_READ_ROLE_DESCRIPTION,
            foundation.READ_ONLY_IAM_PERMISSIONS,
        ),
    )
    put(
        "GET",
        author._url(f"/v1/{mutation}", host="iam.googleapis.com"),
        _role(
            mutation,
            foundation.MUTATION_ROLE_TITLE,
            foundation.MUTATION_ROLE_DESCRIPTION,
            foundation.MUTATION_PERMISSIONS,
        ),
    )
    put(
        "GET",
        author._url(f"/v1/{ancestor_read}", host="iam.googleapis.com"),
        _role(
            ancestor_read,
            foundation.ANCESTOR_READ_ROLE_TITLE,
            foundation.ANCESTOR_READ_ROLE_DESCRIPTION,
            foundation.DIRECT_IAM_ANCESTOR_PERMISSIONS,
        ),
    )
    project_bindings = [{"role": project_read, "members": [member]}]
    if phase == "post_iam":
        project_bindings.append({
            "role": mutation,
            "members": [member],
            "condition": {
                "title": foundation.MUTATION_CONDITION_TITLE,
                "description": foundation.MUTATION_CONDITION_DESCRIPTION,
                "expression": foundation._condition_expression(),
            },
        })
    put(
        "POST",
        author._url(
            f"/v3/projects/{PROJECT_NUMBER}:getIamPolicy",
            host="cloudresourcemanager.googleapis.com",
        ),
        {"version": 3, "bindings": project_bindings, "etag": "project-policy-etag"},
    )
    for node in ancestry.ordered_chain[1:-1]:
        put(
            "POST",
            author._url(
                f"/v3/{node['resource_name']}:getIamPolicy",
                host="cloudresourcemanager.googleapis.com",
            ),
            {"version": 3, "bindings": [], "etag": "folder-policy-etag"},
        )
    put(
        "POST",
        author._url(
            f"/v3/organizations/{ORGANIZATION}:getIamPolicy",
            host="cloudresourcemanager.googleapis.com",
        ),
        {
            "version": 3,
            "bindings": [{"role": ancestor_read, "members": [member]}],
            "etag": "organization-policy-etag",
        },
    )

    assert set(raw) == {author._request_key(item) for item in requests}
    return raw


@pytest.mark.parametrize("phase", ["inert", "post_iam"])
def test_realistic_fixed_rest_fixture_authors_validator_compatible_observation(
    phase: str,
) -> None:
    plan, ancestry, _cloud_key = _context()
    observation = author._unsigned_from_raw(
        plan=plan,
        ancestry_evidence=ancestry,
        phase=phase,
        raw=_raw(plan, ancestry, phase=phase),
        collected_at_unix=NOW,
        package_sha256="3" * 64,
        foundation_identities=_identities(plan, ancestry),
        verified_probe=_verified_probe(phase),
    )

    assert observation["credential_values_read"] is False
    assert observation["iam"]["mutation_binding_present"] is (phase == "post_iam")
    assert observation["release_binding"]["phase"] == phase
    preflight._validate_cloud_unsigned(
        observation,
        plan_sha256=plan.sha256,
        mutation_binding_present=phase == "post_iam",
    )


def _reject(mutator, *, phase: str = "inert", match: str | None = None) -> None:
    plan, ancestry, _cloud_key = _context()
    raw = _raw(plan, ancestry, phase=phase)
    mutator(raw, plan)
    with pytest.raises(author.OwnerGateCloudObservationAuthorError, match=match):
        author._unsigned_from_raw(
            plan=plan,
            ancestry_evidence=ancestry,
            phase=phase,
            raw=raw,
            collected_at_unix=NOW,
            package_sha256="3" * 64,
            foundation_identities=_identities(plan, ancestry),
            verified_probe=_verified_probe(phase),
        )


def test_extra_permission_in_probe_is_rejected() -> None:
    plan, ancestry, _cloud_key = _context()
    probe = _verified_probe("inert")
    probe.permission_probe["disk"]["granted_permissions"].append("compute.disks.update")
    with pytest.raises(
        author.OwnerGateCloudObservationAuthorError,
        match="permission_probe_invalid",
    ):
        author._unsigned_from_raw(
            plan=plan,
            ancestry_evidence=ancestry,
            phase="inert",
            raw=_raw(plan, ancestry, phase="inert"),
            collected_at_unix=NOW,
            package_sha256="3" * 64,
            foundation_identities=_identities(plan, ancestry),
            verified_probe=probe,
        )


def test_wrong_mutation_condition_is_rejected() -> None:
    def mutate(raw, plan) -> None:
        key = next(
            key
            for key in raw
            if key.endswith(f"projects/{PROJECT_NUMBER}:getIamPolicy")
        )
        binding = next(
            item
            for item in raw[key]["bindings"]
            if item["role"] == plan.spec.custom_role
        )
        binding["condition"]["expression"] = "resource.name.startsWith('projects/')"

    _reject(mutate, phase="post_iam", match="iam_invalid")


def test_inherited_forbidden_role_is_rejected() -> None:
    def mutate(raw, _plan) -> None:
        key = next(
            key for key in raw if f"organizations/{ORGANIZATION}:getIamPolicy" in key
        )
        raw[key]["bindings"].append({
            "role": "roles/owner",
            "members": [
                f"serviceAccount:{foundation.SERVICE_ACCOUNT_NAME}@"
                f"{foundation.PROJECT}.iam.gserviceaccount.com"
            ],
        })

    _reject(mutate, match="iam_invalid")


def test_user_managed_service_account_key_is_rejected() -> None:
    def mutate(raw, _plan) -> None:
        key = next(
            key
            for key in raw
            if ".iam.gserviceaccount.com/keys?keyTypes=USER_MANAGED" in key
        )
        raw[key]["keys"] = [
            {
                "name": "projects/p/serviceAccounts/s/keys/1",
                "keyType": "USER_MANAGED",
            }
        ]

    _reject(mutate, match="service_account_invalid")


def test_public_owner_gate_firewall_is_rejected() -> None:
    def mutate(raw, _plan) -> None:
        key = next(key for key in raw if key.endswith("/global/firewalls"))
        raw[key]["items"].append({
            "name": "public-owner-gate",
            "direction": "INGRESS",
            "sourceRanges": ["0.0.0.0/0"],
            "targetTags": [foundation.OWNER_GATE_NETWORK_TAG],
            "allowed": [{"IPProtocol": "tcp", "ports": ["8080"]}],
        })

    _reject(mutate, match="firewall_invalid")


def test_target_numeric_id_drift_is_rejected() -> None:
    def mutate(raw, _plan) -> None:
        key = next(
            key
            for key in raw
            if key.endswith(f"/instances/{foundation.TARGET_INSTANCE}")
        )
        raw[key]["id"] = "9999999999999999999"

    _reject(mutate, match="targets_invalid")


def test_foundation_owner_gate_numeric_id_drift_is_rejected() -> None:
    def mutate(raw, _plan) -> None:
        key = next(
            key for key in raw if key.endswith(f"/instances/{foundation.VM_NAME}")
        )
        raw[key]["id"] = "9999999999999999999"

    _reject(mutate, match="instance_invalid")


@pytest.mark.parametrize(
    ("suffix", "field", "match"),
    (
        (f"subnetworks/{foundation.OWNER_GATE_SUBNET_NAME}", "id", "subnet_invalid"),
        (
            "firewalls/muncho-owner-gate-web-from-production",
            "id",
            "firewall_invalid",
        ),
        (f"disks/{foundation.VM_NAME}", "id", "instance_invalid"),
    ),
)
def test_signed_foundation_numeric_identity_drift_is_rejected(
    suffix: str,
    field: str,
    match: str,
) -> None:
    def mutate(raw, _plan) -> None:
        key = next(
            key for key in raw if key.startswith("GET ") and key.endswith(suffix)
        )
        raw[key][field] = "9999999999999999999"

    _reject(mutate, match=match)


def test_owner_boot_disk_attachment_drift_is_rejected() -> None:
    def mutate(raw, _plan) -> None:
        key = next(
            key for key in raw if key.endswith(f"/instances/{foundation.VM_NAME}")
        )
        raw[key]["disks"][0]["autoDelete"] = False

    _reject(mutate, match="instance_invalid")


@pytest.mark.parametrize("effective", [False, True])
def test_mixed_firewall_source_selectors_are_rejected(effective: bool) -> None:
    def mutate(raw, _plan) -> None:
        if effective:
            key = next(key for key in raw if key.endswith("/getEffectiveFirewalls"))
            rule = next(
                item
                for item in raw[key]["firewalls"]
                if item["name"] == "muncho-owner-gate-web-from-production"
            )
        else:
            key = next(
                key
                for key in raw
                if key.endswith("/firewalls/muncho-owner-gate-web-from-production")
            )
            rule = raw[key]
        rule["sourceRanges"] = ["0.0.0.0/0"]

    _reject(mutate, match="firewall_invalid")


def test_live_ancestry_etag_drift_is_rejected() -> None:
    def mutate(raw, _plan) -> None:
        key = next(key for key in raw if key.endswith(f"/organizations/{ORGANIZATION}"))
        raw[key]["etag"] = "changed-etag"

    _reject(mutate, match="ancestry_invalid")


def test_folder_ancestry_is_read_and_matched_exactly() -> None:
    plan, ancestry, _cloud_key = _context(with_folder=True)
    observation = author._unsigned_from_raw(
        plan=plan,
        ancestry_evidence=ancestry,
        phase="inert",
        raw=_raw(plan, ancestry, phase="inert"),
        collected_at_unix=NOW,
        package_sha256="3" * 64,
        foundation_identities=_identities(plan, ancestry),
        verified_probe=_verified_probe("inert"),
    )
    assert observation["project"] == foundation.PROJECT


@pytest.mark.parametrize("kind", ["secondary", "route", "address", "connector"])
def test_full_network_overlap_sources_are_rejected(kind: str) -> None:
    def mutate(raw, _plan) -> None:
        if kind == "secondary":
            key = next(key for key in raw if key.endswith("/aggregated/subnetworks"))
            production = raw[key]["items"][f"regions/{foundation.REGION}"][
                "subnetworks"
            ][0]
            production["secondaryIpRanges"] = [
                {"ipCidrRange": foundation.OWNER_GATE_SUBNET_CIDR}
            ]
        elif kind == "route":
            key = next(key for key in raw if key.endswith("/global/routes"))
            raw[key]["items"].append({
                "name": "overlap-static",
                "network": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{foundation.PROJECT}/global/networks/{foundation.NETWORK_NAME}"
                ),
                "destRange": "10.80.3.0/29",
            })
        elif kind == "address":
            key = next(key for key in raw if key.endswith("/global/addresses"))
            raw[key]["items"].append({
                "name": "overlap-service-range",
                "purpose": "VPC_PEERING",
                "network": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{foundation.PROJECT}/global/networks/{foundation.NETWORK_NAME}"
                ),
                "address": foundation.OWNER_GATE_SUBNET_CIDR.split("/", 1)[0],
                "prefixLength": 28,
            })
        else:
            key = next(key for key in raw if "/connectors?pageSize=100" in key)
            raw[key]["connectors"].append({
                "name": "overlap-connector",
                "network": foundation.NETWORK_NAME,
                "ipCidrRange": foundation.OWNER_GATE_SUBNET_CIDR,
            })

    _reject(mutate, match="subnet_invalid")


def test_peering_inventory_is_bound_into_signed_network_digest() -> None:
    plan, ancestry, _cloud_key = _context()
    raw = _raw(plan, ancestry, phase="inert")
    baseline = author._unsigned_from_raw(
        plan=plan,
        ancestry_evidence=ancestry,
        phase="inert",
        raw=raw,
        collected_at_unix=NOW,
        package_sha256="3" * 64,
        foundation_identities=_identities(plan, ancestry),
        verified_probe=_verified_probe("inert"),
    )
    network_key = next(
        key for key in raw if key.endswith(f"/networks/{foundation.NETWORK_NAME}")
    )
    raw[network_key]["peerings"] = [{"name": "servicenetworking-googleapis-com"}]
    changed = author._unsigned_from_raw(
        plan=plan,
        ancestry_evidence=ancestry,
        phase="inert",
        raw=raw,
        collected_at_unix=NOW,
        package_sha256="3" * 64,
        foundation_identities=_identities(plan, ancestry),
        verified_probe=_verified_probe("inert"),
    )
    assert (
        baseline["subnet"]["route_inventory_sha256"]
        != changed["subnet"]["route_inventory_sha256"]
    )


def test_request_inventory_and_reader_have_no_generic_surface() -> None:
    plan, ancestry, _ = _context()
    requests = author.request_inventory(
        plan=plan,
        ancestry_evidence=ancestry,
        phase="inert",
    )
    assert all(item["method"] in {"GET", "POST"} for item in requests)
    assert all(
        item["method"] == "GET" or "getIamPolicy" in item["url"] for item in requests
    )
    assert not any("testIamPermissions" in item["url"] for item in requests)
    assert any(
        f"organizations/{ORGANIZATION}:getIamPolicy" in item["url"] for item in requests
    )
    assert any(
        plan.spec.ancestor_read_only_iam_role in item["url"] for item in requests
    )
    token = direct_iam_author._GcloudAccessToken(
        bytearray(b"x" * 40),
        marker=direct_iam_author._TOKEN_MARKER,
    )
    reader = author._FixedCloudFactsReader(
        token=token,
        plan=plan,
        ancestry_evidence=ancestry,
        phase="inert",
        _connection_factories={
            host: (lambda: pytest.fail("network must not be reached"))
            for host in author._ALLOWED_HOSTS
        },
    )
    with pytest.raises(
        author.OwnerGateCloudObservationAuthorError,
        match="request_forbidden",
    ):
        reader._request_exact({
            "method": "GET",
            "url": "https://compute.googleapis.com/compute/v1/projects/other",
        })
    direct_iam_author.wipe_access_token(token)


def test_raw_response_surface_drift_is_rejected() -> None:
    plan, ancestry, _cloud_key = _context()
    raw = _raw(plan, ancestry, phase="inert")
    raw["GET https://compute.googleapis.com/compute/v1/projects/other"] = {}
    with pytest.raises(
        author.OwnerGateCloudObservationAuthorError,
        match="facts_invalid",
    ):
        author._unsigned_from_raw(
            plan=plan,
            ancestry_evidence=ancestry,
            phase="inert",
            raw=raw,
            collected_at_unix=NOW,
            package_sha256="3" * 64,
            foundation_identities=_identities(plan, ancestry),
            verified_probe=_verified_probe("inert"),
        )


class _FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        content_type: str = "application/json; charset=utf-8",
        location: str | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self._headers = {
            "Content-Length": str(len(body)),
            "Content-Type": content_type,
        }
        if location is not None:
            self._headers["Location"] = location

    def getheader(self, name: str) -> str | None:
        return self._headers.get(name)

    def read(self, maximum: int) -> bytes:
        return self._body[:maximum]


class _FakeCloudHttp:
    def __init__(
        self,
        raw: dict[str, object],
        *,
        corrupt_key: str | None = None,
        corrupt_kind: str | None = None,
        unstable_key: str | None = None,
    ) -> None:
        self.raw = raw
        self.corrupt_key = corrupt_key
        self.corrupt_kind = corrupt_kind
        self.unstable_key = unstable_key
        self.calls: list[tuple[str, str, bytes | None, dict[str, str]]] = []
        self.counts: dict[str, int] = {}

    def factory(self, host: str):
        parent = self

        class Connection:
            def request(
                self,
                method: str,
                target: str,
                *,
                body: bytes | None,
                headers: dict[str, str],
            ) -> None:
                self.key = f"{method} https://{host}{target}"
                parent.calls.append((method, self.key, body, dict(headers)))

            def getresponse(self) -> _FakeResponse:
                key = self.key
                parent.counts[key] = parent.counts.get(key, 0) + 1
                value = parent.raw[key]
                if key == parent.unstable_key and parent.counts[key] >= 2:
                    value = {**value, "unstable": True}
                body = foundation.canonical_json_bytes(value)
                if key != parent.corrupt_key:
                    return _FakeResponse(body)
                if parent.corrupt_kind == "redirect":
                    return _FakeResponse(body, location="https://evil.invalid/")
                if parent.corrupt_kind == "content-type":
                    return _FakeResponse(body, content_type="text/html")
                if parent.corrupt_kind == "status":
                    return _FakeResponse(body, status=503)
                raise AssertionError("unknown corrupt kind")

            def close(self) -> None:
                pass

        return Connection()


def _http_reader(
    plan: foundation.OwnerGateFoundationPlan,
    ancestry: project_ancestry.ProjectAncestryEvidence,
    raw: dict[str, object],
    fake: _FakeCloudHttp,
) -> tuple[author._FixedCloudFactsReader, direct_iam_author._GcloudAccessToken]:
    token = direct_iam_author._GcloudAccessToken(
        bytearray(b"opaque-owner-token-value-that-is-long-enough"),
        marker=direct_iam_author._TOKEN_MARKER,
    )
    reader = author._FixedCloudFactsReader(
        token=token,
        plan=plan,
        ancestry_evidence=ancestry,
        phase="inert",
        _connection_factories={
            host: (lambda host=host: fake.factory(host))
            for host in author._ALLOWED_HOSTS
        },
    )
    return reader, token


def test_fixed_http_reader_discovers_all_regions_and_double_reads_every_fact() -> None:
    plan, ancestry, _ = _context()
    raw = _raw(plan, ancestry, phase="inert")
    location_key = next(key for key in raw if key.endswith("/locations?pageSize=100"))
    second_region = "us-central1"
    raw[location_key]["locations"].append({
        "name": f"projects/{foundation.PROJECT}/locations/{second_region}",
        "locationId": second_region,
    })
    second_url = author._url(
        f"/v1/projects/{foundation.PROJECT}/locations/{second_region}/connectors?pageSize=100",
        host="vpcaccess.googleapis.com",
    )
    raw[f"GET {second_url}"] = {"connectors": []}
    fake = _FakeCloudHttp(raw)
    reader, token = _http_reader(plan, ancestry, raw, fake)
    try:
        assert reader.collect() == raw
        assert fake.counts[location_key] == 4
        assert all(
            count == 2 for key, count in fake.counts.items() if key != location_key
        )
        assert all(
            headers["Authorization"].startswith("Bearer opaque-owner-token")
            and headers["Connection"] == "close"
            for _method, _key, _body, headers in fake.calls
        )
    finally:
        direct_iam_author.wipe_access_token(token)


def test_fixed_http_reader_rejects_second_snapshot_drift() -> None:
    plan, ancestry, _ = _context()
    raw = _raw(plan, ancestry, phase="inert")
    project_key = next(
        key for key in raw if key.endswith(f"/compute/v1/projects/{foundation.PROJECT}")
    )
    fake = _FakeCloudHttp(raw, unstable_key=project_key)
    reader, token = _http_reader(plan, ancestry, raw, fake)
    try:
        with pytest.raises(
            author.OwnerGateCloudObservationAuthorError,
            match="facts_unstable",
        ):
            reader.collect()
    finally:
        direct_iam_author.wipe_access_token(token)


@pytest.mark.parametrize("kind", ["redirect", "content-type", "status"])
def test_fixed_http_reader_rejects_http_boundary_drift(kind: str) -> None:
    plan, ancestry, _ = _context()
    raw = _raw(plan, ancestry, phase="inert")
    project_key = next(
        key for key in raw if key.endswith(f"/compute/v1/projects/{foundation.PROJECT}")
    )
    fake = _FakeCloudHttp(raw, corrupt_key=project_key, corrupt_kind=kind)
    reader, token = _http_reader(plan, ancestry, raw, fake)
    try:
        with pytest.raises(
            author.OwnerGateCloudObservationAuthorError,
            match="resource_unavailable",
        ):
            reader.collect()
    finally:
        direct_iam_author.wipe_access_token(token)


def test_fixed_http_reader_enforces_aggregate_snapshot_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, ancestry, _ = _context()
    raw = _raw(plan, ancestry, phase="inert")
    fake = _FakeCloudHttp(raw)
    reader, token = _http_reader(plan, ancestry, raw, fake)
    monkeypatch.setattr(author, "MAX_SNAPSHOT_BYTES", 1024)
    try:
        with pytest.raises(
            author.OwnerGateCloudObservationAuthorError,
            match="http_invalid",
        ):
            reader.collect()
    finally:
        direct_iam_author.wipe_access_token(token)
