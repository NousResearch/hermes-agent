#!/usr/bin/env python3
"""Inert, digest-bound GCE foundation plan for the Muncho owner gate.

The module only authors and validates a plan.  It does not execute ``gcloud``
or fall back to a local Cloud credential.  The plan deliberately separates the
private VM foundation from the later mutation-IAM activation: the owner-gate
service account has no Compute mutation authority until a split-UID/passkey
smoke has succeeded and a separately approved activation step is applied.

No task or message semantics live here.  All decisions are mechanical resource,
identity, network, and permission invariants.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


PROJECT = "adventico-ai-platform"
REGION = "europe-west3"
ZONE = "europe-west3-a"
VM_NAME = "muncho-owner-gate-01"
SERVICE_ACCOUNT_NAME = "muncho-owner-gate-executor"
MUTATION_ROLE_ID = "munchoOwnerGateStorageExecutorV1"
READ_ONLY_IAM_ROLE_ID = "munchoOwnerGateIamObservationReaderV1"
ANCESTOR_READ_ONLY_IAM_ROLE_ID = (
    "munchoOwnerGateHierarchyObservationReaderV1"
)
MUTATION_CONDITION_TITLE = "muncho_owner_gate_exact_storage_v1"
MUTATION_CONDITION_DESCRIPTION = (
    "Exact canary disk and instance resources only"
)
PROJECT_READ_ROLE_TITLE = "Muncho Owner Gate IAM Observation Reader V1"
PROJECT_READ_ROLE_DESCRIPTION = (
    "Exact project and service-account IAM read-only observation"
)
ANCESTOR_READ_ROLE_TITLE = (
    "Muncho Owner Gate Hierarchy Observation Reader V1"
)
ANCESTOR_READ_ROLE_DESCRIPTION = (
    "Exact folder and organization IAM read-only observation"
)
MUTATION_ROLE_TITLE = "Muncho Owner Gate Storage Executor V1"
MUTATION_ROLE_DESCRIPTION = (
    "Exact canary storage disk and instance get resize stop start only"
)
NETWORK_NAME = "ai-platform-vpc"
PRODUCTION_SUBNET_NAME = "ai-platform-europe-west3"
OWNER_GATE_SUBNET_NAME = "muncho-owner-gate-europe-west3"
OWNER_GATE_SUBNET_CIDR = "10.80.3.0/28"
OWNER_GATE_PRIVATE_IP = "10.80.3.2"
PRODUCTION_SOURCE_VM = "ai-platform-runtime-01"
PRODUCTION_SOURCE_VM_ID = "1094477181810932795"
PRODUCTION_SOURCE_SERVICE_ACCOUNT = (
    "ai-platform-runtime@adventico-ai-platform.iam.gserviceaccount.com"
)
OWNER_GATE_NETWORK_TAG = "muncho-owner-gate"
IAP_NETWORK_TAG = "iap-ssh"
IAP_SOURCE_RANGE = "35.235.240.0/20"
PRIVATE_GOOGLE_API_VIP_RANGE = "199.36.153.8/30"
WEB_LISTEN_PORT = 8080
MACHINE_TYPE = "e2-small"
BOOT_DISK_SIZE_GB = 20
BOOT_DISK_TYPE = "pd-balanced"

TARGET_INSTANCE = "muncho-canary-v2-01"
TARGET_INSTANCE_ID = "9153645328899914617"
TARGET_DISK = "muncho-canary-v2-01"
TARGET_BOOT_DEVICE = "persistent-disk-0"
TARGET_DISK_ID = "4195397669213846393"
TARGET_DISK_SELF_LINK = (
    f"projects/{PROJECT}/zones/{ZONE}/disks/{TARGET_DISK}"
)
TARGET_INSTANCE_SELF_LINK = (
    f"projects/{PROJECT}/zones/{ZONE}/instances/{TARGET_INSTANCE}"
)

RELEASE_BASE = Path("/opt/muncho-owner-gate/releases")
CURRENT_RELEASE_LINK = Path("/opt/muncho-owner-gate/current")
PASSKEY_AUTHORITY_SOCKET = Path(
    "/run/muncho-owner-gate/passkey-authority.sock"
)
PRIVILEGED_EXECUTOR_SOCKET = Path(
    "/run/muncho-owner-gate/privileged-executor.sock"
)
AUTHORITY_ENTRYPOINT = CURRENT_RELEASE_LINK / "bin/muncho-passkey-v2-authority"
EXECUTOR_ENTRYPOINT = CURRENT_RELEASE_LINK / "bin/muncho-passkey-v2-executor"
MUTATION_ENABLE_SEAL = Path(
    "/etc/muncho-owner-gate/storage-executor-enabled"
)

NETWORK_EVIDENCE_SCHEMA = "muncho-owner-gate-production-network-evidence.v2"
NETWORK_EVIDENCE_SIGNATURE_DOMAIN = (
    b"muncho-owner-gate/production-network-evidence/v2\x00"
)
PLAN_SCHEMA = "muncho-owner-gate-gce-foundation-plan.v1"
PREFLIGHT_MAX_AGE_SECONDS = 900
MUTATION_PERMISSIONS = (
    "compute.disks.get",
    "compute.disks.resize",
    "compute.instances.get",
    "compute.instances.start",
    "compute.instances.stop",
)
EXECUTION_PERMISSIONS = MUTATION_PERMISSIONS
READ_ONLY_IAM_PERMISSIONS = (
    "iam.roles.get",
    "iam.serviceAccountKeys.list",
    "iam.serviceAccounts.get",
    "iam.serviceAccounts.getIamPolicy",
    "resourcemanager.projects.get",
    "resourcemanager.projects.getIamPolicy",
)
DIRECT_IAM_ANCESTOR_PERMISSIONS = (
    "iam.roles.get",
    "resourcemanager.folders.get",
    "resourcemanager.folders.getIamPolicy",
    "resourcemanager.organizations.get",
    "resourcemanager.organizations.getIamPolicy",
)
OWNER_GATE_OAUTH_SCOPES = (
    "https://www.googleapis.com/auth/compute",
    "https://www.googleapis.com/auth/iam",
    "https://www.googleapis.com/auth/cloudplatformprojects.readonly",
    "https://www.googleapis.com/auth/cloudplatformfolders.readonly",
    "https://www.googleapis.com/auth/cloudplatformorganizations.readonly",
)
PRIVATE_GOOGLE_API_HOSTS = (
    "compute.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
)
FORBIDDEN_PROJECT_ROLES = (
    "roles/owner",
    "roles/editor",
    "roles/compute.admin",
    "roles/iam.serviceAccountKeyAdmin",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_RESOURCE_NAME = re.compile(r"^[a-z][a-z0-9-]{1,61}[a-z0-9]$")
_B64URL_UNPADDED = re.compile(r"^[A-Za-z0-9_-]+$")
_OPAQUE_PROVIDER_TAG = re.compile(r"^[A-Za-z0-9_+/=.-]{1,256}$")
_OWNER_SUBNET_ROUTE_NAME = re.compile(r"^default-route-r-[0-9a-f]{16}$")
_RFC3339_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,9})?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)


class OwnerGateFoundationError(RuntimeError):
    """Stable, secret-free owner-gate foundation error."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise OwnerGateFoundationError(
            "owner_gate_foundation_json_invalid"
        ) from None


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _decode_canonical_ed25519_signature(value: Any) -> bytes:
    """Decode one canonical, unpadded 64-byte Ed25519 signature."""

    if (
        not isinstance(value, str)
        or len(value) != 86
        or _B64URL_UNPADDED.fullmatch(value) is None
    ):
        raise OwnerGateFoundationError(
            "owner_gate_network_evidence_signature_invalid"
        )
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        raise OwnerGateFoundationError(
            "owner_gate_network_evidence_signature_invalid"
        ) from None
    if (
        len(raw) != 64
        or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != value
    ):
        raise OwnerGateFoundationError(
            "owner_gate_network_evidence_signature_invalid"
        )
    return raw


def _strict_keys(value: Mapping[str, Any], fields: frozenset[str], label: str) -> None:
    if frozenset(value) != fields:
        raise OwnerGateFoundationError(f"owner_gate_{label}_fields_invalid")


def _validate_preexisting_owner_gate_network_identities(
    *,
    subnet_identity: Mapping[str, Any] | None,
    route_identity: Mapping[str, Any] | None,
) -> None:
    """Validate the exact, author-observed owner subnet and local route."""

    if subnet_identity is None and route_identity is None:
        return
    if not isinstance(subnet_identity, Mapping) or not isinstance(
        route_identity, Mapping
    ):
        raise OwnerGateFoundationError(
            "owner_gate_network_evidence_invalid"
        )

    subnet_fields = frozenset(
        {
            "resource_type",
            "kind",
            "name",
            "self_link",
            "numeric_id",
            "fingerprint",
            "creation_timestamp",
            "network_self_link",
            "region_self_link",
            "ip_cidr_range",
            "private_ip_google_access",
            "stack_type",
            "purpose",
            "secondary_ip_ranges",
            "allow_subnet_cidr_routes_overlap",
            "gateway_address",
            "private_ipv6_google_access",
        }
    )
    route_fields = frozenset(
        {
            "resource_type",
            "kind",
            "name",
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
        }
    )
    try:
        _strict_keys(subnet_identity, subnet_fields, "network_evidence")
        _strict_keys(route_identity, route_fields, "network_evidence")
    except OwnerGateFoundationError:
        raise OwnerGateFoundationError(
            "owner_gate_network_evidence_invalid"
        ) from None

    network = f"projects/{PROJECT}/global/networks/{NETWORK_NAME}"
    region = f"projects/{PROJECT}/regions/{REGION}"
    subnet = f"{region}/subnetworks/{OWNER_GATE_SUBNET_NAME}"
    route_name = route_identity.get("name")
    route = f"projects/{PROJECT}/global/routes/{route_name}"
    expected_description = (
        f"Default local route to the subnetwork {OWNER_GATE_SUBNET_CIDR}."
    )
    if (
        subnet_identity.get("resource_type") != "compute_subnetwork"
        or subnet_identity.get("kind") != "compute#subnetwork"
        or subnet_identity.get("name") != OWNER_GATE_SUBNET_NAME
        or subnet_identity.get("self_link") != subnet
        or not isinstance(subnet_identity.get("numeric_id"), str)
        or _NUMERIC_ID.fullmatch(
            str(subnet_identity.get("numeric_id", ""))
        )
        is None
        or not isinstance(subnet_identity.get("fingerprint"), str)
        or _OPAQUE_PROVIDER_TAG.fullmatch(
            str(subnet_identity.get("fingerprint", ""))
        )
        is None
        or _RFC3339_TIMESTAMP.fullmatch(
            str(subnet_identity.get("creation_timestamp", ""))
        )
        is None
        or subnet_identity.get("network_self_link") != network
        or subnet_identity.get("region_self_link") != region
        or subnet_identity.get("ip_cidr_range") != OWNER_GATE_SUBNET_CIDR
        or subnet_identity.get("private_ip_google_access") is not True
        or subnet_identity.get("stack_type") != "IPV4_ONLY"
        or subnet_identity.get("purpose") != "PRIVATE"
        or subnet_identity.get("secondary_ip_ranges") != []
        or subnet_identity.get("allow_subnet_cidr_routes_overlap") is not False
        or subnet_identity.get("gateway_address") != "10.80.3.1"
        or subnet_identity.get("private_ipv6_google_access")
        != "DISABLE_GOOGLE_ACCESS"
        or route_identity.get("resource_type") != "compute_route"
        or route_identity.get("kind") != "compute#route"
        or not isinstance(route_name, str)
        or _OWNER_SUBNET_ROUTE_NAME.fullmatch(route_name) is None
        or route_identity.get("self_link") != route
        or not isinstance(route_identity.get("numeric_id"), str)
        or _NUMERIC_ID.fullmatch(
            str(route_identity.get("numeric_id", ""))
        )
        is None
        or _RFC3339_TIMESTAMP.fullmatch(
            str(route_identity.get("creation_timestamp", ""))
        )
        is None
        or route_identity.get("network_self_link") != network
        or route_identity.get("destination_range")
        != OWNER_GATE_SUBNET_CIDR
        or route_identity.get("next_hop_network_self_link") != network
        or type(route_identity.get("priority")) is not int
        or route_identity.get("priority") != 0
        or route_identity.get("description") != expected_description
        or route_identity.get("route_type") != "SUBNET"
        or route_identity.get("tags") != []
    ):
        raise OwnerGateFoundationError(
            "owner_gate_network_evidence_invalid"
        )


@dataclass(frozen=True)
class ProductionNetworkEvidence:
    schema: str
    collected_at_unix: int
    project: str
    zone: str
    source_instance: str
    source_instance_id: str
    source_instance_self_link: str
    source_service_account: str
    network_self_link: str
    subnetwork_self_link: str
    source_internal_ip: str
    subnetwork_cidr: str
    reserved_network_ranges: tuple[str, ...]
    preexisting_owner_gate_subnet_identity: Mapping[str, Any] | None
    preexisting_owner_gate_subnet_route_identity: Mapping[str, Any] | None
    range_inventory_receipts: Mapping[str, str]
    network_connectivity_api_disabled: bool
    private_google_access: bool
    iap_firewall_rule: str
    iap_source_range: str
    collector_public_key_id: str
    signature_ed25519_b64url: str
    evidence_sha256: str

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        public_key: Ed25519PublicKey,
        expected_public_key_id: str,
        now_unix: int | None = None,
    ) -> "ProductionNetworkEvidence":
        fields = frozenset(field.name for field in cls.__dataclass_fields__.values())
        if not isinstance(raw, Mapping):
            raise OwnerGateFoundationError("owner_gate_network_evidence_invalid")
        _strict_keys(raw, fields, "network_evidence")
        try:
            value = cls(**dict(raw))
        except TypeError as exc:
            raise OwnerGateFoundationError(
                "owner_gate_network_evidence_invalid"
            ) from None
        value.validate(now_unix=now_unix)
        value.verify_attestation(
            public_key=public_key,
            expected_public_key_id=expected_public_key_id,
        )
        return value

    def unsigned(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {
                "evidence_sha256",
                "collector_public_key_id",
                "signature_ed25519_b64url",
            }
        }

    def signed_payload(self) -> dict[str, Any]:
        return {
            **self.unsigned(),
            "evidence_sha256": self.evidence_sha256,
            "collector_public_key_id": self.collector_public_key_id,
        }

    def verify_attestation(
        self,
        *,
        public_key: Ed25519PublicKey,
        expected_public_key_id: str,
    ) -> None:
        if not isinstance(public_key, Ed25519PublicKey):
            raise OwnerGateFoundationError(
                "owner_gate_network_evidence_key_invalid"
            )
        key_id = hashlib.sha256(public_key.public_bytes_raw()).hexdigest()
        if (
            self.collector_public_key_id != key_id
            or key_id != expected_public_key_id
        ):
            raise OwnerGateFoundationError(
                "owner_gate_network_evidence_key_invalid"
            )
        try:
            signature = _decode_canonical_ed25519_signature(
                self.signature_ed25519_b64url
            )
            public_key.verify(
                signature,
                NETWORK_EVIDENCE_SIGNATURE_DOMAIN
                + canonical_json_bytes(self.signed_payload()),
            )
        except (InvalidSignature, OwnerGateFoundationError) as exc:
            raise OwnerGateFoundationError(
                "owner_gate_network_evidence_signature_invalid"
            ) from None

    def validate(self, *, now_unix: int | None = None) -> None:
        _validate_preexisting_owner_gate_network_identities(
            subnet_identity=self.preexisting_owner_gate_subnet_identity,
            route_identity=self.preexisting_owner_gate_subnet_route_identity,
        )
        expected_network = (
            f"projects/{PROJECT}/global/networks/{NETWORK_NAME}"
        )
        expected_subnet = (
            f"projects/{PROJECT}/regions/{REGION}/subnetworks/"
            f"{PRODUCTION_SUBNET_NAME}"
        )
        expected_source = (
            f"projects/{PROJECT}/zones/{ZONE}/instances/{PRODUCTION_SOURCE_VM}"
        )
        normalized = {
            "network": self.network_self_link.removeprefix(
                "https://www.googleapis.com/compute/v1/"
            ),
            "subnet": self.subnetwork_self_link.removeprefix(
                "https://www.googleapis.com/compute/v1/"
            ),
            "source": self.source_instance_self_link.removeprefix(
                "https://www.googleapis.com/compute/v1/"
            ),
        }
        if (
            self.schema != NETWORK_EVIDENCE_SCHEMA
            or self.project != PROJECT
            or self.zone != ZONE
            or self.source_instance != PRODUCTION_SOURCE_VM
            or self.source_instance_id != PRODUCTION_SOURCE_VM_ID
            or self.source_internal_ip != "10.80.0.2"
            or self.source_service_account != PRODUCTION_SOURCE_SERVICE_ACCOUNT
            or normalized["network"] != expected_network
            or normalized["subnet"] != expected_subnet
            or normalized["source"] != expected_source
            or self.iap_firewall_rule != "allow-iap-ssh"
            or self.iap_source_range != IAP_SOURCE_RANGE
            or not isinstance(self.collected_at_unix, int)
            or isinstance(self.collected_at_unix, bool)
            or self.collected_at_unix <= 0
            or not isinstance(self.private_google_access, bool)
            or self.network_connectivity_api_disabled is not True
            or not isinstance(self.range_inventory_receipts, Mapping)
            or set(self.range_inventory_receipts) != {
                "aggregate_subnets",
                "routes",
                "peerings",
                "private_service_ranges",
                "serverless_connectors",
                "network_connectivity_service",
                "policy_based_routes",
            }
            or any(
                _SHA256.fullmatch(str(item)) is None
                for item in self.range_inventory_receipts.values()
            )
            or self.range_inventory_receipts.get(
                "network_connectivity_service"
            )
            != sha256_json([])
            or self.range_inventory_receipts.get("policy_based_routes")
            != sha256_json([])
            or _SHA256.fullmatch(self.collector_public_key_id or "") is None
            or not isinstance(self.signature_ed25519_b64url, str)
            or not self.signature_ed25519_b64url
        ):
            raise OwnerGateFoundationError(
                "owner_gate_network_evidence_invalid"
            )
        try:
            address = ipaddress.ip_address(self.source_internal_ip)
            network = ipaddress.ip_network(self.subnetwork_cidr, strict=True)
            reserved = tuple(
                ipaddress.ip_network(item, strict=True)
                for item in self.reserved_network_ranges
            )
            owner_gate_network = ipaddress.ip_network(
                OWNER_GATE_SUBNET_CIDR,
                strict=True,
            )
        except (TypeError, ValueError) as exc:
            raise OwnerGateFoundationError(
                "owner_gate_network_evidence_invalid"
            ) from None
        if (
            address.version != 4
            or network.version != 4
            or not address.is_private
            or address not in network
            or network.prefixlen < 16
            or network.prefixlen > 28
            or not reserved
            or len(reserved) != len(set(reserved))
            or network not in reserved
            or any(owner_gate_network.overlaps(item) for item in reserved)
            or not _SHA256.fullmatch(self.evidence_sha256 or "")
            or sha256_json(self.unsigned()) != self.evidence_sha256
        ):
            raise OwnerGateFoundationError(
                "owner_gate_network_evidence_invalid"
            )
        if now_unix is not None and (
            not isinstance(now_unix, int)
            or isinstance(now_unix, bool)
            or now_unix < self.collected_at_unix
            or now_unix - self.collected_at_unix > PREFLIGHT_MAX_AGE_SECONDS
        ):
            raise OwnerGateFoundationError("owner_gate_network_evidence_stale")


@dataclass(frozen=True)
class OwnerGateSpec:
    release_revision: str
    boot_image_self_link: str
    interpreter_sha256: str
    network_collector_public_key_id: str
    organization_id: str
    ancestry_evidence_sha256: str
    source_tree_oid: str | None = None
    boot_image_numeric_id: str | None = None
    package_inventory_sha256: str | None = None
    cloud_collector_public_key_id: str | None = None
    host_collector_public_key_id: str | None = None
    project: str = PROJECT
    region: str = REGION
    zone: str = ZONE
    vm_name: str = VM_NAME
    service_account_name: str = SERVICE_ACCOUNT_NAME
    machine_type: str = MACHINE_TYPE
    boot_disk_size_gb: int = BOOT_DISK_SIZE_GB
    boot_disk_type: str = BOOT_DISK_TYPE

    @property
    def service_account_email(self) -> str:
        return f"{self.service_account_name}@{self.project}.iam.gserviceaccount.com"

    @property
    def custom_role(self) -> str:
        return f"projects/{self.project}/roles/{MUTATION_ROLE_ID}"

    @property
    def read_only_iam_role(self) -> str:
        return f"projects/{self.project}/roles/{READ_ONLY_IAM_ROLE_ID}"

    @property
    def organization_resource(self) -> str:
        return f"organizations/{self.organization_id}"

    @property
    def ancestor_read_only_iam_role(self) -> str:
        return (
            f"{self.organization_resource}/roles/"
            f"{ANCESTOR_READ_ONLY_IAM_ROLE_ID}"
        )

    @property
    def release_root(self) -> Path:
        return RELEASE_BASE / self.release_revision

    @property
    def final_release_bound(self) -> bool:
        return all(
            isinstance(value, str)
            for value in (
                self.package_inventory_sha256,
                self.cloud_collector_public_key_id,
                self.host_collector_public_key_id,
            )
        )

    @property
    def pre_foundation_bound(self) -> bool:
        return (
            not self.final_release_bound
            and self.package_inventory_sha256 is None
            and self.cloud_collector_public_key_id is None
            and self.host_collector_public_key_id is None
            and isinstance(self.source_tree_oid, str)
            and isinstance(self.boot_image_numeric_id, str)
        )

    def validate(self) -> None:
        if (
            self.project != PROJECT
            or self.region != REGION
            or self.zone != ZONE
            or self.vm_name != VM_NAME
            or self.service_account_name != SERVICE_ACCOUNT_NAME
            or self.machine_type != MACHINE_TYPE
            or self.boot_disk_size_gb != BOOT_DISK_SIZE_GB
            or self.boot_disk_type != BOOT_DISK_TYPE
            or _REVISION.fullmatch(self.release_revision or "") is None
            or _SHA256.fullmatch(self.interpreter_sha256 or "") is None
            or _SHA256.fullmatch(self.network_collector_public_key_id or "") is None
            or _NUMERIC_ID.fullmatch(self.organization_id or "") is None
            or _SHA256.fullmatch(self.ancestry_evidence_sha256 or "") is None
            or (
                not self.final_release_bound
                and not self.pre_foundation_bound
            )
            or (
                self.final_release_bound
                and any(
                    _SHA256.fullmatch(str(value)) is None
                    for value in (
                        self.package_inventory_sha256,
                        self.cloud_collector_public_key_id,
                        self.host_collector_public_key_id,
                    )
                )
            )
            or (
                self.source_tree_oid is not None
                and _REVISION.fullmatch(self.source_tree_oid) is None
            )
            or (
                self.boot_image_numeric_id is not None
                and _NUMERIC_ID.fullmatch(self.boot_image_numeric_id) is None
            )
            or _RESOURCE_NAME.fullmatch(self.vm_name) is None
            or not self.boot_image_self_link.startswith(
                "projects/debian-cloud/global/images/"
            )
            or "family" in self.boot_image_self_link
            or not re.fullmatch(
                r"projects/debian-cloud/global/images/debian-12-bookworm-v[0-9]{8}",
                self.boot_image_self_link,
            )
        ):
            raise OwnerGateFoundationError("owner_gate_spec_invalid")


@dataclass(frozen=True)
class PlanStep:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class OwnerGateFoundationPlan:
    schema: str
    spec: OwnerGateSpec
    network_evidence_sha256: str
    architecture: Mapping[str, Any]
    foundation_steps: tuple[PlanStep, ...]
    deferred_private_api_connectivity_steps: tuple[PlanStep, ...]
    deferred_mutation_iam_steps: tuple[PlanStep, ...]
    post_binding_validation: Mapping[str, Any]
    rollback_steps: tuple[PlanStep, ...]

    def payload(self) -> dict[str, Any]:
        spec_payload = {
            key: value
            for key, value in asdict(self.spec).items()
            if value is not None
        }
        if self.spec.final_release_bound:
            spec_payload["release_root"] = str(self.spec.release_root)
        return {
            "schema": self.schema,
            "spec": {
                **spec_payload,
                "service_account_email": self.spec.service_account_email,
                "custom_role": self.spec.custom_role,
                "read_only_iam_role": self.spec.read_only_iam_role,
                "organization_resource": self.spec.organization_resource,
                "ancestor_read_only_iam_role": (
                    self.spec.ancestor_read_only_iam_role
                ),
            },
            "network_evidence_sha256": self.network_evidence_sha256,
            "architecture": dict(self.architecture),
            "foundation_steps": [asdict(step) for step in self.foundation_steps],
            "deferred_private_api_connectivity_steps": [
                asdict(step)
                for step in self.deferred_private_api_connectivity_steps
            ],
            "deferred_mutation_iam_steps": [
                asdict(step) for step in self.deferred_mutation_iam_steps
            ],
            "post_binding_validation": dict(self.post_binding_validation),
            "rollback_steps": [asdict(step) for step in self.rollback_steps],
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.payload())

    def report(self) -> dict[str, Any]:
        return {**self.payload(), "plan_sha256": self.sha256}


def _condition_expression() -> str:
    return (
        "(resource.type == 'compute.googleapis.com/Disk' && "
        f"resource.name == '{TARGET_DISK_SELF_LINK}') || "
        "(resource.type == 'compute.googleapis.com/Instance' && "
        f"resource.name == '{TARGET_INSTANCE_SELF_LINK}')"
    )


def build_plan(
    *,
    spec: OwnerGateSpec,
    network_evidence: ProductionNetworkEvidence,
    network_collector_public_key: Ed25519PublicKey,
    now_unix: int,
) -> OwnerGateFoundationPlan:
    spec.validate()
    network_evidence.validate(now_unix=now_unix)
    network_evidence.verify_attestation(
        public_key=network_collector_public_key,
        expected_public_key_id=spec.network_collector_public_key_id,
    )
    project = spec.project
    service_account = spec.service_account_email
    network = network_evidence.network_self_link.removeprefix(
        "https://www.googleapis.com/compute/v1/"
    )
    subnet = (
        f"projects/{project}/regions/{REGION}/subnetworks/"
        f"{OWNER_GATE_SUBNET_NAME}"
    )
    condition = _condition_expression()
    return OwnerGateFoundationPlan(
        schema=PLAN_SCHEMA,
        spec=spec,
        network_evidence_sha256=network_evidence.evidence_sha256,
        architecture={
            "network_evidence_collected_at_unix": (
                network_evidence.collected_at_unix
            ),
            "preexisting_owner_gate_subnet_identity": (
                None
                if network_evidence.preexisting_owner_gate_subnet_identity
                is None
                else dict(
                    network_evidence.preexisting_owner_gate_subnet_identity
                )
            ),
            "preexisting_owner_gate_subnet_route_identity": (
                None
                if network_evidence.preexisting_owner_gate_subnet_route_identity
                is None
                else dict(
                    network_evidence.preexisting_owner_gate_subnet_route_identity
                )
            ),
            "network_connectivity_api_disabled": (
                network_evidence.network_connectivity_api_disabled
            ),
            "private_vm": True,
            "external_ip_allowed": False,
            "same_production_vpc": True,
            "same_production_subnet": False,
            "dedicated_owner_gate_subnet": True,
            "owner_gate_subnet": subnet,
            "owner_gate_subnet_cidr": OWNER_GATE_SUBNET_CIDR,
            "owner_gate_private_ip": OWNER_GATE_PRIVATE_IP,
            "admin_transport": "iap_tcp_forwarding_only",
            "generic_ssh_runtime_transport": False,
            "local_gcloud_runtime_fallback": False,
            "caddy_location": PRODUCTION_SOURCE_VM,
            "caddy_origin_unchanged": "https://auth.lomliev.com",
            "caddy_private_upstream_port": WEB_LISTEN_PORT,
            "dedicated_service_account": True,
            "service_account_keys_allowed": False,
            "project_owner_or_editor_allowed": False,
            "mutation_iam_enabled_during_bootstrap": False,
            "mutation_iam_activation_requires_split_uid_smoke": True,
            "activation_sequence": [
                "offline_split_uid_webauthn_concurrency_and_firewall_smoke",
                "bind_exact_resource_mutation_role",
                "executor_api_connectivity_and_numeric_target_repreflight_with_seal_absent",
                "gcp_owner_reauth_installs_topology_and_iam_readiness_seal",
                "web_authn_exact_action_authorization",
                "exact_action_execution_requires_both_seal_and_receipt",
            ],
            "executor_activation_seal": str(MUTATION_ENABLE_SEAL),
            "executor_receipt_public_key_source": (
                "target_generated_key_bound_by_signed_install_and_host_receipts"
            ),
            "web_to_authority_socket": str(PASSKEY_AUTHORITY_SOCKET),
            "authority_to_executor_socket": str(PRIVILEGED_EXECUTOR_SOCKET),
            "private_google_api_vip_range": PRIVATE_GOOGLE_API_VIP_RANGE,
            "private_google_api_hosts": list(PRIVATE_GOOGLE_API_HOSTS),
            "private_google_access_required": True,
            "offline_bootstrap_requires_package_transfer_over_iap": True,
            "bootstrap_apt_or_pip_network_access_required": False,
            "production_subnet_mutation_required": False,
            "allowed_execution_permissions": list(EXECUTION_PERMISSIONS),
            "mutation_role_permissions": list(MUTATION_PERMISSIONS),
            "read_only_iam_role": spec.read_only_iam_role,
            "read_only_iam_role_permissions": list(READ_ONLY_IAM_PERMISSIONS),
            "direct_iam_ancestor_permissions": list(
                DIRECT_IAM_ANCESTOR_PERMISSIONS
            ),
            "organization_resource": spec.organization_resource,
            "ancestry_evidence_sha256": spec.ancestry_evidence_sha256,
            "ancestor_read_only_iam_role": (
                spec.ancestor_read_only_iam_role
            ),
            "direct_iam_ancestor_bindings_require_pinned_external_ids": True,
            "owner_gate_oauth_scopes": list(OWNER_GATE_OAUTH_SCOPES),
            "operation_poll_permission_present": False,
            "operation_completion_observation": (
                "poll_exact_conditioned_disk_and_instance_get"
            ),
            "forbidden_project_roles": list(FORBIDDEN_PROJECT_ROLES),
            "target_disk": TARGET_DISK_SELF_LINK,
            "target_disk_id": TARGET_DISK_ID,
            "target_boot_device": TARGET_BOOT_DEVICE,
            "target_instance": TARGET_INSTANCE_SELF_LINK,
            "target_instance_id": TARGET_INSTANCE_ID,
            "source_runtime_instance_id": PRODUCTION_SOURCE_VM_ID,
            "network_collector_public_key_id": (
                spec.network_collector_public_key_id
            ),
            **(
                {
                    "release_root": str(spec.release_root),
                    "release_root_owner": "root:root",
                    "release_root_mode": "0555",
                    "cloud_collector_public_key_id": (
                        spec.cloud_collector_public_key_id
                    ),
                    "host_collector_public_key_id": (
                        spec.host_collector_public_key_id
                    ),
                }
                if spec.final_release_bound
                else {
                    "pre_foundation_only": True,
                    "final_package_inventory_bound": False,
                    "package_deployment_authorized": False,
                    "service_start_authorized": False,
                    "foundation_source_tree_oid": spec.source_tree_oid,
                    "boot_image_numeric_id": spec.boot_image_numeric_id,
                }
            ),
        },
        foundation_steps=(
            PlanStep(
                "create_dedicated_service_account",
                (
                    "gcloud",
                    "iam",
                    "service-accounts",
                    "create",
                    spec.service_account_name,
                    f"--project={project}",
                    "--display-name=Muncho owner gate executor",
                    "--quiet",
                ),
            ),
            PlanStep(
                "create_narrow_iam_observation_reader_role",
                (
                    "gcloud",
                    "iam",
                    "roles",
                    "create",
                    READ_ONLY_IAM_ROLE_ID,
                    f"--project={project}",
                    f"--title={PROJECT_READ_ROLE_TITLE}",
                    f"--description={PROJECT_READ_ROLE_DESCRIPTION}",
                    f"--permissions={','.join(READ_ONLY_IAM_PERMISSIONS)}",
                    "--stage=GA",
                    "--quiet",
                ),
            ),
            PlanStep(
                "bind_narrow_iam_observation_reader_to_owner_gate_service_account",
                (
                    "owner-gate-provider",
                    "set-iam-binding-cas",
                    f"projects/{project}",
                    f"--member=serviceAccount:{service_account}",
                    f"--role={spec.read_only_iam_role}",
                    "--condition=None",
                ),
            ),
            PlanStep(
                "create_narrow_storage_executor_role",
                (
                    "gcloud",
                    "iam",
                    "roles",
                    "create",
                    MUTATION_ROLE_ID,
                    f"--project={project}",
                    f"--title={MUTATION_ROLE_TITLE}",
                    f"--description={MUTATION_ROLE_DESCRIPTION}",
                    f"--permissions={','.join(MUTATION_PERMISSIONS)}",
                    "--stage=GA",
                    "--quiet",
                ),
            ),
            PlanStep(
                "create_narrow_organization_iam_observation_reader_role",
                (
                    "gcloud",
                    "iam",
                    "roles",
                    "create",
                    ANCESTOR_READ_ONLY_IAM_ROLE_ID,
                    f"--organization={spec.organization_id}",
                    f"--title={ANCESTOR_READ_ROLE_TITLE}",
                    f"--description={ANCESTOR_READ_ROLE_DESCRIPTION}",
                    "--permissions="
                    + ",".join(DIRECT_IAM_ANCESTOR_PERMISSIONS),
                    "--stage=GA",
                    "--quiet",
                ),
            ),
            PlanStep(
                "bind_narrow_organization_iam_observation_reader_to_owner_gate_service_account",
                (
                    "owner-gate-provider",
                    "set-iam-binding-cas",
                    spec.organization_resource,
                    f"--member=serviceAccount:{service_account}",
                    f"--role={spec.ancestor_read_only_iam_role}",
                    "--condition=None",
                ),
            ),
            PlanStep(
                "create_dedicated_private_owner_gate_subnet",
                (
                    "gcloud",
                    "compute",
                    "networks",
                    "subnets",
                    "create",
                    OWNER_GATE_SUBNET_NAME,
                    f"--project={project}",
                    f"--region={REGION}",
                    f"--network={network}",
                    f"--range={OWNER_GATE_SUBNET_CIDR}",
                    "--stack-type=IPV4_ONLY",
                    "--enable-private-ip-google-access",
                    "--quiet",
                ),
            ),
            PlanStep(
                "create_private_owner_gate_vm",
                (
                    "gcloud",
                    "compute",
                    "instances",
                    "create",
                    spec.vm_name,
                    f"--project={project}",
                    f"--zone={spec.zone}",
                    f"--machine-type={spec.machine_type}",
                    f"--network={network}",
                    f"--subnet={subnet}",
                    f"--private-network-ip={OWNER_GATE_PRIVATE_IP}",
                    "--no-address",
                    f"--service-account={service_account}",
                    f"--scopes={','.join(OWNER_GATE_OAUTH_SCOPES)}",
                    f"--image={spec.boot_image_self_link}",
                    f"--boot-disk-size={spec.boot_disk_size_gb}GB",
                    f"--boot-disk-type={spec.boot_disk_type}",
                    f"--tags={IAP_NETWORK_TAG},{OWNER_GATE_NETWORK_TAG}",
                    "--metadata=enable-oslogin=TRUE,block-project-ssh-keys=TRUE,serial-port-enable=FALSE",
                    "--shielded-secure-boot",
                    "--shielded-vtpm",
                    "--shielded-integrity-monitoring",
                    "--maintenance-policy=MIGRATE",
                    "--provisioning-model=STANDARD",
                    "--quiet",
                ),
            ),
            PlanStep(
                "allow_private_web_upstream_from_current_caddy_host",
                (
                    "gcloud",
                    "compute",
                    "firewall-rules",
                    "create",
                    "muncho-owner-gate-web-from-production",
                    f"--project={project}",
                    f"--network={network}",
                    "--direction=INGRESS",
                    "--priority=700",
                    "--action=ALLOW",
                    f"--rules=tcp:{WEB_LISTEN_PORT}",
                    f"--source-service-accounts={network_evidence.source_service_account}",
                    f"--target-service-accounts={service_account}",
                    "--enable-logging",
                    "--quiet",
                ),
            ),
        ),
        deferred_private_api_connectivity_steps=(),
        deferred_mutation_iam_steps=(
            PlanStep(
                "activate_resource_conditioned_mutation_role_after_smoke",
                (
                    "gcloud",
                    "projects",
                    "add-iam-policy-binding",
                    project,
                    f"--member=serviceAccount:{service_account}",
                    f"--role={spec.custom_role}",
                    "--condition="
                    "title=muncho_owner_gate_exact_storage_v1,"
                    "description=Exact canary disk and instance resources only,"
                    f"expression={condition}",
                    "--quiet",
                ),
            ),
        ),
        post_binding_validation={
            "policy_condition_lint_required": True,
            "policy_condition_supported_resource_types": [
                "compute.googleapis.com/Disk",
                "compute.googleapis.com/Instance",
            ],
            "operation_permission_present": False,
            "completion_observed_via_exact_resource_get": True,
            "executor_activation_seal_must_be_absent": True,
            "executor_compute_api_connectivity_required": True,
            "target_instance_numeric_id_required": TARGET_INSTANCE_ID,
            "target_disk_numeric_id_required": TARGET_DISK_ID,
            "effective_permissions_must_equal": list(EXECUTION_PERMISSIONS),
            "mutation_attempt_during_validation_allowed": False,
        },
        rollback_steps=(
            PlanStep(
                "remove_exact_mutation_binding_if_present",
                (
                    "gcloud", "projects", "remove-iam-policy-binding", project,
                    f"--member=serviceAccount:{service_account}",
                    f"--role={spec.custom_role}",
                    "--condition="
                    "title=muncho_owner_gate_exact_storage_v1,"
                    "description=Exact canary disk and instance resources only,"
                    f"expression={condition}", "--quiet",
                ),
            ),
            PlanStep(
                "delete_private_web_firewall_if_created",
                (
                    "gcloud", "compute", "firewall-rules", "delete",
                    "muncho-owner-gate-web-from-production",
                    f"--project={project}", "--quiet",
                ),
            ),
            PlanStep(
                "delete_private_owner_gate_vm_if_created",
                (
                    "gcloud", "compute", "instances", "delete", spec.vm_name,
                    f"--project={project}", f"--zone={spec.zone}",
                    "--delete-disks=all", "--quiet",
                ),
            ),
            PlanStep(
                "delete_dedicated_owner_gate_subnet_if_created",
                (
                    "gcloud", "compute", "networks", "subnets", "delete",
                    OWNER_GATE_SUBNET_NAME, f"--project={project}",
                    f"--region={REGION}", "--quiet",
                ),
            ),
            PlanStep(
                "remove_exact_organization_iam_observation_binding_if_present",
                (
                    "owner-gate-provider", "remove-iam-binding-cas",
                    spec.organization_resource,
                    f"--member=serviceAccount:{service_account}",
                    f"--role={spec.ancestor_read_only_iam_role}",
                    "--condition=None",
                ),
            ),
            PlanStep(
                "delete_organization_iam_observation_role_if_created",
                (
                    "gcloud", "iam", "roles", "delete",
                    ANCESTOR_READ_ONLY_IAM_ROLE_ID,
                    f"--organization={spec.organization_id}", "--quiet",
                ),
            ),
            PlanStep(
                "remove_exact_read_only_iam_observation_binding_if_present",
                (
                    "owner-gate-provider", "remove-iam-binding-cas",
                    f"projects/{project}",
                    f"--member=serviceAccount:{service_account}",
                    f"--role={spec.read_only_iam_role}",
                    "--condition=None",
                ),
            ),
            PlanStep(
                "delete_mutation_custom_role_if_created",
                (
                    "gcloud", "iam", "roles", "delete", MUTATION_ROLE_ID,
                    f"--project={project}", "--quiet",
                ),
            ),
            PlanStep(
                "delete_read_only_iam_custom_role_if_created",
                (
                    "gcloud", "iam", "roles", "delete", READ_ONLY_IAM_ROLE_ID,
                    f"--project={project}", "--quiet",
                ),
            ),
            PlanStep(
                "delete_dedicated_service_account_if_created",
                (
                    "gcloud", "iam", "service-accounts", "delete",
                    service_account, f"--project={project}", "--quiet",
                ),
            ),
        ),
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        if not raw or len(raw) > 1024 * 1024:
            raise ValueError
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise OwnerGateFoundationError(
            "owner_gate_network_evidence_unavailable"
        ) from None
    if not isinstance(value, Mapping):
        raise OwnerGateFoundationError("owner_gate_network_evidence_invalid")
    return value


def _load_ed25519_public_key(path: Path) -> Ed25519PublicKey:
    try:
        raw = path.read_bytes()
        if not raw or len(raw) > 16 * 1024:
            raise ValueError
        key = serialization.load_pem_public_key(raw)
    except (OSError, TypeError, ValueError) as exc:
        raise OwnerGateFoundationError(
            "owner_gate_network_evidence_key_invalid"
        ) from None
    if not isinstance(key, Ed25519PublicKey):
        raise OwnerGateFoundationError(
            "owner_gate_network_evidence_key_invalid"
        )
    return key


def main(argv: Sequence[str] | None = None) -> int:
    # Local imports avoid a module cycle: both validators reuse this module's
    # canonical JSON and project constants.
    from scripts.canary import owner_gate_pre_foundation as pre_foundation
    from scripts.canary import owner_gate_owner_reauth as owner_reauth
    from scripts.canary import owner_gate_trust as release_trust

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-foundation-authority", type=Path, required=True)
    parser.add_argument("--owner-reauth-receipt", type=Path, required=True)
    parser.add_argument("--release-trust-public-key", type=Path, required=True)
    parser.add_argument("--network-collector-public-key", type=Path, required=True)
    parser.add_argument("--network-evidence", type=Path, required=True)
    parser.add_argument("--project-ancestry-evidence", type=Path, required=True)
    parser.add_argument(
        "--project-ancestry-collector-public-key",
        type=Path,
        required=True,
    )
    arguments = parser.parse_args(argv)
    now_unix = int(time.time())
    try:
        release_key = pre_foundation.load_pinned_public_key(
            arguments.release_trust_public_key,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
        )
        authority_raw = release_trust._read_immutable(
            arguments.pre_foundation_authority,
            maximum=pre_foundation.MAX_JSON_BYTES,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
        reauth_raw = release_trust._read_immutable(
            arguments.owner_reauth_receipt,
            maximum=owner_reauth.MAX_CAPTURE_BYTES,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
        reauth_receipt = owner_reauth.decode_canonical_owner_reauth_receipt(
            reauth_raw,
            public_key=release_key,
            now_unix=now_unix,
        )
        ancestry_raw = release_trust._read_immutable(
            arguments.project_ancestry_evidence,
            maximum=pre_foundation.MAX_JSON_BYTES,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
        ancestry_key = _load_ed25519_public_key(
            arguments.project_ancestry_collector_public_key
        )
        authority = pre_foundation.decode_canonical_authority(
            authority_raw,
            public_key=release_key,
            owner_reauthentication_receipt=reauth_receipt,
            now_unix=now_unix,
            project_ancestry_evidence_raw=ancestry_raw,
            project_ancestry_collector_public_key=ancestry_key,
        )
    except (
        OSError,
        pre_foundation.OwnerGatePreFoundationError,
        owner_reauth.OwnerGateOwnerReauthError,
        release_trust.OwnerGateTrustError,
    ) as exc:
        raise OwnerGateFoundationError(
            "owner_gate_pre_foundation_authority_invalid"
        ) from None
    network_key = _load_ed25519_public_key(
        arguments.network_collector_public_key
    )
    evidence = ProductionNetworkEvidence.from_mapping(
        _load_json(arguments.network_evidence),
        public_key=network_key,
        expected_public_key_id=authority["network_collector_public_key_id"],
        now_unix=now_unix,
    )
    plan = build_plan(
        spec=pre_foundation.spec_from_authority(authority),
        network_evidence=evidence,
        network_collector_public_key=network_key,
        now_unix=now_unix,
    )
    pre_foundation.validate_pre_foundation_authority(
        authority,
        public_key=release_key,
        owner_reauthentication_receipt=reauth_receipt,
        now_unix=now_unix,
        expected_plan=plan,
        network_evidence=evidence,
        network_collector_public_key=network_key,
        project_ancestry_evidence_raw=ancestry_raw,
        project_ancestry_collector_public_key=ancestry_key,
    )
    report = {
        **plan.report(),
        "pre_foundation_authority_sha256": authority[
            "pre_foundation_authority_sha256"
        ],
        "inert_plan_sha256": pre_foundation.inert_plan_sha256(plan),
        "mutation_iam_binding_authorized": False,
        "package_deployment_authorized": False,
        "service_start_authorized": False,
    }
    print(canonical_json_bytes(report).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
