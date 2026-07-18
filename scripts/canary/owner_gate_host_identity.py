#!/usr/bin/env python3
"""Sealed first-contact identity collector for the private owner-gate VM.

The live entry point accepts only the validated, signed foundation-A chain.
It then performs two fixed Compute REST projections around two independent
provider-authenticated IAP/OpenSSH no-auth handshakes.  The handshakes can
learn a server key but cannot authenticate, run a remote command, change
metadata, or use OS Login.  One stable ssh-ed25519 key is owner-signed into a
canonical v2 receipt for a distinct source revision B to review and pin.

This module contains only mechanical identity validation and receipt
authoring.  It has no model, task, routing, or mutation decision logic.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import source_artifact_publication as source_publication


MAX_COMPUTE_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_OWNER_INPUT_BYTES = 16 * 1024 * 1024
MAX_OWNER_PUBLIC_KEY_BYTES = 16 * 1024
COMPUTE_API_HOST = "compute.googleapis.com"
OWNER_ACCOUNT = "lomliev@adventico.com"
REGION = foundation.REGION
VM_NAME = foundation.VM_NAME
SERVICE_ACCOUNT_EMAIL = (
    f"{foundation.SERVICE_ACCOUNT_NAME}@{foundation.PROJECT}.iam.gserviceaccount.com"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_PROVIDER_TAG = re.compile(r"^[A-Za-z0-9_+/=-]{1,4096}$")
_TOKEN = re.compile(r"^[\x21-\x7e]{1,16384}$")
_FORBIDDEN_NETWORK_ENVIRONMENT = frozenset({
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "no_proxy",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "SSLKEYLOGFILE",
    "OPENSSL_CONF",
    "OPENSSL_MODULES",
    "CURL_CA_BUNDLE",
    "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",
})
_BOOT_IMAGE = re.compile(
    r"^https://www\.googleapis\.com/compute/v1/projects/debian-cloud/"
    r"global/images/debian-12-bookworm-v[0-9]{8}$"
)


class OwnerGateHostIdentityError(RuntimeError):
    """Stable, secret-free first-contact failure."""


def _error(code: str, exc: BaseException | None = None) -> NoReturn:
    del exc
    raise OwnerGateHostIdentityError(code) from None


def _canonical(value: Any) -> bytes:
    try:
        return foundation.canonical_json_bytes(value)
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_host_identity_json_invalid", exc)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256(_canonical(value))


def _json_object_without_duplicates(
    pairs: Sequence[tuple[str, Any]],
) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _error("owner_gate_host_identity_compute_duplicate_key")
        result[key] = value
    return result


def _reject_nonfinite_compute_json(_value: str) -> None:
    _error("owner_gate_host_identity_compute_invalid")


def _provider_link(relative: str) -> str:
    return "https://www.googleapis.com/compute/v1/" + relative


EXPECTED_VM_SELF_LINK = _provider_link(
    f"projects/{foundation.PROJECT}/zones/{foundation.ZONE}/instances/{VM_NAME}"
)
EXPECTED_MACHINE_TYPE_SELF_LINK = _provider_link(
    f"projects/{foundation.PROJECT}/zones/{foundation.ZONE}/machineTypes/"
    f"{foundation.MACHINE_TYPE}"
)
EXPECTED_NETWORK_SELF_LINK = _provider_link(
    f"projects/{foundation.PROJECT}/global/networks/{foundation.NETWORK_NAME}"
)
EXPECTED_SUBNETWORK_SELF_LINK = _provider_link(
    f"projects/{foundation.PROJECT}/regions/{REGION}/subnetworks/"
    f"{foundation.OWNER_GATE_SUBNET_NAME}"
)
EXPECTED_BOOT_DISK_SELF_LINK = _provider_link(
    f"projects/{foundation.PROJECT}/zones/{foundation.ZONE}/disks/{VM_NAME}"
)
EXPECTED_BOOT_DISK_TYPE_SELF_LINK = _provider_link(
    f"projects/{foundation.PROJECT}/zones/{foundation.ZONE}/diskTypes/"
    f"{foundation.BOOT_DISK_TYPE}"
)
EXPECTED_ZONE_SELF_LINK = _provider_link(
    f"projects/{foundation.PROJECT}/zones/{foundation.ZONE}"
)
EXPECTED_REGION_SELF_LINK = _provider_link(
    f"projects/{foundation.PROJECT}/regions/{REGION}"
)
EXPECTED_OAUTH_SCOPES = tuple(sorted(foundation.OWNER_GATE_OAUTH_SCOPES))
EXPECTED_NETWORK_TAGS = tuple(sorted({foundation.IAP_NETWORK_TAG, foundation.OWNER_GATE_NETWORK_TAG}))
EXPECTED_METADATA = {
    "block-project-ssh-keys": "TRUE",
    "enable-oslogin": "TRUE",
    "serial-port-enable": "FALSE",
}
EXPECTED_SHIELDED_CONFIG = {
    "enableIntegrityMonitoring": True,
    "enableSecureBoot": True,
    "enableVtpm": True,
}
EXPECTED_DEBIAN_LICENSES = (
    "https://www.googleapis.com/compute/v1/projects/"
    "debian-cloud/global/licenses/debian-12-bookworm",
)
EXPECTED_SCHEDULING = {
    "automaticRestart": True,
    "instanceTerminationAction": "DELETE",
    "onHostMaintenance": "MIGRATE",
    "preemptible": False,
    "provisioningModel": "STANDARD",
}


@dataclass(frozen=True)
class _FoundationChainProjection:
    """Projection that can only be derived from the validated signed A chain."""

    foundation_source_revision: str
    foundation_source_tree_oid: str
    owner_reauthentication_receipt_sha256: str
    owner_reauthentication_expires_at_unix: int
    pre_foundation_authority_sha256: str
    foundation_apply_receipt_sha256: str
    direct_iam_authority_sha256: str
    ancestry_evidence_sha256: str
    ancestry_chain_sha256: str
    signed_network_evidence_sha256: str
    network_evidence_sha256: str
    project_number: str
    vm_numeric_id: str
    vm_self_link: str
    vm_creation_timestamp: str
    service_account_unique_id: str
    network_numeric_id: str
    subnetwork_numeric_id: str
    subnetwork_self_link: str
    network_self_link: str
    boot_disk_numeric_id: str
    boot_disk_architecture: str
    boot_disk_physical_block_size_bytes: int
    boot_disk_licenses: tuple[str, ...]
    boot_image_numeric_id: str
    boot_image_self_link: str

    def validate(self) -> None:
        if (
            _REVISION.fullmatch(self.foundation_source_revision or "") is None
            or _REVISION.fullmatch(self.foundation_source_tree_oid or "") is None
            or any(
                _SHA256.fullmatch(value or "") is None
                for value in (
                    self.owner_reauthentication_receipt_sha256,
                    self.pre_foundation_authority_sha256,
                    self.foundation_apply_receipt_sha256,
                    self.direct_iam_authority_sha256,
                    self.ancestry_evidence_sha256,
                    self.ancestry_chain_sha256,
                    self.signed_network_evidence_sha256,
                    self.network_evidence_sha256,
                )
            )
            or type(self.owner_reauthentication_expires_at_unix) is not int
            or self.owner_reauthentication_expires_at_unix <= 0
            or self.project_number != launcher.OWNER_GATE_PROJECT_NUMBER
            or any(
                _NUMERIC_ID.fullmatch(value or "") is None
                for value in (
                    self.vm_numeric_id,
                    self.service_account_unique_id,
                    self.network_numeric_id,
                    self.subnetwork_numeric_id,
                    self.boot_disk_numeric_id,
                    self.boot_image_numeric_id,
                )
            )
            or self.vm_numeric_id == launcher.VM_INSTANCE_ID
            or self.vm_self_link != EXPECTED_VM_SELF_LINK
            or not isinstance(self.vm_creation_timestamp, str)
            or not self.vm_creation_timestamp
            or self.subnetwork_self_link != EXPECTED_SUBNETWORK_SELF_LINK
            or self.network_self_link != EXPECTED_NETWORK_SELF_LINK
            or self.boot_disk_architecture != "X86_64"
            or self.boot_disk_physical_block_size_bytes != 4096
            or self.boot_disk_licenses != EXPECTED_DEBIAN_LICENSES
            or _BOOT_IMAGE.fullmatch(self.boot_image_self_link or "") is None
        ):
            _error("owner_gate_host_identity_foundation_chain_invalid")


def _projection_from_validated_chains(
    *,
    foundation_chain: Any,
    direct_iam_authority_raw: bytes,
    now_unix: int,
    recovery_only: bool = False,
) -> _FoundationChainProjection:
    """Derive every host input from canonical signed/validated A artifacts."""

    from scripts.canary import direct_iam_identity_author as direct_iam
    from scripts.canary import owner_gate_foundation_apply as foundation_apply

    if (
        type(foundation_chain)
        is not foundation_apply.ValidatedFoundationApplyChain
        or getattr(foundation_chain, "_marker", None)
        is not foundation_apply._CHAIN_MARKER
        or type(direct_iam_authority_raw) is not bytes
        or not direct_iam_authority_raw
        or type(now_unix) is not int
        or now_unix <= 0
        or type(recovery_only) is not bool
    ):
        _error("owner_gate_host_identity_foundation_chain_invalid")
    foundation_a = foundation_chain.foundation_a
    if (
        type(foundation_a) is not foundation_apply.ValidatedFoundationAChain
        or getattr(foundation_a, "_marker", None)
        is not foundation_apply._CHAIN_MARKER
    ):
        _error("owner_gate_host_identity_foundation_chain_invalid")
    try:
        canonical_direct = (
            direct_iam._decode_canonical_authority_for_recovery_chain(
                direct_iam_authority_raw,
                foundation_chain=foundation_chain,
            )
            if recovery_only
            else direct_iam.decode_canonical_authority_for_validated_chain(
                direct_iam_authority_raw,
                foundation_chain=foundation_chain,
                now_unix=now_unix,
            )
        )
        if (
            type(canonical_direct)
            is not direct_iam.CanonicalDirectIamAuthority
            or canonical_direct.raw != direct_iam_authority_raw
        ):
            _error("owner_gate_host_identity_foundation_chain_invalid")
        direct = canonical_direct.value
        vm = foundation_chain.owner_gate_vm_identity
        service_account = foundation_chain.service_account_identity
        subnet = foundation_chain.subnet_identity
        ancestry = foundation_a.ancestry_evidence
        expected_ancestor_chain = [
            str(item["resource_name"])
            for item in ancestry.ordered_chain[1:]
        ]
        owner_receipt = foundation_a.owner_reauthentication_receipt
        apply = foundation_chain.apply_receipt
    except direct_iam.DirectIamIdentityAuthorError as exc:
        _error("owner_gate_host_identity_foundation_chain_mismatch", exc)
    except (KeyError, TypeError) as exc:
        _error("owner_gate_host_identity_foundation_chain_invalid", exc)
    if (
        direct["pre_foundation_authority_sha256"]
        != foundation_chain.pre_foundation_authority_sha256
        or direct["foundation_apply_receipt_sha256"]
        != foundation_chain.foundation_apply_receipt_sha256
        or direct["owner_reauthentication_receipt_sha256"]
        != foundation_chain.owner_reauthentication_receipt_sha256
        or direct["owner_gate_vm_numeric_id"] != vm.get("numeric_id")
        or direct["owner_gate_service_account_unique_id"]
        != service_account.get("unique_id")
        or direct["resource_ancestor_chain"] != expected_ancestor_chain
        or direct["project_number"] != ancestry.project_number
        or vm.get("subnetwork_numeric_id") != subnet.get("numeric_id")
        or vm.get("subnetwork_self_link") != subnet.get("self_link")
        or direct["collected_at_unix"] < apply["completed_at_unix"]
        or direct["collected_at_unix"]
        < owner_receipt["issued_at_unix"]
        or direct["collected_at_unix"]
        > owner_receipt["expires_at_unix"]
        or (
            not recovery_only
            and now_unix > owner_receipt["expires_at_unix"]
        )
    ):
        _error("owner_gate_host_identity_foundation_chain_mismatch")
    value = _FoundationChainProjection(
        foundation_source_revision=foundation_chain.foundation_source_revision,
        foundation_source_tree_oid=foundation_chain.foundation_source_tree_oid,
        owner_reauthentication_receipt_sha256=(
            foundation_chain.owner_reauthentication_receipt_sha256
        ),
        owner_reauthentication_expires_at_unix=int(
            owner_receipt["expires_at_unix"]
        ),
        pre_foundation_authority_sha256=(
            foundation_chain.pre_foundation_authority_sha256
        ),
        foundation_apply_receipt_sha256=(
            foundation_chain.foundation_apply_receipt_sha256
        ),
        direct_iam_authority_sha256=canonical_direct.raw_sha256,
        ancestry_evidence_sha256=foundation_a.ancestry_evidence_sha256,
        ancestry_chain_sha256=str(
            foundation_a.authority["ancestry_chain_sha256"]
        ),
        signed_network_evidence_sha256=(
            foundation_a.signed_network_evidence_sha256
        ),
        network_evidence_sha256=foundation_a.network_evidence_sha256,
        project_number=str(ancestry.project_number),
        vm_numeric_id=str(vm["numeric_id"]),
        vm_self_link=str(vm["self_link"]),
        vm_creation_timestamp=str(vm["creation_timestamp"]),
        service_account_unique_id=str(service_account["unique_id"]),
        network_numeric_id=str(vm["network_numeric_id"]),
        subnetwork_numeric_id=str(subnet["numeric_id"]),
        subnetwork_self_link=str(subnet["self_link"]),
        network_self_link=str(vm["network_self_link"]),
        boot_disk_numeric_id=str(vm["boot_disk_numeric_id"]),
        boot_disk_architecture=str(vm["boot_image_architecture"]),
        boot_disk_physical_block_size_bytes=4096,
        boot_disk_licenses=tuple(vm["boot_image_license_self_links"]),
        boot_image_numeric_id=str(vm["boot_image_numeric_id"]),
        boot_image_self_link=str(vm["boot_image_self_link"]),
    )
    value.validate()
    return value


@dataclass(frozen=True)
class _DirectComputeIdentity:
    value: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return _sha256_json(self.value)


def _mapping(value: Any, *, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error(code)
    return value


def _one_mapping(value: Any, *, code: str) -> Mapping[str, Any]:
    if not isinstance(value, list) or len(value) != 1:
        _error(code)
    return _mapping(value[0], code=code)


def _numeric(value: Any, *, code: str) -> str:
    text = str(value)
    if _NUMERIC_ID.fullmatch(text) is None:
        _error(code)
    return text


def _tag(value: Any, *, code: str) -> str:
    if not isinstance(value, str) or _PROVIDER_TAG.fullmatch(value) is None:
        _error(code)
    return value


def _positive_int(value: Any, *, code: str) -> int:
    if isinstance(value, bool):
        _error(code)
    try:
        number = int(str(value))
    except (TypeError, ValueError) as exc:
        _error(code, exc)
    if number <= 0 or str(number) != str(value):
        _error(code)
    return number


def _metadata_projection(value: Any) -> Mapping[str, str]:
    metadata = _mapping(value, code="owner_gate_host_identity_instance_invalid")
    _tag(metadata.get("fingerprint"), code="owner_gate_host_identity_instance_invalid")
    items = metadata.get("items")
    if not isinstance(items, list) or len(items) != len(EXPECTED_METADATA):
        _error("owner_gate_host_identity_instance_invalid")
    result: dict[str, str] = {}
    for raw in items:
        item = _mapping(raw, code="owner_gate_host_identity_instance_invalid")
        if set(item) != {"key", "value"}:
            _error("owner_gate_host_identity_instance_invalid")
        key = item.get("key")
        item_value = item.get("value")
        if (
            not isinstance(key, str)
            or not isinstance(item_value, str)
            or key in result
        ):
            _error("owner_gate_host_identity_instance_invalid")
        result[key] = item_value
    if result != EXPECTED_METADATA:
        _error("owner_gate_host_identity_instance_invalid")
    return dict(sorted(result.items()))


def _direct_compute_identity(
    responses: Mapping[str, Any],
    *,
    chain: _FoundationChainProjection,
) -> _DirectComputeIdentity:
    """Validate one exact direct Compute snapshot and return its projection."""

    chain.validate()
    if set(responses) != {"instance", "disk", "image", "network", "subnetwork"}:
        _error("owner_gate_host_identity_compute_snapshot_invalid")
    instance = _mapping(
        responses["instance"], code="owner_gate_host_identity_instance_invalid"
    )
    disk = _mapping(responses["disk"], code="owner_gate_host_identity_disk_invalid")
    image = _mapping(responses["image"], code="owner_gate_host_identity_image_invalid")
    network = _mapping(
        responses["network"], code="owner_gate_host_identity_network_invalid"
    )
    subnet = _mapping(
        responses["subnetwork"], code="owner_gate_host_identity_subnetwork_invalid"
    )

    instance_id = _numeric(
        instance.get("id"), code="owner_gate_host_identity_instance_invalid"
    )
    if (
        instance.get("kind") != "compute#instance"
        or instance.get("name") != VM_NAME
        or instance.get("selfLink") != EXPECTED_VM_SELF_LINK
        or instance.get("zone") != EXPECTED_ZONE_SELF_LINK
        or instance.get("machineType") != EXPECTED_MACHINE_TYPE_SELF_LINK
        or instance.get("status") != "RUNNING"
        or instance.get("creationTimestamp") != chain.vm_creation_timestamp
        or instance.get("canIpForward") is not False
        or instance.get("deletionProtection") is not False
        or instance_id != chain.vm_numeric_id
        or instance.get("selfLink") != chain.vm_self_link
    ):
        _error("owner_gate_host_identity_instance_invalid")

    scheduling_raw = _mapping(
        instance.get("scheduling"),
        code="owner_gate_host_identity_instance_invalid",
    )
    scheduling = {
        key: scheduling_raw.get(key)
        for key in EXPECTED_SCHEDULING
    }
    confidential = _mapping(
        instance.get("confidentialInstanceConfig", {
            "enableConfidentialCompute": False,
        }),
        code="owner_gate_host_identity_instance_invalid",
    )
    _tag(
        instance.get("labelFingerprint"),
        code="owner_gate_host_identity_instance_invalid",
    )
    if (
        scheduling != EXPECTED_SCHEDULING
        or instance.get("labels", {}) != {}
        or instance.get("resourcePolicies", []) != []
        or instance.get("minCpuPlatform", "Automatic") != "Automatic"
        or confidential != {"enableConfidentialCompute": False}
    ):
        _error("owner_gate_host_identity_instance_invalid")

    tags = _mapping(
        instance.get("tags"), code="owner_gate_host_identity_instance_invalid"
    )
    _tag(tags.get("fingerprint"), code="owner_gate_host_identity_instance_invalid")
    tag_items = tags.get("items")
    if (
        not isinstance(tag_items, list)
        or any(not isinstance(item, str) for item in tag_items)
        or tuple(sorted(tag_items)) != EXPECTED_NETWORK_TAGS
        or len(tag_items) != len(set(tag_items))
    ):
        _error("owner_gate_host_identity_instance_invalid")
    metadata = _metadata_projection(instance.get("metadata"))
    shielded = _mapping(
        instance.get("shieldedInstanceConfig"),
        code="owner_gate_host_identity_instance_invalid",
    )
    if dict(shielded) != EXPECTED_SHIELDED_CONFIG:
        _error("owner_gate_host_identity_instance_invalid")

    service_account = _one_mapping(
        instance.get("serviceAccounts"),
        code="owner_gate_host_identity_instance_invalid",
    )
    scopes = service_account.get("scopes")
    if (
        service_account.get("email") != SERVICE_ACCOUNT_EMAIL
        or not isinstance(scopes, list)
        or any(not isinstance(item, str) for item in scopes)
        or tuple(sorted(scopes)) != EXPECTED_OAUTH_SCOPES
        or len(scopes) != len(set(scopes))
    ):
        _error("owner_gate_host_identity_instance_invalid")

    interface = _one_mapping(
        instance.get("networkInterfaces"),
        code="owner_gate_host_identity_instance_invalid",
    )
    access_configs = interface.get("accessConfigs", [])
    ipv6_access_configs = interface.get("ipv6AccessConfigs", [])
    if (
        interface.get("network") != EXPECTED_NETWORK_SELF_LINK
        or interface.get("network") != chain.network_self_link
        or interface.get("subnetwork") != EXPECTED_SUBNETWORK_SELF_LINK
        or interface.get("subnetwork") != chain.subnetwork_self_link
        or interface.get("networkIP") != foundation.OWNER_GATE_PRIVATE_IP
        or interface.get("aliasIpRanges", []) != []
        or access_configs != []
        or ipv6_access_configs != []
        or interface.get("stackType", "IPV4_ONLY") != "IPV4_ONLY"
    ):
        _error("owner_gate_host_identity_instance_invalid")

    attached_disk = _one_mapping(
        instance.get("disks"), code="owner_gate_host_identity_instance_invalid"
    )
    if (
        attached_disk.get("boot") is not True
        or attached_disk.get("autoDelete") is not True
        or attached_disk.get("mode") != "READ_WRITE"
        or attached_disk.get("type") != "PERSISTENT"
        or attached_disk.get("interface") != "SCSI"
        or attached_disk.get("index") != 0
        or attached_disk.get("source") != EXPECTED_BOOT_DISK_SELF_LINK
        or attached_disk.get("deviceName") != VM_NAME
        or _positive_int(
            attached_disk.get("diskSizeGb"),
            code="owner_gate_host_identity_instance_invalid",
        ) != foundation.BOOT_DISK_SIZE_GB
    ):
        _error("owner_gate_host_identity_instance_invalid")

    disk_id = _numeric(disk.get("id"), code="owner_gate_host_identity_disk_invalid")
    if (
        disk.get("kind") != "compute#disk"
        or disk.get("name") != VM_NAME
        or disk.get("selfLink") != EXPECTED_BOOT_DISK_SELF_LINK
        or disk.get("zone") != EXPECTED_ZONE_SELF_LINK
        or disk.get("status") != "READY"
        or disk.get("type") != EXPECTED_BOOT_DISK_TYPE_SELF_LINK
        or _positive_int(
            disk.get("sizeGb"), code="owner_gate_host_identity_disk_invalid"
        ) != foundation.BOOT_DISK_SIZE_GB
        or disk.get("sourceImage") != chain.boot_image_self_link
        or str(disk.get("sourceImageId")) != chain.boot_image_numeric_id
        or disk.get("users") != [EXPECTED_VM_SELF_LINK]
        or disk_id != chain.boot_disk_numeric_id
        or disk.get("architecture") != chain.boot_disk_architecture
        or _positive_int(
            disk.get("physicalBlockSizeBytes"),
            code="owner_gate_host_identity_disk_invalid",
        ) != chain.boot_disk_physical_block_size_bytes
        or tuple(disk.get("licenses", [])) != chain.boot_disk_licenses
    ):
        _error("owner_gate_host_identity_disk_invalid")

    image_id = _numeric(
        image.get("id"), code="owner_gate_host_identity_image_invalid"
    )
    if (
        image.get("kind") != "compute#image"
        or image.get("selfLink") != chain.boot_image_self_link
        or image_id != chain.boot_image_numeric_id
        or image.get("status") != "READY"
        or image.get("family") != "debian-12"
        or image.get("architecture") != "X86_64"
        or tuple(image.get("licenses", [])) != EXPECTED_DEBIAN_LICENSES
        or image.get("deprecated") is not None
    ):
        _error("owner_gate_host_identity_image_invalid")

    network_id = _numeric(
        network.get("id"), code="owner_gate_host_identity_network_invalid"
    )
    if (
        network.get("kind") != "compute#network"
        or network.get("name") != foundation.NETWORK_NAME
        or network.get("selfLink") != EXPECTED_NETWORK_SELF_LINK
        or network.get("selfLink") != chain.network_self_link
        or network.get("autoCreateSubnetworks") is not False
        or network_id != chain.network_numeric_id
    ):
        _error("owner_gate_host_identity_network_invalid")

    subnetwork_id = _numeric(
        subnet.get("id"), code="owner_gate_host_identity_subnetwork_invalid"
    )
    if (
        subnet.get("kind") != "compute#subnetwork"
        or subnet.get("name") != foundation.OWNER_GATE_SUBNET_NAME
        or subnet.get("selfLink") != EXPECTED_SUBNETWORK_SELF_LINK
        or subnet.get("selfLink") != chain.subnetwork_self_link
        or subnetwork_id != chain.subnetwork_numeric_id
        or subnet.get("network") != EXPECTED_NETWORK_SELF_LINK
        or subnet.get("region") != EXPECTED_REGION_SELF_LINK
        or subnet.get("ipCidrRange") != foundation.OWNER_GATE_SUBNET_CIDR
        or subnet.get("privateIpGoogleAccess") is not True
        or subnet.get("stackType", "IPV4_ONLY") != "IPV4_ONLY"
        or subnet.get("purpose", "PRIVATE") != "PRIVATE"
        or subnet.get("secondaryIpRanges", []) != []
    ):
        _error("owner_gate_host_identity_subnetwork_invalid")

    value = {
        "project": foundation.PROJECT,
        "project_number": chain.project_number,
        "zone": foundation.ZONE,
        "vm_name": VM_NAME,
        "vm_self_link": EXPECTED_VM_SELF_LINK,
        "vm_numeric_id": instance_id,
        "vm_creation_timestamp": chain.vm_creation_timestamp,
        "machine_type_self_link": EXPECTED_MACHINE_TYPE_SELF_LINK,
        "scheduling": dict(sorted(scheduling.items())),
        "labels": {},
        "resource_policies": [],
        "minimum_cpu_platform": "Automatic",
        "confidential_compute": False,
        "owner_gate_service_account_email": SERVICE_ACCOUNT_EMAIL,
        "owner_gate_service_account_unique_id": chain.service_account_unique_id,
        "oauth_scopes": list(EXPECTED_OAUTH_SCOPES),
        "network_tags": list(EXPECTED_NETWORK_TAGS),
        "instance_metadata": metadata,
        "shielded_instance_config": dict(sorted(shielded.items())),
        "can_ip_forward": False,
        "external_ip_present": False,
        "internal_ip": foundation.OWNER_GATE_PRIVATE_IP,
        "network_self_link": EXPECTED_NETWORK_SELF_LINK,
        "network_numeric_id": network_id,
        "subnetwork_self_link": EXPECTED_SUBNETWORK_SELF_LINK,
        "subnetwork_numeric_id": subnetwork_id,
        "boot_disk_self_link": EXPECTED_BOOT_DISK_SELF_LINK,
        "boot_disk_numeric_id": disk_id,
        "boot_disk_type_self_link": EXPECTED_BOOT_DISK_TYPE_SELF_LINK,
        "boot_disk_size_gb": foundation.BOOT_DISK_SIZE_GB,
        "boot_disk_auto_delete": True,
        "boot_disk_architecture": chain.boot_disk_architecture,
        "boot_disk_physical_block_size_bytes": (
            chain.boot_disk_physical_block_size_bytes
        ),
        "boot_disk_licenses": list(chain.boot_disk_licenses),
        "boot_image_self_link": chain.boot_image_self_link,
        "boot_image_numeric_id": image_id,
        "boot_image_architecture": "X86_64",
        "boot_image_licenses": list(EXPECTED_DEBIAN_LICENSES),
    }
    return _DirectComputeIdentity(value)


ComputeRequester = Callable[[str, Mapping[str, str], float], tuple[int, bytes]]


class _WipeableAccessToken:
    """One in-memory token buffer, invalidated on every terminal path."""

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
            _error("owner_gate_host_identity_access_token_invalid")
        self._buffer = bytearray(value.encode("ascii", errors="strict"))
        self._active = True

    def authorization_header(self) -> str:
        if not self._active or not self._buffer:
            _error("owner_gate_host_identity_access_token_retired")
        try:
            return "Bearer " + bytes(self._buffer).decode("ascii", errors="strict")
        except UnicodeError as exc:
            _error("owner_gate_host_identity_access_token_invalid", exc)

    def wipe(self) -> None:
        for index in range(len(self._buffer)):
            self._buffer[index] = 0
        self._buffer.clear()
        self._active = False

    @property
    def retired(self) -> bool:
        return not self._active and not self._buffer


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _json_content_type(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parts = tuple(item.strip() for item in value.split(";"))
    if not parts or parts[0].casefold() != "application/json":
        return False
    return len(parts) == 1 or (
        len(parts) == 2
        and parts[1].casefold().replace(" ", "") == "charset=utf-8"
    )


def _default_compute_request(
    url: str,
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[int, bytes]:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != COMPUTE_API_HOST
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or set(headers) != {"Accept", "Authorization"}
        or headers.get("Accept") != "application/json"
        or not isinstance(headers.get("Authorization"), str)
        or not str(headers["Authorization"]).startswith("Bearer ")
        or _TOKEN.fullmatch(str(headers["Authorization"])[7:]) is None
        or any(
            ord(character) < 0x20 or ord(character) > 0x7E
            for character in str(headers["Authorization"])
        )
        or isinstance(timeout_seconds, bool)
        or not 0 < timeout_seconds <= 60
    ):
        _error("owner_gate_host_identity_compute_config_invalid")
    if any(os.environ.get(name) for name in _FORBIDDEN_NETWORK_ENVIRONMENT):
        _error("owner_gate_host_identity_compute_tls_invalid")
    try:
        context = launcher._pinned_system_tls_context()
    except launcher.OwnerLauncherError as exc:
        _error("owner_gate_host_identity_compute_tls_invalid", exc)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context),
        _NoRedirectHandler(),
    )
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            content_length = response.headers.get("Content-Length")
            declared_length: int | None = None
            if content_length is not None:
                if not content_length.isdecimal():
                    _error("owner_gate_host_identity_compute_invalid")
                declared_length = int(content_length)
            if (
                type(response.status) is not int
                or response.status != 200
                or response.geturl() != url
                or response.headers.get("Location") is not None
                or not _json_content_type(response.headers.get("Content-Type"))
                or (
                    declared_length is not None
                    and not 0 < declared_length <= MAX_COMPUTE_RESPONSE_BYTES
                )
            ):
                _error("owner_gate_host_identity_compute_invalid")
            payload = response.read(MAX_COMPUTE_RESPONSE_BYTES + 1)
            if (
                not payload
                or len(payload) > MAX_COMPUTE_RESPONSE_BYTES
                or (
                    declared_length is not None
                    and len(payload) != declared_length
                )
            ):
                _error("owner_gate_host_identity_compute_invalid")
            return response.status, payload
    except OwnerGateHostIdentityError:
        raise
    except urllib.error.HTTPError as exc:
        _error("owner_gate_host_identity_compute_unavailable", exc)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        _error("owner_gate_host_identity_compute_unavailable", exc)


def _compute_urls(chain: _FoundationChainProjection) -> Mapping[str, str]:
    image_relative = chain.boot_image_self_link.removeprefix(
        "https://www.googleapis.com/compute/v1/"
    )
    return {
        "instance": (
            f"https://{COMPUTE_API_HOST}/compute/v1/projects/{foundation.PROJECT}/"
            f"zones/{foundation.ZONE}/instances/{VM_NAME}"
        ),
        "disk": (
            f"https://{COMPUTE_API_HOST}/compute/v1/projects/{foundation.PROJECT}/"
            f"zones/{foundation.ZONE}/disks/{VM_NAME}"
        ),
        "image": f"https://{COMPUTE_API_HOST}/compute/v1/{image_relative}",
        "network": (
            f"https://{COMPUTE_API_HOST}/compute/v1/projects/{foundation.PROJECT}/"
            f"global/networks/{foundation.NETWORK_NAME}"
        ),
        "subnetwork": (
            f"https://{COMPUTE_API_HOST}/compute/v1/projects/{foundation.PROJECT}/"
            f"regions/{REGION}/subnetworks/{foundation.OWNER_GATE_SUBNET_NAME}"
        ),
    }


def _read_compute_snapshot(
    *,
    chain: _FoundationChainProjection,
    access_token: _WipeableAccessToken,
    requester: ComputeRequester,
    timeout_seconds: float = 30.0,
) -> _DirectComputeIdentity:
    if (
        type(access_token) is not _WipeableAccessToken
        or not callable(requester)
        or isinstance(timeout_seconds, bool)
        or not 0 < timeout_seconds <= 60
    ):
        _error("owner_gate_host_identity_compute_config_invalid")
    responses: dict[str, Any] = {}
    for name, url in _compute_urls(chain).items():
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != COMPUTE_API_HOST
            or parsed.port is not None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed.query
        ):
            _error("owner_gate_host_identity_compute_url_invalid")
        headers = {
            "Accept": "application/json",
            "Authorization": access_token.authorization_header(),
        }
        try:
            status, raw = requester(url, headers, float(timeout_seconds))
        finally:
            headers.clear()
        if (
            type(status) is not int
            or status != 200
            or type(raw) is not bytes
            or not raw
            or len(raw) > MAX_COMPUTE_RESPONSE_BYTES
        ):
            _error("owner_gate_host_identity_compute_unavailable")
        try:
            value = json.loads(
                raw.decode("utf-8", errors="strict"),
                object_pairs_hook=_json_object_without_duplicates,
                parse_constant=_reject_nonfinite_compute_json,
            )
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            _error("owner_gate_host_identity_compute_invalid", exc)
        if not isinstance(value, Mapping):
            _error("owner_gate_host_identity_compute_invalid")
        responses[name] = value
    return _direct_compute_identity(responses, chain=chain)


class TrustedOwnerGateSshExecutable:
    """Pin the system OpenSSH client and the shell used by ProxyCommand."""

    def __init__(self) -> None:
        self._ssh = launcher._PinnedExecutablePath(
            "/usr/bin/ssh",
            invalid_code="owner_gate_host_identity_ssh_invalid",
            changed_code="owner_gate_host_identity_ssh_changed",
        )
        self._shell = launcher._PinnedExecutablePath(
            "/bin/sh",
            invalid_code="owner_gate_host_identity_shell_invalid",
            changed_code="owner_gate_host_identity_shell_changed",
        )
        self._paths_identity = self._capture_paths()
        self._toolchain_identity = self._capture_toolchain()

    def _capture_paths(self) -> tuple[str, str]:
        return self._ssh.absolute_path(), self._shell.absolute_path()

    def paths(self) -> tuple[str, str]:
        current = self._capture_paths()
        if current != self._paths_identity:
            _error("owner_gate_host_identity_ssh_changed")
        return current

    @staticmethod
    def _content_sha256(path: str) -> str:
        descriptor: int | None = None
        try:
            before = os.lstat(path)
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or (before.st_dev, before.st_ino)
                != (opened.st_dev, opened.st_ino)
            or opened.st_uid not in {0, os.getuid()}  # windows-footgun: ok — POSIX owner boundary
                or opened.st_mode & 0o022
                or not opened.st_mode & 0o100
                or opened.st_nlink < 1
                or opened.st_size <= 0
                or opened.st_size > 64 * 1024 * 1024
            ):
                _error("owner_gate_host_identity_toolchain_invalid")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_size,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_size,
            ):
                _error("owner_gate_host_identity_toolchain_changed")
            return digest.hexdigest()
        except OwnerGateHostIdentityError:
            raise
        except OSError as exc:
            _error("owner_gate_host_identity_toolchain_invalid", exc)
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @staticmethod
    def _version(argv: Sequence[str], *, stderr: bool) -> str:
        try:
            completed = subprocess.run(
                tuple(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    "PATH": "/usr/bin:/bin",
                    "LANG": "C",
                    "LC_ALL": "C",
                },
                shell=False,
                close_fds=True,
                timeout=10.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _error("owner_gate_host_identity_toolchain_version_invalid", exc)
        raw = completed.stderr if stderr else completed.stdout
        if (
            completed.returncode != 0
            or not isinstance(raw, bytes)
            or not raw.endswith(b"\n")
            or raw.count(b"\n") != 1
            or len(raw) > 512
        ):
            _error("owner_gate_host_identity_toolchain_version_invalid")
        try:
            value = raw[:-1].decode("ascii", errors="strict")
        except UnicodeError as exc:
            _error("owner_gate_host_identity_toolchain_version_invalid", exc)
        if not value or any(ord(item) < 0x20 or ord(item) > 0x7E for item in value):
            _error("owner_gate_host_identity_toolchain_version_invalid")
        return value

    def _capture_toolchain(self) -> Mapping[str, str]:
        ssh_path, shell_path = self._capture_paths()
        value = {
            "ssh_executable_sha256": self._content_sha256(ssh_path),
            "ssh_version": self._version((ssh_path, "-V"), stderr=True),
            "shell_executable_sha256": self._content_sha256(shell_path),
            "shell_version": self._version(
                (
                    shell_path,
                    "-c",
                    'printf "%s\\n" "${BASH_VERSION-${ZSH_VERSION-${KSH_VERSION-unknown}}}"',
                ),
                stderr=False,
            ),
        }
        if value["shell_version"] == "unknown":
            _error("owner_gate_host_identity_toolchain_version_invalid")
        return value

    def identity(self) -> Mapping[str, str]:
        self.paths()
        current = self._capture_toolchain()
        if current != self._toolchain_identity:
            _error("owner_gate_host_identity_toolchain_changed")
        return dict(current)


SshRunner = Callable[..., subprocess.CompletedProcess[bytes]]


def _temporary_known_hosts() -> tuple[str, str, tuple[int, ...]]:
    base = "/private/tmp" if sys_platform_is_darwin() else "/tmp"
    directory = tempfile.mkdtemp(prefix="muncho-owner-gate-host-key-", dir=base)
    try:
        os.chmod(directory, 0o700)
        metadata = os.lstat(directory)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid()  # windows-footgun: ok — POSIX owner boundary
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            _error("owner_gate_host_identity_known_hosts_invalid")
        path = os.path.join(directory, "known_hosts")
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.fsync(descriptor)
            opened = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_uid,
            opened.st_gid,
            opened.st_nlink,
            stat.S_IMODE(opened.st_mode),
        )
        if opened.st_uid != os.getuid() or opened.st_nlink != 1 or identity[-1] != 0o600:  # windows-footgun: ok — POSIX owner boundary
            _error("owner_gate_host_identity_known_hosts_invalid")
        return directory, path, identity
    except BaseException:
        try:
            os.rmdir(directory)
        except OSError:
            pass
        raise


def sys_platform_is_darwin() -> bool:
    return os.uname().sysname == "Darwin"  # windows-footgun: owner path is POSIX-only


def _known_hosts_key(path: str, identity: tuple[int, ...], instance_id: str) -> str:
    try:
        metadata = os.lstat(path)
        if (
            (metadata.st_dev, metadata.st_ino) != identity[:2]
            or metadata.st_uid != identity[2]
            or metadata.st_gid != identity[3]
            or metadata.st_nlink != identity[4]
            or stat.S_IMODE(metadata.st_mode) != identity[5]
            or metadata.st_size <= 0
            or metadata.st_size > 16 * 1024
        ):
            _error("owner_gate_host_identity_known_hosts_changed")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != identity[:2]:
                _error("owner_gate_host_identity_known_hosts_changed")
            raw = os.read(descriptor, 16 * 1024 + 1)
            if os.read(descriptor, 1):
                _error("owner_gate_host_identity_known_hosts_invalid")
        finally:
            os.close(descriptor)
    except OwnerGateHostIdentityError:
        raise
    except OSError as exc:
        _error("owner_gate_host_identity_known_hosts_invalid", exc)
    if (
        not raw.endswith(b"\n")
        or raw.count(b"\n") != 1
        or b"\r" in raw
        or b"\x00" in raw
    ):
        _error("owner_gate_host_identity_known_hosts_invalid")
    try:
        line = raw[:-1].decode("ascii", errors="strict")
    except UnicodeError as exc:
        _error("owner_gate_host_identity_known_hosts_invalid", exc)
    pieces = line.split(" ")
    if (
        len(pieces) != 3
        or pieces[0] != f"compute.{instance_id}"
        or pieces[1] != "ssh-ed25519"
        or not pieces[2]
    ):
        _error("owner_gate_host_identity_known_hosts_invalid")
    try:
        launcher.PinnedOwnerGateHostIdentityReceipt._validate_host_key(pieces[2])
    except launcher.OwnerLauncherError as exc:
        _error("owner_gate_host_identity_known_hosts_invalid", exc)
    return pieces[2]


def _iap_ssh_argv(
    *,
    runtime: launcher.TrustedGcloudExecutable,
    configuration: launcher.PinnedGcloudConfiguration,
    ssh: TrustedOwnerGateSshExecutable,
    known_hosts_path: str,
    instance_id: str,
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    if (
        type(runtime) is not launcher.TrustedGcloudExecutable
        or type(configuration) is not launcher.PinnedGcloudConfiguration
        or type(ssh) is not TrustedOwnerGateSshExecutable
        or _NUMERIC_ID.fullmatch(instance_id or "") is None
        or not os.path.isabs(known_hosts_path)
    ):
        _error("owner_gate_host_identity_capability_invalid")
    prefix = runtime.trusted_command_prefix()
    ssh_path, _shell_path = ssh.paths()
    account = configuration.account
    if account != OWNER_ACCOUNT:
        _error("owner_gate_host_identity_owner_invalid")
    proxy_command = shlex.join((
        *prefix,
        "compute",
        "start-iap-tunnel",
        VM_NAME,
        "22",
        "--listen-on-stdin",
        f"--project={foundation.PROJECT}",
        f"--zone={foundation.ZONE}",
        f"--account={OWNER_ACCOUNT}",
        "--configuration=adventico-ai-platform-admin",
        "--quiet",
    ))
    argv = (
        ssh_path,
        "-F",
        "/dev/null",
        "-N",
        "-T",
        "-oBatchMode=yes",
        "-oIdentitiesOnly=yes",
        "-oIdentityAgent=none",
        "-oCertificateFile=none",
        "-oPreferredAuthentications=none",
        "-oPubkeyAuthentication=no",
        "-oPasswordAuthentication=no",
        "-oKbdInteractiveAuthentication=no",
        "-oGSSAPIAuthentication=no",
        "-oHostbasedAuthentication=no",
        "-oNumberOfPasswordPrompts=0",
        "-oPermitLocalCommand=no",
        "-oClearAllForwardings=yes",
        "-oControlMaster=no",
        "-oControlPath=none",
        "-oKnownHostsCommand=none",
        "-oCanonicalizeHostname=no",
        "-oForwardAgent=no",
        "-oEscapeChar=none",
        "-oRequestTTY=no",
        "-oStrictHostKeyChecking=accept-new",
        "-oHashKnownHosts=no",
        f"-oHostKeyAlias=compute.{instance_id}",
        f"-oUserKnownHostsFile={known_hosts_path}",
        "-oGlobalKnownHostsFile=/dev/null",
        "-oUpdateHostKeys=no",
        "-oVerifyHostKeyDNS=no",
        "-oHostKeyAlgorithms=ssh-ed25519",
        "-oConnectionAttempts=1",
        "-oConnectTimeout=20",
        "-oLogLevel=ERROR",
        f"-oProxyCommand={proxy_command}",
        f"nobody@compute.{instance_id}",
    )
    environment = dict(launcher._owner_gcloud_environment(configuration, prefix[0]))
    if (
        set(environment) != launcher.OwnerGateIapTransport._ENVIRONMENT_KEYS
        or any("proxy" in name.casefold() for name in environment)
        or environment.get("CLOUDSDK_CORE_PROJECT") != foundation.PROJECT
        or environment.get("CLOUDSDK_COMPUTE_ZONE") != foundation.ZONE
    ):
        _error("owner_gate_host_identity_environment_invalid")
    return argv, environment


def _capture_host_key_once(
    *,
    runtime: launcher.TrustedGcloudExecutable,
    configuration: launcher.PinnedGcloudConfiguration,
    ssh: TrustedOwnerGateSshExecutable,
    instance_id: str,
    runner: SshRunner,
) -> str:
    directory, known_hosts, identity = _temporary_known_hosts()
    try:
        runtime_before = runtime.trusted_command_prefix()
        ssh_before = ssh.paths()
        argv, environment = _iap_ssh_argv(
            runtime=runtime,
            configuration=configuration,
            ssh=ssh,
            known_hosts_path=known_hosts,
            instance_id=instance_id,
        )
        try:
            completed = runner(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=dict(environment),
                shell=False,
                close_fds=True,
                timeout=45.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _error("owner_gate_host_identity_iap_handshake_failed", exc)
        finally:
            stability_errors: list[BaseException] = []
            runtime_after: tuple[str, ...] | None = None
            ssh_after: tuple[str, str] | None = None
            try:
                runtime_after = runtime.trusted_command_prefix()
            except BaseException as exc:
                stability_errors.append(exc)
            try:
                configuration.assert_stable()
            except BaseException as exc:
                stability_errors.append(exc)
            try:
                ssh_after = ssh.paths()
            except BaseException as exc:
                stability_errors.append(exc)
            if (
                stability_errors
                or runtime_after != runtime_before
                or ssh_after != ssh_before
            ):
                _error("owner_gate_host_identity_runtime_changed")
        # Authentication is deliberately impossible.  OpenSSH records the
        # host key first, then exits 255 because every client auth method is
        # disabled.  Any zero exit would mean an unexpected remote session.
        if completed.returncode != 255:
            _error("owner_gate_host_identity_iap_handshake_invalid")
        return _known_hosts_key(known_hosts, identity, instance_id)
    finally:
        try:
            os.unlink(known_hosts)
        except OSError:
            pass
        try:
            os.rmdir(directory)
        except OSError:
            pass


def _stable_iap_host_key(
    *,
    runtime: launcher.TrustedGcloudExecutable,
    configuration: launcher.PinnedGcloudConfiguration,
    ssh: TrustedOwnerGateSshExecutable,
    instance_id: str,
    runner: SshRunner,
) -> str:
    first = _capture_host_key_once(
        runtime=runtime,
        configuration=configuration,
        ssh=ssh,
        instance_id=instance_id,
        runner=runner,
    )
    second = _capture_host_key_once(
        runtime=runtime,
        configuration=configuration,
        ssh=ssh,
        instance_id=instance_id,
        runner=runner,
    )
    if second != first:
        _error("owner_gate_host_identity_host_key_changed")
    return first


def _author_receipt(
    *,
    chain: _FoundationChainProjection,
    identity: _DirectComputeIdentity,
    host_key_base64: str,
    direct_observed_before_unix: int,
    host_key_observed_at_unix: int,
    direct_observed_after_unix: int,
    sealed_runtime_identity_sha256: str,
    toolchain_identity: Mapping[str, str],
    owner_signer: launcher._PhaseBOwnerExternalSigner,
) -> Mapping[str, Any]:
    chain.validate()
    if type(owner_signer) is not launcher._PhaseBOwnerExternalSigner:
        _error("owner_gate_host_identity_owner_signer_invalid")
    if (
        type(direct_observed_before_unix) is not int
        or type(host_key_observed_at_unix) is not int
        or type(direct_observed_after_unix) is not int
        or direct_observed_before_unix <= 0
        or not direct_observed_before_unix
        <= host_key_observed_at_unix
        <= direct_observed_after_unix
        or direct_observed_after_unix - direct_observed_before_unix > 300
        or direct_observed_after_unix
        > chain.owner_reauthentication_expires_at_unix
        or _SHA256.fullmatch(sealed_runtime_identity_sha256 or "") is None
        or not isinstance(toolchain_identity, Mapping)
        or set(toolchain_identity)
        != {
            "ssh_executable_sha256",
            "ssh_version",
            "shell_executable_sha256",
            "shell_version",
        }
        or any(
            _SHA256.fullmatch(str(toolchain_identity.get(name, ""))) is None
            for name in (
                "ssh_executable_sha256",
                "shell_executable_sha256",
            )
        )
    ):
        _error("owner_gate_host_identity_time_invalid")
    try:
        launcher.PinnedOwnerGateHostIdentityReceipt._validate_host_key(
            host_key_base64
        )
    except launcher.OwnerLauncherError as exc:
        _error("owner_gate_host_identity_host_key_invalid", exc)
    authority = owner_signer.inspect()
    toolchain = {
        "sealed_runtime_identity_sha256": sealed_runtime_identity_sha256,
        **dict(toolchain_identity),
    }
    unsigned = {
        "schema": launcher.OWNER_GATE_HOST_IDENTITY_RECEIPT_SCHEMA,
        "foundation_source_revision": chain.foundation_source_revision,
        "foundation_source_tree_oid": chain.foundation_source_tree_oid,
        "owner_reauthentication_receipt_sha256": (
            chain.owner_reauthentication_receipt_sha256
        ),
        "pre_foundation_authority_sha256": chain.pre_foundation_authority_sha256,
        "foundation_apply_receipt_sha256": chain.foundation_apply_receipt_sha256,
        "direct_iam_authority_sha256": chain.direct_iam_authority_sha256,
        "ancestry_evidence_sha256": chain.ancestry_evidence_sha256,
        "ancestry_chain_sha256": chain.ancestry_chain_sha256,
        "signed_network_evidence_sha256": chain.signed_network_evidence_sha256,
        "network_evidence_sha256": chain.network_evidence_sha256,
        **dict(identity.value),
        "owner_account": OWNER_ACCOUNT,
        "host_key_algorithm": "ssh-ed25519",
        "host_key_base64": host_key_base64,
        "collection_method": launcher.OWNER_GATE_HOST_IDENTITY_COLLECTION_METHOD,
        "direct_identity_sha256": identity.sha256,
        "direct_observed_before_unix": direct_observed_before_unix,
        "host_key_observed_at_unix": host_key_observed_at_unix,
        "direct_observed_after_unix": direct_observed_after_unix,
        "owner_reauthentication_expires_at_unix": (
            chain.owner_reauthentication_expires_at_unix
        ),
        **toolchain,
        "first_contact_toolchain_sha256": _sha256_json(toolchain),
        "owner_public_key_id": authority.key_id,
    }
    digest = _sha256_json(unsigned)
    signed = {**unsigned, "receipt_sha256": digest}
    try:
        signature = owner_signer.sign(
            _canonical(signed),
            namespace=launcher.OWNER_GATE_HOST_IDENTITY_SSHSIG_NAMESPACE,
            expected_authority=authority,
        )
    except launcher.OwnerLauncherError as exc:
        _error("owner_gate_host_identity_owner_signing_failed", exc)
    return {**signed, "signature_sshsig": signature}


def _collect_with_capabilities(
    *,
    chain: _FoundationChainProjection,
    runtime: launcher.TrustedGcloudExecutable,
    configuration: launcher.PinnedGcloudConfiguration,
    owner_identity: launcher.GcloudOwnerAccessToken,
    ssh: TrustedOwnerGateSshExecutable,
    owner_signer: launcher._PhaseBOwnerExternalSigner,
    compute_requester: ComputeRequester,
    ssh_runner: SshRunner,
    clock: Callable[[], float],
) -> Mapping[str, Any]:
    """Private test seam after every live capability has been exactly bound."""

    chain.validate()
    if (
        type(runtime) is not launcher.TrustedGcloudExecutable
        or type(configuration) is not launcher.PinnedGcloudConfiguration
        or type(owner_identity) is not launcher.GcloudOwnerAccessToken
        or type(ssh) is not TrustedOwnerGateSshExecutable
        or type(owner_signer) is not launcher._PhaseBOwnerExternalSigner
        or owner_identity.gcloud_configuration is not configuration
        or getattr(owner_identity, "_gcloud_executable", None) is not runtime
        or not callable(compute_requester)
        or not callable(ssh_runner)
        or not callable(clock)
    ):
        _error("owner_gate_host_identity_capability_invalid")
    now = int(clock())
    if now <= 0 or now > chain.owner_reauthentication_expires_at_unix:
        _error("owner_gate_host_identity_owner_reauth_expired")
    runtime_before = runtime.sealed_runtime_identity(
        expected_release_sha=chain.foundation_source_revision,
    )
    if (
        not isinstance(runtime_before, Mapping)
        or _SHA256.fullmatch(str(runtime_before.get("identity_sha256", "")))
        is None
    ):
        _error("owner_gate_host_identity_runtime_invalid")
    toolchain_before = ssh.identity()
    owner_bound = False
    access_token: _WipeableAccessToken | None = None
    try:
        owner_identity.bind_approved_subject(
            _sha256(OWNER_ACCOUNT.encode("ascii"))
        )
        owner_bound = True
        if owner_identity.approved_account != OWNER_ACCOUNT:
            _error("owner_gate_host_identity_owner_invalid")
        raw_access_token = owner_identity()
        access_token = _WipeableAccessToken(raw_access_token)
        raw_access_token = ""
        owner_identity.require_stable()
        before_unix = int(clock())
        before = _read_compute_snapshot(
            chain=chain,
            access_token=access_token,
            requester=compute_requester,
        )
        host_key = _stable_iap_host_key(
            runtime=runtime,
            configuration=configuration,
            ssh=ssh,
            instance_id=chain.vm_numeric_id,
            runner=ssh_runner,
        )
        host_key_unix = int(clock())
        after = _read_compute_snapshot(
            chain=chain,
            access_token=access_token,
            requester=compute_requester,
        )
        after_unix = int(clock())
        if after != before:
            _error("owner_gate_host_identity_compute_changed")
        return _author_receipt(
            chain=chain,
            identity=after,
            host_key_base64=host_key,
            direct_observed_before_unix=before_unix,
            host_key_observed_at_unix=host_key_unix,
            direct_observed_after_unix=after_unix,
            sealed_runtime_identity_sha256=str(
                runtime_before["identity_sha256"]
            ),
            toolchain_identity=toolchain_before,
            owner_signer=owner_signer,
        )
    finally:
        if access_token is not None:
            access_token.wipe()
        stability_errors: list[BaseException] = []
        checks: list[tuple[str, Callable[[], Any]]] = [
            (
                "runtime",
                lambda: runtime.sealed_runtime_identity(
                    expected_release_sha=chain.foundation_source_revision,
                ),
            ),
            ("configuration", configuration.assert_stable),
            ("ssh", ssh.identity),
            ("owner_signer", owner_signer.inspect),
        ]
        if owner_bound:
            checks.append(("owner_identity", owner_identity.require_stable))
        results: dict[str, Any] = {}
        for name, check in checks:
            try:
                results[name] = check()
            except BaseException as exc:
                stability_errors.append(exc)
        if (
            stability_errors
            or results.get("runtime") != runtime_before
            or results.get("ssh") != toolchain_before
        ):
            _error("owner_gate_host_identity_capability_changed")


def canonical_receipt_bytes(
    receipt: Mapping[str, Any],
    *,
    owner_signer: launcher._PhaseBOwnerExternalSigner,
) -> bytes:
    """Verify the authored owner signature and return one canonical line."""

    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != launcher.PinnedOwnerGateHostIdentityReceipt._FIELDS
        or type(owner_signer) is not launcher._PhaseBOwnerExternalSigner
    ):
        _error("owner_gate_host_identity_receipt_invalid")
    unsigned = {
        name: value
        for name, value in receipt.items()
        if name not in {"receipt_sha256", "signature_sshsig"}
    }
    digest = receipt.get("receipt_sha256")
    authority = owner_signer.inspect()
    signed = {**unsigned, "receipt_sha256": digest}
    if (
        receipt.get("schema") != launcher.OWNER_GATE_HOST_IDENTITY_RECEIPT_SCHEMA
        or receipt.get("owner_public_key_id") != authority.key_id
        or not isinstance(digest, str)
        or digest != _sha256_json(unsigned)
        or not isinstance(receipt.get("signature_sshsig"), str)
        or not receipt["signature_sshsig"]
    ):
        _error("owner_gate_host_identity_receipt_invalid")
    try:
        launcher._verify_owner_ed25519_sshsig(
            str(receipt["signature_sshsig"]),
            message=_canonical(signed),
            public_key_ed25519_hex=authority.public_key_ed25519_hex,
            namespace=launcher.OWNER_GATE_HOST_IDENTITY_SSHSIG_NAMESPACE,
            code="owner_gate_host_identity_receipt_signature_invalid",
        )
    except launcher.OwnerLauncherError as exc:
        _error("owner_gate_host_identity_receipt_signature_invalid", exc)
    raw = _canonical(receipt) + b"\n"
    if len(raw) > launcher.PinnedOwnerGateHostIdentityReceipt._MAX_BYTES:
        _error("owner_gate_host_identity_receipt_invalid")
    return raw


def _host_publication_chain(
    chain: _FoundationChainProjection,
    *,
    owner_public_key_id: str,
) -> Mapping[str, Any]:
    chain.validate()
    if _SHA256.fullmatch(owner_public_key_id or "") is None:
        _error("owner_gate_host_identity_owner_signer_invalid")
    return {
        "foundation_source_revision": chain.foundation_source_revision,
        "foundation_source_tree_oid": chain.foundation_source_tree_oid,
        "owner_reauthentication_receipt_sha256": (
            chain.owner_reauthentication_receipt_sha256
        ),
        "pre_foundation_authority_sha256": (
            chain.pre_foundation_authority_sha256
        ),
        "foundation_apply_receipt_sha256": (
            chain.foundation_apply_receipt_sha256
        ),
        "direct_iam_authority_sha256": chain.direct_iam_authority_sha256,
        "ancestry_evidence_sha256": chain.ancestry_evidence_sha256,
        "ancestry_chain_sha256": chain.ancestry_chain_sha256,
        "signed_network_evidence_sha256": (
            chain.signed_network_evidence_sha256
        ),
        "network_evidence_sha256": chain.network_evidence_sha256,
        "project_number": chain.project_number,
        "vm_numeric_id": chain.vm_numeric_id,
        "service_account_unique_id": chain.service_account_unique_id,
        "network_numeric_id": chain.network_numeric_id,
        "subnetwork_numeric_id": chain.subnetwork_numeric_id,
        "boot_disk_numeric_id": chain.boot_disk_numeric_id,
        "boot_image_numeric_id": chain.boot_image_numeric_id,
        "owner_public_key_id": owner_public_key_id,
    }


def _decode_candidate_receipt(
    raw: bytes,
    *,
    chain: _FoundationChainProjection,
    owner_signer: launcher._PhaseBOwnerExternalSigner,
) -> Mapping[str, Any]:
    if (
        type(raw) is not bytes
        or not raw.endswith(b"\n")
        or raw.count(b"\n") != 1
        or len(raw) > launcher.PinnedOwnerGateHostIdentityReceipt._MAX_BYTES
    ):
        _error("owner_gate_host_identity_receipt_invalid")
    try:
        value = json.loads(
            raw[:-1].decode("ascii", errors="strict"),
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_reject_nonfinite_compute_json,
        )
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        _error("owner_gate_host_identity_receipt_invalid", exc)
    if (
        not isinstance(value, Mapping)
        or canonical_receipt_bytes(value, owner_signer=owner_signer) != raw
    ):
        _error("owner_gate_host_identity_receipt_invalid")
    before = value.get("direct_observed_before_unix")
    host_observed = value.get("host_key_observed_at_unix")
    after = value.get("direct_observed_after_unix")
    exact = {
        "foundation_source_revision": chain.foundation_source_revision,
        "foundation_source_tree_oid": chain.foundation_source_tree_oid,
        "owner_reauthentication_receipt_sha256": (
            chain.owner_reauthentication_receipt_sha256
        ),
        "pre_foundation_authority_sha256": (
            chain.pre_foundation_authority_sha256
        ),
        "foundation_apply_receipt_sha256": (
            chain.foundation_apply_receipt_sha256
        ),
        "direct_iam_authority_sha256": chain.direct_iam_authority_sha256,
        "ancestry_evidence_sha256": chain.ancestry_evidence_sha256,
        "ancestry_chain_sha256": chain.ancestry_chain_sha256,
        "signed_network_evidence_sha256": (
            chain.signed_network_evidence_sha256
        ),
        "network_evidence_sha256": chain.network_evidence_sha256,
        "project": foundation.PROJECT,
        "project_number": chain.project_number,
        "zone": foundation.ZONE,
        "vm_name": VM_NAME,
        "vm_self_link": chain.vm_self_link,
        "vm_numeric_id": chain.vm_numeric_id,
        "vm_creation_timestamp": chain.vm_creation_timestamp,
        "owner_gate_service_account_email": SERVICE_ACCOUNT_EMAIL,
        "owner_gate_service_account_unique_id": chain.service_account_unique_id,
        "network_self_link": chain.network_self_link,
        "network_numeric_id": chain.network_numeric_id,
        "subnetwork_self_link": chain.subnetwork_self_link,
        "subnetwork_numeric_id": chain.subnetwork_numeric_id,
        "boot_disk_numeric_id": chain.boot_disk_numeric_id,
        "boot_disk_architecture": chain.boot_disk_architecture,
        "boot_disk_physical_block_size_bytes": (
            chain.boot_disk_physical_block_size_bytes
        ),
        "boot_disk_licenses": list(chain.boot_disk_licenses),
        "boot_image_self_link": chain.boot_image_self_link,
        "boot_image_numeric_id": chain.boot_image_numeric_id,
        "owner_reauthentication_expires_at_unix": (
            chain.owner_reauthentication_expires_at_unix
        ),
        "owner_account": OWNER_ACCOUNT,
        "collection_method": (
            launcher.OWNER_GATE_HOST_IDENTITY_COLLECTION_METHOD
        ),
    }
    if (
        any(value.get(name) != expected for name, expected in exact.items())
        or any(type(item) is not int or item <= 0 for item in (before, host_observed, after))
        or not before <= host_observed <= after
        or after - before > 300
        or after > chain.owner_reauthentication_expires_at_unix
    ):
        _error("owner_gate_host_identity_receipt_chain_mismatch")
    return dict(value)


def collect_and_publish_owner_gate_host_identity_v2(
    *,
    pre_foundation_authority_raw: bytes,
    owner_reauthentication_receipt_raw: bytes,
    network_evidence_raw: bytes,
    project_ancestry_evidence_raw: bytes,
    direct_iam_authority_raw: bytes,
    release_public_key: Ed25519PublicKey,
    network_collector_public_key: Ed25519PublicKey,
    project_ancestry_collector_public_key: Ed25519PublicKey,
) -> Mapping[str, Any]:
    """Live boundary: revalidate raw A artifacts, collect, sign, publish once."""

    from scripts.canary import owner_gate_foundation_apply as foundation_apply

    if (
        any(
            type(value) is not bytes or not value
            for value in (
                pre_foundation_authority_raw,
                owner_reauthentication_receipt_raw,
                network_evidence_raw,
                project_ancestry_evidence_raw,
                direct_iam_authority_raw,
            )
        )
        or not isinstance(release_public_key, Ed25519PublicKey)
        or not isinstance(network_collector_public_key, Ed25519PublicKey)
        or not isinstance(
            project_ancestry_collector_public_key,
            Ed25519PublicKey,
        )
    ):
        _error("owner_gate_host_identity_foundation_chain_invalid")
    now_unix = int(time.time())
    recovery_only = False
    try:
        foundation_a = foundation_apply.decode_validated_foundation_a_chain(
            pre_foundation_authority_raw=pre_foundation_authority_raw,
            owner_reauthentication_receipt_raw=(
                owner_reauthentication_receipt_raw
            ),
            network_evidence_raw=network_evidence_raw,
            project_ancestry_evidence_raw=project_ancestry_evidence_raw,
            release_public_key=release_public_key,
            network_collector_public_key=network_collector_public_key,
            project_ancestry_collector_public_key=(
                project_ancestry_collector_public_key
            ),
            now_unix=now_unix,
        )
        apply_chain = foundation_apply.load_validated_foundation_apply_chain(
            foundation_a
        )
        chain = _projection_from_validated_chains(
            foundation_chain=apply_chain,
            direct_iam_authority_raw=direct_iam_authority_raw,
            now_unix=now_unix,
        )
    except (
        OwnerGateHostIdentityError,
        foundation_apply.OwnerGateFoundationApplyError,
    ) as fresh_error:
        try:
            apply_chain = (
                foundation_apply._load_validated_foundation_apply_chain_for_source_recovery(
                    pre_foundation_authority_raw=(
                        pre_foundation_authority_raw
                    ),
                    owner_reauthentication_receipt_raw=(
                        owner_reauthentication_receipt_raw
                    ),
                    network_evidence_raw=network_evidence_raw,
                    project_ancestry_evidence_raw=(
                        project_ancestry_evidence_raw
                    ),
                    release_public_key=release_public_key,
                    network_collector_public_key=(
                        network_collector_public_key
                    ),
                    project_ancestry_collector_public_key=(
                        project_ancestry_collector_public_key
                    ),
                )
            )
            chain = _projection_from_validated_chains(
                foundation_chain=apply_chain,
                direct_iam_authority_raw=direct_iam_authority_raw,
                now_unix=int(apply_chain.apply_receipt["completed_at_unix"]),
                recovery_only=True,
            )
        except (
            OwnerGateHostIdentityError,
            foundation_apply.OwnerGateFoundationApplyError,
        ):
            if isinstance(fresh_error, OwnerGateHostIdentityError):
                raise fresh_error
            _error("owner_gate_host_identity_foundation_chain_invalid", fresh_error)
        recovery_only = True
    owner_signer = launcher._PhaseBOwnerExternalSigner()
    owner_authority = owner_signer.inspect()

    def validate(raw: bytes) -> source_publication._ValidatedArtifact:
        receipt = _decode_candidate_receipt(
            raw,
            chain=chain,
            owner_signer=owner_signer,
        )
        return source_publication._ValidatedArtifact(
            value=receipt,
            logical_sha256=str(receipt["receipt_sha256"]),
        )

    def collect() -> bytes:
        runtime = launcher.TrustedGcloudExecutable(
            release_sha=chain.foundation_source_revision,
        )
        configuration = launcher.PinnedGcloudConfiguration()
        owner_identity = launcher.GcloudOwnerAccessToken(
            gcloud_executable=runtime,
            gcloud_configuration=configuration,
        )
        ssh = TrustedOwnerGateSshExecutable()
        receipt = _collect_with_capabilities(
            chain=chain,
            runtime=runtime,
            configuration=configuration,
            owner_identity=owner_identity,
            ssh=ssh,
            owner_signer=owner_signer,
            compute_requester=_default_compute_request,
            ssh_runner=subprocess.run,
            clock=time.time,
        )
        return canonical_receipt_bytes(receipt, owner_signer=owner_signer)

    try:
        result = source_publication._run_host_identity(
            owner_home=Path(launcher._canonical_owner_home()),
            chain=_host_publication_chain(
                chain,
                owner_public_key_id=owner_authority.key_id,
            ),
            maximum=launcher.PinnedOwnerGateHostIdentityReceipt._MAX_BYTES,
            validator=validate,
            collector=collect,
            _recovery_only=recovery_only,
        )
    except source_publication._SourceArtifactPublicationError as exc:
        _error("owner_gate_host_identity_publication_failed", exc)
    receipt = result.value
    if not isinstance(receipt, Mapping):
        _error("owner_gate_host_identity_publication_failed")
    publication = {
        "path": result.path,
        "receipt_sha256": result.logical_sha256,
        "receipt_file_sha256": result.file_sha256,
    }
    return {
        "receipt": dict(receipt),
        "publication": publication,
    }


def _read_owner_input(path: Path, *, maximum: int) -> bytes:
    """Read one immutable owner artifact without following any path alias."""

    from scripts.canary import owner_gate_trust as release_trust

    if (
        type(path) is not type(Path())
        or not path.is_absolute()
        or ".." in path.parts
        or os.path.realpath(path) != str(path)
        or type(maximum) is not int
        or not 0 < maximum <= MAX_OWNER_INPUT_BYTES
    ):
        _error("owner_gate_host_identity_owner_input_invalid")
    try:
        return release_trust._read_immutable(
            path,
            maximum=maximum,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
    except release_trust.OwnerGateTrustError as exc:
        _error("owner_gate_host_identity_owner_input_invalid", exc)


def _load_owner_collector_public_key(path: Path) -> Ed25519PublicKey:
    raw = _read_owner_input(path, maximum=MAX_OWNER_PUBLIC_KEY_BYTES)
    try:
        if len(raw) == 32:
            key = Ed25519PublicKey.from_public_bytes(raw)
        else:
            key = serialization.load_pem_public_key(raw)
    except (TypeError, ValueError) as exc:
        _error("owner_gate_host_identity_owner_public_key_invalid", exc)
    if not isinstance(key, Ed25519PublicKey):
        _error("owner_gate_host_identity_owner_public_key_invalid")
    return key


def main(argv: Sequence[str] | None = None) -> int:
    """Owner-only fixed first-contact CLI; the receipt destination is fixed."""

    from scripts.canary import owner_gate_pre_foundation as pre_foundation

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
    parser.add_argument("--direct-iam-authority", type=Path, required=True)
    arguments = parser.parse_args(argv)

    try:
        release_public_key = pre_foundation.load_pinned_public_key(
            arguments.release_trust_public_key,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
        )
    except pre_foundation.OwnerGatePreFoundationError as exc:
        _error("owner_gate_host_identity_owner_public_key_invalid", exc)
    network_public_key = _load_owner_collector_public_key(
        arguments.network_collector_public_key
    )
    ancestry_public_key = _load_owner_collector_public_key(
        arguments.project_ancestry_collector_public_key
    )
    result = collect_and_publish_owner_gate_host_identity_v2(
        pre_foundation_authority_raw=_read_owner_input(
            arguments.pre_foundation_authority,
            maximum=MAX_OWNER_INPUT_BYTES,
        ),
        owner_reauthentication_receipt_raw=_read_owner_input(
            arguments.owner_reauth_receipt,
            maximum=MAX_OWNER_INPUT_BYTES,
        ),
        network_evidence_raw=_read_owner_input(
            arguments.network_evidence,
            maximum=MAX_OWNER_INPUT_BYTES,
        ),
        project_ancestry_evidence_raw=_read_owner_input(
            arguments.project_ancestry_evidence,
            maximum=MAX_OWNER_INPUT_BYTES,
        ),
        direct_iam_authority_raw=_read_owner_input(
            arguments.direct_iam_authority,
            maximum=MAX_OWNER_INPUT_BYTES,
        ),
        release_public_key=release_public_key,
        network_collector_public_key=network_public_key,
        project_ancestry_collector_public_key=ancestry_public_key,
    )
    publication = result.get("publication") if isinstance(result, Mapping) else None
    expected_output = str(
        Path(launcher._canonical_owner_home())
        / launcher.OWNER_GATE_HOST_IDENTITY_RECEIPT_RELATIVE
    )
    if (
        not isinstance(publication, Mapping)
        or set(publication)
        != {"path", "receipt_sha256", "receipt_file_sha256"}
        or publication.get("path") != expected_output
        or _SHA256.fullmatch(str(publication.get("receipt_sha256", ""))) is None
        or _SHA256.fullmatch(str(publication.get("receipt_file_sha256", "")))
        is None
    ):
        _error("owner_gate_host_identity_publication_invalid")
    summary = {
        "schema": "muncho-owner-gate-iap-host-identity-publication.v2",
        "receipt_published": True,
        **dict(publication),
    }
    sys.stdout.write(_canonical(summary).decode("utf-8") + "\n")
    return 0


__all__ = [
    "OwnerGateHostIdentityError",
    "TrustedOwnerGateSshExecutable",
    "canonical_receipt_bytes",
    "collect_and_publish_owner_gate_host_identity_v2",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
