#!/usr/bin/env python3
"""Dual-attested, read-only storage-growth observation boundary.

The launcher may assemble a candidate storage observation, but it has no
attestation authority.  The fixed canary-host signer and the fixed owner-gate
cloud signer independently recollect the facts in their trust domains before
signing the same canonical observation bundle core.  This module contains no
mutation operation, generic remote command, or local ``gcloud`` fallback.

Private signing keys are raw 32-byte Ed25519 seeds in fixed, non-symlink,
mode-0400 files.  They are read only by the role-specific signer process and
are never accepted through argv, environment variables, or protocol frames.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import http.client
import json
import os
import re
import socket
import ssl
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Protocol, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import owner_gate_foundation as owner_gate_foundation
from scripts.canary import storage_growth_contract as contract
from scripts.canary import storage_growth_evidence as evidence
from scripts.canary.runtime_units import CANARY_RUNTIME_UNITS


ATTESTATION_REQUEST_SCHEMA = (
    "muncho-storage-growth-trusted-attestation-request.v1"
)
ATTESTATION_RESPONSE_SCHEMA = (
    "muncho-storage-growth-trusted-attestation-response.v1"
)
ATTESTATION_REPLAY_SCHEMA = (
    "muncho-storage-growth-trusted-attestation-replay.v1"
)
TRUSTED_IAM_PROJECTION_SCHEMA = (
    "muncho-storage-growth-trusted-iam-projection.v1"
)
TRUSTED_IAM_AUTHORITY_SCOPE = (
    "exact_allow_grants_with_inherited_sensitive_inventory_sa_policy_and_key_snapshot"
)
HOST_CONFIG_SCHEMA = "muncho-storage-growth-host-attestor-config.v1"
CLOUD_CONFIG_SCHEMA = "muncho-storage-growth-cloud-attestor-config.v1"

MAX_FRAME_BYTES = 1024 * 1024
MAX_CONFIG_BYTES = 32 * 1024
MAX_COMMAND_OUTPUT_BYTES = 256 * 1024
MAX_HTTP_BODY_BYTES = 512 * 1024
MAX_TRUSTED_CA_BUNDLE_BYTES = 4 * 1024 * 1024
COMMAND_TIMEOUT_SECONDS = 15
HTTP_TIMEOUT_SECONDS = 10

FIXED_DEBIAN_CA_BUNDLE_PATH = Path("/etc/ssl/certs/ca-certificates.crt")
_TRUSTED_CA_OWNER_UID = 0
_FORBIDDEN_TLS_ENVIRONMENT = frozenset({
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

HOST_CONFIG_PATH = Path(
    "/etc/muncho/trusted-observation/host-attestor.json"
)
HOST_PRIVATE_KEY_PATH = Path(
    "/etc/muncho/trusted-observation/host-observation-attestation.key"
)
HOST_REPLAY_DIRECTORY = Path(
    "/var/lib/muncho/trusted-observation/host-attestations"
)
CLOUD_CONFIG_PATH = Path(
    "/etc/muncho-owner-gate/cloud-observation-attestor.json"
)
CLOUD_PRIVATE_KEY_PATH = Path(
    "/etc/muncho-owner-gate/executor-keys/cloud-observation-attestation.key"
)
CLOUD_REPLAY_DIRECTORY = Path(
    "/var/lib/muncho-owner-gate/executor/cloud-attestations"
)
TRUSTED_IAM_TTL_SECONDS = 300

HOST_RECEIPT_PATH = Path("/etc/muncho/full-canary/host-identity.json")
STOPPED_RELEASE_RECEIPT_PATH = Path(
    "/var/lib/muncho-canary-release-evidence"
) / contract.CURRENT_STOPPED_RELEASE_SHA / "stopped-release-publication.json"
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
FINDMNT = "/usr/bin/findmnt"
SYSTEMCTL = "/usr/bin/systemctl"

METADATA_HOST = "169.254.169.254"
COMPUTE_HOST = "compute.googleapis.com"
CLOUD_RESOURCE_MANAGER_HOST = "cloudresourcemanager.googleapis.com"
IAM_HOST = "iam.googleapis.com"
METADATA_TOKEN_PATH = (
    "/computeMetadata/v1/instance/service-accounts/default/token"
)
METADATA_SCOPES_PATH = (
    "/computeMetadata/v1/instance/service-accounts/default/scopes"
)
METADATA_INSTANCE_ID_PATH = "/computeMetadata/v1/instance/id"
METADATA_SERVICE_ACCOUNT_EMAIL_PATH = (
    "/computeMetadata/v1/instance/service-accounts/default/email"
)
METADATA_TEXT_MEDIA_TYPE = "application/text"
OWNER_GATE_METADATA_SCOPES = (
    "https://www.googleapis.com/auth/compute",
    "https://www.googleapis.com/auth/iam",
    "https://www.googleapis.com/auth/cloudplatformprojects.readonly",
    "https://www.googleapis.com/auth/cloudplatformfolders.readonly",
    "https://www.googleapis.com/auth/cloudplatformorganizations.readonly",
)
OWNER_GATE_VM_NAME = "muncho-owner-gate-01"
OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT = (
    "muncho-owner-gate-executor@adventico-ai-platform."
    "iam.gserviceaccount.com"
)
OWNER_GATE_PROJECT_READ_ROLE = (
    f"projects/{contract.PROJECT}/roles/"
    f"{owner_gate_foundation.READ_ONLY_IAM_ROLE_ID}"
)
OWNER_GATE_MUTATION_ROLE = (
    f"projects/{contract.PROJECT}/roles/"
    f"{owner_gate_foundation.MUTATION_ROLE_ID}"
)
OWNER_GATE_MUTATION_CONDITION = {
    "title": "muncho_owner_gate_exact_storage_v1",
    "description": "Exact canary disk and instance resources only",
    "expression": owner_gate_foundation._condition_expression(),
}
OWNER_GATE_PROJECT_READ_ROLE_TITLE = (
    "Muncho Owner Gate IAM Observation Reader V1"
)
OWNER_GATE_PROJECT_READ_ROLE_DESCRIPTION = (
    "Exact project and service-account IAM read-only observation"
)
OWNER_GATE_ANCESTOR_READ_ROLE_TITLE = (
    "Muncho Owner Gate Hierarchy Observation Reader V1"
)
OWNER_GATE_ANCESTOR_READ_ROLE_DESCRIPTION = (
    "Exact folder and organization IAM read-only observation"
)
OWNER_GATE_MUTATION_ROLE_TITLE = "Muncho Owner Gate Storage Executor V1"
OWNER_GATE_MUTATION_ROLE_DESCRIPTION = (
    "Exact canary storage disk and instance get resize stop start only"
)
TARGET_RUNTIME_AUTHORITY_SCHEMA = (
    "muncho-storage-growth-target-runtime-allow-authority.v1"
)
COLLECTOR_AUTHORITY_SCHEMA = (
    "muncho-storage-growth-owner-gate-allow-authority.v1"
)
MAX_POLICY_ROLE_DEFINITIONS = 256
_ROLE_ID = r"[A-Za-z0-9_.]{1,128}"
_BUILTIN_ROLE = re.compile(rf"^roles/{_ROLE_ID}$")
_PROJECT_ROLE = re.compile(
    rf"^projects/{re.escape(contract.PROJECT)}/roles/{_ROLE_ID}$"
)
_ORGANIZATION_ROLE = re.compile(
    rf"^organizations/([1-9][0-9]{{5,31}})/roles/{_ROLE_ID}$"
)
_ROLE_STAGES = {
    "ALPHA", "BETA", "GA", "DEPRECATED", "DISABLED", "EAP",
}
INSTANCE_PATH = (
    f"/compute/v1/projects/{contract.PROJECT}/zones/{contract.ZONE}/"
    f"instances/{contract.VM_NAME}"
)
DISK_PATH = (
    f"/compute/v1/projects/{contract.PROJECT}/zones/{contract.ZONE}/"
    f"disks/{contract.DISK_NAME}"
)
OWNER_GATE_INSTANCE_PATH = (
    f"/compute/v1/projects/{contract.PROJECT}/zones/{contract.ZONE}/"
    f"instances/{OWNER_GATE_VM_NAME}"
)
PROJECT_IAM_POLICY_PATH = f"/v1/projects/{contract.PROJECT}:getIamPolicy"
TARGET_SERVICE_ACCOUNT_PATH = (
    f"/v1/projects/{contract.PROJECT}/serviceAccounts/"
    f"{contract.RUNTIME_SERVICE_ACCOUNT}"
)
COLLECTOR_SERVICE_ACCOUNT_PATH = (
    f"/v1/projects/{contract.PROJECT}/serviceAccounts/"
    f"{OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT}"
)
COLLECTOR_SERVICE_ACCOUNT_IAM_POLICY_PATH = (
    f"{COLLECTOR_SERVICE_ACCOUNT_PATH}:getIamPolicy"
    "?options.requestedPolicyVersion=3"
)
COLLECTOR_SERVICE_ACCOUNT_USER_KEYS_PATH = (
    f"{COLLECTOR_SERVICE_ACCOUNT_PATH}/keys?keyTypes=USER_MANAGED"
)
ROLE_PATHS = tuple(
    f"/v1/{role}" for role in contract.RUNTIME_ROLES
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_BOOT_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_CHECKPOINT_STATES = {
    "source": ("await_source", frozenset({"source_ready"})),
    "post_resize": (
        "await_post_resize",
        frozenset({"resize_complete_boot_required", "target_ready"}),
    ),
    "post_stop": (
        "await_post_stop",
        frozenset({"terminated_after_growth_intent"}),
    ),
    "post_start": ("await_post_start", frozenset({"target_ready"})),
}
_OBSERVATION_REQUEST_FIELDS = {
    "schema",
    "transaction_id",
    "checkpoint",
    "canonical_state",
    "prior_event_head_sha256",
    "request_binding_sha256",
    "observation_nonce_sha256",
    "collection_attempt_id",
    "collection_attempt_sequence",
    "collection_attempt_issued_at_unix",
    "collection_attempt_expires_at_unix",
    "release_sha",
    "plan_sha256",
    "observation_request_sha256",
}
_ATTESTATION_REQUEST_FIELDS = {
    "schema",
    "role",
    "observation_request",
    "candidate_observation",
    "trusted_iam_projection",
    "frame_sha256",
}
_ATTESTATION_RESPONSE_FIELDS = {
    "schema",
    "role",
    "attestation_request_sha256",
    "observation_request_sha256",
    "observation_sha256",
    "trusted_iam_projection",
    "trusted_iam_projection_sha256",
    "bundle_core_sha256",
    "recollection_sha256",
    "attestation",
    "response_sha256",
}
_ATTESTATION_FIELDS = {
    "schema",
    "public_key_id",
    "signature_ed25519_b64url",
}
_TRUSTED_IAM_PROJECTION_FIELDS = {
    "schema",
    "authority_scope",
    "transaction_id",
    "checkpoint",
    "observation_request_sha256",
    "collection_attempt_id",
    "collection_attempt_sequence",
    "collection_attempt_issued_at_unix",
    "collection_attempt_expires_at_unix",
    "candidate_external_iam_receipt_sha256",
    "candidate_external_iam_policy_sha256",
    "project",
    "project_number",
    "zone",
    "instance",
    "service_account",
    "scopes",
    "member",
    "roles",
    "permissions",
    "project_policy_version",
    "project_policy_etag",
    "project_policy_sha256",
    "resource_ancestor_chain",
    "resource_hierarchy",
    "resource_hierarchy_sha256",
    "resource_policies",
    "resource_policies_sha256",
    "relevant_bindings",
    "relevant_bindings_sha256",
    "role_definitions",
    "role_definitions_sha256",
    "policy_role_definitions",
    "policy_role_definitions_sha256",
    "residual_external_bindings",
    "residual_external_bindings_sha256",
    "service_account_resource",
    "service_account_resource_sha256",
    "collector_runtime_identity",
    "collector_runtime_identity_sha256",
    "target_runtime_authority",
    "target_runtime_authority_sha256",
    "collector_service_account_authority",
    "collector_service_account_authority_sha256",
    "external_gcp_admin_trust_root",
    "external_gcp_admin_trust_root_sha256",
    "collected_at_unix",
    "expires_at_unix",
    "projection_sha256",
}
_CONFIG_FIELDS = {
    "schema",
    "role",
    "private_key_path",
    "private_key_uid",
    "private_key_gid",
    "private_key_mode",
    "public_key_id",
    "replay_directory",
    "replay_directory_uid",
    "replay_directory_gid",
    "replay_directory_mode",
}
_DIRECT_IAM_PIN_FIELDS = {
    "expected_project_number",
    "expected_ancestor_chain",
    "expected_runtime_instance_numeric_id",
    "expected_runtime_service_account_email",
    "expected_runtime_service_account_unique_id",
    "expected_target_service_account_unique_id",
    "expected_metadata_scopes",
    "expected_external_gcp_admin_trust_root",
}
_SERVICE_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "MainPID",
    "FragmentPath",
    "DropInPaths",
)


class TrustedObservationError(RuntimeError):
    """Stable, secret-free trusted observation boundary failure."""


class HostFactsReader(Protocol):
    def collect(self) -> Mapping[str, Any]: ...


class CloudFactsReader(Protocol):
    def collect(
        self,
        observation: Mapping[str, Any],
        observation_request: Mapping[str, Any],
        now_unix: int,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]: ...


def _sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _copy_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(protocol.canonical_json_bytes(value).decode("utf-8"))


def validate_observation_request(value: Any) -> Mapping[str, Any]:
    """Validate the exact self-digested request returned by the executor.

    A collector never derives the checkpoint from a candidate observation.
    Terminal requests are deliberately unusable for collection.
    """

    request = classify_observation_request(value)
    if request["canonical_state"] == "terminal":
        raise TrustedObservationError(
            "trusted_observation_request_terminal"
        )
    checkpoint = request.get("checkpoint")
    if not isinstance(checkpoint, str):
        raise TrustedObservationError("trusted_observation_request_invalid")
    expected = _CHECKPOINT_STATES.get(checkpoint)
    if expected is None or request.get("canonical_state") != expected[0]:
        raise TrustedObservationError("trusted_observation_request_invalid")
    binding = evidence.observation_request_binding_sha256(
        transaction_id=request["transaction_id"],
        checkpoint=checkpoint,
        prior_event_head_sha256=request["prior_event_head_sha256"],
        release_sha=request["release_sha"],
        plan_sha256=request["plan_sha256"],
    )
    nonce = evidence.observation_nonce_sha256(
        request_binding_sha256=binding,
        transaction_id=request["transaction_id"],
        checkpoint=checkpoint,
    )
    if (
        request.get("request_binding_sha256") != binding
        or request.get("observation_nonce_sha256") != nonce
    ):
        raise TrustedObservationError(
            "trusted_observation_request_binding_invalid"
        )
    return request


def classify_observation_request(value: Any) -> Mapping[str, Any]:
    """Validate the executor response and identify terminal vs collection.

    A terminal response is returned to the launcher as a short-circuit signal;
    it is never accepted by either signer or the bundle combiner.
    """

    if not isinstance(value, Mapping) or set(value) != _OBSERVATION_REQUEST_FIELDS:
        raise TrustedObservationError(
            "trusted_observation_request_fields_invalid"
        )
    request = _copy_json(value)
    unsigned = {
        name: item
        for name, item in request.items()
        if name != "observation_request_sha256"
    }
    if (
        request.get("schema")
        != "muncho-storage-growth-observation-request.v1"
        or not _sha(request.get("transaction_id"))
        or not _sha(request.get("prior_event_head_sha256"))
        or _REVISION.fullmatch(str(request.get("release_sha", ""))) is None
        or request.get("plan_sha256") != contract.canonical_plan_sha256()
        or request.get("observation_request_sha256")
        != protocol.sha256_json(unsigned)
    ):
        raise TrustedObservationError("trusted_observation_request_invalid")
    if request["canonical_state"] == "terminal":
        if (
            request.get("checkpoint") is not None
            or request.get("request_binding_sha256") is not None
            or request.get("observation_nonce_sha256") is not None
            or request.get("collection_attempt_id") is not None
            or request.get("collection_attempt_sequence") is not None
            or request.get("collection_attempt_issued_at_unix") is not None
            or request.get("collection_attempt_expires_at_unix") is not None
        ):
            raise TrustedObservationError(
                "trusted_observation_terminal_request_invalid"
            )
        return request
    context = {
        "schema": "muncho-storage-growth-collection-context.v1",
        "transaction_id": request.get("transaction_id"),
        "checkpoint": request.get("checkpoint"),
        "prior_event_head_sha256": request.get(
            "prior_event_head_sha256"
        ),
        "release_sha": request.get("release_sha"),
        "plan_sha256": request.get("plan_sha256"),
    }
    identity = {
        "schema": "muncho-storage-growth-collection-attempt-id.v1",
        "context_sha256": protocol.sha256_json(context),
        "context_sequence": request.get("collection_attempt_sequence"),
        "issued_at_unix": request.get(
            "collection_attempt_issued_at_unix"
        ),
    }
    if (
        request.get("checkpoint") not in _CHECKPOINT_STATES
        or request.get("canonical_state")
        != _CHECKPOINT_STATES[request["checkpoint"]][0]
        or not _sha(request.get("request_binding_sha256"))
        or not _sha(request.get("observation_nonce_sha256"))
        or request.get("collection_attempt_id")
        != protocol.sha256_json(identity)
        or type(request.get("collection_attempt_sequence")) is not int
        or request["collection_attempt_sequence"] < 1
        or type(request.get("collection_attempt_issued_at_unix")) is not int
        or request["collection_attempt_issued_at_unix"] < 1
        or request.get("collection_attempt_expires_at_unix")
        != request["collection_attempt_issued_at_unix"]
        + evidence.OBSERVATION_BUNDLE_TTL_SECONDS
    ):
        raise TrustedObservationError("trusted_observation_request_invalid")
    return request


def _validated_observation(
    request: Mapping[str, Any],
    value: Any,
    *,
    now_unix: int,
) -> Mapping[str, Any]:
    if type(now_unix) is not int or now_unix < 1:
        raise TrustedObservationError("trusted_observation_clock_invalid")
    if not (
        request["collection_attempt_issued_at_unix"]
        <= now_unix
        < request["collection_attempt_expires_at_unix"]
    ):
        raise TrustedObservationError(
            "trusted_observation_collection_attempt_expired"
        )
    expected = _CHECKPOINT_STATES[str(request["checkpoint"])]
    try:
        observation = evidence.validate_observation(
            value,
            now_unix=now_unix,
            require_fresh=True,
            allowed_states=expected[1],
        )
    except evidence.StorageGrowthEvidenceError as exc:
        raise TrustedObservationError(
            "trusted_observation_candidate_invalid"
        ) from None
    collected = observation["collected_at_unix"]
    if not collected <= now_unix < (
        collected + evidence.OBSERVATION_BUNDLE_TTL_SECONDS
    ):
        raise TrustedObservationError(
            "trusted_observation_candidate_expired"
        )
    return _copy_json(observation)


def _iam_policy_sha256(receipt: Mapping[str, Any]) -> str:
    return protocol.sha256_json({
        name: receipt[name]
        for name in (
            "project",
            "zone",
            "instance",
            "service_account",
            "scopes",
            "roles",
            "permissions",
            "foundation_plan_sha256",
            "host_plan_sha256",
        )
    })


def _string_set(value: Any) -> tuple[str, ...] | None:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        return None
    return tuple(sorted(value))


def _valid_policy_member(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and value
        and len(value) <= 1024
        and all(ord(character) >= 0x20 and ord(character) != 0x7F for character in value)
    )


def _normalized_condition(
    value: Any,
    *,
    error: str,
) -> Mapping[str, str] | None:
    if value is None:
        return None
    if (
        not isinstance(value, Mapping)
        or not {"title", "expression"} <= set(value)
        or set(value) - {"title", "description", "expression"}
    ):
        raise TrustedObservationError(error)
    normalized = {
        "title": value.get("title"),
        "description": value.get("description", ""),
        "expression": value.get("expression"),
    }
    if (
        not isinstance(normalized["title"], str)
        or not normalized["title"]
        or not isinstance(normalized["description"], str)
        or not isinstance(normalized["expression"], str)
        or not normalized["expression"]
        or any(
            len(item) > 4096
            or any(ord(character) < 0x20 for character in item)
            for item in normalized.values()
        )
    ):
        raise TrustedObservationError(error)
    return normalized


def _policy_role_allowed(
    role: Any,
    *,
    ancestor_chain: Sequence[str],
) -> bool:
    if not isinstance(role, str):
        return False
    if _BUILTIN_ROLE.fullmatch(role) or _PROJECT_ROLE.fullmatch(role):
        return True
    match = _ORGANIZATION_ROLE.fullmatch(role)
    return bool(
        match
        and ancestor_chain
        and ancestor_chain[-1] == f"organizations/{match.group(1)}"
    )


def _bounded_policy_role_names(
    resource_policies: Sequence[Mapping[str, Any]],
    *,
    ancestor_chain: Sequence[str],
) -> tuple[str, ...]:
    roles: set[str] = set()
    if not isinstance(resource_policies, Sequence):
        raise TrustedObservationError(
            "trusted_observation_iam_policy_role_inventory_invalid"
        )
    for policy in resource_policies:
        bindings = policy.get("bindings") if isinstance(policy, Mapping) else None
        if not isinstance(bindings, list):
            raise TrustedObservationError(
                "trusted_observation_iam_policy_role_inventory_invalid"
            )
        for binding in bindings:
            role = binding.get("role") if isinstance(binding, Mapping) else None
            if not _policy_role_allowed(role, ancestor_chain=ancestor_chain):
                raise TrustedObservationError(
                    "trusted_observation_iam_policy_role_inventory_invalid"
                )
            roles.add(str(role))
            if len(roles) > MAX_POLICY_ROLE_DEFINITIONS:
                raise TrustedObservationError(
                    "trusted_observation_iam_policy_role_inventory_invalid"
                )
    return tuple(sorted(roles))


def _normalize_policy_role_definition(
    resource: Mapping[str, Any],
    *,
    expected_role: str,
    ancestor_chain: Sequence[str],
) -> Mapping[str, Any]:
    error = "trusted_observation_iam_policy_role_definition_invalid"
    if (
        not isinstance(resource, Mapping)
        or not _policy_role_allowed(
            expected_role, ancestor_chain=ancestor_chain
        )
    ):
        raise TrustedObservationError(error)
    permissions = _string_set(resource.get("includedPermissions"))
    deleted = resource.get("deleted", False)
    etag = resource.get("etag")
    title = resource.get("title")
    description = resource.get("description", "")
    if (
        resource.get("name") != expected_role
        or permissions is None
        or resource.get("stage") not in _ROLE_STAGES
        or type(deleted) is not bool
        or deleted
        or not isinstance(title, str)
        or not title
        or len(title) > 4096
        or not isinstance(description, str)
        or len(description) > 16 * 1024
        or (
            etag is not None
            and (
                not isinstance(etag, str)
                or not etag
                or len(etag) > 4096
            )
        )
    ):
        raise TrustedObservationError(error)
    return {
        "name": expected_role,
        "title": title,
        "description": description,
        "included_permissions": list(permissions),
        "stage": resource["stage"],
        "deleted": False,
        "etag": etag,
        "raw_sha256": protocol.sha256_json(resource),
    }


def _semantic_role_definition(
    definition: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "name": definition["name"],
        "title": definition["title"],
        "description": definition["description"],
        "included_permissions": list(definition["included_permissions"]),
        "stage": definition["stage"],
        "deleted": False,
        "etag": definition["etag"],
    }


def validate_external_gcp_admin_trust_root(
    value: Any,
    *,
    ancestor_chain: Sequence[str],
) -> Mapping[str, Any]:
    """Validate the owner-reauthored, release-signed residual IAM roots."""

    error = "trusted_observation_external_gcp_admin_trust_root_invalid"
    fields = {
        "inventory_complete",
        "structural_partition_complete",
        "passkey_protects_against_external_gcp_admins",
        "passkey_protects_against_pinned_external_roots",
        "google_provider_control_plane_outside_passkey",
        "collected_under_owner_reauthentication_receipt_sha256",
        "resource_policy_generations",
        "allowed_residual_bindings",
        "allowed_residual_role_definitions",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TrustedObservationError(error)
    generations = value.get("resource_policy_generations")
    bindings = value.get("allowed_residual_bindings")
    definitions = value.get("allowed_residual_role_definitions")
    allowed_resources = {
        f"projects/{contract.PROJECT}",
        *(str(item) for item in ancestor_chain),
    }
    if (
        value.get("inventory_complete") is not True
        or value.get("structural_partition_complete") is not True
        or value.get("passkey_protects_against_external_gcp_admins") is not False
        or value.get("passkey_protects_against_pinned_external_roots") is not False
        or value.get("google_provider_control_plane_outside_passkey") is not True
        or not _sha(
            value.get(
                "collected_under_owner_reauthentication_receipt_sha256"
            )
        )
        or not isinstance(generations, list)
        or not isinstance(bindings, list)
        or len(bindings) > MAX_POLICY_ROLE_DEFINITIONS
        or not isinstance(definitions, list)
        or len(definitions) > MAX_POLICY_ROLE_DEFINITIONS
    ):
        raise TrustedObservationError(error)
    expected_generation_resources = [
        f"projects/{contract.PROJECT}",
        *(str(item) for item in ancestor_chain),
    ]
    if len(generations) != len(expected_generation_resources):
        raise TrustedObservationError(error)
    for expected_resource, generation in zip(
        expected_generation_resources,
        generations,
        strict=True,
    ):
        if (
            not isinstance(generation, Mapping)
            or set(generation)
            != {"resource", "version", "etag", "audit_configs"}
            or generation.get("resource") != expected_resource
            or type(generation.get("version")) is not int
            or generation["version"] not in {1, 3}
            or not isinstance(generation.get("etag"), str)
            or not generation["etag"]
            or len(generation["etag"]) > 4096
            or not isinstance(generation.get("audit_configs"), list)
        ):
            raise TrustedObservationError(error)
        protocol.canonical_json_bytes(generation["audit_configs"])
    canonical_bindings: list[bytes] = []
    binding_roles: set[str] = set()
    for binding in bindings:
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"resource", "role", "members", "condition"}
            or binding.get("resource") not in allowed_resources
            or not _policy_role_allowed(
                binding.get("role"), ancestor_chain=ancestor_chain
            )
        ):
            raise TrustedObservationError(error)
        members = _string_set(binding.get("members"))
        if (
            members is None
            or not members
            or list(members) != binding.get("members")
            or any(
                not member
                or len(member) > 1024
                or any(
                    ord(character) < 0x20 or ord(character) == 0x7F
                    for character in member
                )
                for member in members
            )
        ):
            raise TrustedObservationError(error)
        condition = _normalized_condition(binding.get("condition"), error=error)
        if condition != binding.get("condition"):
            raise TrustedObservationError(error)
        canonical_bindings.append(protocol.canonical_json_bytes(binding))
        binding_roles.add(str(binding["role"]))
    if canonical_bindings != sorted(set(canonical_bindings)):
        raise TrustedObservationError(error)
    definition_names: list[str] = []
    for definition in definitions:
        if (
            not isinstance(definition, Mapping)
            or set(definition)
            != {
                "name", "title", "description", "included_permissions",
                "stage", "deleted", "etag",
            }
            or not _policy_role_allowed(
                definition.get("name"), ancestor_chain=ancestor_chain
            )
        ):
            raise TrustedObservationError(error)
        permissions = _string_set(definition.get("included_permissions"))
        if (
            not isinstance(definition.get("title"), str)
            or not definition["title"]
            or not isinstance(definition.get("description"), str)
            or permissions is None
            or not permissions
            or list(permissions) != definition.get("included_permissions")
            or definition.get("stage") not in _ROLE_STAGES
            or definition.get("deleted") is not False
            or (
                definition.get("etag") is not None
                and (
                    not isinstance(definition["etag"], str)
                    or not definition["etag"]
                    or len(definition["etag"]) > 4096
                )
            )
        ):
            raise TrustedObservationError(error)
        definition_names.append(str(definition["name"]))
    if (
        definition_names != sorted(set(definition_names))
        or set(definition_names) != binding_roles
    ):
        raise TrustedObservationError(error)
    return _copy_json(value)


def _owner_gate_ancestor_read_role(
    ancestor_chain: Sequence[str],
) -> str:
    if (
        not ancestor_chain
        or re.fullmatch(
            r"organizations/[1-9][0-9]{5,31}",
            str(ancestor_chain[-1]),
        )
        is None
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_authority_pin_invalid"
        )
    organization_id = str(ancestor_chain[-1]).split("/", 1)[1]
    return (
        f"organizations/{organization_id}/roles/"
        "munchoOwnerGateHierarchyObservationReaderV1"
    )


def _owner_gate_collector_roles(
    ancestor_chain: Sequence[str],
) -> tuple[str, str, str]:
    return (
        OWNER_GATE_PROJECT_READ_ROLE,
        _owner_gate_ancestor_read_role(ancestor_chain),
        OWNER_GATE_MUTATION_ROLE,
    )


def _owner_gate_collector_role_contracts(
    ancestor_chain: Sequence[str],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    roles = _owner_gate_collector_roles(ancestor_chain)
    return (
        {
            "name": roles[0],
            "title": OWNER_GATE_PROJECT_READ_ROLE_TITLE,
            "description": OWNER_GATE_PROJECT_READ_ROLE_DESCRIPTION,
            "included_permissions": list(
                sorted(owner_gate_foundation.READ_ONLY_IAM_PERMISSIONS)
            ),
        },
        {
            "name": roles[1],
            "title": OWNER_GATE_ANCESTOR_READ_ROLE_TITLE,
            "description": OWNER_GATE_ANCESTOR_READ_ROLE_DESCRIPTION,
            "included_permissions": list(
                sorted(
                    owner_gate_foundation.DIRECT_IAM_ANCESTOR_PERMISSIONS
                )
            ),
        },
        {
            "name": roles[2],
            "title": OWNER_GATE_MUTATION_ROLE_TITLE,
            "description": OWNER_GATE_MUTATION_ROLE_DESCRIPTION,
            "included_permissions": list(
                sorted(owner_gate_foundation.MUTATION_PERMISSIONS)
            ),
        },
    )


def _normalize_role_definition(
    resource: Mapping[str, Any],
    *,
    expected_role: str,
    error: str,
) -> Mapping[str, Any]:
    if not isinstance(resource, Mapping):
        raise TrustedObservationError(error)
    permissions = _string_set(resource.get("includedPermissions"))
    deleted = resource.get("deleted", False)
    etag = resource.get("etag", "")
    if (
        resource.get("name") != expected_role
        or permissions is None
        or resource.get("stage") not in {"ALPHA", "BETA", "GA"}
        or type(deleted) is not bool
        or deleted
        or not isinstance(etag, str)
        or len(etag) > 4096
    ):
        raise TrustedObservationError(error)
    return {
        "name": expected_role,
        "included_permissions": list(permissions),
        "stage": resource["stage"],
        "deleted": deleted,
        "etag": etag,
        "raw_sha256": protocol.sha256_json(resource),
    }


def build_trusted_iam_projection(
    *,
    observation_request: Mapping[str, Any],
    candidate_observation: Mapping[str, Any],
    expected_project_number: str,
    expected_ancestor_chain: Sequence[str],
    expected_runtime_instance_numeric_id: str,
    expected_runtime_service_account_email: str,
    expected_runtime_service_account_unique_id: str,
    expected_target_service_account_unique_id: str,
    metadata_instance_id: str,
    metadata_service_account_email: str,
    metadata_scopes: Sequence[str],
    collector_instance_resource: Mapping[str, Any],
    collector_service_account_resource: Mapping[str, Any],
    collector_service_account_policy: Mapping[str, Any],
    collector_user_managed_keys: Mapping[str, Any],
    project_resource: Mapping[str, Any],
    ancestor_resources: Sequence[Mapping[str, Any]],
    resource_policies: Sequence[Mapping[str, Any]],
    role_resources: Sequence[Mapping[str, Any]],
    collector_role_resources: Sequence[Mapping[str, Any]],
    additional_policy_role_resources: Sequence[Mapping[str, Any]],
    service_account_resource: Mapping[str, Any],
    expected_external_gcp_admin_trust_root: Mapping[str, Any],
    expected_mutation_binding_present: bool,
    activation_seal_sha256: str | None,
    now_unix: int,
) -> Mapping[str, Any]:
    """Normalize exact live IAM responses inside the cloud signer boundary."""

    request = validate_observation_request(observation_request)
    observation = _validated_observation(
        request, candidate_observation, now_unix=now_unix
    )
    candidate_iam = observation["external_iam_receipt"]
    member = f"serviceAccount:{contract.RUNTIME_SERVICE_ACCOUNT}"
    collector_member = (
        f"serviceAccount:{OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT}"
    )
    ancestors = tuple(expected_ancestor_chain)
    if (
        re.fullmatch(r"[1-9][0-9]{5,31}", expected_project_number)
        is None
        or re.fullmatch(
            r"[1-9][0-9]{5,31}",
            expected_runtime_instance_numeric_id,
        )
        is None
        or expected_runtime_service_account_email
        != OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT
        or re.fullmatch(
            r"[1-9][0-9]{5,31}",
            expected_runtime_service_account_unique_id,
        )
        is None
        or re.fullmatch(
            r"[1-9][0-9]{5,31}",
            expected_target_service_account_unique_id,
        )
        is None
        or not ancestors
        or re.fullmatch(r"organizations/[1-9][0-9]{5,31}", ancestors[-1])
        is None
        or any(
            re.fullmatch(r"folders/[1-9][0-9]{5,31}", item) is None
            for item in ancestors[:-1]
        )
        or type(expected_mutation_binding_present) is not bool
        or (
            expected_mutation_binding_present
            and not _sha(activation_seal_sha256)
        )
        or (
            not expected_mutation_binding_present
            and activation_seal_sha256 is not None
        )
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_authority_pin_invalid"
        )
    collector_roles = _owner_gate_collector_roles(ancestors)
    external_gcp_admin_trust_root = validate_external_gcp_admin_trust_root(
        expected_external_gcp_admin_trust_root,
        ancestor_chain=ancestors,
    )
    resources = (project_resource, *ancestor_resources)
    resource_names = (f"projects/{expected_project_number}", *ancestors)
    if len(resources) != len(resource_names):
        raise TrustedObservationError(
            "trusted_observation_iam_hierarchy_invalid"
        )
    hierarchy: list[Mapping[str, Any]] = []
    for index, (expected_name, resource) in enumerate(
        zip(resource_names, resources, strict=True)
    ):
        if not isinstance(resource, Mapping):
            raise TrustedObservationError(
                "trusted_observation_iam_hierarchy_invalid"
            )
        expected_parent = (
            resource_names[index + 1]
            if index + 1 < len(resource_names)
            else None
        )
        parent = resource.get("parent")
        etag = resource.get("etag")
        if (
            resource.get("name") != expected_name
            or resource.get("state") != "ACTIVE"
            or not isinstance(etag, str)
            or not etag
            or len(etag) > 4096
            or (
                expected_parent is not None
                and parent != expected_parent
            )
            or (
                expected_parent is None
                and parent not in {None, ""}
            )
            or (
                index == 0
                and resource.get("projectId") != contract.PROJECT
            )
        ):
            raise TrustedObservationError(
                "trusted_observation_iam_hierarchy_invalid"
            )
        hierarchy.append({
            "name": expected_name,
            "parent": expected_parent,
            "state": "ACTIVE",
            "etag": etag,
            "project_id": (
                contract.PROJECT if index == 0 else None
            ),
            "raw_sha256": protocol.sha256_json(resource),
        })

    if len(resource_policies) != len(resource_names):
        raise TrustedObservationError(
            "trusted_observation_iam_policy_invalid"
        )
    normalized_policies: list[Mapping[str, Any]] = []
    for resource_name, policy in zip(
        resource_names, resource_policies, strict=True
    ):
        if not isinstance(policy, Mapping):
            raise TrustedObservationError(
                "trusted_observation_iam_policy_invalid"
            )
        bindings_value = policy.get("bindings")
        version = policy.get("version", 1)
        etag = policy.get("etag")
        audit_configs = policy.get("auditConfigs", [])
        if (
            not isinstance(bindings_value, list)
            or type(version) is not int
            or version not in {1, 3}
            or not isinstance(etag, str)
            or not etag
            or len(etag) > 4096
            or not isinstance(audit_configs, list)
            or set(policy) - {"version", "bindings", "etag", "auditConfigs"}
        ):
            raise TrustedObservationError(
                "trusted_observation_iam_policy_invalid"
            )
        normalized_bindings: list[Mapping[str, Any]] = []
        for binding in bindings_value:
            members = (
                _string_set(binding.get("members"))
                if isinstance(binding, Mapping)
                else None
            )
            if (
                not isinstance(binding, Mapping)
                or not isinstance(binding.get("role"), str)
                or members is None
                or set(binding) - {"role", "members", "condition"}
            ):
                raise TrustedObservationError(
                    "trusted_observation_iam_policy_invalid"
                )
            for policy_member in members:
                if (
                    not _valid_policy_member(policy_member)
                    or (
                        contract.RUNTIME_SERVICE_ACCOUNT in policy_member
                        and policy_member != member
                    )
                    or (
                        OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT
                        in policy_member
                        and policy_member != collector_member
                    )
                ):
                    raise TrustedObservationError(
                        "trusted_observation_iam_ambiguous_principal"
                    )
            if resource_name != resource_names[0] and member in members:
                raise TrustedObservationError(
                    "trusted_observation_iam_inherited_binding_drift"
                )
            condition = _normalized_condition(
                binding.get("condition"),
                error="trusted_observation_iam_policy_invalid",
            )
            normalized_binding = {
                "role": binding["role"],
                "members": list(members),
                "condition": condition,
            }
            normalized_bindings.append(normalized_binding)
        normalized_bindings.sort(
            key=lambda item: protocol.canonical_json_bytes(item)
        )
        normalized_policies.append({
            "resource": resource_name,
            "version": version,
            "etag": etag,
            "bindings": normalized_bindings,
            "audit_configs": _copy_json(audit_configs),
            "raw_sha256": protocol.sha256_json(policy),
        })

    project_policy = resource_policies[0]
    bindings = project_policy.get("bindings")
    policy_version = project_policy.get("version", 1)
    policy_etag = project_policy.get("etag")
    if (
        not isinstance(bindings, list)
        or type(policy_version) is not int
        or policy_version not in {1, 3}
        or not isinstance(policy_etag, str)
        or not policy_etag
        or len(policy_etag) > 4096
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_policy_invalid"
        )
    relevant_by_role: dict[str, Mapping[str, Any]] = {}
    for binding in bindings:
        if (
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("role"), str)
            or _string_set(binding.get("members")) is None
            or set(binding) - {"role", "members", "condition"}
        ):
            raise TrustedObservationError(
                "trusted_observation_iam_policy_invalid"
            )
        members = _string_set(binding["members"])
        assert members is not None
        if member not in members:
            continue
        role = binding["role"]
        if (
            role not in contract.RUNTIME_ROLES
            or role in relevant_by_role
            or binding.get("condition") is not None
        ):
            raise TrustedObservationError(
                "trusted_observation_iam_binding_drift"
            )
        relevant_by_role[role] = {
            "role": role,
            "members": list(members),
            "condition": None,
        }
    if tuple(relevant_by_role) != contract.RUNTIME_ROLES:
        # The API does not guarantee binding order; normalize only after
        # proving the exact role set and exactly one binding per role.
        if set(relevant_by_role) != set(contract.RUNTIME_ROLES):
            raise TrustedObservationError(
                "trusted_observation_iam_binding_drift"
            )
    relevant_bindings = [
        relevant_by_role[role] for role in contract.RUNTIME_ROLES
    ]

    if len(role_resources) != len(contract.RUNTIME_ROLES):
        raise TrustedObservationError(
            "trusted_observation_iam_role_invalid"
        )
    definitions: list[Mapping[str, Any]] = []
    permission_union: set[str] = set()
    for expected_role, resource in zip(
        contract.RUNTIME_ROLES, role_resources, strict=True
    ):
        definition = _normalize_role_definition(
            resource,
            expected_role=expected_role,
            error="trusted_observation_iam_role_invalid",
        )
        permission_union.update(definition["included_permissions"])
        definitions.append(definition)
    if tuple(sorted(permission_union)) != tuple(
        sorted(contract.RUNTIME_PERMISSIONS)
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_permission_drift"
        )

    if len(collector_role_resources) != len(collector_roles):
        raise TrustedObservationError(
            "trusted_observation_iam_collector_role_invalid"
        )
    collector_role_contracts = _owner_gate_collector_role_contracts(
        ancestors
    )
    collector_definitions: list[Mapping[str, Any]] = []
    for role_contract, resource in zip(
        collector_role_contracts,
        collector_role_resources,
        strict=True,
    ):
        definition = _normalize_role_definition(
            resource,
            expected_role=str(role_contract["name"]),
            error="trusted_observation_iam_collector_role_invalid",
        )
        if (
            definition["included_permissions"]
            != role_contract["included_permissions"]
            or definition["stage"] != "GA"
            or resource.get("title") != role_contract["title"]
            or resource.get("description")
            != role_contract["description"]
        ):
            raise TrustedObservationError(
                "trusted_observation_iam_collector_permission_drift"
            )
        collector_definitions.append({
            **definition,
            "title": role_contract["title"],
            "description": role_contract["description"],
        })

    policy_role_names = _bounded_policy_role_names(
        resource_policies,
        ancestor_chain=ancestors,
    )
    known_role_resources: dict[str, Mapping[str, Any]] = {}
    for resource in (*role_resources, *collector_role_resources):
        name = resource.get("name") if isinstance(resource, Mapping) else None
        if (
            not isinstance(name, str)
            or name in known_role_resources
            or not _policy_role_allowed(name, ancestor_chain=ancestors)
        ):
            raise TrustedObservationError(
                "trusted_observation_iam_policy_role_definition_invalid"
            )
        known_role_resources[name] = resource
    additional_by_name: dict[str, Mapping[str, Any]] = {}
    for resource in additional_policy_role_resources:
        name = resource.get("name") if isinstance(resource, Mapping) else None
        if (
            not isinstance(name, str)
            or name in additional_by_name
            or name in known_role_resources
            or not _policy_role_allowed(name, ancestor_chain=ancestors)
        ):
            raise TrustedObservationError(
                "trusted_observation_iam_policy_role_definition_invalid"
            )
        additional_by_name[name] = resource
    expected_additional_roles = (
        set(policy_role_names) - set(known_role_resources)
    )
    if set(additional_by_name) != expected_additional_roles:
        raise TrustedObservationError(
            "trusted_observation_iam_policy_role_inventory_invalid"
        )
    all_role_resources = {**known_role_resources, **additional_by_name}
    policy_role_definitions = [
        _normalize_policy_role_definition(
            all_role_resources[role],
            expected_role=role,
            ancestor_chain=ancestors,
        )
        for role in policy_role_names
    ]
    policy_definition_by_name = {
        str(definition["name"]): definition
        for definition in policy_role_definitions
    }

    expected_collector_binding_keys: dict[
        str,
        tuple[str, str, Mapping[str, str] | None],
    ] = {
        "project_read": (
            resource_names[0],
            collector_roles[0],
            None,
        ),
        "ancestor_read": (
            resource_names[-1],
            collector_roles[1],
            None,
        ),
    }
    if expected_mutation_binding_present:
        expected_collector_binding_keys["mutation"] = (
            resource_names[0],
            collector_roles[2],
            OWNER_GATE_MUTATION_CONDITION,
        )
    collector_binding_by_name: dict[str, Mapping[str, Any]] = {}
    for policy in normalized_policies:
        for binding in policy["bindings"]:
            if collector_member not in binding["members"]:
                continue
            matches = [
                name
                for name, (resource_name, role, condition) in (
                    expected_collector_binding_keys.items()
                )
                if policy["resource"] == resource_name
                and binding["role"] == role
                and binding["condition"] == condition
                and binding["members"] == [collector_member]
            ]
            if len(matches) != 1 or matches[0] in collector_binding_by_name:
                raise TrustedObservationError(
                    "trusted_observation_iam_collector_binding_drift"
                )
            collector_binding_by_name[matches[0]] = {
                "resource": policy["resource"],
                **binding,
            }
    if set(collector_binding_by_name) != set(
        expected_collector_binding_keys
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_collector_binding_drift"
        )
    collector_bindings = [
        collector_binding_by_name[name]
        for name in ("project_read", "ancestor_read", "mutation")
        if name in collector_binding_by_name
    ]

    internal_binding_bytes = {
        protocol.canonical_json_bytes({
            **binding,
            "resource": (
                f"projects/{contract.PROJECT}"
                if binding["resource"] == resource_names[0]
                else binding["resource"]
            ),
        })
        for binding in collector_bindings
    }
    residual_external_bindings: list[Mapping[str, Any]] = []
    for policy in normalized_policies:
        external_resource = (
            f"projects/{contract.PROJECT}"
            if policy["resource"] == resource_names[0]
            else policy["resource"]
        )
        for binding in policy["bindings"]:
            if policy_definition_by_name.get(binding["role"]) is None:
                raise TrustedObservationError(
                    "trusted_observation_iam_policy_role_inventory_invalid"
                )
            external_binding = {
                "resource": external_resource,
                **binding,
            }
            if protocol.canonical_json_bytes(external_binding) in internal_binding_bytes:
                continue
            residual_external_bindings.append(external_binding)
    residual_external_bindings.sort(key=protocol.canonical_json_bytes)
    residual_role_names = sorted({
        str(binding["role"]) for binding in residual_external_bindings
    })
    residual_role_definitions = [
        _semantic_role_definition(policy_definition_by_name[role])
        for role in residual_role_names
    ]
    resource_policy_generations = [
        {
            "resource": (
                f"projects/{contract.PROJECT}"
                if policy["resource"] == resource_names[0]
                else policy["resource"]
            ),
            "version": policy["version"],
            "etag": policy["etag"],
            "audit_configs": _copy_json(policy["audit_configs"]),
        }
        for policy in normalized_policies
    ]
    if (
        residual_external_bindings
        != external_gcp_admin_trust_root["allowed_residual_bindings"]
        or residual_role_definitions
        != external_gcp_admin_trust_root[
            "allowed_residual_role_definitions"
        ]
        or resource_policy_generations
        != external_gcp_admin_trust_root["resource_policy_generations"]
    ):
        raise TrustedObservationError(
            "trusted_observation_external_gcp_admin_trust_root_drift"
        )

    collector_instance = collector_instance_resource
    if not isinstance(collector_instance, Mapping):
        raise TrustedObservationError(
            "trusted_observation_iam_collector_identity_invalid"
        )
    collector_accounts = collector_instance.get("serviceAccounts")
    collector_account_entry = (
        collector_accounts[0]
        if isinstance(collector_accounts, list)
        and len(collector_accounts) == 1
        and isinstance(collector_accounts[0], Mapping)
        else None
    )
    collector_account = collector_service_account_resource
    collector_account_name = (
        f"projects/{contract.PROJECT}/serviceAccounts/"
        f"{OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT}"
    )
    metadata_scope_tuple = _string_set(list(metadata_scopes))
    collector_scopes = (
        _string_set(collector_account_entry.get("scopes"))
        if collector_account_entry is not None
        else None
    )
    expected_scope_set = tuple(sorted(OWNER_GATE_METADATA_SCOPES))
    if (
        metadata_instance_id != expected_runtime_instance_numeric_id
        or metadata_service_account_email
        != expected_runtime_service_account_email
        or metadata_scope_tuple != expected_scope_set
        or collector_instance.get("id")
        != expected_runtime_instance_numeric_id
        or collector_instance.get("name") != OWNER_GATE_VM_NAME
        or collector_instance.get("status") != "RUNNING"
        or not _url_suffix(
            collector_instance.get("zone"), f"/zones/{contract.ZONE}"
        )
        or collector_account_entry is None
        or collector_scopes is None
        or collector_account_entry.get("email")
        != expected_runtime_service_account_email
        or collector_scopes != expected_scope_set
        or not isinstance(collector_account, Mapping)
        or collector_account.get("name") != collector_account_name
        or collector_account.get("projectId") != contract.PROJECT
        or collector_account.get("email")
        != expected_runtime_service_account_email
        or collector_account.get("uniqueId")
        != expected_runtime_service_account_unique_id
        or collector_account.get("disabled", False) is not False
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_collector_identity_invalid"
        )
    policy_value = collector_service_account_policy
    if not isinstance(policy_value, Mapping):
        raise TrustedObservationError(
            "trusted_observation_iam_collector_impersonation_invalid"
        )
    policy_bindings = policy_value.get("bindings", [])
    policy_audit_configs = policy_value.get("auditConfigs", [])
    service_account_policy_version = policy_value.get("version", 1)
    service_account_policy_etag = policy_value.get("etag", "")
    if (
        set(policy_value) - {
            "version", "bindings", "etag", "auditConfigs"
        }
        or type(service_account_policy_version) is not int
        or service_account_policy_version not in {1, 3}
        or not isinstance(policy_bindings, list)
        or policy_bindings
        or not isinstance(policy_audit_configs, list)
        or policy_audit_configs
        or not isinstance(service_account_policy_etag, str)
        or len(service_account_policy_etag) > 4096
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_collector_impersonation_invalid"
        )
    collector_impersonation_policy = {
        "version": service_account_policy_version,
        "etag": service_account_policy_etag,
        "bindings": [],
        "audit_configs": [],
        "allowed_impersonation_bindings": [],
        "raw_sha256": protocol.sha256_json(policy_value),
    }
    keys_value = collector_user_managed_keys
    if (
        not isinstance(keys_value, Mapping)
        or set(keys_value) - {"keys"}
        or not isinstance(keys_value.get("keys", []), list)
        or keys_value.get("keys", [])
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_collector_user_keys_invalid"
        )
    collector_key_snapshot = {
        "query_key_types": ["USER_MANAGED"],
        "user_managed_key_count": 0,
        "system_managed_keys_are_provider_managed": True,
        "raw_sha256": protocol.sha256_json(keys_value),
    }
    collector_runtime_identity = {
        "instance_id": expected_runtime_instance_numeric_id,
        "instance_name": OWNER_GATE_VM_NAME,
        "service_account": expected_runtime_service_account_email,
        "service_account_unique_id": (
            expected_runtime_service_account_unique_id
        ),
        "scopes": list(OWNER_GATE_METADATA_SCOPES),
        "metadata_instance_id_sha256": protocol.sha256_bytes(
            metadata_instance_id.encode("utf-8")
        ),
        "metadata_service_account_email_sha256": protocol.sha256_bytes(
            metadata_service_account_email.encode("utf-8")
        ),
        "metadata_scopes_sha256": protocol.sha256_json(
            list(OWNER_GATE_METADATA_SCOPES)
        ),
        "compute_instance_sha256": protocol.sha256_json(
            collector_instance
        ),
        "iam_service_account_sha256": protocol.sha256_json(
            collector_account
        ),
        "service_account_iam_policy_sha256": protocol.sha256_json(
            collector_impersonation_policy
        ),
        "user_managed_key_snapshot_sha256": protocol.sha256_json(
            collector_key_snapshot
        ),
    }

    account = service_account_resource
    account_name = (
        f"projects/{contract.PROJECT}/serviceAccounts/"
        f"{contract.RUNTIME_SERVICE_ACCOUNT}"
    )
    if (
        not isinstance(account, Mapping)
        or account.get("name") != account_name
        or account.get("projectId") != contract.PROJECT
        or account.get("email") != contract.RUNTIME_SERVICE_ACCOUNT
        or not isinstance(account.get("uniqueId"), str)
        or account["uniqueId"]
        != expected_target_service_account_unique_id
        or account.get("disabled", False) is not False
        or not isinstance(account.get("oauth2ClientId", ""), str)
        or (
            account.get("oauth2ClientId", "")
            and not account["oauth2ClientId"].isdigit()
        )
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_service_account_invalid"
        )
    account_projection = {
        "name": account_name,
        "project_id": contract.PROJECT,
        "unique_id": account["uniqueId"],
        "email": contract.RUNTIME_SERVICE_ACCOUNT,
        "disabled": False,
        "oauth2_client_id": account.get("oauth2ClientId", ""),
        "raw_sha256": protocol.sha256_json(account),
    }
    target_runtime_authority = {
        "schema": TARGET_RUNTIME_AUTHORITY_SCHEMA,
        "member": member,
        "service_account": contract.RUNTIME_SERVICE_ACCOUNT,
        "service_account_resource_sha256": protocol.sha256_json(
            account_projection
        ),
        "direct_binding_resource": resource_names[0],
        "roles": list(contract.RUNTIME_ROLES),
        "permissions": list(contract.RUNTIME_PERMISSIONS),
        "relevant_bindings_sha256": protocol.sha256_json(
            relevant_bindings
        ),
        "role_definitions_sha256": protocol.sha256_json(definitions),
    }
    collector_service_account_authority = {
        "schema": COLLECTOR_AUTHORITY_SCHEMA,
        "member": collector_member,
        "service_account": OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT,
        "runtime_identity_sha256": protocol.sha256_json(
            collector_runtime_identity
        ),
        "project_read_role": collector_roles[0],
        "ancestor_read_role": collector_roles[1],
        "mutation_role": collector_roles[2],
        "mutation_binding_present": expected_mutation_binding_present,
        "mutation_condition": (
            dict(OWNER_GATE_MUTATION_CONDITION)
            if expected_mutation_binding_present
            else None
        ),
        "activation_seal_sha256": activation_seal_sha256,
        "service_account_iam_policy": collector_impersonation_policy,
        "service_account_iam_policy_sha256": protocol.sha256_json(
            collector_impersonation_policy
        ),
        "credential_key_snapshot": collector_key_snapshot,
        "credential_key_snapshot_sha256": protocol.sha256_json(
            collector_key_snapshot
        ),
        "bindings": collector_bindings,
        "bindings_sha256": protocol.sha256_json(collector_bindings),
        "role_definitions": collector_definitions,
        "role_definitions_sha256": protocol.sha256_json(
            collector_definitions
        ),
        "project_read_permissions": list(
            owner_gate_foundation.READ_ONLY_IAM_PERMISSIONS
        ),
        "ancestor_read_permissions": list(
            owner_gate_foundation.DIRECT_IAM_ANCESTOR_PERMISSIONS
        ),
        "mutation_permissions": list(
            owner_gate_foundation.MUTATION_PERMISSIONS
        ),
    }
    if (
        _iam_policy_sha256(candidate_iam)
        != contract.EXTERNAL_IAM_POLICY_SHA256
        or observation["external_iam_policy_sha256"]
        != contract.EXTERNAL_IAM_POLICY_SHA256
        or candidate_iam.get("project") != contract.PROJECT
        or candidate_iam.get("zone") != contract.ZONE
        or candidate_iam.get("instance") != contract.VM_NAME
        or candidate_iam.get("service_account")
        != contract.RUNTIME_SERVICE_ACCOUNT
        or tuple(candidate_iam.get("scopes") or ())
        != contract.RUNTIME_SCOPES
        or tuple(candidate_iam.get("roles") or ())
        != contract.RUNTIME_ROLES
        or tuple(candidate_iam.get("permissions") or ())
        != contract.RUNTIME_PERMISSIONS
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_candidate_drift"
        )
    expires_at = min(
        now_unix + TRUSTED_IAM_TTL_SECONDS,
        request["collection_attempt_expires_at_unix"],
        candidate_iam["expires_at_unix"],
    )
    if expires_at <= now_unix:
        raise TrustedObservationError(
            "trusted_observation_iam_projection_expired"
        )
    body = {
        "schema": TRUSTED_IAM_PROJECTION_SCHEMA,
        "authority_scope": TRUSTED_IAM_AUTHORITY_SCOPE,
        "transaction_id": request["transaction_id"],
        "checkpoint": request["checkpoint"],
        "observation_request_sha256": request[
            "observation_request_sha256"
        ],
        "collection_attempt_id": request["collection_attempt_id"],
        "collection_attempt_sequence": request[
            "collection_attempt_sequence"
        ],
        "collection_attempt_issued_at_unix": request[
            "collection_attempt_issued_at_unix"
        ],
        "collection_attempt_expires_at_unix": request[
            "collection_attempt_expires_at_unix"
        ],
        "candidate_external_iam_receipt_sha256": observation[
            "external_iam_receipt_sha256"
        ],
        "candidate_external_iam_policy_sha256": observation[
            "external_iam_policy_sha256"
        ],
        "project": contract.PROJECT,
        "project_number": expected_project_number,
        "zone": contract.ZONE,
        "instance": contract.VM_NAME,
        "service_account": contract.RUNTIME_SERVICE_ACCOUNT,
        "scopes": list(contract.RUNTIME_SCOPES),
        "member": member,
        "roles": list(contract.RUNTIME_ROLES),
        "permissions": list(contract.RUNTIME_PERMISSIONS),
        "project_policy_version": policy_version,
        "project_policy_etag": policy_etag,
        "project_policy_sha256": protocol.sha256_json(project_policy),
        "resource_ancestor_chain": list(ancestors),
        "resource_hierarchy": hierarchy,
        "resource_hierarchy_sha256": protocol.sha256_json(hierarchy),
        "resource_policies": normalized_policies,
        "resource_policies_sha256": protocol.sha256_json(
            normalized_policies
        ),
        "relevant_bindings": relevant_bindings,
        "relevant_bindings_sha256": protocol.sha256_json(
            relevant_bindings
        ),
        "role_definitions": definitions,
        "role_definitions_sha256": protocol.sha256_json(definitions),
        "policy_role_definitions": policy_role_definitions,
        "policy_role_definitions_sha256": protocol.sha256_json(
            policy_role_definitions
        ),
        "residual_external_bindings": residual_external_bindings,
        "residual_external_bindings_sha256": protocol.sha256_json(
            residual_external_bindings
        ),
        "service_account_resource": account_projection,
        "service_account_resource_sha256": protocol.sha256_json(
            account_projection
        ),
        "collector_runtime_identity": collector_runtime_identity,
        "collector_runtime_identity_sha256": protocol.sha256_json(
            collector_runtime_identity
        ),
        "target_runtime_authority": target_runtime_authority,
        "target_runtime_authority_sha256": protocol.sha256_json(
            target_runtime_authority
        ),
        "collector_service_account_authority": (
            collector_service_account_authority
        ),
        "collector_service_account_authority_sha256": protocol.sha256_json(
            collector_service_account_authority
        ),
        "external_gcp_admin_trust_root": external_gcp_admin_trust_root,
        "external_gcp_admin_trust_root_sha256": protocol.sha256_json(
            external_gcp_admin_trust_root
        ),
        "collected_at_unix": now_unix,
        "expires_at_unix": expires_at,
    }
    return {**body, "projection_sha256": protocol.sha256_json(body)}


def validate_trusted_iam_projection(
    value: Any,
    *,
    observation_request: Mapping[str, Any],
    candidate_observation: Mapping[str, Any],
    now_unix: int,
    expected_project_number: str | None = None,
    expected_ancestor_chain: Sequence[str] | None = None,
    expected_runtime_instance_numeric_id: str | None = None,
    expected_runtime_service_account_email: str | None = None,
    expected_runtime_service_account_unique_id: str | None = None,
    expected_target_service_account_unique_id: str | None = None,
    expected_mutation_binding_present: bool | None = None,
    expected_activation_seal_sha256: str | None = None,
    expected_external_gcp_admin_trust_root: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate the exact attempt-bound IAM projection sealed by cloud."""

    request = validate_observation_request(observation_request)
    observation = _validated_observation(
        request, candidate_observation, now_unix=now_unix
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != _TRUSTED_IAM_PROJECTION_FIELDS
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_projection_fields_invalid"
        )
    projection = _copy_json(value)
    candidate_iam = observation["external_iam_receipt"]
    collected_at = projection.get("collected_at_unix")
    expires_at = projection.get("expires_at_unix")
    member = f"serviceAccount:{contract.RUNTIME_SERVICE_ACCOUNT}"
    relevant = projection.get("relevant_bindings")
    definitions = projection.get("role_definitions")
    policy_definitions = projection.get("policy_role_definitions")
    residual_external_bindings = projection.get(
        "residual_external_bindings"
    )
    account = projection.get("service_account_resource")
    project_number = projection.get("project_number")
    ancestors = projection.get("resource_ancestor_chain")
    hierarchy = projection.get("resource_hierarchy")
    policies = projection.get("resource_policies")
    collector_identity = projection.get("collector_runtime_identity")
    target_authority = projection.get("target_runtime_authority")
    collector_authority = projection.get(
        "collector_service_account_authority"
    )
    if (
        type(collected_at) is not int
        or type(expires_at) is not int
        or not isinstance(relevant, list)
        or len(relevant) != len(contract.RUNTIME_ROLES)
        or not isinstance(definitions, list)
        or len(definitions) != len(contract.RUNTIME_ROLES)
        or not isinstance(policy_definitions, list)
        or not policy_definitions
        or len(policy_definitions) > MAX_POLICY_ROLE_DEFINITIONS
        or not isinstance(residual_external_bindings, list)
        or not isinstance(account, Mapping)
        or not isinstance(project_number, str)
        or re.fullmatch(r"[1-9][0-9]{5,31}", project_number) is None
        or not isinstance(ancestors, list)
        or not ancestors
        or re.fullmatch(
            r"organizations/[1-9][0-9]{5,31}", ancestors[-1]
        )
        is None
        or any(
            not isinstance(item, str)
            or re.fullmatch(r"folders/[1-9][0-9]{5,31}", item)
            is None
            for item in ancestors[:-1]
        )
        or not isinstance(hierarchy, list)
        or not isinstance(policies, list)
        or len(hierarchy) != len(ancestors) + 1
        or len(policies) != len(hierarchy)
        or not isinstance(collector_identity, Mapping)
        or not isinstance(target_authority, Mapping)
        or not isinstance(collector_authority, Mapping)
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_projection_invalid"
        )
    external_gcp_admin_trust_root = validate_external_gcp_admin_trust_root(
        projection.get("external_gcp_admin_trust_root"),
        ancestor_chain=ancestors,
    )
    for expected_role, binding, definition in zip(
        contract.RUNTIME_ROLES, relevant, definitions, strict=True
    ):
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"role", "members", "condition"}
            or binding.get("role") != expected_role
            or binding.get("condition") is not None
            or _string_set(binding.get("members")) is None
            or member not in binding["members"]
            or not isinstance(definition, Mapping)
            or set(definition)
            != {
                "name", "included_permissions", "stage", "deleted",
                "etag", "raw_sha256",
            }
            or definition.get("name") != expected_role
            or _string_set(definition.get("included_permissions")) is None
            or definition.get("stage") not in {"ALPHA", "BETA", "GA"}
            or definition.get("deleted") is not False
            or (
                definition.get("etag") is not None
                and not isinstance(definition.get("etag"), str)
            )
            or not _sha(definition.get("raw_sha256"))
        ):
            raise TrustedObservationError(
                "trusted_observation_iam_projection_invalid"
            )
    expanded_permissions = {
        permission
        for definition in definitions
        for permission in definition["included_permissions"]
    }
    policy_definition_by_name: dict[str, Mapping[str, Any]] = {}
    previous_policy_definition_name: str | None = None
    for definition in policy_definitions:
        if (
            not isinstance(definition, Mapping)
            or set(definition)
            != {
                "name", "title", "description", "included_permissions",
                "stage", "deleted", "etag", "raw_sha256",
            }
            or not _policy_role_allowed(
                definition.get("name"), ancestor_chain=ancestors
            )
            or not isinstance(definition.get("title"), str)
            or not definition["title"]
            or not isinstance(definition.get("description"), str)
            or _string_set(definition.get("included_permissions")) is None
            or definition["included_permissions"]
            != list(_string_set(definition["included_permissions"]) or ())
            or definition.get("stage") not in _ROLE_STAGES
            or definition.get("deleted") is not False
            or (
                definition.get("etag") is not None
                and (
                    not isinstance(definition.get("etag"), str)
                    or not definition["etag"]
                )
            )
            or not _sha(definition.get("raw_sha256"))
        ):
            raise TrustedObservationError(
                "trusted_observation_iam_policy_role_definition_invalid"
            )
        name = str(definition["name"])
        if (
            previous_policy_definition_name is not None
            and name <= previous_policy_definition_name
        ):
            raise TrustedObservationError(
                "trusted_observation_iam_policy_role_definition_invalid"
            )
        previous_policy_definition_name = name
        policy_definition_by_name[name] = definition
    expected_account_name = (
        f"projects/{contract.PROJECT}/serviceAccounts/"
        f"{contract.RUNTIME_SERVICE_ACCOUNT}"
    )
    resource_names = (f"projects/{project_number}", *ancestors)
    collector_member = (
        f"serviceAccount:{OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT}"
    )
    collector_roles = _owner_gate_collector_roles(ancestors)
    direct_target_bindings: dict[str, Mapping[str, Any]] = {}
    direct_collector_bindings: dict[str, Mapping[str, Any]] = {}
    all_binding_roles: set[str] = set()
    for index, (resource_name, resource, policy) in enumerate(
        zip(resource_names, hierarchy, policies, strict=True)
    ):
        expected_parent = (
            resource_names[index + 1]
            if index + 1 < len(resource_names)
            else None
        )
        if (
            not isinstance(resource, Mapping)
            or set(resource)
            != {
                "name", "parent", "state", "etag", "project_id",
                "raw_sha256",
            }
            or resource.get("name") != resource_name
            or resource.get("parent") != expected_parent
            or resource.get("state") != "ACTIVE"
            or not isinstance(resource.get("etag"), str)
            or not resource["etag"]
            or resource.get("project_id")
            != (contract.PROJECT if index == 0 else None)
            or not _sha(resource.get("raw_sha256"))
            or not isinstance(policy, Mapping)
            or set(policy)
            != {
                "resource", "version", "etag", "bindings",
                "audit_configs", "raw_sha256",
            }
            or policy.get("resource") != resource_name
            or policy.get("version") not in {1, 3}
            or not isinstance(policy.get("etag"), str)
            or not policy["etag"]
            or not isinstance(policy.get("bindings"), list)
            or not isinstance(policy.get("audit_configs"), list)
            or not _sha(policy.get("raw_sha256"))
        ):
            raise TrustedObservationError(
                "trusted_observation_iam_projection_invalid"
            )
        previous: bytes | None = None
        for binding in policy["bindings"]:
            if (
                not isinstance(binding, Mapping)
                or set(binding) != {"role", "members", "condition"}
                or not isinstance(binding.get("role"), str)
                or _string_set(binding.get("members")) is None
                or binding["members"]
                != list(_string_set(binding["members"]) or ())
                or (
                    binding.get("condition") is not None
                    and _normalized_condition(
                        binding.get("condition"),
                        error="trusted_observation_iam_projection_invalid",
                    )
                    != binding.get("condition")
                )
            ):
                raise TrustedObservationError(
                    "trusted_observation_iam_projection_invalid"
                )
            encoded = protocol.canonical_json_bytes(binding)
            if previous is not None and encoded <= previous:
                raise TrustedObservationError(
                    "trusted_observation_iam_projection_invalid"
                )
            previous = encoded
            all_binding_roles.add(str(binding["role"]))
            for policy_member in binding["members"]:
                if (
                    not _valid_policy_member(policy_member)
                    or (
                        contract.RUNTIME_SERVICE_ACCOUNT in policy_member
                        and policy_member != member
                    )
                    or (
                        OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT
                        in policy_member
                        and policy_member != collector_member
                    )
                ):
                    raise TrustedObservationError(
                        "trusted_observation_iam_ambiguous_principal"
                    )
            if member in binding["members"]:
                if index != 0:
                    raise TrustedObservationError(
                        "trusted_observation_iam_inherited_binding_drift"
                    )
                role = binding["role"]
                if (
                    role not in contract.RUNTIME_ROLES
                    or role in direct_target_bindings
                    or binding.get("condition") is not None
                ):
                    raise TrustedObservationError(
                        "trusted_observation_iam_binding_drift"
                    )
                direct_target_bindings[role] = binding
            if collector_member in binding["members"]:
                binding_key: str | None = None
                if (
                    index == 0
                    and binding["role"] == collector_roles[0]
                    and binding.get("condition") is None
                    and binding["members"] == [collector_member]
                ):
                    binding_key = "project_read"
                elif (
                    index == len(resource_names) - 1
                    and binding["role"] == collector_roles[1]
                    and binding.get("condition") is None
                    and binding["members"] == [collector_member]
                ):
                    binding_key = "ancestor_read"
                elif (
                    index == 0
                    and binding["role"] == collector_roles[2]
                    and binding.get("condition")
                    == OWNER_GATE_MUTATION_CONDITION
                    and binding["members"] == [collector_member]
                ):
                    binding_key = "mutation"
                if (
                    binding_key is None
                    or binding_key in direct_collector_bindings
                ):
                    raise TrustedObservationError(
                        "trusted_observation_iam_collector_binding_drift"
                    )
                direct_collector_bindings[binding_key] = {
                    "resource": resource_name,
                    **binding,
                }
    if set(policy_definition_by_name) != all_binding_roles:
        raise TrustedObservationError(
            "trusted_observation_iam_policy_role_inventory_invalid"
        )
    reconstructed_relevant = [
        direct_target_bindings.get(role)
        for role in contract.RUNTIME_ROLES
    ]
    if (
        any(item is None for item in reconstructed_relevant)
        or reconstructed_relevant != relevant
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_binding_drift"
        )
    mutation_binding_present = collector_authority.get(
        "mutation_binding_present"
    )
    if type(mutation_binding_present) is not bool:
        raise TrustedObservationError(
            "trusted_observation_iam_collector_authority_invalid"
        )
    if expected_external_gcp_admin_trust_root is not None:
        expected_external_root = validate_external_gcp_admin_trust_root(
            expected_external_gcp_admin_trust_root,
            ancestor_chain=ancestors,
        )
        if external_gcp_admin_trust_root != expected_external_root:
            raise TrustedObservationError(
                "trusted_observation_external_gcp_admin_trust_root_drift"
            )
    internal_binding_bytes = {
        protocol.canonical_json_bytes({
            **binding,
            "resource": (
                f"projects/{contract.PROJECT}"
                if binding["resource"] == resource_names[0]
                else binding["resource"]
            ),
        })
        for binding in direct_collector_bindings.values()
    }
    reconstructed_residual_bindings: list[Mapping[str, Any]] = []
    for policy in policies:
        external_resource = (
            f"projects/{contract.PROJECT}"
            if policy["resource"] == resource_names[0]
            else policy["resource"]
        )
        for binding in policy["bindings"]:
            external_binding = {
                "resource": external_resource,
                **binding,
            }
            if protocol.canonical_json_bytes(external_binding) in internal_binding_bytes:
                continue
            reconstructed_residual_bindings.append(external_binding)
    reconstructed_residual_bindings.sort(key=protocol.canonical_json_bytes)
    residual_role_names = sorted({
        str(binding["role"])
        for binding in reconstructed_residual_bindings
    })
    reconstructed_residual_definitions = [
        _semantic_role_definition(policy_definition_by_name[role])
        for role in residual_role_names
    ]
    reconstructed_policy_generations = [
        {
            "resource": (
                f"projects/{contract.PROJECT}"
                if policy["resource"] == resource_names[0]
                else policy["resource"]
            ),
            "version": policy["version"],
            "etag": policy["etag"],
            "audit_configs": _copy_json(policy["audit_configs"]),
        }
        for policy in policies
    ]
    if (
        residual_external_bindings != reconstructed_residual_bindings
        or residual_external_bindings
        != external_gcp_admin_trust_root["allowed_residual_bindings"]
        or reconstructed_residual_definitions
        != external_gcp_admin_trust_root[
            "allowed_residual_role_definitions"
        ]
        or reconstructed_policy_generations
        != external_gcp_admin_trust_root["resource_policy_generations"]
        or projection.get("residual_external_bindings_sha256")
        != protocol.sha256_json(residual_external_bindings)
        or projection.get("policy_role_definitions_sha256")
        != protocol.sha256_json(policy_definitions)
    ):
        raise TrustedObservationError(
            "trusted_observation_external_gcp_admin_trust_root_drift"
        )
    expected_collector_binding_names = [
        "project_read", "ancestor_read",
    ]
    if mutation_binding_present:
        expected_collector_binding_names.append("mutation")
    reconstructed_collector_bindings = [
        direct_collector_bindings.get(name)
        for name in expected_collector_binding_names
    ]
    if (
        any(item is None for item in reconstructed_collector_bindings)
        or set(direct_collector_bindings)
        != set(expected_collector_binding_names)
        or reconstructed_collector_bindings
        != collector_authority.get("bindings")
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_collector_binding_drift"
        )
    collector_definitions = collector_authority.get("role_definitions")
    collector_role_contracts = _owner_gate_collector_role_contracts(
        ancestors
    )
    if (
        not isinstance(collector_definitions, list)
        or len(collector_definitions) != len(collector_roles)
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_collector_authority_invalid"
        )
    for role_contract, definition in zip(
        collector_role_contracts,
        collector_definitions,
        strict=True,
    ):
        if (
            not isinstance(definition, Mapping)
            or set(definition)
            != {
                "name", "included_permissions", "stage", "deleted",
                "etag", "raw_sha256", "title", "description",
            }
            or definition.get("name") != role_contract["name"]
            or definition.get("included_permissions")
            != role_contract["included_permissions"]
            or definition.get("title") != role_contract["title"]
            or definition.get("description")
            != role_contract["description"]
            or definition.get("stage") != "GA"
            or definition.get("deleted") is not False
            or not isinstance(definition.get("etag"), str)
            or not _sha(definition.get("raw_sha256"))
        ):
            raise TrustedObservationError(
                "trusted_observation_iam_collector_authority_invalid"
            )
    impersonation_policy = collector_authority.get(
        "service_account_iam_policy"
    )
    key_snapshot = collector_authority.get("credential_key_snapshot")
    if (
        not isinstance(impersonation_policy, Mapping)
        or set(impersonation_policy)
        != {
            "version", "etag", "bindings", "audit_configs",
            "allowed_impersonation_bindings", "raw_sha256",
        }
        or impersonation_policy.get("version") not in {1, 3}
        or not isinstance(impersonation_policy.get("etag"), str)
        or impersonation_policy.get("bindings") != []
        or impersonation_policy.get("audit_configs") != []
        or impersonation_policy.get("allowed_impersonation_bindings")
        != []
        or not _sha(impersonation_policy.get("raw_sha256"))
        or not isinstance(key_snapshot, Mapping)
        or key_snapshot
        != {
            "query_key_types": ["USER_MANAGED"],
            "user_managed_key_count": 0,
            "system_managed_keys_are_provider_managed": True,
            "raw_sha256": key_snapshot.get("raw_sha256"),
        }
        or not _sha(key_snapshot.get("raw_sha256"))
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_collector_credentials_invalid"
        )
    expected_target_authority = {
        "schema": TARGET_RUNTIME_AUTHORITY_SCHEMA,
        "member": member,
        "service_account": contract.RUNTIME_SERVICE_ACCOUNT,
        "service_account_resource_sha256": protocol.sha256_json(account),
        "direct_binding_resource": resource_names[0],
        "roles": list(contract.RUNTIME_ROLES),
        "permissions": list(contract.RUNTIME_PERMISSIONS),
        "relevant_bindings_sha256": protocol.sha256_json(relevant),
        "role_definitions_sha256": protocol.sha256_json(definitions),
    }
    expected_activation_hash = collector_authority.get(
        "activation_seal_sha256"
    )
    expected_collector_authority_fields = {
        "schema", "member", "service_account",
        "runtime_identity_sha256", "project_read_role",
        "ancestor_read_role", "mutation_role",
        "mutation_binding_present", "mutation_condition",
        "activation_seal_sha256", "bindings", "bindings_sha256",
        "service_account_iam_policy",
        "service_account_iam_policy_sha256",
        "credential_key_snapshot", "credential_key_snapshot_sha256",
        "role_definitions", "role_definitions_sha256",
        "project_read_permissions", "ancestor_read_permissions",
        "mutation_permissions",
    }
    if (
        target_authority != expected_target_authority
        or projection.get("target_runtime_authority_sha256")
        != protocol.sha256_json(target_authority)
        or set(collector_authority)
        != expected_collector_authority_fields
        or collector_authority.get("schema")
        != COLLECTOR_AUTHORITY_SCHEMA
        or collector_authority.get("member") != collector_member
        or collector_authority.get("service_account")
        != OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT
        or collector_authority.get("runtime_identity_sha256")
        != protocol.sha256_json(collector_identity)
        or collector_authority.get("project_read_role")
        != collector_roles[0]
        or collector_authority.get("ancestor_read_role")
        != collector_roles[1]
        or collector_authority.get("mutation_role")
        != collector_roles[2]
        or collector_authority.get("mutation_condition")
        != (
            OWNER_GATE_MUTATION_CONDITION
            if mutation_binding_present
            else None
        )
        or (
            mutation_binding_present
            and not _sha(expected_activation_hash)
        )
        or (
            not mutation_binding_present
            and expected_activation_hash is not None
        )
        or (
            expected_mutation_binding_present is not None
            and mutation_binding_present
            is not expected_mutation_binding_present
        )
        or (
            expected_mutation_binding_present is True
            and expected_activation_hash
            != expected_activation_seal_sha256
        )
        or (
            expected_mutation_binding_present is False
            and expected_activation_seal_sha256 is not None
        )
        or collector_authority.get("service_account_iam_policy_sha256")
        != protocol.sha256_json(impersonation_policy)
        or collector_authority.get("credential_key_snapshot_sha256")
        != protocol.sha256_json(key_snapshot)
        or collector_authority.get("bindings_sha256")
        != protocol.sha256_json(reconstructed_collector_bindings)
        or collector_authority.get("role_definitions_sha256")
        != protocol.sha256_json(collector_definitions)
        or collector_authority.get("project_read_permissions")
        != list(owner_gate_foundation.READ_ONLY_IAM_PERMISSIONS)
        or collector_authority.get("ancestor_read_permissions")
        != list(owner_gate_foundation.DIRECT_IAM_ANCESTOR_PERMISSIONS)
        or collector_authority.get("mutation_permissions")
        != list(owner_gate_foundation.MUTATION_PERMISSIONS)
        or projection.get("collector_service_account_authority_sha256")
        != protocol.sha256_json(collector_authority)
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_collector_authority_invalid"
        )
    body = {
        key: item
        for key, item in projection.items()
        if key != "projection_sha256"
    }
    expected_expiry = min(
        collected_at + TRUSTED_IAM_TTL_SECONDS,
        request["collection_attempt_expires_at_unix"],
        candidate_iam["expires_at_unix"],
    )
    if (
        set(collector_identity)
        != {
            "instance_id", "instance_name", "service_account",
            "service_account_unique_id", "scopes",
            "metadata_instance_id_sha256",
            "metadata_service_account_email_sha256",
            "metadata_scopes_sha256", "compute_instance_sha256",
            "iam_service_account_sha256",
            "service_account_iam_policy_sha256",
            "user_managed_key_snapshot_sha256",
        }
        or re.fullmatch(
            r"[1-9][0-9]{5,31}",
            str(collector_identity.get("instance_id")),
        )
        is None
        or collector_identity.get("instance_name") != OWNER_GATE_VM_NAME
        or collector_identity.get("service_account")
        != OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT
        or re.fullmatch(
            r"[1-9][0-9]{5,31}",
            str(collector_identity.get("service_account_unique_id")),
        )
        is None
        or collector_identity.get("scopes")
        != list(OWNER_GATE_METADATA_SCOPES)
        or collector_identity.get("metadata_instance_id_sha256")
        != protocol.sha256_bytes(
            str(collector_identity.get("instance_id")).encode("utf-8")
        )
        or collector_identity.get("metadata_service_account_email_sha256")
        != protocol.sha256_bytes(
            str(collector_identity.get("service_account")).encode("utf-8")
        )
        or collector_identity.get("metadata_scopes_sha256")
        != protocol.sha256_json(collector_identity.get("scopes"))
        or any(
            not _sha(collector_identity.get(name))
            for name in (
                "metadata_instance_id_sha256",
                "metadata_service_account_email_sha256",
                "metadata_scopes_sha256",
                "compute_instance_sha256",
                "iam_service_account_sha256",
                "service_account_iam_policy_sha256",
                "user_managed_key_snapshot_sha256",
            )
        )
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_collector_identity_invalid"
        )
    if (
        projection.get("schema") != TRUSTED_IAM_PROJECTION_SCHEMA
        or projection.get("authority_scope")
        != TRUSTED_IAM_AUTHORITY_SCOPE
        or projection.get("external_gcp_admin_trust_root")
        != external_gcp_admin_trust_root
        or projection.get("external_gcp_admin_trust_root_sha256")
        != protocol.sha256_json(external_gcp_admin_trust_root)
        or projection.get("transaction_id") != request["transaction_id"]
        or projection.get("checkpoint") != request["checkpoint"]
        or projection.get("observation_request_sha256")
        != request["observation_request_sha256"]
        or projection.get("collection_attempt_id")
        != request["collection_attempt_id"]
        or projection.get("collection_attempt_sequence")
        != request["collection_attempt_sequence"]
        or projection.get("collection_attempt_issued_at_unix")
        != request["collection_attempt_issued_at_unix"]
        or projection.get("collection_attempt_expires_at_unix")
        != request["collection_attempt_expires_at_unix"]
        or projection.get("candidate_external_iam_receipt_sha256")
        != observation["external_iam_receipt_sha256"]
        or projection.get("candidate_external_iam_policy_sha256")
        != contract.EXTERNAL_IAM_POLICY_SHA256
        or projection.get("project") != contract.PROJECT
        or (
            expected_project_number is not None
            and project_number != expected_project_number
        )
        or (
            expected_ancestor_chain is not None
            and ancestors != list(expected_ancestor_chain)
        )
        or projection.get("project_policy_version")
        != policies[0]["version"]
        or projection.get("project_policy_etag")
        != policies[0]["etag"]
        or projection.get("project_policy_sha256")
        != policies[0]["raw_sha256"]
        or projection.get("resource_hierarchy_sha256")
        != protocol.sha256_json(hierarchy)
        or projection.get("resource_policies_sha256")
        != protocol.sha256_json(policies)
        or projection.get("zone") != contract.ZONE
        or projection.get("instance") != contract.VM_NAME
        or projection.get("service_account")
        != contract.RUNTIME_SERVICE_ACCOUNT
        or projection.get("scopes") != list(contract.RUNTIME_SCOPES)
        or projection.get("member") != member
        or projection.get("roles") != list(contract.RUNTIME_ROLES)
        or projection.get("permissions")
        != list(contract.RUNTIME_PERMISSIONS)
        or expanded_permissions != set(contract.RUNTIME_PERMISSIONS)
        or projection.get("project_policy_version") not in {1, 3}
        or not isinstance(projection.get("project_policy_etag"), str)
        or not projection["project_policy_etag"]
        or not _sha(projection.get("project_policy_sha256"))
        or projection.get("relevant_bindings_sha256")
        != protocol.sha256_json(relevant)
        or projection.get("role_definitions_sha256")
        != protocol.sha256_json(definitions)
        or set(account)
        != {
            "name", "project_id", "unique_id", "email", "disabled",
            "oauth2_client_id", "raw_sha256",
        }
        or account.get("name") != expected_account_name
        or account.get("project_id") != contract.PROJECT
        or account.get("email") != contract.RUNTIME_SERVICE_ACCOUNT
        or (
            expected_runtime_instance_numeric_id is not None
            and collector_identity.get("instance_id")
            != expected_runtime_instance_numeric_id
        )
        or (
            expected_runtime_service_account_email is not None
            and collector_identity.get("service_account")
            != expected_runtime_service_account_email
        )
        or (
            expected_runtime_service_account_unique_id is not None
            and collector_identity.get("service_account_unique_id")
            != expected_runtime_service_account_unique_id
        )
        or projection.get("collector_runtime_identity_sha256")
        != protocol.sha256_json(collector_identity)
        or (
            expected_target_service_account_unique_id is not None
            and account.get("unique_id")
            != expected_target_service_account_unique_id
        )
        or account.get("disabled") is not False
        or not isinstance(account.get("unique_id"), str)
        or not account["unique_id"].isdigit()
        or not isinstance(account.get("oauth2_client_id"), str)
        or (account["oauth2_client_id"] and not account["oauth2_client_id"].isdigit())
        or not _sha(account.get("raw_sha256"))
        or projection.get("service_account_resource_sha256")
        != protocol.sha256_json(account)
        or not request["collection_attempt_issued_at_unix"]
        <= collected_at
        <= now_unix
        or expires_at != expected_expiry
        or not now_unix < expires_at
        or projection.get("projection_sha256")
        != protocol.sha256_json(body)
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_projection_invalid"
        )
    return projection


def _bundle_core(
    request: Mapping[str, Any],
    observation: Mapping[str, Any],
    trusted_iam_projection: Mapping[str, Any],
) -> Mapping[str, Any]:
    observed = int(observation["collected_at_unix"])
    return {
        "schema": evidence.ATTESTED_OBSERVATION_SCHEMA,
        "transaction_id": request["transaction_id"],
        "checkpoint": request["checkpoint"],
        "observation_nonce_sha256": request["observation_nonce_sha256"],
        "request_binding_sha256": request["request_binding_sha256"],
        "prior_event_head_sha256": request["prior_event_head_sha256"],
        "observation_request_sha256": request[
            "observation_request_sha256"
        ],
        "collection_attempt_id": request["collection_attempt_id"],
        "collection_attempt_sequence": request[
            "collection_attempt_sequence"
        ],
        "collection_attempt_issued_at_unix": request[
            "collection_attempt_issued_at_unix"
        ],
        "collection_attempt_expires_at_unix": request[
            "collection_attempt_expires_at_unix"
        ],
        "observed_at_unix": observed,
        "expires_at_unix": min(
            observed + evidence.OBSERVATION_BUNDLE_TTL_SECONDS,
            request["collection_attempt_expires_at_unix"],
            trusted_iam_projection["expires_at_unix"],
        ),
        "observation": _copy_json(observation),
        "observation_sha256": protocol.sha256_json(observation),
        "trusted_iam_projection": _copy_json(
            trusted_iam_projection
        ),
        "trusted_iam_projection_sha256": protocol.sha256_json(
            trusted_iam_projection
        ),
    }


def _role_payload(core: Mapping[str, Any], *, role: str) -> Mapping[str, Any]:
    if role not in {"cloud", "host"}:
        raise TrustedObservationError("trusted_observation_role_invalid")
    return {
        "schema": evidence.ATTESTED_OBSERVATION_SCHEMA,
        "role": role,
        **{
            name: core[name]
            for name in (
                "transaction_id",
                "checkpoint",
                "observation_nonce_sha256",
                "request_binding_sha256",
                "prior_event_head_sha256",
                "observation_request_sha256",
                "collection_attempt_id",
                "collection_attempt_sequence",
                "collection_attempt_issued_at_unix",
                "collection_attempt_expires_at_unix",
                "observed_at_unix",
                "expires_at_unix",
                "observation_sha256",
                "observation",
                "trusted_iam_projection_sha256",
                "trusted_iam_projection",
            )
        },
    }


def build_attestation_request(
    observation_request: Mapping[str, Any],
    candidate_observation: Mapping[str, Any],
    *,
    role: str,
    trusted_iam_projection: Mapping[str, Any] | None = None,
    now_unix: int,
) -> Mapping[str, Any]:
    request = validate_observation_request(observation_request)
    observation = _validated_observation(
        request, candidate_observation, now_unix=now_unix
    )
    if role not in {"cloud", "host"}:
        raise TrustedObservationError("trusted_observation_role_invalid")
    if role == "cloud":
        if trusted_iam_projection is not None:
            raise TrustedObservationError(
                "trusted_observation_iam_projection_forbidden"
            )
    else:
        validate_trusted_iam_projection(
            trusted_iam_projection,
            observation_request=request,
            candidate_observation=observation,
            now_unix=now_unix,
        )
    if role == "host" and not host_attestation_required(
        request, observation, now_unix=now_unix
    ):
        raise TrustedObservationError(
            "trusted_observation_host_attestation_forbidden"
        )
    unsigned = {
        "schema": ATTESTATION_REQUEST_SCHEMA,
        "role": role,
        "observation_request": request,
        "candidate_observation": observation,
        "trusted_iam_projection": (
            None
            if trusted_iam_projection is None
            else _copy_json(trusted_iam_projection)
        ),
    }
    return {**unsigned, "frame_sha256": protocol.sha256_json(unsigned)}


def _validate_attestation_request(
    value: Any,
    *,
    expected_role: str,
    now_unix: int,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any] | None,
]:
    if not isinstance(value, Mapping) or set(value) != _ATTESTATION_REQUEST_FIELDS:
        raise TrustedObservationError(
            "trusted_observation_attestation_request_fields_invalid"
        )
    frame = _copy_json(value)
    unsigned = {
        name: item for name, item in frame.items() if name != "frame_sha256"
    }
    if (
        frame.get("schema") != ATTESTATION_REQUEST_SCHEMA
        or frame.get("role") != expected_role
        or frame.get("frame_sha256") != protocol.sha256_json(unsigned)
    ):
        raise TrustedObservationError(
            "trusted_observation_attestation_request_invalid"
        )
    request = validate_observation_request(frame["observation_request"])
    observation = _validated_observation(
        request, frame["candidate_observation"], now_unix=now_unix
    )
    trusted_iam_projection = frame.get("trusted_iam_projection")
    if expected_role == "cloud":
        if trusted_iam_projection is not None:
            raise TrustedObservationError(
                "trusted_observation_iam_projection_forbidden"
            )
    else:
        trusted_iam_projection = validate_trusted_iam_projection(
            trusted_iam_projection,
            observation_request=request,
            candidate_observation=observation,
            now_unix=now_unix,
        )
    if expected_role == "host" and not host_attestation_required(
        request, observation, now_unix=now_unix
    ):
        raise TrustedObservationError(
            "trusted_observation_host_attestation_forbidden"
        )
    return (
        frame,
        request,
        observation,
        None
        if trusted_iam_projection is None
        else _copy_json(trusted_iam_projection),
    )


def host_attestation_required(
    observation_request: Mapping[str, Any],
    candidate_observation: Mapping[str, Any],
    *,
    now_unix: int,
) -> bool:
    request = validate_observation_request(observation_request)
    observation = _validated_observation(
        request, candidate_observation, now_unix=now_unix
    )
    return not (
        request["checkpoint"] == "post_stop"
        and observation["state"] == "terminated_after_growth_intent"
        and observation["canonical_receipt_source"]
        == "durable_signed_source_snapshot_for_stopped_vm"
    )


def _read_regular_file(
    path: Path,
    *,
    maximum: int,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    expected_mode: int | None = None,
) -> bytes:
    if not path.is_absolute() or ".." in path.parts:
        raise TrustedObservationError("trusted_observation_file_path_invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size < 1
            or info.st_size > maximum
            or (expected_uid is not None and info.st_uid != expected_uid)
            or (expected_gid is not None and info.st_gid != expected_gid)
            or (
                expected_mode is not None
                and stat.S_IMODE(info.st_mode) != expected_mode
            )
        ):
            raise TrustedObservationError(
                "trusted_observation_file_metadata_invalid"
            )
        chunks = bytearray()
        while len(chunks) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        after = os.fstat(descriptor)
        if (
            len(chunks) != info.st_size
            or len(chunks) > maximum
            or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise TrustedObservationError(
                "trusted_observation_file_changed"
            )
        return bytes(chunks)
    except TrustedObservationError:
        raise
    except OSError as exc:
        raise TrustedObservationError(
            "trusted_observation_file_unavailable"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_pseudofile(path: Path, *, maximum: int) -> bytes:
    """Read an exact bounded proc/sys-style file whose stat size may be zero."""

    if not path.is_absolute() or ".." in path.parts:
        raise TrustedObservationError("trusted_observation_file_path_invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise TrustedObservationError(
                "trusted_observation_file_metadata_invalid"
            )
        raw = os.read(descriptor, maximum + 1)
        if not raw or len(raw) > maximum or os.read(descriptor, 1):
            raise TrustedObservationError(
                "trusted_observation_file_metadata_invalid"
            )
        return raw
    except TrustedObservationError:
        raise
    except OSError as exc:
        raise TrustedObservationError(
            "trusted_observation_file_unavailable"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _load_config(
    path: Path,
    *,
    role: str,
    expected_uid: int,
    expected_path: Path,
    expected_private_key_path: Path,
    expected_replay_directory: Path,
) -> Mapping[str, Any]:
    if path != expected_path:
        raise TrustedObservationError("trusted_observation_config_path_invalid")
    raw = _read_regular_file(
        path,
        maximum=MAX_CONFIG_BYTES,
        expected_uid=0,
        expected_gid=0,
        expected_mode=0o444,
    )
    try:
        value = protocol.decode_canonical_json(raw, maximum_bytes=MAX_CONFIG_BYTES)
    except protocol.PasskeyV2ProtocolError as exc:
        raise TrustedObservationError(
            "trusted_observation_config_invalid"
        ) from None
    config = validate_attestor_config(value, expected_role=role)
    if (
        Path(config["private_key_path"]) != expected_private_key_path
        or Path(config["replay_directory"]) != expected_replay_directory
        or config["private_key_uid"] != expected_uid
        or config["private_key_gid"] != expected_uid
        or config["replay_directory_uid"] != expected_uid
        or config["replay_directory_gid"] != expected_uid
    ):
        raise TrustedObservationError("trusted_observation_config_identity_invalid")
    return config


def validate_attestor_config(
    value: Any, *, expected_role: str
) -> Mapping[str, Any]:
    if (
        expected_role not in {"host", "cloud"}
        or not isinstance(value, Mapping)
        or set(value) != _CONFIG_FIELDS
    ):
        raise TrustedObservationError("trusted_observation_config_fields_invalid")
    config = _copy_json(value)
    expected_schema = (
        HOST_CONFIG_SCHEMA if expected_role == "host" else CLOUD_CONFIG_SCHEMA
    )
    private_path = Path(str(config.get("private_key_path", "")))
    replay_path = Path(str(config.get("replay_directory", "")))
    if (
        config.get("schema") != expected_schema
        or config.get("role") != expected_role
        or not private_path.is_absolute()
        or ".." in private_path.parts
        or not replay_path.is_absolute()
        or ".." in replay_path.parts
        or type(config.get("private_key_uid")) is not int
        or type(config.get("private_key_gid")) is not int
        or config.get("private_key_mode") != "0400"
        or not _sha(config.get("public_key_id"))
        or type(config.get("replay_directory_uid")) is not int
        or type(config.get("replay_directory_gid")) is not int
        or config.get("replay_directory_mode") != "0700"
    ):
        raise TrustedObservationError("trusted_observation_config_invalid")
    return config


def _load_private_key(
    config: Mapping[str, Any]
) -> tuple[Ed25519PrivateKey, Ed25519PublicKey, str]:
    raw = _read_regular_file(
        Path(config["private_key_path"]),
        maximum=32,
        expected_uid=int(config["private_key_uid"]),
        expected_gid=int(config["private_key_gid"]),
        expected_mode=0o400,
    )
    if len(raw) != 32:
        raise TrustedObservationError("trusted_observation_private_key_invalid")
    try:
        private_key = Ed25519PrivateKey.from_private_bytes(raw)
    except ValueError as exc:
        raise TrustedObservationError(
            "trusted_observation_private_key_invalid"
        ) from None
    public_key = private_key.public_key()
    key_id = protocol.sha256_bytes(public_key.public_bytes_raw())
    if key_id != config["public_key_id"]:
        raise TrustedObservationError(
            "trusted_observation_private_key_identity_invalid"
        )
    return private_key, public_key, key_id


def _validate_replay_directory(
    config: Mapping[str, Any]
) -> tuple[Path, int]:
    path = Path(config["replay_directory"])
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TrustedObservationError(
            "trusted_observation_replay_directory_unavailable"
        ) from None
    info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != config["replay_directory_uid"]
        or info.st_gid != config["replay_directory_gid"]
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise TrustedObservationError(
            "trusted_observation_replay_directory_invalid"
        )
    return path, descriptor


def _read_at(directory_fd: int, name: str, *, maximum: int) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise TrustedObservationError(
            "trusted_observation_replay_receipt_unavailable"
        ) from None
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size < 1
            or info.st_size > maximum
        ):
            raise TrustedObservationError(
                "trusted_observation_replay_receipt_invalid"
            )
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        if len(raw) != info.st_size or len(raw) > maximum:
            raise TrustedObservationError(
                "trusted_observation_replay_receipt_invalid"
            )
        return bytes(raw)
    finally:
        os.close(descriptor)


def _write_at_atomic(directory_fd: int, name: str, raw: bytes) -> None:
    temporary = f".{name}.{os.getpid()}.{protocol.sha256_bytes(raw)[:16]}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.rename(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except OSError as exc:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except OSError:
            pass
        raise TrustedObservationError(
            "trusted_observation_replay_receipt_write_failed"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _store_or_replay(
    response: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    public_key: Ed25519PublicKey,
    core: Mapping[str, Any],
) -> Mapping[str, Any]:
    _path, directory_fd = _validate_replay_directory(config)
    role = str(response["role"])
    request_sha = str(response["observation_request_sha256"])
    name = f"{role}-{request_sha}.json"
    try:
        lock_fd = os.open(
            ".attestation.lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        try:
            os.fchmod(lock_fd, 0o600)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            existing_raw = _read_at(
                directory_fd, name, maximum=MAX_FRAME_BYTES
            )
            if existing_raw is not None:
                try:
                    existing = protocol.decode_canonical_json(
                        existing_raw, maximum_bytes=MAX_FRAME_BYTES
                    )
                except protocol.PasskeyV2ProtocolError as exc:
                    raise TrustedObservationError(
                        "trusted_observation_replay_receipt_invalid"
                    ) from None
                checked = _validate_role_response(
                    existing,
                    expected_role=role,
                    expected_request_sha256=request_sha,
                    expected_frame_sha256=str(
                        response["attestation_request_sha256"]
                    ),
                    core=core,
                    public_key=public_key,
                )
                if protocol.canonical_json_bytes(checked) != protocol.canonical_json_bytes(
                    response
                ):
                    raise TrustedObservationError(
                        "trusted_observation_replay_conflict"
                    )
                return checked
            _write_at_atomic(
                directory_fd, name, protocol.canonical_json_bytes(response)
            )
            return response
        finally:
            os.close(lock_fd)
    finally:
        os.close(directory_fd)


def _default_command_runner(argv: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            tuple(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C.UTF-8"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TrustedObservationError(
            "trusted_observation_host_command_failed"
        ) from None
    if (
        completed.returncode != 0
        or not completed.stdout
        or len(completed.stdout) > MAX_COMMAND_OUTPUT_BYTES
        or completed.stderr
    ):
        raise TrustedObservationError(
            "trusted_observation_host_command_failed"
        )
    return completed.stdout


def _json_no_duplicates(raw: bytes, *, label: str) -> Any:
    if not raw or len(raw) > MAX_COMMAND_OUTPUT_BYTES:
        raise TrustedObservationError(f"trusted_observation_{label}_invalid")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in items:
            if name in value:
                raise ValueError("duplicate")
            value[name] = item
        return value

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
            parse_float=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise TrustedObservationError(
            f"trusted_observation_{label}_invalid"
        ) from None


def _nonnegative_integer(value: Any) -> int | None:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    return None


class FixedHostFactsReader:
    """Read the exact host-only facts from the running canary."""

    def __init__(
        self,
        command_runner: Callable[[Sequence[str]], bytes] = _default_command_runner,
    ) -> None:
        if not callable(command_runner):
            raise TrustedObservationError(
                "trusted_observation_host_reader_invalid"
            )
        self._run = command_runner

    def _boot_id(self) -> str:
        raw = _read_pseudofile(BOOT_ID_PATH, maximum=128)
        try:
            boot_id = raw.decode("ascii", errors="strict").strip()
        except UnicodeError as exc:
            raise TrustedObservationError(
                "trusted_observation_host_boot_invalid"
            ) from None
        if _BOOT_ID.fullmatch(boot_id) is None:
            raise TrustedObservationError(
                "trusted_observation_host_boot_invalid"
            )
        return hashlib.sha256(boot_id.encode("ascii")).hexdigest()

    def _root(self) -> Mapping[str, Any]:
        raw = self._run((
            FINDMNT,
            "--json",
            "--bytes",
            "--output=SOURCE,FSTYPE,SIZE,AVAIL,TARGET",
            "/",
        ))
        value = _json_no_duplicates(raw, label="host_root")
        if not isinstance(value, Mapping) or set(value) != {"filesystems"}:
            raise TrustedObservationError("trusted_observation_host_root_invalid")
        filesystems = value["filesystems"]
        if not isinstance(filesystems, list) or len(filesystems) != 1:
            raise TrustedObservationError("trusted_observation_host_root_invalid")
        item = filesystems[0]
        if not isinstance(item, Mapping):
            raise TrustedObservationError("trusted_observation_host_root_invalid")
        size = _nonnegative_integer(item.get("size"))
        available = _nonnegative_integer(item.get("avail"))
        if (
            item.get("source") != contract.ROOT_SOURCE
            or item.get("fstype") != contract.ROOT_FILESYSTEM
            or item.get("target") != contract.ROOT_MOUNTPOINT
            or size is None
            or available is None
            or available > size
        ):
            raise TrustedObservationError("trusted_observation_host_root_invalid")
        return {
            "root_source": item["source"],
            "root_filesystem": item["fstype"],
            "root_mountpoint": item["target"],
            "root_size_bytes": size,
            "root_available_bytes": available,
        }

    def _service(self, unit: str) -> Mapping[str, Any]:
        if unit not in CANARY_RUNTIME_UNITS:
            raise TrustedObservationError(
                "trusted_observation_host_service_invalid"
            )
        raw = self._run((
            SYSTEMCTL,
            "show",
            "--no-pager",
            *(f"--property={name}" for name in _SERVICE_PROPERTIES),
            unit,
        ))
        try:
            lines = raw.decode("utf-8", errors="strict").splitlines()
        except UnicodeError as exc:
            raise TrustedObservationError(
                "trusted_observation_host_service_invalid"
            ) from None
        properties: dict[str, str] = {}
        for line in lines:
            if "=" not in line:
                raise TrustedObservationError(
                    "trusted_observation_host_service_invalid"
                )
            name, item = line.split("=", 1)
            if name not in _SERVICE_PROPERTIES or name in properties:
                raise TrustedObservationError(
                    "trusted_observation_host_service_invalid"
                )
            properties[name] = item
        if set(properties) != set(_SERVICE_PROPERTIES):
            raise TrustedObservationError(
                "trusted_observation_host_service_invalid"
            )
        absent = {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "",
            "MainPID": "0",
            "FragmentPath": "",
            "DropInPaths": "",
        }
        disabled = {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "disabled",
            "MainPID": "0",
            "FragmentPath": f"/etc/systemd/system/{unit}",
            "DropInPaths": "",
        }
        if properties == absent:
            state = "absent"
        elif properties == disabled:
            state = "disabled_inactive"
        else:
            raise TrustedObservationError(
                "trusted_observation_host_service_invalid"
            )
        return {"unit": unit, "state": state, "properties": properties}

    def collect(self) -> Mapping[str, Any]:
        host_receipt = _read_regular_file(
            HOST_RECEIPT_PATH, maximum=MAX_COMMAND_OUTPUT_BYTES
        )
        stopped_receipt = _read_regular_file(
            STOPPED_RELEASE_RECEIPT_PATH, maximum=MAX_COMMAND_OUTPUT_BYTES
        )
        services = [self._service(unit) for unit in CANARY_RUNTIME_UNITS]
        return {
            "schema": "muncho-storage-growth-host-recollection.v1",
            "boot_id_sha256": self._boot_id(),
            **self._root(),
            "service_states": services,
            "service_states_sha256": protocol.sha256_json(services),
            "canonical_receipt_source": "fresh_running_vm_receipt_bytes",
            "current_stopped_release_sha": contract.CURRENT_STOPPED_RELEASE_SHA,
            "current_host_receipt_file_sha256": hashlib.sha256(
                host_receipt
            ).hexdigest(),
            "current_host_receipt_sha256": contract.CURRENT_HOST_RECEIPT_SHA256,
            "current_stopped_release_receipt_file_sha256": hashlib.sha256(
                stopped_receipt
            ).hexdigest(),
            "current_stopped_release_receipt_sha256": (
                contract.CURRENT_STOPPED_RELEASE_RECEIPT_SHA256
            ),
        }


def _expected_host_recollection(
    observation: Mapping[str, Any]
) -> Mapping[str, Any]:
    if observation["instance_status"] != "RUNNING":
        raise TrustedObservationError("trusted_observation_host_state_invalid")
    return {
        "schema": "muncho-storage-growth-host-recollection.v1",
        **{
            name: observation[name]
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


def _ca_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_dev,
        value.st_ino,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _fixed_ca_parent_identities(
    path: Path,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    current = Path(path.anchor)
    result: list[tuple[str, tuple[int, ...]]] = []
    for component in path.parts[1:-1]:
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise TrustedObservationError(
                "trusted_observation_cloud_tls_invalid"
            ) from None
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid not in {0, _TRUSTED_CA_OWNER_UID}
            or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise TrustedObservationError(
                "trusted_observation_cloud_tls_invalid"
            )
        result.append((str(current), _ca_stat_identity(metadata)))
    return tuple(result)


def _read_fixed_debian_ca_bundle() -> bytes:
    if any(os.environ.get(name) for name in _FORBIDDEN_TLS_ENVIRONMENT):
        raise TrustedObservationError("trusted_observation_cloud_tls_invalid")
    path = FIXED_DEBIAN_CA_BUNDLE_PATH
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or ".." in path.parts
        or str(path) != os.path.normpath(str(path))
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_CLOEXEC")
    ):
        raise TrustedObservationError("trusted_observation_cloud_tls_invalid")
    descriptor = -1
    try:
        parents_before = _fixed_ca_parent_identities(path)
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != _TRUSTED_CA_OWNER_UID
            or before.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            or before.st_nlink != 1
            or not 0 < before.st_size <= MAX_TRUSTED_CA_BUNDLE_BYTES
        ):
            raise TrustedObservationError(
                "trusted_observation_cloud_tls_invalid"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        opened = os.fstat(descriptor)
        if _ca_stat_identity(opened) != _ca_stat_identity(before):
            raise TrustedObservationError(
                "trusted_observation_cloud_tls_changed"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_TRUSTED_CA_BUNDLE_BYTES:
                raise TrustedObservationError(
                    "trusted_observation_cloud_tls_invalid"
                )
            chunks.append(chunk)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
        parents_after = _fixed_ca_parent_identities(path)
        if (
            total != opened.st_size
            or _ca_stat_identity(after) != _ca_stat_identity(opened)
            or _ca_stat_identity(path_after) != _ca_stat_identity(before)
            or parents_after != parents_before
        ):
            raise TrustedObservationError(
                "trusted_observation_cloud_tls_changed"
            )
        payload = b"".join(chunks)
    except TrustedObservationError:
        raise
    except OSError as exc:
        raise TrustedObservationError(
            "trusted_observation_cloud_tls_invalid"
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        not payload
        or b"-----BEGIN CERTIFICATE-----" not in payload
        or b"-----END CERTIFICATE-----" not in payload
    ):
        raise TrustedObservationError("trusted_observation_cloud_tls_invalid")
    return payload


def fixed_debian_tls_context() -> ssl.SSLContext:
    """Build one no-env TLS context from the fixed descriptor-read Debian CA."""

    payload = _read_fixed_debian_ca_bundle()
    try:
        pem = payload.decode("ascii", errors="strict")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        context.load_verify_locations(cadata=pem)
    except (UnicodeError, ValueError, ssl.SSLError) as exc:
        raise TrustedObservationError(
            "trusted_observation_cloud_tls_invalid"
        ) from None
    if context.keylog_filename is not None:
        raise TrustedObservationError("trusted_observation_cloud_tls_invalid")
    return context


def _bounded_http_body(response: http.client.HTTPResponse) -> bytes:
    raw_length = response.getheader("Content-Length")
    if raw_length is not None:
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise TrustedObservationError(
                "trusted_observation_cloud_http_invalid"
            ) from None
        if length < 1 or length > MAX_HTTP_BODY_BYTES:
            raise TrustedObservationError(
                "trusted_observation_cloud_http_invalid"
            )
    body = response.read(MAX_HTTP_BODY_BYTES + 1)
    if not body or len(body) > MAX_HTTP_BODY_BYTES:
        raise TrustedObservationError(
            "trusted_observation_cloud_http_invalid"
        )
    return body


def _response_content_type(value: Any, *, media_type: str) -> bool:
    if not isinstance(value, str):
        return False
    parts = tuple(item.strip() for item in value.split(";"))
    if not parts or parts[0].lower() != media_type:
        return False
    if len(parts) == 1:
        return True
    return (
        len(parts) == 2
        and parts[1].lower().replace(" ", "") == "charset=utf-8"
    )


class FixedComputeFactsReader:
    """Direct no-proxy reader for pinned Compute and IAM allow truth."""

    def __init__(
        self,
        *,
        expected_project_number: str,
        expected_ancestor_chain: Sequence[str],
        expected_runtime_instance_numeric_id: str,
        expected_runtime_service_account_email: str,
        expected_runtime_service_account_unique_id: str,
        expected_target_service_account_unique_id: str,
        expected_metadata_scopes: Sequence[str],
        expected_external_gcp_admin_trust_root: Mapping[str, Any],
        expected_mutation_binding_present: bool,
        activation_seal_sha256: str | None,
        metadata_connection_factory: Callable[[], Any] | None = None,
        compute_connection_factory: Callable[[], Any] | None = None,
        resource_manager_connection_factory: Callable[[], Any] | None = None,
        iam_connection_factory: Callable[[], Any] | None = None,
    ) -> None:
        ancestor_chain = tuple(expected_ancestor_chain)
        scopes = tuple(expected_metadata_scopes)
        if (
            re.fullmatch(r"[1-9][0-9]{5,31}", expected_project_number)
            is None
            or re.fullmatch(
                r"[1-9][0-9]{5,31}",
                expected_runtime_instance_numeric_id,
            )
            is None
            or expected_runtime_service_account_email
            != OWNER_GATE_COLLECTOR_SERVICE_ACCOUNT
            or re.fullmatch(
                r"[1-9][0-9]{5,31}",
                expected_runtime_service_account_unique_id,
            )
            is None
            or re.fullmatch(
                r"[1-9][0-9]{5,31}",
                expected_target_service_account_unique_id,
            )
            is None
            or not ancestor_chain
            or re.fullmatch(
                r"organizations/[1-9][0-9]{5,31}",
                ancestor_chain[-1],
            )
            is None
            or any(
                re.fullmatch(r"folders/[1-9][0-9]{5,31}", item)
                is None
                for item in ancestor_chain[:-1]
            )
            or scopes != OWNER_GATE_METADATA_SCOPES
            or type(expected_mutation_binding_present) is not bool
            or (
                expected_mutation_binding_present
                and not _sha(activation_seal_sha256)
            )
            or (
                not expected_mutation_binding_present
                and activation_seal_sha256 is not None
            )
        ):
            raise TrustedObservationError(
                "trusted_observation_iam_authority_pin_invalid"
            )
        self._project_number = expected_project_number
        self._ancestor_chain = ancestor_chain
        self._runtime_instance_numeric_id = (
            expected_runtime_instance_numeric_id
        )
        self._runtime_service_account_email = (
            expected_runtime_service_account_email
        )
        self._runtime_service_account_unique_id = (
            expected_runtime_service_account_unique_id
        )
        self._target_service_account_unique_id = (
            expected_target_service_account_unique_id
        )
        self._metadata_scopes = scopes
        self._external_gcp_admin_trust_root = (
            validate_external_gcp_admin_trust_root(
                expected_external_gcp_admin_trust_root,
                ancestor_chain=ancestor_chain,
            )
        )
        self._mutation_binding_present = (
            expected_mutation_binding_present
        )
        self._activation_seal_sha256 = activation_seal_sha256
        self._resource_names = (
            f"projects/{expected_project_number}",
            *ancestor_chain,
        )
        self._collector_roles = _owner_gate_collector_roles(
            ancestor_chain
        )
        self._iam_paths = {
            *ROLE_PATHS,
            *(f"/v1/{role}" for role in self._collector_roles),
            TARGET_SERVICE_ACCOUNT_PATH,
            COLLECTOR_SERVICE_ACCOUNT_PATH,
            COLLECTOR_SERVICE_ACCOUNT_USER_KEYS_PATH,
        }
        self._metadata_factory = metadata_connection_factory or (
            lambda: http.client.HTTPConnection(
                METADATA_HOST, 80, timeout=HTTP_TIMEOUT_SECONDS
            )
        )
        context = (
            fixed_debian_tls_context()
            if any(
                factory is None
                for factory in (
                    compute_connection_factory,
                    resource_manager_connection_factory,
                    iam_connection_factory,
                )
            )
            else None
        )
        self._compute_factory = compute_connection_factory or (
            lambda: http.client.HTTPSConnection(
                COMPUTE_HOST,
                443,
                timeout=HTTP_TIMEOUT_SECONDS,
                context=context,
            )
        )
        self._resource_manager_factory = (
            resource_manager_connection_factory
            or (
                lambda: http.client.HTTPSConnection(
                    CLOUD_RESOURCE_MANAGER_HOST,
                    443,
                    timeout=HTTP_TIMEOUT_SECONDS,
                    context=context,
                )
            )
        )
        self._iam_factory = iam_connection_factory or (
            lambda: http.client.HTTPSConnection(
                IAM_HOST,
                443,
                timeout=HTTP_TIMEOUT_SECONDS,
                context=context,
            )
        )

    def _metadata(self, path: str, *, accept: str) -> bytes:
        if path not in {
            METADATA_TOKEN_PATH,
            METADATA_SCOPES_PATH,
            METADATA_INSTANCE_ID_PATH,
            METADATA_SERVICE_ACCOUNT_EMAIL_PATH,
        }:
            raise TrustedObservationError(
                "trusted_observation_metadata_resource_forbidden"
            )
        connection = self._metadata_factory()
        try:
            connection.request(
                "GET",
                path,
                headers={
                    "Metadata-Flavor": "Google",
                    "Accept": accept,
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            body = _bounded_http_body(response)
            if (
                response.status != 200
                or response.getheader("Metadata-Flavor") != "Google"
                or response.getheader("Location") is not None
                or not _response_content_type(
                    response.getheader("Content-Type"),
                    media_type=accept,
                )
            ):
                raise TrustedObservationError(
                    "trusted_observation_metadata_token_invalid"
                )
        except TrustedObservationError:
            raise
        except (OSError, socket.timeout, http.client.HTTPException) as exc:
            raise TrustedObservationError(
                "trusted_observation_metadata_unavailable"
            ) from None
        finally:
            connection.close()
        return body

    def _token(self) -> str:
        body = self._metadata(
            METADATA_TOKEN_PATH, accept="application/json"
        )
        value = _json_no_duplicates(body, label="metadata_token")
        if (
            not isinstance(value, Mapping)
            or set(value) != {"access_token", "expires_in", "token_type"}
            or not isinstance(value.get("access_token"), str)
            or not value["access_token"]
            or len(value["access_token"]) > 16 * 1024
            or value.get("token_type") != "Bearer"
            or type(value.get("expires_in")) is not int
            or value["expires_in"] < 120
        ):
            raise TrustedObservationError(
                "trusted_observation_metadata_token_invalid"
            )
        return value["access_token"]

    def _metadata_text(self, path: str) -> str:
        body = self._metadata(path, accept=METADATA_TEXT_MEDIA_TYPE)
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TrustedObservationError(
                "trusted_observation_metadata_identity_invalid"
            ) from None
        if (
            not text
            or text != text.strip()
            or "\n" in text
            or "\r" in text
            or len(text) > 4096
        ):
            raise TrustedObservationError(
                "trusted_observation_metadata_identity_invalid"
            )
        return text

    def _require_scopes(self) -> tuple[str, ...]:
        body = self._metadata(
            METADATA_SCOPES_PATH, accept=METADATA_TEXT_MEDIA_TYPE
        )
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TrustedObservationError(
                "trusted_observation_metadata_scopes_invalid"
            ) from None
        scopes = tuple(line for line in text.splitlines() if line)
        if (
            len(scopes) != len(set(scopes))
            or tuple(sorted(scopes))
            != tuple(sorted(self._metadata_scopes))
        ):
            raise TrustedObservationError(
                "trusted_observation_metadata_scopes_invalid"
            )
        return self._metadata_scopes

    @staticmethod
    def _api_json(
        *,
        connection_factory: Callable[[], Any],
        method: str,
        path: str,
        token: str,
        body: bytes | None = None,
    ) -> Mapping[str, Any]:
        connection = connection_factory()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Connection": "close",
        }
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        try:
            connection.request(
                method,
                path,
                body=body,
                headers=headers,
            )
            response = connection.getresponse()
            response_body = _bounded_http_body(response)
            if (
                response.status != 200
                or response.getheader("Location") is not None
                or not _response_content_type(
                    response.getheader("Content-Type"),
                    media_type="application/json",
                )
            ):
                raise TrustedObservationError(
                    "trusted_observation_cloud_resource_unavailable"
                )
        except TrustedObservationError:
            raise
        except (
            OSError,
            socket.timeout,
            ssl.SSLError,
            http.client.HTTPException,
        ) as exc:
            raise TrustedObservationError(
                "trusted_observation_cloud_resource_unavailable"
            ) from None
        finally:
            connection.close()
        value = _json_no_duplicates(
            response_body, label="cloud_resource"
        )
        if not isinstance(value, Mapping):
            raise TrustedObservationError(
                "trusted_observation_cloud_resource_invalid"
            )
        return dict(value)

    def _compute_get(
        self, path: str, *, token: str
    ) -> Mapping[str, Any]:
        if path not in {INSTANCE_PATH, DISK_PATH, OWNER_GATE_INSTANCE_PATH}:
            raise TrustedObservationError(
                "trusted_observation_cloud_resource_forbidden"
            )
        return self._api_json(
            connection_factory=self._compute_factory,
            method="GET",
            path=path,
            token=token,
        )

    def _resource_get(
        self, resource: str, *, token: str
    ) -> Mapping[str, Any]:
        if resource not in self._resource_names:
            raise TrustedObservationError(
                "trusted_observation_iam_resource_forbidden"
            )
        return self._api_json(
            connection_factory=self._resource_manager_factory,
            method="GET",
            path=f"/v3/{resource}",
            token=token,
        )

    def _resource_policy(
        self, resource: str, *, token: str
    ) -> Mapping[str, Any]:
        if resource not in self._resource_names:
            raise TrustedObservationError(
                "trusted_observation_iam_resource_forbidden"
            )
        body = protocol.canonical_json_bytes({
            "options": {"requestedPolicyVersion": 3}
        })
        return self._api_json(
            connection_factory=self._resource_manager_factory,
            method="POST",
            path=f"/v3/{resource}:getIamPolicy",
            token=token,
            body=body,
        )

    def _iam_get(
        self, path: str, *, token: str
    ) -> Mapping[str, Any]:
        if path not in self._iam_paths:
            raise TrustedObservationError(
                "trusted_observation_iam_resource_forbidden"
            )
        return self._api_json(
            connection_factory=self._iam_factory,
            method="GET",
            path=path,
            token=token,
        )

    def _iam_role_get(
        self, role: str, *, token: str
    ) -> Mapping[str, Any]:
        if not _policy_role_allowed(
            role, ancestor_chain=self._ancestor_chain
        ):
            raise TrustedObservationError(
                "trusted_observation_iam_resource_forbidden"
            )
        return self._api_json(
            connection_factory=self._iam_factory,
            method="GET",
            path=f"/v1/{role}",
            token=token,
        )

    def _collector_service_account_policy(
        self, *, token: str
    ) -> Mapping[str, Any]:
        return self._api_json(
            connection_factory=self._iam_factory,
            method="POST",
            path=COLLECTOR_SERVICE_ACCOUNT_IAM_POLICY_PATH,
            token=token,
            body=None,
        )

    def collect(
        self,
        observation: Mapping[str, Any],
        observation_request: Mapping[str, Any],
        now_unix: int,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        metadata_instance_id = self._metadata_text(
            METADATA_INSTANCE_ID_PATH
        )
        metadata_service_account_email = self._metadata_text(
            METADATA_SERVICE_ACCOUNT_EMAIL_PATH
        )
        metadata_scopes = self._require_scopes()
        if (
            metadata_instance_id != self._runtime_instance_numeric_id
            or metadata_service_account_email
            != self._runtime_service_account_email
        ):
            raise TrustedObservationError(
                "trusted_observation_metadata_identity_invalid"
            )
        token = self._token()
        try:
            instance = self._compute_get(INSTANCE_PATH, token=token)
            disk = self._compute_get(DISK_PATH, token=token)
            collector_instance = self._compute_get(
                OWNER_GATE_INSTANCE_PATH, token=token
            )
            resources = [
                self._resource_get(resource, token=token)
                for resource in self._resource_names
            ]
            policies = [
                self._resource_policy(resource, token=token)
                for resource in self._resource_names
            ]
            roles = [
                self._iam_get(path, token=token)
                for path in ROLE_PATHS
            ]
            collector_roles = [
                self._iam_get(f"/v1/{role}", token=token)
                for role in self._collector_roles
            ]
            policy_role_names = _bounded_policy_role_names(
                policies,
                ancestor_chain=self._ancestor_chain,
            )
            known_roles = {
                *contract.RUNTIME_ROLES,
                *self._collector_roles,
            }
            additional_policy_roles = [
                self._iam_role_get(role, token=token)
                for role in policy_role_names
                if role not in known_roles
            ]
            account = self._iam_get(
                TARGET_SERVICE_ACCOUNT_PATH, token=token
            )
            collector_account = self._iam_get(
                COLLECTOR_SERVICE_ACCOUNT_PATH, token=token
            )
            collector_policy = self._collector_service_account_policy(
                token=token
            )
            collector_user_keys = self._iam_get(
                COLLECTOR_SERVICE_ACCOUNT_USER_KEYS_PATH,
                token=token,
            )
        finally:
            # Do not retain the bearer token on the object or in a receipt.
            token = ""
        recollection = _cloud_recollection(
            instance, disk, observation=observation
        )
        projection = build_trusted_iam_projection(
            observation_request=observation_request,
            candidate_observation=observation,
            expected_project_number=self._project_number,
            expected_ancestor_chain=self._ancestor_chain,
            expected_runtime_instance_numeric_id=(
                self._runtime_instance_numeric_id
            ),
            expected_runtime_service_account_email=(
                self._runtime_service_account_email
            ),
            expected_runtime_service_account_unique_id=(
                self._runtime_service_account_unique_id
            ),
            expected_target_service_account_unique_id=(
                self._target_service_account_unique_id
            ),
            metadata_instance_id=metadata_instance_id,
            metadata_service_account_email=(
                metadata_service_account_email
            ),
            metadata_scopes=metadata_scopes,
            collector_instance_resource=collector_instance,
            collector_service_account_resource=collector_account,
            collector_service_account_policy=collector_policy,
            collector_user_managed_keys=collector_user_keys,
            project_resource=resources[0],
            ancestor_resources=resources[1:],
            resource_policies=policies,
            role_resources=roles,
            collector_role_resources=collector_roles,
            additional_policy_role_resources=additional_policy_roles,
            service_account_resource=account,
            expected_external_gcp_admin_trust_root=(
                self._external_gcp_admin_trust_root
            ),
            expected_mutation_binding_present=(
                self._mutation_binding_present
            ),
            activation_seal_sha256=self._activation_seal_sha256,
            now_unix=now_unix,
        )
        return recollection, projection


def _url_suffix(value: Any, suffix: str) -> bool:
    return isinstance(value, str) and value.endswith(suffix)


def _cloud_recollection(
    instance: Mapping[str, Any],
    disk: Mapping[str, Any],
    *,
    observation: Mapping[str, Any],
) -> Mapping[str, Any]:
    status = instance.get("status")
    service_accounts = instance.get("serviceAccounts")
    attachments = instance.get("disks")
    if (
        instance.get("id") != contract.VM_INSTANCE_ID
        or instance.get("name") != contract.VM_NAME
        or status not in {"RUNNING", "TERMINATED"}
        or status != observation.get("instance_status")
        or not _url_suffix(instance.get("zone"), f"/zones/{contract.ZONE}")
        or not isinstance(service_accounts, list)
        or len(service_accounts) != 1
        or not isinstance(service_accounts[0], Mapping)
        or service_accounts[0].get("email") != contract.RUNTIME_SERVICE_ACCOUNT
        or service_accounts[0].get("scopes") != list(contract.RUNTIME_SCOPES)
        or not isinstance(attachments, list)
        or len(attachments) != 1
        or not isinstance(attachments[0], Mapping)
    ):
        raise TrustedObservationError(
            "trusted_observation_cloud_instance_invalid"
        )
    attachment = attachments[0]
    if (
        attachment.get("boot") is not True
        or attachment.get("autoDelete") is not True
        or attachment.get("deviceName") != contract.BOOT_DEVICE_NAME
        or attachment.get("mode") != "READ_WRITE"
        or attachment.get("type") != "PERSISTENT"
        or not _url_suffix(
            attachment.get("source"), f"/disks/{contract.DISK_NAME}"
        )
    ):
        raise TrustedObservationError(
            "trusted_observation_cloud_attachment_invalid"
        )
    disk_size = _nonnegative_integer(disk.get("sizeGb"))
    if (
        disk.get("id") != contract.DISK_ID
        or disk.get("name") != contract.DISK_NAME
        or disk.get("status") != "READY"
        or disk_size not in {contract.SOURCE_SIZE_GB, contract.TARGET_SIZE_GB}
        or disk_size != observation.get("disk_size_gb")
        or not _url_suffix(disk.get("zone"), f"/zones/{contract.ZONE}")
        or not _url_suffix(
            disk.get("type"), f"/diskTypes/{contract.DISK_TYPE}"
        )
        or not _url_suffix(
            disk.get("sourceImage"),
            f"/projects/{contract.SOURCE_IMAGE_PROJECT}/global/images/"
            f"{contract.SOURCE_IMAGE}",
        )
        or disk.get("architecture") != "X86_64"
        or str(disk.get("physicalBlockSizeBytes")) != "4096"
        or disk.get("users")
        != [
            "https://www.googleapis.com/compute/v1/projects/"
            f"{contract.PROJECT}/zones/{contract.ZONE}/instances/"
            f"{contract.VM_NAME}"
        ]
    ):
        raise TrustedObservationError(
            "trusted_observation_cloud_disk_invalid"
        )
    iam = observation.get("external_iam_receipt")
    if (
        not isinstance(iam, Mapping)
        or iam.get("service_account") != service_accounts[0]["email"]
        or iam.get("scopes") != service_accounts[0]["scopes"]
        or tuple(iam.get("roles") or ()) != contract.RUNTIME_ROLES
        or tuple(iam.get("permissions") or ()) != contract.RUNTIME_PERMISSIONS
        or observation.get("external_iam_policy_sha256")
        != contract.EXTERNAL_IAM_POLICY_SHA256
        or observation.get("external_iam_receipt_sha256")
        != protocol.sha256_json(iam)
    ):
        raise TrustedObservationError(
            "trusted_observation_cloud_iam_invalid"
        )
    projection = {
        "schema": "muncho-storage-growth-cloud-recollection.v1",
        "project": contract.PROJECT,
        "zone": contract.ZONE,
        "vm_name": instance["name"],
        "vm_instance_id": instance["id"],
        "instance_status": status,
        "disk_name": disk["name"],
        "disk_id": disk["id"],
        "disk_size_gb": disk_size,
        "boot_device_name": attachment["deviceName"],
        "runtime_service_account": service_accounts[0]["email"],
        "runtime_scopes": service_accounts[0]["scopes"],
        "external_iam_receipt_sha256": observation[
            "external_iam_receipt_sha256"
        ],
        "external_iam_policy_sha256": observation[
            "external_iam_policy_sha256"
        ],
    }
    expected = _expected_cloud_recollection(observation)
    if protocol.canonical_json_bytes(projection) != protocol.canonical_json_bytes(
        expected
    ):
        raise TrustedObservationError(
            "trusted_observation_cloud_candidate_mismatch"
        )
    return projection


def _expected_cloud_recollection(
    observation: Mapping[str, Any]
) -> Mapping[str, Any]:
    iam = observation["external_iam_receipt"]
    return {
        "schema": "muncho-storage-growth-cloud-recollection.v1",
        "project": observation["project"],
        "zone": observation["zone"],
        "vm_name": observation["vm_name"],
        "vm_instance_id": observation["vm_instance_id"],
        "instance_status": observation["instance_status"],
        "disk_name": observation["disk_name"],
        "disk_id": observation["disk_id"],
        "disk_size_gb": observation["disk_size_gb"],
        "boot_device_name": observation["boot_device_name"],
        "runtime_service_account": iam["service_account"],
        "runtime_scopes": iam["scopes"],
        "external_iam_receipt_sha256": observation[
            "external_iam_receipt_sha256"
        ],
        "external_iam_policy_sha256": observation[
            "external_iam_policy_sha256"
        ],
    }


def _seal_role_response(
    *,
    frame: Mapping[str, Any],
    request: Mapping[str, Any],
    observation: Mapping[str, Any],
    recollection: Mapping[str, Any],
    trusted_iam_projection: Mapping[str, Any],
    role: str,
    private_key: Ed25519PrivateKey,
    key_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    core = _bundle_core(
        request, observation, trusted_iam_projection
    )
    signature = private_key.sign(
        protocol.canonical_json_bytes(_role_payload(core, role=role))
    )
    if len(signature) != 64:
        raise TrustedObservationError(
            "trusted_observation_signature_invalid"
        )
    attestation = {
        "schema": evidence.ATTESTATION_SCHEMA,
        "public_key_id": key_id,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(
            signature
        ).rstrip(b"=").decode("ascii"),
    }
    unsigned = {
        "schema": ATTESTATION_RESPONSE_SCHEMA,
        "role": role,
        "attestation_request_sha256": frame["frame_sha256"],
        "observation_request_sha256": request[
            "observation_request_sha256"
        ],
        "observation_sha256": core["observation_sha256"],
        "trusted_iam_projection": core["trusted_iam_projection"],
        "trusted_iam_projection_sha256": core[
            "trusted_iam_projection_sha256"
        ],
        "bundle_core_sha256": protocol.sha256_json(core),
        "recollection_sha256": protocol.sha256_json(recollection),
        "attestation": attestation,
    }
    return (
        {**unsigned, "response_sha256": protocol.sha256_json(unsigned)},
        core,
    )


def _run_attestor(
    frame: Mapping[str, Any],
    *,
    role: str,
    config: Mapping[str, Any],
    host_collect: Callable[[], Mapping[str, Any]] | None,
    cloud_collect: Callable[
        [Mapping[str, Any], Mapping[str, Any], int],
        tuple[Mapping[str, Any], Mapping[str, Any]],
    ] | None,
    now_unix: int,
) -> Mapping[str, Any]:
    if role not in {"host", "cloud"}:
        raise TrustedObservationError("trusted_observation_role_invalid")
    checked_config = validate_attestor_config(config, expected_role=role)
    checked_frame, request, observation, trusted_iam_projection = (
        _validate_attestation_request(
        frame, expected_role=role, now_unix=now_unix
        )
    )
    if role == "host":
        if host_collect is None or cloud_collect is not None:
            raise TrustedObservationError("trusted_observation_role_invalid")
        recollection = host_collect()
        expected_recollection = _expected_host_recollection(observation)
        if protocol.canonical_json_bytes(recollection) != protocol.canonical_json_bytes(
            expected_recollection
        ):
            raise TrustedObservationError(
                "trusted_observation_host_candidate_mismatch"
            )
    else:
        if cloud_collect is None or host_collect is not None:
            raise TrustedObservationError("trusted_observation_role_invalid")
        collected = cloud_collect(observation, request, now_unix)
        if (
            not isinstance(collected, tuple)
            or len(collected) != 2
        ):
            raise TrustedObservationError(
                "trusted_observation_cloud_recollection_invalid"
            )
        recollection, trusted_iam_projection = collected
        expected_recollection = _expected_cloud_recollection(observation)
        if protocol.canonical_json_bytes(recollection) != protocol.canonical_json_bytes(
            expected_recollection
        ):
            raise TrustedObservationError(
                "trusted_observation_cloud_candidate_mismatch"
            )
        trusted_iam_projection = validate_trusted_iam_projection(
            trusted_iam_projection,
            observation_request=request,
            candidate_observation=observation,
            now_unix=now_unix,
        )
    if not isinstance(trusted_iam_projection, Mapping):
        raise TrustedObservationError(
            "trusted_observation_iam_projection_invalid"
        )
    # Re-validate every caller-controlled byte after the independent reads and
    # before signing.  The launcher also fetches the executor request again
    # after both signers return; combine_trusted_attestations requires equality.
    (
        frame_after,
        request_after,
        observation_after,
        trusted_iam_frame_after,
    ) = _validate_attestation_request(
        checked_frame, expected_role=role, now_unix=now_unix
    )
    if (
        protocol.canonical_json_bytes(frame_after)
        != protocol.canonical_json_bytes(checked_frame)
        or protocol.canonical_json_bytes(request_after)
        != protocol.canonical_json_bytes(request)
        or protocol.canonical_json_bytes(observation_after)
        != protocol.canonical_json_bytes(observation)
        or protocol.canonical_json_bytes(trusted_iam_frame_after)
        != protocol.canonical_json_bytes(
            checked_frame["trusted_iam_projection"]
        )
    ):
        raise TrustedObservationError(
            "trusted_observation_request_changed"
        )
    private_key, public_key, key_id = _load_private_key(checked_config)
    response, core = _seal_role_response(
        frame=checked_frame,
        request=request,
        observation=observation,
        recollection=recollection,
        trusted_iam_projection=trusted_iam_projection,
        role=role,
        private_key=private_key,
        key_id=key_id,
    )
    return _store_or_replay(
        response,
        config=checked_config,
        public_key=public_key,
        core=core,
    )


def run_host_attestor(
    frame: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    facts_reader: HostFactsReader,
    now_unix: int,
) -> Mapping[str, Any]:
    return _run_attestor(
        frame,
        role="host",
        config=config,
        host_collect=facts_reader.collect,
        cloud_collect=None,
        now_unix=now_unix,
    )


def run_cloud_attestor(
    frame: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    facts_reader: CloudFactsReader,
    now_unix: int,
) -> Mapping[str, Any]:
    return _run_attestor(
        frame,
        role="cloud",
        config=config,
        host_collect=None,
        cloud_collect=facts_reader.collect,
        now_unix=now_unix,
    )


def _decode_signature(value: Any) -> bytes:
    if not isinstance(value, str) or not value or "=" in value:
        raise TrustedObservationError(
            "trusted_observation_signature_invalid"
        )
    try:
        signature = base64.urlsafe_b64decode(
            (value + "=" * (-len(value) % 4)).encode("ascii")
        )
    except (UnicodeError, ValueError) as exc:
        raise TrustedObservationError(
            "trusted_observation_signature_invalid"
        ) from None
    if (
        len(signature) != 64
        or base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
        != value
    ):
        raise TrustedObservationError(
            "trusted_observation_signature_invalid"
        )
    return signature


def _validate_role_response(
    value: Any,
    *,
    expected_role: str,
    expected_request_sha256: str,
    expected_frame_sha256: str,
    core: Mapping[str, Any],
    public_key: Ed25519PublicKey,
) -> Mapping[str, Any]:
    if (
        expected_role not in {"cloud", "host"}
        or not isinstance(value, Mapping)
        or set(value) != _ATTESTATION_RESPONSE_FIELDS
        or not isinstance(public_key, Ed25519PublicKey)
    ):
        raise TrustedObservationError(
            "trusted_observation_attestation_response_fields_invalid"
        )
    response = _copy_json(value)
    unsigned = {
        name: item
        for name, item in response.items()
        if name != "response_sha256"
    }
    attestation = response.get("attestation")
    key_id = protocol.sha256_bytes(public_key.public_bytes_raw())
    if (
        response.get("schema") != ATTESTATION_RESPONSE_SCHEMA
        or response.get("role") != expected_role
        or response.get("attestation_request_sha256")
        != expected_frame_sha256
        or response.get("observation_request_sha256")
        != expected_request_sha256
        or response.get("observation_sha256") != core["observation_sha256"]
        or response.get("trusted_iam_projection")
        != core["trusted_iam_projection"]
        or response.get("trusted_iam_projection_sha256")
        != core["trusted_iam_projection_sha256"]
        or response.get("bundle_core_sha256") != protocol.sha256_json(core)
        or not _sha(response.get("recollection_sha256"))
        or response.get("response_sha256") != protocol.sha256_json(unsigned)
        or not isinstance(attestation, Mapping)
        or set(attestation) != _ATTESTATION_FIELDS
        or attestation.get("schema") != evidence.ATTESTATION_SCHEMA
        or attestation.get("public_key_id") != key_id
    ):
        raise TrustedObservationError(
            "trusted_observation_attestation_response_invalid"
        )
    signature = _decode_signature(
        attestation.get("signature_ed25519_b64url")
    )
    try:
        public_key.verify(
            signature,
            protocol.canonical_json_bytes(
                _role_payload(core, role=expected_role)
            ),
        )
    except InvalidSignature as exc:
        raise TrustedObservationError(
            "trusted_observation_signature_invalid"
        ) from None
    return response


def combine_trusted_attestations(
    *,
    observation_request: Mapping[str, Any],
    refreshed_observation_request: Mapping[str, Any],
    candidate_observation: Mapping[str, Any],
    cloud_response: Mapping[str, Any],
    cloud_public_key: Ed25519PublicKey,
    host_response: Mapping[str, Any] | None,
    host_public_key: Ed25519PublicKey,
    now_unix: int,
) -> Mapping[str, Any]:
    """Combine only independently verified signatures over one exact core.

    ``refreshed_observation_request`` must be fetched from the remote executor
    after both signers return.  Byte inequality proves the ledger checkpoint
    or prior event head moved and invalidates the entire collection attempt.
    """

    request = validate_observation_request(observation_request)
    refreshed = validate_observation_request(refreshed_observation_request)
    if protocol.canonical_json_bytes(request) != protocol.canonical_json_bytes(
        refreshed
    ):
        raise TrustedObservationError(
            "trusted_observation_request_changed"
        )
    observation = _validated_observation(
        request, candidate_observation, now_unix=now_unix
    )
    if not isinstance(cloud_response, Mapping):
        raise TrustedObservationError(
            "trusted_observation_attestation_response_fields_invalid"
        )
    trusted_iam_projection = validate_trusted_iam_projection(
        cloud_response.get("trusted_iam_projection"),
        observation_request=request,
        candidate_observation=observation,
        now_unix=now_unix,
    )
    core = _bundle_core(
        request, observation, trusted_iam_projection
    )
    cloud_frame = build_attestation_request(
        request,
        observation,
        role="cloud",
        trusted_iam_projection=None,
        now_unix=now_unix,
    )
    checked_cloud = _validate_role_response(
        cloud_response,
        expected_role="cloud",
        expected_request_sha256=request["observation_request_sha256"],
        expected_frame_sha256=cloud_frame["frame_sha256"],
        core=core,
        public_key=cloud_public_key,
    )
    needs_host = host_attestation_required(
        request, observation, now_unix=now_unix
    )
    if needs_host:
        if host_response is None:
            raise TrustedObservationError(
                "trusted_observation_host_attestor_unavailable"
            )
        host_frame = build_attestation_request(
            request,
            observation,
            role="host",
            trusted_iam_projection=trusted_iam_projection,
            now_unix=now_unix,
        )
        checked_host = _validate_role_response(
            host_response,
            expected_role="host",
            expected_request_sha256=request["observation_request_sha256"],
            expected_frame_sha256=host_frame["frame_sha256"],
            core=core,
            public_key=host_public_key,
        )
        host_attestation: Mapping[str, Any] | None = checked_host[
            "attestation"
        ]
    else:
        if host_response is not None:
            raise TrustedObservationError(
                "trusted_observation_host_attestation_forbidden"
            )
        host_attestation = None
    body = {
        **core,
        "cloud_attestation": checked_cloud["attestation"],
        "host_attestation": host_attestation,
    }
    bundle = {**body, "bundle_sha256": protocol.sha256_json(body)}
    try:
        return evidence.validate_attested_observation_structure(bundle)
    except evidence.StorageGrowthEvidenceError as exc:
        raise TrustedObservationError(
            "trusted_observation_bundle_invalid"
        ) from None


def _read_canonical_line(stream: BinaryIO) -> Mapping[str, Any]:
    raw = stream.read(MAX_FRAME_BYTES + 2)
    if (
        not isinstance(raw, bytes)
        or not raw
        or len(raw) > MAX_FRAME_BYTES + 1
        or not raw.endswith(b"\n")
        or b"\n" in raw[:-1]
    ):
        raise TrustedObservationError(
            "trusted_observation_stdin_frame_invalid"
        )
    try:
        value = protocol.decode_canonical_json(
            raw[:-1], maximum_bytes=MAX_FRAME_BYTES
        )
    except protocol.PasskeyV2ProtocolError as exc:
        raise TrustedObservationError(
            "trusted_observation_stdin_frame_invalid"
        ) from None
    if not isinstance(value, Mapping):
        raise TrustedObservationError(
            "trusted_observation_stdin_frame_invalid"
        )
    return value


def _write_canonical_line(stream: BinaryIO, value: Mapping[str, Any]) -> None:
    raw = protocol.canonical_json_bytes(value) + b"\n"
    if len(raw) > MAX_FRAME_BYTES + 1:
        raise TrustedObservationError(
            "trusted_observation_stdout_frame_invalid"
        )
    stream.write(raw)
    stream.flush()


def serve_host_attestor_once(
    *,
    stdin: BinaryIO,
    stdout: BinaryIO,
    config: Mapping[str, Any],
    facts_reader: HostFactsReader,
    now_unix: int,
) -> int:
    frame = _read_canonical_line(stdin)
    response = run_host_attestor(
        frame,
        config=config,
        facts_reader=facts_reader,
        now_unix=now_unix,
    )
    _write_canonical_line(stdout, response)
    return 0


def serve_cloud_attestor_once(
    *,
    stdin: BinaryIO,
    stdout: BinaryIO,
    config: Mapping[str, Any],
    facts_reader: CloudFactsReader,
    now_unix: int,
) -> int:
    frame = _read_canonical_line(stdin)
    response = run_cloud_attestor(
        frame,
        config=config,
        facts_reader=facts_reader,
        now_unix=now_unix,
    )
    _write_canonical_line(stdout, response)
    return 0


def host_attestor_main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments or os.geteuid() != 0:  # windows-footgun: ok — Debian root boundary
        raise TrustedObservationError(
            "trusted_observation_host_entrypoint_invalid"
        )
    config = _load_config(
        HOST_CONFIG_PATH,
        role="host",
        expected_uid=0,
        expected_path=HOST_CONFIG_PATH,
        expected_private_key_path=HOST_PRIVATE_KEY_PATH,
        expected_replay_directory=HOST_REPLAY_DIRECTORY,
    )
    return serve_host_attestor_once(
        stdin=sys.stdin.buffer,
        stdout=sys.stdout.buffer,
        config=config,
        facts_reader=FixedHostFactsReader(),
        now_unix=int(__import__("time").time()),
    )


def cloud_attestor_main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments or os.geteuid() != contract.AUTHORITATIVE_EXECUTOR_UID:  # windows-footgun: ok — Debian runtime boundary
        raise TrustedObservationError(
            "trusted_observation_cloud_entrypoint_invalid"
        )
    # Production invokes ``attest_cloud_observation_fixed`` in the executor
    # process so the strict release-pinned IAM authority values are supplied
    # from the already validated executor config.  This standalone entrypoint
    # has no caller-selectable config or safe source for those pins.
    raise TrustedObservationError(
        "trusted_observation_iam_authority_pin_unavailable"
    )


def load_cloud_attestor_config() -> Mapping[str, Any]:
    """Load only the executor-owned production cloud-attestor config."""

    return _load_config(
        CLOUD_CONFIG_PATH,
        role="cloud",
        expected_uid=contract.AUTHORITATIVE_EXECUTOR_UID,
        expected_path=CLOUD_CONFIG_PATH,
        expected_private_key_path=CLOUD_PRIVATE_KEY_PATH,
        expected_replay_directory=CLOUD_REPLAY_DIRECTORY,
    )


def attest_cloud_observation_fixed(
    frame: Mapping[str, Any],
    *,
    now_unix: int,
    direct_iam_pins: Mapping[str, Any],
    expected_mutation_binding_present: bool,
    activation_seal_sha256: str | None,
) -> Mapping[str, Any]:
    """Executor-only fixed production callback; no caller-selected paths."""

    if os.geteuid() != contract.AUTHORITATIVE_EXECUTOR_UID:  # windows-footgun: ok — Debian runtime boundary
        raise TrustedObservationError(
            "trusted_observation_cloud_entrypoint_invalid"
        )
    if (
        not isinstance(direct_iam_pins, Mapping)
        or set(direct_iam_pins) != _DIRECT_IAM_PIN_FIELDS
    ):
        raise TrustedObservationError(
            "trusted_observation_iam_authority_pin_invalid"
        )
    return run_cloud_attestor(
        frame,
        config=load_cloud_attestor_config(),
        facts_reader=FixedComputeFactsReader(
            expected_project_number=str(
                direct_iam_pins["expected_project_number"]
            ),
            expected_ancestor_chain=direct_iam_pins[
                "expected_ancestor_chain"
            ],
            expected_runtime_instance_numeric_id=str(
                direct_iam_pins[
                    "expected_runtime_instance_numeric_id"
                ]
            ),
            expected_runtime_service_account_email=str(
                direct_iam_pins[
                    "expected_runtime_service_account_email"
                ]
            ),
            expected_runtime_service_account_unique_id=str(
                direct_iam_pins[
                    "expected_runtime_service_account_unique_id"
                ]
            ),
            expected_target_service_account_unique_id=str(
                direct_iam_pins[
                    "expected_target_service_account_unique_id"
                ]
            ),
            expected_metadata_scopes=direct_iam_pins[
                "expected_metadata_scopes"
            ],
            expected_external_gcp_admin_trust_root=direct_iam_pins[
                "expected_external_gcp_admin_trust_root"
            ],
            expected_mutation_binding_present=(
                expected_mutation_binding_present
            ),
            activation_seal_sha256=activation_seal_sha256,
        ),
        now_unix=now_unix,
    )


__all__ = [
    "ATTESTATION_REQUEST_SCHEMA",
    "ATTESTATION_RESPONSE_SCHEMA",
    "CLOUD_CONFIG_SCHEMA",
    "FixedComputeFactsReader",
    "FixedHostFactsReader",
    "HOST_CONFIG_SCHEMA",
    "TrustedObservationError",
    "build_attestation_request",
    "attest_cloud_observation_fixed",
    "classify_observation_request",
    "cloud_attestor_main",
    "combine_trusted_attestations",
    "fixed_debian_tls_context",
    "host_attestation_required",
    "host_attestor_main",
    "load_cloud_attestor_config",
    "build_trusted_iam_projection",
    "run_cloud_attestor",
    "run_host_attestor",
    "serve_cloud_attestor_once",
    "serve_host_attestor_once",
    "validate_attestor_config",
    "validate_external_gcp_admin_trust_root",
    "validate_observation_request",
    "validate_trusted_iam_projection",
]
