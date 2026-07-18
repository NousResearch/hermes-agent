#!/usr/bin/env python3
"""Canonical release-bound pins for the owner-gate direct IAM reader.

The values in this asset are collected under an exact owner reauthentication
after the service account and VM exist.  The release trust manifest signs the
asset digest; package, stage zero, bootstrap, and runtime may therefore render
identity placeholders only from these bytes.  Missing values never default or
fall back to names inferred by the runtime.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from scripts.canary import owner_gate_foundation as foundation


SCHEMA = "muncho-owner-gate-direct-iam-identity-authority.v1"
PROJECT_NUMBER = "39589465056"
OWNER_GATE_SERVICE_ACCOUNT_EMAIL = (
    "muncho-owner-gate-executor@adventico-ai-platform.iam.gserviceaccount.com"
)
TARGET_SERVICE_ACCOUNT_EMAIL = (
    "muncho-canary-v2-runtime@adventico-ai-platform.iam.gserviceaccount.com"
)
MAX_BYTES = 256 * 1024

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_FOLDER = re.compile(r"^folders/[1-9][0-9]{5,30}$")
_ORGANIZATION = re.compile(r"^organizations/[1-9][0-9]{5,30}$")
_ROLE = re.compile(
    r"^(?:roles/[A-Za-z0-9_.]{1,128}|"
    r"(?:projects/[a-z][a-z0-9-]{4,62}|organizations/[1-9][0-9]{5,30})"
    r"/roles/[A-Za-z0-9_.]{1,128})$"
)
_PERMISSION = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{2,255}$")
class DirectIamIdentityAuthorityError(RuntimeError):
    """Stable validation failure for the non-secret signed pin asset."""


def _validate_external_admin_trust_root(
    value: Any,
    *,
    chain: list[Any],
    owner_receipt_sha256: Any,
) -> None:
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
        raise DirectIamIdentityAuthorityError(
            "direct_iam_identity_authority_external_admin_invalid"
        )
    generations = value.get("resource_policy_generations")
    bindings = value.get("allowed_residual_bindings")
    definitions = value.get("allowed_residual_role_definitions")
    allowed_resources = {
        f"projects/{foundation.PROJECT}",
        *(str(item) for item in chain),
    }
    if (
        value.get("inventory_complete") is not True
        or value.get("structural_partition_complete") is not True
        or value.get("passkey_protects_against_external_gcp_admins") is not False
        or value.get("passkey_protects_against_pinned_external_roots") is not False
        or value.get("google_provider_control_plane_outside_passkey") is not True
        or value.get("collected_under_owner_reauthentication_receipt_sha256")
        != owner_receipt_sha256
        or not isinstance(generations, list)
        or not isinstance(bindings, list)
        or not isinstance(definitions, list)
    ):
        raise DirectIamIdentityAuthorityError(
            "direct_iam_identity_authority_external_admin_invalid"
        )
    expected_generation_resources = [
        f"projects/{foundation.PROJECT}",
        *(str(item) for item in chain),
    ]
    if len(generations) != len(expected_generation_resources):
        raise DirectIamIdentityAuthorityError(
            "direct_iam_identity_authority_external_admin_invalid"
        )
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
            raise DirectIamIdentityAuthorityError(
                "direct_iam_identity_authority_external_admin_invalid"
            )
        try:
            foundation.canonical_json_bytes(generation["audit_configs"])
        except foundation.OwnerGateFoundationError as exc:
            raise DirectIamIdentityAuthorityError(
                "direct_iam_identity_authority_external_admin_invalid"
            ) from None

    canonical_bindings: list[bytes] = []
    binding_roles: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) != {
            "resource",
            "role",
            "members",
            "condition",
        }:
            raise DirectIamIdentityAuthorityError(
                "direct_iam_identity_authority_external_admin_invalid"
            )
        role = binding.get("role")
        members = binding.get("members")
        condition = binding.get("condition")
        if (
            binding.get("resource") not in allowed_resources
            or not isinstance(role, str)
            or _ROLE.fullmatch(role) is None
            or not isinstance(members, list)
            or not members
            or any(
                not isinstance(member, str)
                or not member
                or len(member) > 1024
                or any(
                    ord(character) < 0x20 or ord(character) == 0x7F
                    for character in member
                )
                for member in members
            )
            or members != sorted(members)
            or len(members) != len(set(members))
        ):
            raise DirectIamIdentityAuthorityError(
                "direct_iam_identity_authority_external_admin_invalid"
            )
        if condition is not None and (
            not isinstance(condition, Mapping)
            or set(condition) != {"title", "description", "expression"}
            or not isinstance(condition.get("title"), str)
            or not condition["title"]
            or not isinstance(condition.get("description"), str)
            or not isinstance(condition.get("expression"), str)
            or not condition["expression"]
            or any(
                len(condition[key]) > 4096
                or any(ord(character) < 0x20 for character in condition[key])
                for key in ("title", "description", "expression")
            )
        ):
            raise DirectIamIdentityAuthorityError(
                "direct_iam_identity_authority_external_admin_invalid"
            )
        canonical_bindings.append(foundation.canonical_json_bytes(binding))
        binding_roles.add(role)
    if canonical_bindings != sorted(set(canonical_bindings)):
        raise DirectIamIdentityAuthorityError(
            "direct_iam_identity_authority_external_admin_invalid"
        )
    definition_names: list[str] = []
    for definition in definitions:
        if not isinstance(definition, Mapping) or set(definition) != {
            "name",
            "title",
            "description",
            "included_permissions",
            "stage",
            "deleted",
            "etag",
        }:
            raise DirectIamIdentityAuthorityError(
                "direct_iam_identity_authority_external_admin_invalid"
            )
        name = definition.get("name")
        permissions = definition.get("included_permissions")
        if (
            not isinstance(name, str)
            or _ROLE.fullmatch(name) is None
            or not isinstance(definition.get("title"), str)
            or not definition["title"]
            or not isinstance(definition.get("description"), str)
            or not isinstance(permissions, list)
            or not permissions
            or any(
                not isinstance(permission, str)
                or _PERMISSION.fullmatch(permission) is None
                for permission in permissions
            )
            or permissions != sorted(permissions)
            or len(permissions) != len(set(permissions))
            or definition.get("stage")
            not in {"ALPHA", "BETA", "GA", "DEPRECATED", "DISABLED", "EAP"}
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
            raise DirectIamIdentityAuthorityError(
                "direct_iam_identity_authority_external_admin_invalid"
            )
        definition_names.append(name)
    if (
        definition_names != sorted(set(definition_names))
        or set(definition_names) != binding_roles
    ):
        raise DirectIamIdentityAuthorityError(
            "direct_iam_identity_authority_external_admin_invalid"
        )


def validate(value: Any, *, release_revision: str | None = None) -> Mapping[str, Any]:
    fields = {
        "schema",
        "release_revision",
        "project_id",
        "project_number",
        "owner_gate_vm_name",
        "owner_gate_vm_numeric_id",
        "owner_gate_service_account_email",
        "owner_gate_service_account_unique_id",
        "target_service_account_email",
        "target_service_account_unique_id",
        "resource_ancestor_chain",
        "project_read_role",
        "project_read_role_title",
        "project_read_role_description",
        "project_read_role_etag",
        "project_read_permissions",
        "project_read_binding_member",
        "project_read_binding_present",
        "ancestor_read_role",
        "ancestor_read_role_title",
        "ancestor_read_role_description",
        "ancestor_read_role_etag",
        "ancestor_read_permissions",
        "ancestor_binding_member",
        "ancestor_binding_present",
        "mutation_role",
        "mutation_role_title",
        "mutation_role_description",
        "mutation_role_etag",
        "mutation_permissions",
        "mutation_condition",
        "mutation_binding_member",
        "mutation_binding_present",
        "mutation_activation_seal",
        "mutation_activation_seal_present",
        "allowed_owner_gate_impersonators",
        "owner_gate_user_managed_key_inventory",
        "external_gcp_admin_trust_root",
        "metadata_oauth_scopes",
        "private_google_api_hosts",
        "private_google_api_vip_range",
        "owner_reauthentication_receipt_sha256",
        "pre_foundation_authority_sha256",
        "foundation_apply_receipt_sha256",
        "collected_at_unix",
        "authority_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DirectIamIdentityAuthorityError(
            "direct_iam_identity_authority_fields_invalid"
        )
    unsigned = {key: item for key, item in value.items() if key != "authority_sha256"}
    chain = value.get("resource_ancestor_chain")
    if (
        not isinstance(chain, list)
        or not chain
        or len(chain) != len(set(chain))
    ):
        raise DirectIamIdentityAuthorityError(
            "direct_iam_identity_authority_chain_invalid"
        )
    if (
        _ORGANIZATION.fullmatch(str(chain[-1])) is None
        or any(_FOLDER.fullmatch(str(item)) is None for item in chain[:-1])
    ):
        raise DirectIamIdentityAuthorityError(
            "direct_iam_identity_authority_chain_invalid"
        )
    organization_id = str(chain[-1]).split("/", 1)[1]
    expected_ancestor_role = (
        f"organizations/{organization_id}/roles/"
        f"{foundation.ANCESTOR_READ_ONLY_IAM_ROLE_ID}"
    )
    expected_member = f"serviceAccount:{OWNER_GATE_SERVICE_ACCOUNT_EMAIL}"
    expected_mutation_condition = {
        "title": foundation.MUTATION_CONDITION_TITLE,
        "description": foundation.MUTATION_CONDITION_DESCRIPTION,
        "expression": foundation._condition_expression(),
    }
    _validate_external_admin_trust_root(
        value.get("external_gcp_admin_trust_root"),
        chain=chain,
        owner_receipt_sha256=value.get("owner_reauthentication_receipt_sha256"),
    )
    if (
        value.get("schema") != SCHEMA
        or _REVISION.fullmatch(str(value.get("release_revision", ""))) is None
        or (
            release_revision is not None
            and value.get("release_revision") != release_revision
        )
        or value.get("project_id") != foundation.PROJECT
        or value.get("project_number") != PROJECT_NUMBER
        or value.get("owner_gate_vm_name") != foundation.VM_NAME
        or _NUMERIC_ID.fullmatch(
            str(value.get("owner_gate_vm_numeric_id", ""))
        )
        is None
        or value.get("owner_gate_service_account_email")
        != OWNER_GATE_SERVICE_ACCOUNT_EMAIL
        or _NUMERIC_ID.fullmatch(
            str(value.get("owner_gate_service_account_unique_id", ""))
        )
        is None
        or value.get("target_service_account_email")
        != TARGET_SERVICE_ACCOUNT_EMAIL
        or _NUMERIC_ID.fullmatch(
            str(value.get("target_service_account_unique_id", ""))
        )
        is None
        or value.get("project_read_role")
        != (
            f"projects/{foundation.PROJECT}/roles/"
            f"{foundation.READ_ONLY_IAM_ROLE_ID}"
        )
        or value.get("project_read_permissions")
        != list(foundation.READ_ONLY_IAM_PERMISSIONS)
        or value.get("project_read_role_title")
        != foundation.PROJECT_READ_ROLE_TITLE
        or value.get("project_read_role_description")
        != foundation.PROJECT_READ_ROLE_DESCRIPTION
        or (
            value.get("project_read_role_etag") is not None
            and (
                not isinstance(value["project_read_role_etag"], str)
                or not value["project_read_role_etag"]
                or len(value["project_read_role_etag"]) > 4096
            )
        )
        or value.get("project_read_binding_member") != expected_member
        or value.get("project_read_binding_present") is not True
        or value.get("ancestor_read_role") != expected_ancestor_role
        or value.get("ancestor_read_permissions")
        != list(foundation.DIRECT_IAM_ANCESTOR_PERMISSIONS)
        or value.get("ancestor_read_role_title")
        != foundation.ANCESTOR_READ_ROLE_TITLE
        or value.get("ancestor_read_role_description")
        != foundation.ANCESTOR_READ_ROLE_DESCRIPTION
        or (
            value.get("ancestor_read_role_etag") is not None
            and (
                not isinstance(value["ancestor_read_role_etag"], str)
                or not value["ancestor_read_role_etag"]
                or len(value["ancestor_read_role_etag"]) > 4096
            )
        )
        or value.get("ancestor_binding_member") != expected_member
        or value.get("ancestor_binding_present") is not True
        or value.get("mutation_role")
        != (
            f"projects/{foundation.PROJECT}/roles/"
            f"{foundation.MUTATION_ROLE_ID}"
        )
        or value.get("mutation_permissions")
        != list(foundation.MUTATION_PERMISSIONS)
        or value.get("mutation_role_title") != foundation.MUTATION_ROLE_TITLE
        or value.get("mutation_role_description")
        != foundation.MUTATION_ROLE_DESCRIPTION
        or (
            value.get("mutation_role_etag") is not None
            and (
                not isinstance(value["mutation_role_etag"], str)
                or not value["mutation_role_etag"]
                or len(value["mutation_role_etag"]) > 4096
            )
        )
        or value.get("mutation_condition") != expected_mutation_condition
        or value.get("mutation_binding_member") != expected_member
        or value.get("mutation_binding_present") is not False
        or value.get("mutation_activation_seal")
        != str(foundation.MUTATION_ENABLE_SEAL)
        or value.get("mutation_activation_seal_present") is not False
        or value.get("allowed_owner_gate_impersonators") != []
        or value.get("owner_gate_user_managed_key_inventory")
        != {
            "requested_key_types": ["USER_MANAGED"],
            "allowed_key_names": [],
        }
        or value.get("metadata_oauth_scopes")
        != list(foundation.OWNER_GATE_OAUTH_SCOPES)
        or value.get("private_google_api_hosts")
        != list(foundation.PRIVATE_GOOGLE_API_HOSTS)
        or value.get("private_google_api_vip_range")
        != foundation.PRIVATE_GOOGLE_API_VIP_RANGE
        or _SHA256.fullmatch(
            str(value.get("owner_reauthentication_receipt_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(value.get("pre_foundation_authority_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(value.get("foundation_apply_receipt_sha256", ""))
        )
        is None
        or type(value.get("collected_at_unix")) is not int
        or value["collected_at_unix"] <= 0
        or value.get("authority_sha256") != foundation.sha256_json(unsigned)
    ):
        raise DirectIamIdentityAuthorityError(
            "direct_iam_identity_authority_invalid"
        )
    return dict(value)


def decode_canonical(raw: bytes, *, release_revision: str | None = None) -> Mapping[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > MAX_BYTES:
        raise DirectIamIdentityAuthorityError(
            "direct_iam_identity_authority_invalid"
        )
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise DirectIamIdentityAuthorityError(
            "direct_iam_identity_authority_invalid"
        ) from None
    checked = validate(value, release_revision=release_revision)
    if foundation.canonical_json_bytes(checked) != raw:
        raise DirectIamIdentityAuthorityError(
            "direct_iam_identity_authority_not_canonical"
        )
    return checked


__all__ = [
    "DirectIamIdentityAuthorityError",
    "MAX_BYTES",
    "OWNER_GATE_SERVICE_ACCOUNT_EMAIL",
    "PROJECT_NUMBER",
    "SCHEMA",
    "TARGET_SERVICE_ACCOUNT_EMAIL",
    "decode_canonical",
    "validate",
]
