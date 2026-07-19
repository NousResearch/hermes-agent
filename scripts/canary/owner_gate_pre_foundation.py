#!/usr/bin/env python3
"""Signed, inert authority for creating the private Muncho owner-gate VM.

The final release package cannot exist until the VM and service account have
stable provider-assigned identities.  This module deliberately breaks that
cycle with a smaller, domain-separated authority which can authorize only the
exact inert foundation plan.  It cannot authorize package deployment, service
start, or the deferred Compute mutation binding.

The module contains validators and receipt construction only.  It does not
execute gcloud, a shell, or any remote command.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import ipaddress
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Never, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_owner_reauth as owner_reauth
from scripts.canary import owner_gate_project_ancestry as project_ancestry
from scripts.canary import owner_gate_trust as release_trust


AUTHORITY_SCHEMA = "muncho-owner-gate-pre-foundation-authority.v1"
APPLY_RECEIPT_SCHEMA = "muncho-owner-gate-foundation-apply-receipt.v1"
INERT_PLAN_SCHEMA = "muncho-owner-gate-inert-foundation-plan.v1"
AUTHORITY_PURPOSE = "muncho_owner_gate_exact_inert_pre_foundation"
APPLY_RECEIPT_PURPOSE = "muncho_owner_gate_exact_inert_foundation_apply"
AUTHORITY_SIGNATURE_DOMAIN = b"muncho-owner-gate/pre-foundation-authority/v1\x00"
APPLY_RECEIPT_SIGNATURE_DOMAIN = b"muncho-owner-gate/foundation-apply-receipt/v1\x00"
OWNER_ACCOUNT = "lomliev@adventico.com"
PYTHON_VERSION = "3.11.2"
MAX_AUTHORITY_TTL_SECONDS = foundation.PREFLIGHT_MAX_AGE_SECONDS
MAX_JSON_BYTES = 4 * 1024 * 1024

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_OWNER_SUBNET_ROUTE_NAME = re.compile(r"^default-route-r-[0-9a-f]{16}$")
_IMAGE = re.compile(
    r"^projects/debian-cloud/global/images/"
    r"debian-12-bookworm-v[0-9]{8}$"
)
_B64URL = re.compile(r"^[A-Za-z0-9_-]{86}$")
_OPAQUE_PROVIDER_TAG = re.compile(r"^[A-Za-z0-9_+/=.-]{1,256}$")
_RFC3339_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,9})?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)

_AUTHORITY_BODY_FIELDS = frozenset({
    "schema",
    "purpose",
    "foundation_source_revision",
    "foundation_source_tree_oid",
    "project",
    "region",
    "zone",
    "project_number",
    "organization_id",
    "ancestry_evidence_sha256",
    "ancestry_chain_sha256",
    "ancestry_collector_public_key_id",
    "ancestry_evidence_collected_at_unix",
    "ancestry_evidence_expires_at_unix",
    "signed_network_evidence_sha256",
    "network_evidence_sha256",
    "network_collector_public_key_id",
    "network_evidence_collected_at_unix",
    "interpreter_image",
    "inert_plan_sha256",
    "owner_reauthentication",
    "issued_at_unix",
    "expires_at_unix",
    "mutation_iam_binding_authorized",
    "package_deployment_authorized",
    "service_start_authorized",
    "final_package_inventory_present",
    "signer_key_id",
})
_AUTHORITY_FIELDS = _AUTHORITY_BODY_FIELDS | frozenset({
    "pre_foundation_authority_sha256",
    "signature_ed25519_b64url",
})
_STEP_RECEIPT_FIELDS = frozenset({
    "step_name",
    "argv_sha256",
    "disposition",
    "operation_receipt_sha256",
    "postcondition_receipt_sha256",
    "resource_identity",
})
_APPLY_BODY_FIELDS = frozenset({
    "schema",
    "purpose",
    "pre_foundation_authority_sha256",
    "inert_plan_sha256",
    "foundation_source_revision",
    "foundation_source_tree_oid",
    "project",
    "region",
    "zone",
    "project_number",
    "organization_id",
    "ancestry_evidence_sha256",
    "ancestry_chain_sha256",
    "ancestry_collector_public_key_id",
    "owner_reauthentication_receipt_sha256",
    "started_at_unix",
    "completed_at_unix",
    "applied_steps",
    "partial_unknown_state",
    "mutation_iam_binding_created",
    "package_deployed",
    "service_started",
    "signer_key_id",
})
_APPLY_FIELDS = _APPLY_BODY_FIELDS | frozenset({
    "foundation_apply_receipt_sha256",
    "signature_ed25519_b64url",
})


class OwnerGatePreFoundationError(RuntimeError):
    """Stable, secret-free pre-foundation authority failure."""


def _error(code: str, exc: BaseException | None = None) -> Never:
    del exc
    raise OwnerGatePreFoundationError(code) from None


def _canonical(value: Any) -> bytes:
    try:
        return foundation.canonical_json_bytes(value)
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_pre_foundation_json_invalid", exc)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _strict_mapping(
    value: Any,
    fields: frozenset[str],
    *,
    code: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != fields:
        _error(code)
    return value


def _decode_signature(value: Any, *, code: str) -> bytes:
    if not isinstance(value, str) or _B64URL.fullmatch(value) is None:
        _error(code)
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        _error(code, exc)
    if (
        len(raw) != 64
        or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        != value
    ):
        _error(code)
    return raw


def _key_id(public_key: Ed25519PublicKey) -> str:
    if not isinstance(public_key, Ed25519PublicKey):
        _error("owner_gate_pre_foundation_signer_invalid")
    return hashlib.sha256(public_key.public_bytes_raw()).hexdigest()


def _require_pinned_public_key(public_key: Ed25519PublicKey) -> str:
    key_id = _key_id(public_key)
    if (
        _SHA256.fullmatch(
            release_trust.PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256 or ""
        )
        is None
        or key_id != release_trust.PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256
    ):
        _error("owner_gate_pre_foundation_signer_not_pinned")
    return key_id


def load_pinned_public_key(
    path: Path,
    *,
    expected_uid: int | None = None,
) -> Ed25519PublicKey:
    """Load the raw release-author public key under the fixed fork pin."""

    uid = os.geteuid() if expected_uid is None else expected_uid  # windows-footgun: ok — POSIX owner boundary
    try:
        raw = release_trust._read_immutable(
            path,
            maximum=32,
            expected_uid=uid,
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
        key = Ed25519PublicKey.from_public_bytes(raw)
    except (ValueError, release_trust.OwnerGateTrustError) as exc:
        _error("owner_gate_pre_foundation_public_key_invalid", exc)
    _require_pinned_public_key(key)
    return key


def _signed_network_evidence_sha256(
    evidence: foundation.ProductionNetworkEvidence,
) -> str:
    return _sha256_json(asdict(evidence))


def _validated_project_ancestry(
    *,
    raw: bytes,
    collector_public_key: Ed25519PublicKey,
    owner_reauthentication_receipt: Mapping[str, Any],
    owner_reauthentication_public_key: Ed25519PublicKey,
    expected_release_revision: str,
    now_unix: int,
) -> project_ancestry.ProjectAncestryEvidence:
    try:
        return project_ancestry.decode_canonical_project_ancestry_evidence(
            raw,
            collector_public_key=collector_public_key,
            owner_reauthentication_receipt=owner_reauthentication_receipt,
            owner_reauthentication_public_key=(
                owner_reauthentication_public_key
            ),
            expected_release_revision=expected_release_revision,
            now_unix=now_unix,
        )
    except project_ancestry.OwnerGateProjectAncestryError as exc:
        _error("owner_gate_pre_foundation_ancestry_evidence_invalid", exc)


def inert_plan_projection(
    plan: foundation.OwnerGateFoundationPlan,
) -> Mapping[str, Any]:
    """Project only the exact pre-foundation operations that are authorized."""

    spec = plan.spec
    try:
        spec.validate()
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_pre_foundation_plan_invalid", exc)
    if (
        plan.schema != foundation.PLAN_SCHEMA
        or not spec.pre_foundation_bound
        or spec.final_release_bound
        or plan.deferred_private_api_connectivity_steps
        or plan.architecture.get("pre_foundation_only") is not True
        or plan.architecture.get("final_package_inventory_bound") is not False
        or plan.architecture.get("package_deployment_authorized") is not False
        or plan.architecture.get("service_start_authorized") is not False
        or plan.architecture.get("mutation_iam_enabled_during_bootstrap") is not False
        or plan.architecture.get("network_connectivity_api_disabled") is not True
        or type(
            plan.architecture.get("network_evidence_collected_at_unix")
        )
        is not int
        or plan.architecture["network_evidence_collected_at_unix"] <= 0
        or "preexisting_owner_gate_subnet_identity" not in plan.architecture
        or "preexisting_owner_gate_subnet_route_identity"
        not in plan.architecture
    ):
        _error("owner_gate_pre_foundation_plan_invalid")
    spec_payload = {
        key: value
        for key, value in asdict(spec).items()
        if value is not None
    }
    if any(
        key in spec_payload
        for key in (
            "package_inventory_sha256",
            "cloud_collector_public_key_id",
            "host_collector_public_key_id",
        )
    ):
        _error("owner_gate_pre_foundation_plan_invalid")
    return {
        "schema": INERT_PLAN_SCHEMA,
        "foundation_plan_schema": plan.schema,
        "spec": spec_payload,
        "network_evidence_sha256": plan.network_evidence_sha256,
        "architecture": dict(plan.architecture),
        "foundation_steps": [asdict(step) for step in plan.foundation_steps],
        "deferred_private_api_connectivity_steps": [],
        "deferred_mutation_iam_steps_authorized": False,
        "rollback_steps_authorized_without_fresh_owner_reauthentication": False,
        "mutation_iam_binding_authorized": False,
        "package_deployment_authorized": False,
        "service_start_authorized": False,
    }


def inert_plan_sha256(plan: foundation.OwnerGateFoundationPlan) -> str:
    return _sha256_json(inert_plan_projection(plan))


def _interpreter_image_from_spec(
    spec: foundation.OwnerGateSpec,
) -> Mapping[str, Any]:
    if not spec.pre_foundation_bound:
        _error("owner_gate_pre_foundation_plan_invalid")
    image_name = spec.boot_image_self_link.rsplit("/", 1)[-1]
    return {
        "project": "debian-cloud",
        "image_name": image_name,
        "image_numeric_id": spec.boot_image_numeric_id,
        "image_self_link": (
            "https://www.googleapis.com/compute/v1/"
            + spec.boot_image_self_link
        ),
        "python_version": PYTHON_VERSION,
        "interpreter_sha256": spec.interpreter_sha256,
    }


def build_authority_body(
    *,
    plan: foundation.OwnerGateFoundationPlan,
    network_evidence: foundation.ProductionNetworkEvidence,
    network_collector_public_key: Ed25519PublicKey,
    project_ancestry_evidence_raw: bytes,
    project_ancestry_collector_public_key: Ed25519PublicKey,
    owner_reauthentication_receipt: Mapping[str, Any],
    owner_reauthentication_public_key: Ed25519PublicKey,
    issued_at_unix: int,
    expires_at_unix: int,
    signer_key_id: str,
) -> Mapping[str, Any]:
    """Build the canonical unsigned authority body for offline signing."""

    projection_sha256 = inert_plan_sha256(plan)
    spec = plan.spec
    try:
        network_evidence.validate(now_unix=issued_at_unix)
        network_evidence.verify_attestation(
            public_key=network_collector_public_key,
            expected_public_key_id=spec.network_collector_public_key_id,
        )
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_pre_foundation_network_evidence_invalid", exc)
    try:
        checked_reauth = owner_reauth.validate_owner_reauth_receipt(
            owner_reauthentication_receipt,
            public_key=owner_reauthentication_public_key,
            now_unix=issued_at_unix,
        )
    except owner_reauth.OwnerGateOwnerReauthError as exc:
        _error("owner_gate_pre_foundation_owner_reauth_invalid", exc)
    checked_ancestry = _validated_project_ancestry(
        raw=project_ancestry_evidence_raw,
        collector_public_key=project_ancestry_collector_public_key,
        owner_reauthentication_receipt=owner_reauthentication_receipt,
        owner_reauthentication_public_key=(
            owner_reauthentication_public_key
        ),
        expected_release_revision=spec.release_revision,
        now_unix=issued_at_unix,
    )
    ancestry_value = checked_ancestry.value
    if (
        checked_ancestry.organization_id != spec.organization_id
        or checked_ancestry.signed_evidence_sha256
        != spec.ancestry_evidence_sha256
        or checked_ancestry.collector_public_key_id
        != network_evidence.collector_public_key_id
        or project_ancestry_collector_public_key.public_bytes_raw()
        != network_collector_public_key.public_bytes_raw()
    ):
        _error("owner_gate_pre_foundation_ancestry_evidence_mismatch")
    body = {
        "schema": AUTHORITY_SCHEMA,
        "purpose": AUTHORITY_PURPOSE,
        "foundation_source_revision": spec.release_revision,
        "foundation_source_tree_oid": spec.source_tree_oid,
        "project": spec.project,
        "region": spec.region,
        "zone": spec.zone,
        "project_number": checked_ancestry.project_number,
        "organization_id": checked_ancestry.organization_id,
        "ancestry_evidence_sha256": (
            checked_ancestry.signed_evidence_sha256
        ),
        "ancestry_chain_sha256": ancestry_value["stable_chain_sha256"],
        "ancestry_collector_public_key_id": (
            checked_ancestry.collector_public_key_id
        ),
        "ancestry_evidence_collected_at_unix": ancestry_value[
            "collected_at_unix"
        ],
        "ancestry_evidence_expires_at_unix": ancestry_value[
            "expires_at_unix"
        ],
        "signed_network_evidence_sha256": (
            _signed_network_evidence_sha256(network_evidence)
        ),
        "network_evidence_sha256": network_evidence.evidence_sha256,
        "network_collector_public_key_id": (
            network_evidence.collector_public_key_id
        ),
        "network_evidence_collected_at_unix": (
            network_evidence.collected_at_unix
        ),
        "interpreter_image": _interpreter_image_from_spec(spec),
        "inert_plan_sha256": projection_sha256,
        "owner_reauthentication": {
            "account": OWNER_ACCOUNT,
            "receipt_sha256": checked_reauth[
                "owner_reauthentication_receipt_sha256"
            ],
            "expires_at_unix": checked_reauth["expires_at_unix"],
        },
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
        "mutation_iam_binding_authorized": False,
        "package_deployment_authorized": False,
        "service_start_authorized": False,
        "final_package_inventory_present": False,
        "signer_key_id": signer_key_id,
    }
    _validate_authority_body(body, now_unix=issued_at_unix)
    return body


def _validate_authority_body(
    value: Any,
    *,
    now_unix: int | None,
) -> Mapping[str, Any]:
    body = _strict_mapping(
        value,
        _AUTHORITY_BODY_FIELDS,
        code="owner_gate_pre_foundation_authority_invalid",
    )
    image = _strict_mapping(
        body.get("interpreter_image"),
        frozenset({
            "project",
            "image_name",
            "image_numeric_id",
            "image_self_link",
            "python_version",
            "interpreter_sha256",
        }),
        code="owner_gate_pre_foundation_authority_invalid",
    )
    owner = _strict_mapping(
        body.get("owner_reauthentication"),
        frozenset({"account", "receipt_sha256", "expires_at_unix"}),
        code="owner_gate_pre_foundation_authority_invalid",
    )
    issued = body.get("issued_at_unix")
    expires = body.get("expires_at_unix")
    owner_expires = owner.get("expires_at_unix")
    ancestry_collected = body.get("ancestry_evidence_collected_at_unix")
    ancestry_expires = body.get("ancestry_evidence_expires_at_unix")
    boot_link = str(image.get("image_self_link", ""))
    short_link = boot_link.removeprefix(
        "https://www.googleapis.com/compute/v1/"
    )
    if (
        body.get("schema") != AUTHORITY_SCHEMA
        or body.get("purpose") != AUTHORITY_PURPOSE
        or _REVISION.fullmatch(
            str(body.get("foundation_source_revision", ""))
        )
        is None
        or _REVISION.fullmatch(
            str(body.get("foundation_source_tree_oid", ""))
        )
        is None
        or body.get("project") != foundation.PROJECT
        or body.get("region") != foundation.REGION
        or body.get("zone") != foundation.ZONE
        or _NUMERIC_ID.fullmatch(str(body.get("project_number", "")))
        is None
        or _NUMERIC_ID.fullmatch(str(body.get("organization_id", "")))
        is None
        or any(
            _SHA256.fullmatch(str(body.get(field, ""))) is None
            for field in (
                "ancestry_evidence_sha256",
                "ancestry_chain_sha256",
                "ancestry_collector_public_key_id",
                "signed_network_evidence_sha256",
                "network_evidence_sha256",
                "network_collector_public_key_id",
                "inert_plan_sha256",
                "signer_key_id",
            )
        )
        or type(body.get("network_evidence_collected_at_unix")) is not int
        or type(ancestry_collected) is not int
        or type(ancestry_expires) is not int
        or image.get("project") != "debian-cloud"
        or not isinstance(image.get("image_name"), str)
        or _IMAGE.fullmatch(short_link) is None
        or image.get("image_name") != short_link.rsplit("/", 1)[-1]
        or _NUMERIC_ID.fullmatch(str(image.get("image_numeric_id", "")))
        is None
        or boot_link
        != "https://www.googleapis.com/compute/v1/" + short_link
        or image.get("python_version") != "3.11.2"
        or _SHA256.fullmatch(str(image.get("interpreter_sha256", "")))
        is None
        or owner.get("account") != OWNER_ACCOUNT
        or _SHA256.fullmatch(str(owner.get("receipt_sha256", ""))) is None
        or type(issued) is not int
        or type(expires) is not int
        or type(owner_expires) is not int
        or issued <= 0
        or expires <= issued
        or expires - issued > MAX_AUTHORITY_TTL_SECONDS
        or owner_expires < expires
        or ancestry_collected > issued
        or issued - ancestry_collected
        > foundation.PREFLIGHT_MAX_AGE_SECONDS
        or ancestry_expires < expires
        or body.get("network_evidence_collected_at_unix") > issued
        or issued - body["network_evidence_collected_at_unix"]
        > foundation.PREFLIGHT_MAX_AGE_SECONDS
        or body.get("mutation_iam_binding_authorized") is not False
        or body.get("package_deployment_authorized") is not False
        or body.get("service_start_authorized") is not False
        or body.get("final_package_inventory_present") is not False
    ):
        _error("owner_gate_pre_foundation_authority_invalid")
    if now_unix is not None and (
        type(now_unix) is not int
        or now_unix < issued
        or now_unix > expires
    ):
        _error("owner_gate_pre_foundation_authority_expired")
    return dict(body)


def sign_pre_foundation_authority(
    body: Mapping[str, Any],
    *,
    private_key: Ed25519PrivateKey,
    owner_reauthentication_receipt: Mapping[str, Any],
    project_ancestry_evidence_raw: bytes,
    project_ancestry_collector_public_key: Ed25519PublicKey,
) -> Mapping[str, Any]:
    """Sign one validated authority with the fixed release-author key."""

    checked = _validate_authority_body(body, now_unix=body.get("issued_at_unix"))
    if not isinstance(private_key, Ed25519PrivateKey):
        _error("owner_gate_pre_foundation_signer_invalid")
    key_id = _require_pinned_public_key(private_key.public_key())
    try:
        checked_reauth = owner_reauth.validate_owner_reauth_receipt(
            owner_reauthentication_receipt,
            public_key=private_key.public_key(),
            now_unix=checked["issued_at_unix"],
        )
    except owner_reauth.OwnerGateOwnerReauthError as exc:
        _error("owner_gate_pre_foundation_owner_reauth_invalid", exc)
    if (
        checked["signer_key_id"] != key_id
        or checked["owner_reauthentication"]
        != {
            "account": OWNER_ACCOUNT,
            "receipt_sha256": checked_reauth[
                "owner_reauthentication_receipt_sha256"
            ],
            "expires_at_unix": checked_reauth["expires_at_unix"],
        }
    ):
        _error("owner_gate_pre_foundation_signer_invalid")
    checked_ancestry = _validated_project_ancestry(
        raw=project_ancestry_evidence_raw,
        collector_public_key=project_ancestry_collector_public_key,
        owner_reauthentication_receipt=owner_reauthentication_receipt,
        owner_reauthentication_public_key=private_key.public_key(),
        expected_release_revision=checked["foundation_source_revision"],
        now_unix=checked["issued_at_unix"],
    )
    if (
        checked["project_number"] != checked_ancestry.project_number
        or checked["organization_id"] != checked_ancestry.organization_id
        or checked["ancestry_evidence_sha256"]
        != checked_ancestry.signed_evidence_sha256
        or checked["ancestry_chain_sha256"]
        != checked_ancestry.value["stable_chain_sha256"]
        or checked["ancestry_collector_public_key_id"]
        != checked_ancestry.collector_public_key_id
    ):
        _error("owner_gate_pre_foundation_ancestry_evidence_mismatch")
    digest = _sha256_json(checked)
    signed_payload = {
        **checked,
        "pre_foundation_authority_sha256": digest,
    }
    signature = private_key.sign(
        AUTHORITY_SIGNATURE_DOMAIN + _canonical(signed_payload)
    )
    if len(signature) != 64:
        _error("owner_gate_pre_foundation_signature_invalid")
    return {
        **signed_payload,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode("ascii"),
    }


def validate_pre_foundation_authority(
    value: Any,
    *,
    public_key: Ed25519PublicKey,
    owner_reauthentication_receipt: Mapping[str, Any],
    now_unix: int | None = None,
    expected_plan: foundation.OwnerGateFoundationPlan | None = None,
    network_evidence: foundation.ProductionNetworkEvidence | None = None,
    network_collector_public_key: Ed25519PublicKey | None = None,
    project_ancestry_evidence_raw: bytes,
    project_ancestry_collector_public_key: Ed25519PublicKey,
) -> Mapping[str, Any]:
    """Validate signature, freshness, evidence, and the rebuilt inert plan."""

    authority = _strict_mapping(
        value,
        _AUTHORITY_FIELDS,
        code="owner_gate_pre_foundation_authority_invalid",
    )
    body = {
        key: item
        for key, item in authority.items()
        if key not in {
            "pre_foundation_authority_sha256",
            "signature_ed25519_b64url",
        }
    }
    checked = _validate_authority_body(body, now_unix=now_unix)
    key_id = _require_pinned_public_key(public_key)
    try:
        checked_reauth = owner_reauth.validate_owner_reauth_receipt(
            owner_reauthentication_receipt,
            public_key=public_key,
            now_unix=checked["issued_at_unix"],
        )
    except owner_reauth.OwnerGateOwnerReauthError as exc:
        _error("owner_gate_pre_foundation_owner_reauth_invalid", exc)
    signed_payload = {
        **checked,
        "pre_foundation_authority_sha256": authority.get(
            "pre_foundation_authority_sha256"
        ),
    }
    if (
        checked["signer_key_id"] != key_id
        or authority.get("pre_foundation_authority_sha256")
        != _sha256_json(checked)
        or checked["owner_reauthentication"]
        != {
            "account": OWNER_ACCOUNT,
            "receipt_sha256": checked_reauth[
                "owner_reauthentication_receipt_sha256"
            ],
            "expires_at_unix": checked_reauth["expires_at_unix"],
        }
    ):
        _error("owner_gate_pre_foundation_authority_invalid")
    try:
        public_key.verify(
            _decode_signature(
                authority.get("signature_ed25519_b64url"),
                code="owner_gate_pre_foundation_signature_invalid",
            ),
            AUTHORITY_SIGNATURE_DOMAIN + _canonical(signed_payload),
        )
    except InvalidSignature as exc:
        _error("owner_gate_pre_foundation_signature_invalid", exc)
    checked_ancestry = _validated_project_ancestry(
        raw=project_ancestry_evidence_raw,
        collector_public_key=project_ancestry_collector_public_key,
        owner_reauthentication_receipt=owner_reauthentication_receipt,
        owner_reauthentication_public_key=public_key,
        expected_release_revision=checked["foundation_source_revision"],
        now_unix=(checked["issued_at_unix"] if now_unix is None else now_unix),
    )
    if (
        checked["project_number"] != checked_ancestry.project_number
        or checked["organization_id"] != checked_ancestry.organization_id
        or checked["ancestry_evidence_sha256"]
        != checked_ancestry.signed_evidence_sha256
        or checked["ancestry_chain_sha256"]
        != checked_ancestry.value["stable_chain_sha256"]
        or checked["ancestry_collector_public_key_id"]
        != checked_ancestry.collector_public_key_id
        or checked["ancestry_evidence_collected_at_unix"]
        != checked_ancestry.value["collected_at_unix"]
        or checked["ancestry_evidence_expires_at_unix"]
        != checked_ancestry.value["expires_at_unix"]
    ):
        _error("owner_gate_pre_foundation_ancestry_evidence_mismatch")
    if expected_plan is not None and (
        authority["inert_plan_sha256"] != inert_plan_sha256(expected_plan)
        or authority["foundation_source_revision"]
        != expected_plan.spec.release_revision
        or authority["foundation_source_tree_oid"]
        != expected_plan.spec.source_tree_oid
        or authority["organization_id"]
        != expected_plan.spec.organization_id
        or authority["ancestry_evidence_sha256"]
        != expected_plan.spec.ancestry_evidence_sha256
        or authority["interpreter_image"]
        != _interpreter_image_from_spec(expected_plan.spec)
    ):
        _error("owner_gate_pre_foundation_plan_mismatch")
    if (network_evidence is None) != (network_collector_public_key is None):
        _error("owner_gate_pre_foundation_network_evidence_invalid")
    if network_evidence is not None:
        assert network_collector_public_key is not None
        try:
            network_evidence.validate(now_unix=now_unix)
            network_evidence.verify_attestation(
                public_key=network_collector_public_key,
                expected_public_key_id=authority[
                    "network_collector_public_key_id"
                ],
            )
        except foundation.OwnerGateFoundationError as exc:
            _error("owner_gate_pre_foundation_network_evidence_invalid", exc)
        if (
            authority["signed_network_evidence_sha256"]
            != _signed_network_evidence_sha256(network_evidence)
            or authority["network_evidence_sha256"]
            != network_evidence.evidence_sha256
            or authority["network_evidence_collected_at_unix"]
            != network_evidence.collected_at_unix
            or network_collector_public_key.public_bytes_raw()
            != project_ancestry_collector_public_key.public_bytes_raw()
            or authority["network_collector_public_key_id"]
            != authority["ancestry_collector_public_key_id"]
        ):
            _error("owner_gate_pre_foundation_network_evidence_mismatch")
    return dict(authority)


def spec_from_authority(value: Mapping[str, Any]) -> foundation.OwnerGateSpec:
    """Recover only the pre-foundation spec; no final package fields exist."""

    image = value.get("interpreter_image")
    if not isinstance(image, Mapping):
        _error("owner_gate_pre_foundation_authority_invalid")
    spec = foundation.OwnerGateSpec(
        release_revision=str(value.get("foundation_source_revision", "")),
        source_tree_oid=str(value.get("foundation_source_tree_oid", "")),
        boot_image_self_link=str(image.get("image_self_link", "")).removeprefix(
            "https://www.googleapis.com/compute/v1/"
        ),
        boot_image_numeric_id=str(image.get("image_numeric_id", "")),
        interpreter_sha256=str(image.get("interpreter_sha256", "")),
        network_collector_public_key_id=str(
            value.get("network_collector_public_key_id", "")
        ),
        organization_id=str(value.get("organization_id", "")),
        ancestry_evidence_sha256=str(
            value.get("ancestry_evidence_sha256", "")
        ),
    )
    try:
        spec.validate()
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_pre_foundation_authority_invalid", exc)
    return spec


def _provider_link(relative: str) -> str:
    return "https://www.googleapis.com/compute/v1/" + relative


def _provider_relative(value: Any) -> str:
    prefix = "https://www.googleapis.com/compute/v1/"
    if not isinstance(value, str) or not value.startswith(prefix):
        _error("owner_gate_foundation_apply_identity_invalid")
    relative = value.removeprefix(prefix)
    if not relative.startswith("projects/"):
        _error("owner_gate_foundation_apply_identity_invalid")
    return relative


def _provider_tag(value: Any) -> bool:
    return (
        isinstance(value, str)
        and _OPAQUE_PROVIDER_TAG.fullmatch(value) is not None
    )


def _validate_resource_identity(
    step_name: str,
    value: Any,
    *,
    plan: foundation.OwnerGateFoundationPlan,
) -> Mapping[str, Any]:
    spec = plan.spec
    member = f"serviceAccount:{spec.service_account_email}"
    network = _provider_link(
        f"projects/{spec.project}/global/networks/"
        f"{foundation.NETWORK_NAME}"
    )
    subnet_relative = (
        f"projects/{spec.project}/regions/{spec.region}/subnetworks/"
        f"{foundation.OWNER_GATE_SUBNET_NAME}"
    )
    if step_name == "create_dedicated_service_account":
        item = _strict_mapping(
            value,
            frozenset({
                "resource_type",
                "resource_name",
                "email",
                "unique_id",
                "etag",
                "disabled",
                "user_managed_key_count",
                "user_managed_keys",
            }),
            code="owner_gate_foundation_apply_identity_invalid",
        )
        if (
            item.get("resource_type") != "iam_service_account"
            or item.get("resource_name")
            != (
                f"projects/{spec.project}/serviceAccounts/"
                f"{spec.service_account_email}"
            )
            or item.get("email") != spec.service_account_email
            or _NUMERIC_ID.fullmatch(str(item.get("unique_id", ""))) is None
            or not _provider_tag(item.get("etag"))
            or item.get("disabled") is not False
            or item.get("user_managed_key_count") != 0
            or item.get("user_managed_keys") != []
        ):
            _error("owner_gate_foundation_apply_identity_invalid")
        return dict(item)
    role_contracts = {
        "create_narrow_iam_observation_reader_role": (
            "project_custom_role",
            spec.read_only_iam_role,
            foundation.PROJECT_READ_ROLE_TITLE,
            foundation.PROJECT_READ_ROLE_DESCRIPTION,
            list(foundation.READ_ONLY_IAM_PERMISSIONS),
        ),
        "create_narrow_storage_executor_role": (
            "project_custom_role",
            spec.custom_role,
            foundation.MUTATION_ROLE_TITLE,
            foundation.MUTATION_ROLE_DESCRIPTION,
            list(foundation.MUTATION_PERMISSIONS),
        ),
        "create_narrow_organization_iam_observation_reader_role": (
            "organization_custom_role",
            spec.ancestor_read_only_iam_role,
            foundation.ANCESTOR_READ_ROLE_TITLE,
            foundation.ANCESTOR_READ_ROLE_DESCRIPTION,
            list(foundation.DIRECT_IAM_ANCESTOR_PERMISSIONS),
        ),
    }
    if step_name in role_contracts:
        item = _strict_mapping(
            value,
            frozenset({
                "resource_type",
                "name",
                "etag",
                "title",
                "description",
                "stage",
                "included_permissions",
                "deleted",
            }),
            code="owner_gate_foundation_apply_identity_invalid",
        )
        kind, name, title, description, permissions = role_contracts[step_name]
        if (
            item.get("resource_type") != kind
            or item.get("name") != name
            or not _provider_tag(item.get("etag"))
            or item.get("title") != title
            or item.get("description") != description
            or item.get("stage") != "GA"
            or item.get("included_permissions") != permissions
            or item.get("deleted") is not False
        ):
            _error("owner_gate_foundation_apply_identity_invalid")
        return dict(item)
    binding_contracts = {
        "bind_narrow_iam_observation_reader_to_owner_gate_service_account": (
            "project_iam_binding",
            f"projects/{spec.project}",
            spec.read_only_iam_role,
        ),
        "bind_narrow_organization_iam_observation_reader_to_owner_gate_service_account": (
            "organization_iam_binding",
            spec.organization_resource,
            spec.ancestor_read_only_iam_role,
        ),
    }
    if step_name in binding_contracts:
        item = _strict_mapping(
            value,
            frozenset({
                "resource_type",
                "resource_name",
                "role",
                "member",
                "condition",
                "policy_etag",
                "policy_version",
                "matching_binding_count",
                "matching_member_occurrences",
                "binding_members",
            }),
            code="owner_gate_foundation_apply_identity_invalid",
        )
        kind, resource_name, role = binding_contracts[step_name]
        if (
            item.get("resource_type") != kind
            or item.get("resource_name") != resource_name
            or item.get("role") != role
            or item.get("member") != member
            or item.get("condition") is not None
            or not _provider_tag(item.get("policy_etag"))
            or item.get("policy_version") not in {1, 3}
            or item.get("matching_binding_count") != 1
            or item.get("matching_member_occurrences") != 1
            or item.get("binding_members") != [member]
        ):
            _error("owner_gate_foundation_apply_identity_invalid")
        return dict(item)
    if step_name == "create_dedicated_private_owner_gate_subnet":
        item = _strict_mapping(
            value,
            frozenset({
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
                "local_route_identity",
            }),
            code="owner_gate_foundation_apply_identity_invalid",
        )
        route = _strict_mapping(
            item.get("local_route_identity"),
            frozenset({
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
            }),
            code="owner_gate_foundation_apply_identity_invalid",
        )
        route_name = route.get("name")
        route_self_link = _provider_link(
            f"projects/{spec.project}/global/routes/{route_name}"
        )
        expected_route_description = (
            "Default local route to the subnetwork "
            f"{foundation.OWNER_GATE_SUBNET_CIDR}."
        )
        gateway_address = str(
            ipaddress.ip_network(
                foundation.OWNER_GATE_SUBNET_CIDR,
                strict=True,
            ).network_address
            + 1
        )
        if (
            item.get("resource_type") != "compute_subnetwork"
            or item.get("kind") != "compute#subnetwork"
            or item.get("name") != foundation.OWNER_GATE_SUBNET_NAME
            or item.get("self_link") != _provider_link(subnet_relative)
            or _NUMERIC_ID.fullmatch(str(item.get("numeric_id", ""))) is None
            or not _provider_tag(item.get("fingerprint"))
            or not isinstance(item.get("creation_timestamp"), str)
            or _RFC3339_TIMESTAMP.fullmatch(item["creation_timestamp"])
            is None
            or item.get("network_self_link") != network
            or item.get("region_self_link")
            != _provider_link(
                f"projects/{spec.project}/regions/{spec.region}"
            )
            or item.get("ip_cidr_range") != foundation.OWNER_GATE_SUBNET_CIDR
            or item.get("private_ip_google_access") is not True
            or item.get("stack_type") != "IPV4_ONLY"
            or item.get("purpose") != "PRIVATE"
            or item.get("secondary_ip_ranges") != []
            or item.get("allow_subnet_cidr_routes_overlap") is not False
            or item.get("gateway_address") != gateway_address
            or item.get("private_ipv6_google_access")
            != "DISABLE_GOOGLE_ACCESS"
            or route.get("resource_type") != "compute_route"
            or route.get("kind") != "compute#route"
            or not isinstance(route_name, str)
            or _OWNER_SUBNET_ROUTE_NAME.fullmatch(route_name) is None
            or route.get("self_link") != route_self_link
            or _NUMERIC_ID.fullmatch(str(route.get("numeric_id", "")))
            is None
            or not isinstance(route.get("creation_timestamp"), str)
            or _RFC3339_TIMESTAMP.fullmatch(route["creation_timestamp"])
            is None
            or route.get("network_self_link") != network
            or route.get("destination_range")
            != foundation.OWNER_GATE_SUBNET_CIDR
            or route.get("next_hop_network_self_link")
            != network
            or type(route.get("priority")) is not int
            or route.get("priority") != 0
            or route.get("description") != expected_route_description
            or route.get("route_type") != "SUBNET"
            or route.get("tags") != []
        ):
            _error("owner_gate_foundation_apply_identity_invalid")

        architecture = plan.architecture
        if not isinstance(architecture, Mapping):
            _error("owner_gate_foundation_apply_identity_invalid")
        subnet_key = "preexisting_owner_gate_subnet_identity"
        route_key = "preexisting_owner_gate_subnet_route_identity"
        subnet_bound = subnet_key in architecture
        route_bound = route_key in architecture
        if not subnet_bound or not route_bound:
            _error("owner_gate_foundation_apply_identity_invalid")
        signed_subnet = architecture[subnet_key]
        signed_route = architecture[route_key]
        if (signed_subnet is None) != (signed_route is None):
            _error("owner_gate_foundation_apply_identity_invalid")
        if signed_subnet is not None:
            if not isinstance(signed_subnet, Mapping) or not isinstance(
                signed_route,
                Mapping,
            ):
                _error("owner_gate_foundation_apply_identity_invalid")
            observed_subnet = {
                key: entry
                for key, entry in item.items()
                if key != "local_route_identity"
            }
            for field in (
                "self_link",
                "network_self_link",
                "region_self_link",
            ):
                observed_subnet[field] = _provider_relative(
                    observed_subnet[field]
                )
            observed_route = dict(route)
            for field in (
                "self_link",
                "network_self_link",
                "next_hop_network_self_link",
            ):
                observed_route[field] = _provider_relative(
                    observed_route[field]
                )
            if (
                dict(signed_subnet) != observed_subnet
                or dict(signed_route) != observed_route
            ):
                _error("owner_gate_foundation_apply_identity_invalid")
        return dict(item)
    if step_name == "create_private_owner_gate_vm":
        item = _strict_mapping(
            value,
            frozenset({
                "resource_type",
                "name",
                "self_link",
                "numeric_id",
                "metadata_fingerprint",
                "machine_type",
                "network_self_link",
                "network_numeric_id",
                "internal_ip",
                "subnetwork_self_link",
                "subnetwork_numeric_id",
                "network_stack_type",
                "access_configs",
                "service_account_email",
                "deletion_protection",
                "boot_image_numeric_id",
                "boot_image_self_link",
                "boot_image_architecture",
                "boot_image_license_self_links",
                "tags",
                "metadata",
                "shielded_instance_config",
                "oauth_scopes",
                "can_ip_forward",
                "maintenance_policy",
                "provisioning_model",
                "automatic_restart",
                "preemptible",
                "instance_termination_action",
                "network_interface_count",
                "alias_ip_ranges",
                "creation_timestamp",
                "labels",
                "resource_policies",
                "min_cpu_platform",
                "confidential_instance_config",
                "boot_disk_name",
                "boot_disk_self_link",
                "boot_disk_numeric_id",
                "boot_disk_size_gb",
                "boot_disk_type_self_link",
                "boot_disk_auto_delete",
                "boot_disk_boot",
                "boot_disk_mode",
                "boot_disk_interface",
                "boot_disk_attachment_type",
                "boot_disk_attachment_index",
            }),
            code="owner_gate_foundation_apply_identity_invalid",
        )
        if (
            item.get("resource_type") != "compute_instance"
            or item.get("name") != spec.vm_name
            or item.get("self_link")
            != _provider_link(
                f"projects/{spec.project}/zones/{spec.zone}/instances/"
                f"{spec.vm_name}"
            )
            or _NUMERIC_ID.fullmatch(str(item.get("numeric_id", ""))) is None
            or not _provider_tag(item.get("metadata_fingerprint"))
            or item.get("machine_type")
            != _provider_link(
                f"projects/{spec.project}/zones/{spec.zone}/machineTypes/"
                f"{spec.machine_type}"
            )
            or item.get("network_self_link") != network
            or _NUMERIC_ID.fullmatch(
                str(item.get("network_numeric_id", ""))
            )
            is None
            or item.get("internal_ip") != foundation.OWNER_GATE_PRIVATE_IP
            or item.get("subnetwork_self_link") != _provider_link(subnet_relative)
            or _NUMERIC_ID.fullmatch(
                str(item.get("subnetwork_numeric_id", ""))
            )
            is None
            or item.get("network_stack_type") != "IPV4_ONLY"
            or item.get("access_configs") != []
            or item.get("service_account_email")
            != spec.service_account_email
            or item.get("deletion_protection") is not False
            or item.get("boot_image_numeric_id")
            != spec.boot_image_numeric_id
            or item.get("boot_image_self_link")
            != _provider_link(spec.boot_image_self_link)
            or item.get("boot_image_architecture") != "X86_64"
            or item.get("boot_image_license_self_links")
            != [_provider_link(
                "projects/debian-cloud/global/licenses/debian-12-bookworm"
            )]
            or item.get("tags")
            != sorted([
                foundation.IAP_NETWORK_TAG,
                foundation.OWNER_GATE_NETWORK_TAG,
            ])
            or item.get("metadata")
            != {
                "block-project-ssh-keys": "TRUE",
                "enable-oslogin": "TRUE",
                "serial-port-enable": "FALSE",
            }
            or item.get("shielded_instance_config")
            != {
                "enable_integrity_monitoring": True,
                "enable_secure_boot": True,
                "enable_vtpm": True,
            }
            or item.get("oauth_scopes")
            != sorted(foundation.OWNER_GATE_OAUTH_SCOPES)
            or item.get("can_ip_forward") is not False
            or item.get("maintenance_policy") != "MIGRATE"
            or item.get("provisioning_model") != "STANDARD"
            or item.get("automatic_restart") is not True
            or item.get("preemptible") is not False
            or item.get("instance_termination_action") != "DELETE"
            or item.get("network_interface_count") != 1
            or item.get("alias_ip_ranges") != []
            or not isinstance(item.get("creation_timestamp"), str)
            or _RFC3339_TIMESTAMP.fullmatch(item["creation_timestamp"]) is None
            or item.get("labels") != {}
            or item.get("resource_policies") != []
            or item.get("min_cpu_platform") != "Automatic"
            or item.get("confidential_instance_config")
            != {"enable_confidential_compute": False}
            or item.get("boot_disk_name") != spec.vm_name
            or item.get("boot_disk_self_link")
            != _provider_link(
                f"projects/{spec.project}/zones/{spec.zone}/disks/"
                f"{spec.vm_name}"
            )
            or _NUMERIC_ID.fullmatch(
                str(item.get("boot_disk_numeric_id", ""))
            )
            is None
            or item.get("boot_disk_size_gb") != spec.boot_disk_size_gb
            or item.get("boot_disk_type_self_link")
            != _provider_link(
                f"projects/{spec.project}/zones/{spec.zone}/diskTypes/"
                f"{spec.boot_disk_type}"
            )
            or item.get("boot_disk_auto_delete") is not True
            or item.get("boot_disk_boot") is not True
            or item.get("boot_disk_mode") != "READ_WRITE"
            or item.get("boot_disk_interface") != "SCSI"
            or item.get("boot_disk_attachment_type") != "PERSISTENT"
            or item.get("boot_disk_attachment_index") != 0
        ):
            _error("owner_gate_foundation_apply_identity_invalid")
        return dict(item)
    if step_name == "allow_private_web_upstream_from_current_caddy_host":
        item = _strict_mapping(
            value,
            frozenset({
                "resource_type",
                "name",
                "self_link",
                "numeric_id",
                "creation_timestamp",
                "network_self_link",
                "direction",
                "priority",
                "disabled",
                "action",
                "allowed",
                "denied",
                "source_ranges",
                "destination_ranges",
                "source_tags",
                "target_tags",
                "source_service_accounts",
                "target_service_accounts",
                "log_config",
            }),
            code="owner_gate_foundation_apply_identity_invalid",
        )
        name = "muncho-owner-gate-web-from-production"
        if (
            item.get("resource_type") != "compute_firewall"
            or item.get("name") != name
            or item.get("self_link")
            != _provider_link(
                f"projects/{spec.project}/global/firewalls/{name}"
            )
            or _NUMERIC_ID.fullmatch(str(item.get("numeric_id", ""))) is None
            or not isinstance(item.get("creation_timestamp"), str)
            or _RFC3339_TIMESTAMP.fullmatch(item["creation_timestamp"]) is None
            or item.get("network_self_link") != network
            or item.get("direction") != "INGRESS"
            or item.get("priority") != 700
            or item.get("disabled") is not False
            or item.get("action") != "ALLOW"
            or item.get("allowed")
            != [{
                "ip_protocol": "tcp",
                "ports": [str(foundation.WEB_LISTEN_PORT)],
            }]
            or item.get("denied") != []
            or item.get("source_ranges") != []
            or item.get("destination_ranges") != []
            or item.get("source_tags") != []
            or item.get("target_tags") != []
            or item.get("source_service_accounts")
            != [foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT]
            or item.get("target_service_accounts")
            != [spec.service_account_email]
            or item.get("log_config") != {"enable": True}
        ):
            _error("owner_gate_foundation_apply_identity_invalid")
        return dict(item)
    _error("owner_gate_foundation_apply_identity_invalid")


def _validate_step_receipts(
    value: Any,
    *,
    plan: foundation.OwnerGateFoundationPlan,
) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != len(plan.foundation_steps):
        _error("owner_gate_foundation_apply_steps_invalid")
    checked: list[Mapping[str, Any]] = []
    for receipt, step in zip(value, plan.foundation_steps, strict=True):
        item = _strict_mapping(
            receipt,
            _STEP_RECEIPT_FIELDS,
            code="owner_gate_foundation_apply_steps_invalid",
        )
        if (
            item.get("step_name") != step.name
            or item.get("argv_sha256") != _sha256_json(list(step.argv))
            or item.get("disposition") not in {"created", "preexisting_exact"}
            or _SHA256.fullmatch(
                str(item.get("operation_receipt_sha256", ""))
            )
            is None
            or _SHA256.fullmatch(
                str(item.get("postcondition_receipt_sha256", ""))
            )
            is None
        ):
            _error("owner_gate_foundation_apply_steps_invalid")
        checked.append({
            **dict(item),
            "resource_identity": _validate_resource_identity(
                step.name,
                item.get("resource_identity"),
                plan=plan,
            ),
        })
    return checked


def _build_apply_receipt_body_from_execution(
    *,
    authority: Mapping[str, Any],
    owner_reauthentication_receipt: Mapping[str, Any],
    owner_reauthentication_public_key: Ed25519PublicKey,
    project_ancestry_evidence_raw: bytes,
    project_ancestry_collector_public_key: Ed25519PublicKey,
    plan: foundation.OwnerGateFoundationPlan,
    execution: Any,
) -> Mapping[str, Any]:
    """Build only from the exact provider execution result class."""

    foundation_apply = importlib.import_module(
        "scripts.canary.owner_gate_foundation_apply"
    )

    if type(execution) is not foundation_apply._ProviderExecutionResult:
        _error("owner_gate_foundation_apply_execution_invalid")
    started_at_unix = execution.started_at_unix
    completed_at_unix = execution.completed_at_unix
    steps = _validate_step_receipts(
        list(execution.step_receipts),
        plan=plan,
    )
    owner = authority.get("owner_reauthentication")
    if not isinstance(owner, Mapping):
        _error("owner_gate_foundation_apply_receipt_invalid")
    try:
        checked_reauth = owner_reauth.validate_owner_reauth_receipt(
            owner_reauthentication_receipt,
            public_key=owner_reauthentication_public_key,
            now_unix=started_at_unix,
        )
    except owner_reauth.OwnerGateOwnerReauthError as exc:
        _error("owner_gate_foundation_owner_reauth_invalid", exc)
    if owner.get("receipt_sha256") != checked_reauth[
        "owner_reauthentication_receipt_sha256"
    ]:
        _error("owner_gate_foundation_owner_reauth_invalid")
    checked_ancestry = _validated_project_ancestry(
        raw=project_ancestry_evidence_raw,
        collector_public_key=project_ancestry_collector_public_key,
        owner_reauthentication_receipt=owner_reauthentication_receipt,
        owner_reauthentication_public_key=(
            owner_reauthentication_public_key
        ),
        expected_release_revision=plan.spec.release_revision,
        now_unix=started_at_unix,
    )
    if (
        authority.get("project_number") != checked_ancestry.project_number
        or authority.get("organization_id")
        != checked_ancestry.organization_id
        or authority.get("ancestry_evidence_sha256")
        != checked_ancestry.signed_evidence_sha256
        or authority.get("ancestry_chain_sha256")
        != checked_ancestry.value["stable_chain_sha256"]
    ):
        _error("owner_gate_foundation_ancestry_evidence_invalid")
    body = {
        "schema": APPLY_RECEIPT_SCHEMA,
        "purpose": APPLY_RECEIPT_PURPOSE,
        "pre_foundation_authority_sha256": authority.get(
            "pre_foundation_authority_sha256"
        ),
        "inert_plan_sha256": inert_plan_sha256(plan),
        "foundation_source_revision": plan.spec.release_revision,
        "foundation_source_tree_oid": plan.spec.source_tree_oid,
        "project": plan.spec.project,
        "region": plan.spec.region,
        "zone": plan.spec.zone,
        "project_number": authority.get("project_number"),
        "organization_id": plan.spec.organization_id,
        "ancestry_evidence_sha256": plan.spec.ancestry_evidence_sha256,
        "ancestry_chain_sha256": authority.get("ancestry_chain_sha256"),
        "ancestry_collector_public_key_id": authority.get(
            "ancestry_collector_public_key_id"
        ),
        "owner_reauthentication_receipt_sha256": owner.get("receipt_sha256"),
        "started_at_unix": started_at_unix,
        "completed_at_unix": completed_at_unix,
        "applied_steps": steps,
        "partial_unknown_state": False,
        "mutation_iam_binding_created": False,
        "package_deployed": False,
        "service_started": False,
        "signer_key_id": authority.get("signer_key_id"),
    }
    _validate_apply_body(body, authority=authority, plan=plan)
    return body


def _validate_apply_body(
    value: Any,
    *,
    authority: Mapping[str, Any],
    plan: foundation.OwnerGateFoundationPlan,
) -> Mapping[str, Any]:
    body = _strict_mapping(
        value,
        _APPLY_BODY_FIELDS,
        code="owner_gate_foundation_apply_receipt_invalid",
    )
    owner = authority.get("owner_reauthentication")
    if not isinstance(owner, Mapping):
        _error("owner_gate_foundation_apply_receipt_invalid")
    started = body.get("started_at_unix")
    completed = body.get("completed_at_unix")
    if (
        body.get("schema") != APPLY_RECEIPT_SCHEMA
        or body.get("purpose") != APPLY_RECEIPT_PURPOSE
        or body.get("pre_foundation_authority_sha256")
        != authority.get("pre_foundation_authority_sha256")
        or body.get("inert_plan_sha256") != inert_plan_sha256(plan)
        or body.get("inert_plan_sha256")
        != authority.get("inert_plan_sha256")
        or body.get("foundation_source_revision")
        != authority.get("foundation_source_revision")
        or body.get("foundation_source_tree_oid")
        != authority.get("foundation_source_tree_oid")
        or body.get("project") != foundation.PROJECT
        or body.get("region") != foundation.REGION
        or body.get("zone") != foundation.ZONE
        or body.get("project_number") != authority.get("project_number")
        or body.get("organization_id") != authority.get("organization_id")
        or body.get("ancestry_evidence_sha256")
        != authority.get("ancestry_evidence_sha256")
        or body.get("ancestry_chain_sha256")
        != authority.get("ancestry_chain_sha256")
        or body.get("ancestry_collector_public_key_id")
        != authority.get("ancestry_collector_public_key_id")
        or body.get("owner_reauthentication_receipt_sha256")
        != owner.get("receipt_sha256")
        or type(started) is not int
        or type(completed) is not int
        or started < authority.get("issued_at_unix", 0)
        or completed < started
        or completed > authority.get("expires_at_unix", 0)
        or body.get("partial_unknown_state") is not False
        or body.get("mutation_iam_binding_created") is not False
        or body.get("package_deployed") is not False
        or body.get("service_started") is not False
        or body.get("signer_key_id") != authority.get("signer_key_id")
    ):
        _error("owner_gate_foundation_apply_receipt_invalid")
    _validate_step_receipts(body.get("applied_steps"), plan=plan)
    return dict(body)


def _sign_foundation_apply_receipt_body(
    body: Mapping[str, Any],
    *,
    private_key: Ed25519PrivateKey,
    authority: Mapping[str, Any],
    owner_reauthentication_receipt: Mapping[str, Any],
    project_ancestry_evidence_raw: bytes,
    project_ancestry_collector_public_key: Ed25519PublicKey,
    plan: foundation.OwnerGateFoundationPlan,
) -> Mapping[str, Any]:
    if not isinstance(private_key, Ed25519PrivateKey):
        _error("owner_gate_pre_foundation_signer_invalid")
    completed = body.get("completed_at_unix")
    if type(completed) is not int:
        _error("owner_gate_foundation_apply_receipt_invalid")
    checked_authority = validate_pre_foundation_authority(
        authority,
        public_key=private_key.public_key(),
        owner_reauthentication_receipt=owner_reauthentication_receipt,
        now_unix=completed,
        expected_plan=plan,
        project_ancestry_evidence_raw=project_ancestry_evidence_raw,
        project_ancestry_collector_public_key=(
            project_ancestry_collector_public_key
        ),
    )
    checked = _validate_apply_body(
        body,
        authority=checked_authority,
        plan=plan,
    )
    key_id = _require_pinned_public_key(private_key.public_key())
    if checked["signer_key_id"] != key_id:
        _error("owner_gate_pre_foundation_signer_invalid")
    digest = _sha256_json(checked)
    signed_payload = {**checked, "foundation_apply_receipt_sha256": digest}
    signature = private_key.sign(
        APPLY_RECEIPT_SIGNATURE_DOMAIN + _canonical(signed_payload)
    )
    if len(signature) != 64:
        _error("owner_gate_foundation_apply_signature_invalid")
    return {
        **signed_payload,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode("ascii"),
    }


def _sign_foundation_apply_execution(
    execution: Any,
    *,
    private_key: Ed25519PrivateKey,
    authority: Mapping[str, Any],
    owner_reauthentication_receipt: Mapping[str, Any],
    project_ancestry_evidence_raw: bytes,
    project_ancestry_collector_public_key: Ed25519PublicKey,
    plan: foundation.OwnerGateFoundationPlan,
) -> Mapping[str, Any]:
    """Sign only step receipts produced by the bounded provider executor."""

    body = _build_apply_receipt_body_from_execution(
        authority=authority,
        owner_reauthentication_receipt=owner_reauthentication_receipt,
        owner_reauthentication_public_key=private_key.public_key(),
        project_ancestry_evidence_raw=project_ancestry_evidence_raw,
        project_ancestry_collector_public_key=(
            project_ancestry_collector_public_key
        ),
        plan=plan,
        execution=execution,
    )
    return _sign_foundation_apply_receipt_body(
        body,
        private_key=private_key,
        authority=authority,
        owner_reauthentication_receipt=owner_reauthentication_receipt,
        project_ancestry_evidence_raw=project_ancestry_evidence_raw,
        project_ancestry_collector_public_key=(
            project_ancestry_collector_public_key
        ),
        plan=plan,
    )


def validate_foundation_apply_receipt(
    value: Any,
    *,
    public_key: Ed25519PublicKey,
    authority: Mapping[str, Any],
    owner_reauthentication_receipt: Mapping[str, Any],
    project_ancestry_evidence_raw: bytes,
    project_ancestry_collector_public_key: Ed25519PublicKey,
    plan: foundation.OwnerGateFoundationPlan,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    receipt = _strict_mapping(
        value,
        _APPLY_FIELDS,
        code="owner_gate_foundation_apply_receipt_invalid",
    )
    completed = receipt.get("completed_at_unix")
    if type(completed) is not int:
        _error("owner_gate_foundation_apply_receipt_invalid")
    checked_authority = validate_pre_foundation_authority(
        authority,
        public_key=public_key,
        owner_reauthentication_receipt=owner_reauthentication_receipt,
        now_unix=completed,
        expected_plan=plan,
        project_ancestry_evidence_raw=project_ancestry_evidence_raw,
        project_ancestry_collector_public_key=(
            project_ancestry_collector_public_key
        ),
    )
    if now_unix is not None and (
        type(now_unix) is not int or now_unix < completed
    ):
        _error("owner_gate_foundation_apply_receipt_invalid")
    body = {
        key: item
        for key, item in receipt.items()
        if key not in {
            "foundation_apply_receipt_sha256",
            "signature_ed25519_b64url",
        }
    }
    checked = _validate_apply_body(
        body,
        authority=checked_authority,
        plan=plan,
    )
    key_id = _require_pinned_public_key(public_key)
    signed_payload = {
        **checked,
        "foundation_apply_receipt_sha256": receipt.get(
            "foundation_apply_receipt_sha256"
        ),
    }
    if (
        checked["signer_key_id"] != key_id
        or receipt.get("foundation_apply_receipt_sha256")
        != _sha256_json(checked)
    ):
        _error("owner_gate_foundation_apply_receipt_invalid")
    try:
        public_key.verify(
            _decode_signature(
                receipt.get("signature_ed25519_b64url"),
                code="owner_gate_foundation_apply_signature_invalid",
            ),
            APPLY_RECEIPT_SIGNATURE_DOMAIN + _canonical(signed_payload),
        )
    except InvalidSignature as exc:
        _error("owner_gate_foundation_apply_signature_invalid", exc)
    return dict(receipt)


def _decode_canonical(raw: bytes, *, code: str) -> Mapping[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > MAX_JSON_BYTES:
        _error(code)
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error(code, exc)
    if not isinstance(value, Mapping) or _canonical(value) != raw:
        _error(code)
    return value


def decode_canonical_authority(
    raw: bytes,
    **validation: Any,
) -> Mapping[str, Any]:
    value = _decode_canonical(
        raw, code="owner_gate_pre_foundation_authority_invalid"
    )
    return validate_pre_foundation_authority(value, **validation)


def decode_canonical_apply_receipt(
    raw: bytes,
    **validation: Any,
) -> Mapping[str, Any]:
    value = _decode_canonical(
        raw, code="owner_gate_foundation_apply_receipt_invalid"
    )
    return validate_foundation_apply_receipt(value, **validation)


__all__ = [
    "APPLY_RECEIPT_SCHEMA",
    "AUTHORITY_SCHEMA",
    "INERT_PLAN_SCHEMA",
    "OwnerGatePreFoundationError",
    "build_authority_body",
    "decode_canonical_apply_receipt",
    "decode_canonical_authority",
    "inert_plan_projection",
    "inert_plan_sha256",
    "load_pinned_public_key",
    "sign_pre_foundation_authority",
    "spec_from_authority",
    "validate_foundation_apply_receipt",
    "validate_pre_foundation_authority",
]
