#!/usr/bin/env python3
"""Portable signed projection of one fully-authored production FreezePlan.

The complete FreezePlan semantic validator deliberately remains in
``gateway.canonical_writer_production_cutover``. It runs while the owner
authors the publication and again when production root stages it. The
dedicated owner-gate only needs a narrow proof that the intervening bytes are
exact, self-digested, target-bound, and signed by the key embedded in that
exact plan. Keeping this projection here avoids importing the gateway and
agent graph into the sealed WebAuthn runtime.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


FREEZE_PLAN_SCHEMA = "muncho-production-legacy-freeze-plan.v3"
APPROVAL_SCHEMA = "muncho-production-legacy-cutover-approval.v1"
PUBLICATION_SCHEMA = "muncho-production-cutover-publication.v1"
PROJECT = "adventico-ai-platform"
ZONE = "europe-west3-a"
VM_NAME = "ai-platform-runtime-01"
DATABASE = "ai_platform_brain"
MAX_JSON_BYTES = 16 * 1024 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SIGNATURE = re.compile(r"^[0-9a-f]{128}$")
_ROLE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_PUBLICATION_FIELDS = frozenset({
    "schema", "action", "release_revision", "documents",
    "secret_material_recorded", "secret_digest_recorded",
    "publication_sha256",
})
_FREEZE_FIELDS = frozenset({
    "schema", "release_revision", "target", "owner_subject_sha256",
    "owner_public_key_ed25519_hex", "owner_key_id", "gateway_before",
    "writer_before", "connector_before", "initial_snapshot",
    "owner_runtime_attestation", "observe_artifact", "cutover_authority",
    "database_recovery_receipt_sha256", "states",
    "secret_material_recorded", "plan_sha256",
})
_TARGET_FIELDS = frozenset({
    "project", "zone", "vm", "database", "sql_instance", "sql_host",
    "tls_server_name", "port", "writer_login",
})
_APPROVAL_FIELDS = frozenset({
    "schema", "plan_kind", "purpose", "sequence",
    "previous_approval_sha256", "plan_sha256", "owner_subject_sha256",
    "owner_public_key_ed25519_hex", "owner_key_id", "nonce_sha256",
    "issued_at_unix", "expires_at_unix", "approved",
    "signature_ed25519_hex", "approval_sha256",
})
_APPROVAL_SIGNED_FIELDS = _APPROVAL_FIELDS - {
    "signature_ed25519_hex", "approval_sha256",
}


class PortableCutoverContractError(ValueError):
    """Stable failure at the sealed portable cutover boundary."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, UnicodeError, ValueError):
        raise PortableCutoverContractError(
            "portable_cutover_json_invalid"
        ) from None
    if len(raw) > MAX_JSON_BYTES:
        raise PortableCutoverContractError(
            "portable_cutover_json_invalid"
        )
    return raw


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_hashed(
    value: Any,
    *,
    fields: frozenset[str],
    digest_field: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PortableCutoverContractError(
            "portable_cutover_document_invalid"
        )
    raw = copy.deepcopy(dict(value))
    digest = raw.get(digest_field)
    unsigned = {
        key: item for key, item in raw.items() if key != digest_field
    }
    if (
        not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or digest != sha256_json(unsigned)
    ):
        raise PortableCutoverContractError(
            "portable_cutover_document_invalid"
        )
    return raw


@dataclass(frozen=True)
class PortableFreezePlan:
    value: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return str(self.value["plan_sha256"])

    def to_mapping(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.value))


def validate_freeze_plan(value: Any) -> PortableFreezePlan:
    """Validate the exact signed-plan projection, not its full semantics."""

    raw = _exact_hashed(
        value,
        fields=_FREEZE_FIELDS,
        digest_field="plan_sha256",
    )
    target = raw.get("target")
    public_hex = raw.get("owner_public_key_ed25519_hex")
    if not isinstance(target, Mapping) or set(target) != _TARGET_FIELDS:
        raise PortableCutoverContractError(
            "portable_cutover_plan_invalid"
        )
    if (
        raw.get("schema") != FREEZE_PLAN_SCHEMA
        or _REVISION.fullmatch(str(raw.get("release_revision", ""))) is None
        or _SHA256.fullmatch(str(raw.get("owner_subject_sha256", "")))
        is None
        or not isinstance(public_hex, str)
        or _SHA256.fullmatch(public_hex) is None
        or raw.get("owner_key_id")
        != hashlib.sha256(bytes.fromhex(public_hex)).hexdigest()
        or raw.get("states")
        != ["authority", "gateway_stopped", "final_tail_captured"]
        or raw.get("secret_material_recorded") is not False
        or _SHA256.fullmatch(
            str(raw.get("database_recovery_receipt_sha256", ""))
        ) is None
        or target.get("project") != PROJECT
        or target.get("zone") != ZONE
        or target.get("vm") != VM_NAME
        or target.get("database") != DATABASE
        or not isinstance(target.get("sql_instance"), str)
        or not target["sql_instance"]
        or not isinstance(target.get("sql_host"), str)
        or not target["sql_host"]
        or not isinstance(target.get("tls_server_name"), str)
        or not target["tls_server_name"]
        or target.get("port") != 5432
        or not isinstance(target.get("writer_login"), str)
        or _ROLE.fullmatch(target["writer_login"]) is None
    ):
        raise PortableCutoverContractError(
            "portable_cutover_plan_invalid"
        )
    return PortableFreezePlan(raw)


def approval_signature_payload(value: Mapping[str, Any]) -> bytes:
    if set(value) != _APPROVAL_FIELDS:
        raise PortableCutoverContractError(
            "portable_cutover_approval_invalid"
        )
    return canonical_json_bytes({
        key: value[key] for key in _APPROVAL_SIGNED_FIELDS
    })


def validate_freeze_approval(
    value: Any,
    *,
    plan: PortableFreezePlan,
    now_unix: int,
) -> Mapping[str, Any]:
    raw = _exact_hashed(
        value,
        fields=_APPROVAL_FIELDS,
        digest_field="approval_sha256",
    )
    public_hex = plan.value["owner_public_key_ed25519_hex"]
    sequence = raw.get("sequence")
    expected_purpose = (
        "freeze_apply" if sequence == 0 else "freeze_resume"
    )
    if (
        type(now_unix) is not int
        or now_unix <= 0
        or raw.get("schema") != APPROVAL_SCHEMA
        or raw.get("plan_kind") != "freeze"
        or raw.get("purpose") != expected_purpose
        or type(sequence) is not int
        or sequence < 0
        or (sequence == 0 and raw.get("previous_approval_sha256") is not None)
        or (
            sequence > 0
            and _SHA256.fullmatch(
                str(raw.get("previous_approval_sha256", ""))
            ) is None
        )
        or raw.get("plan_sha256") != plan.sha256
        or raw.get("owner_subject_sha256")
        != plan.value["owner_subject_sha256"]
        or raw.get("owner_public_key_ed25519_hex") != public_hex
        or raw.get("owner_key_id") != plan.value["owner_key_id"]
        or _SHA256.fullmatch(str(raw.get("nonce_sha256", ""))) is None
        or type(raw.get("issued_at_unix")) is not int
        or type(raw.get("expires_at_unix")) is not int
        or not raw["issued_at_unix"] <= now_unix < raw["expires_at_unix"]
        or not 1
        <= raw["expires_at_unix"] - raw["issued_at_unix"]
        <= 3600
        or raw.get("approved") is not True
        or _SIGNATURE.fullmatch(str(raw.get("signature_ed25519_hex", "")))
        is None
    ):
        raise PortableCutoverContractError(
            "portable_cutover_approval_invalid"
        )
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex)).verify(
            bytes.fromhex(raw["signature_ed25519_hex"]),
            approval_signature_payload(raw),
        )
    except (InvalidSignature, ValueError):
        raise PortableCutoverContractError(
            "portable_cutover_approval_invalid"
        ) from None
    return raw


def validate_freeze_publication(
    value: Any,
    *,
    now_unix: int,
) -> tuple[Mapping[str, Any], PortableFreezePlan, Mapping[str, Any]]:
    publication = _exact_hashed(
        value,
        fields=_PUBLICATION_FIELDS,
        digest_field="publication_sha256",
    )
    documents = publication.get("documents")
    if (
        publication.get("schema") != PUBLICATION_SCHEMA
        or publication.get("action") != "freeze-authority"
        or _REVISION.fullmatch(
            str(publication.get("release_revision", ""))
        ) is None
        or publication.get("secret_material_recorded") is not False
        or publication.get("secret_digest_recorded") is not False
        or not isinstance(documents, Mapping)
        or set(documents) != {"plan", "approval"}
    ):
        raise PortableCutoverContractError(
            "portable_cutover_publication_invalid"
        )
    plan = validate_freeze_plan(documents["plan"])
    approval = validate_freeze_approval(
        documents["approval"], plan=plan, now_unix=now_unix
    )
    if plan.value["release_revision"] != publication["release_revision"]:
        raise PortableCutoverContractError(
            "portable_cutover_publication_invalid"
        )
    return publication, plan, approval


__all__ = [
    "PortableCutoverContractError",
    "PortableFreezePlan",
    "approval_signature_payload",
    "canonical_json_bytes",
    "sha256_json",
    "validate_freeze_approval",
    "validate_freeze_plan",
    "validate_freeze_publication",
]
