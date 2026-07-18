#!/usr/bin/env python3
"""Verify the out-of-band authority for one exact owner-gate release.

Inventory manifests are not authority: a caller can hash arbitrary bytes.  A
release becomes installable only when this module's stable fork-pinned signer
public-key hash validates a canonical Ed25519-signed external release trust
manifest.  The manifest is signed *after* the release commit exists, avoiding
an impossible self-reference between a commit and a hash embedded in itself.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from scripts.canary import owner_gate_foundation as foundation


TRUST_SCHEMA = "muncho-owner-gate-release-trust.v2"
FORK_REPOSITORY = "lomliev/hermes-agent"
ATTESTATION_PURPOSE = "muncho_owner_gate_exact_offline_release_supply_chain"

# This stable owner release-signing key is the immutable fork anchor.  It is
# configured in a reviewed key-bootstrap commit independent of any release it
# later signs.  The private seed lives only in Emil's fixed owner authority
# directory outside every repository/worktree.
PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256 = (
    "302bd03b449a4f46476d9d2dc8026acedaca17334154ba2cf8ba2a68c72992a0"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_TREE_OID = re.compile(r"^[0-9a-f]{40}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_FOLDER_RESOURCE = re.compile(r"^folders/[1-9][0-9]{5,30}$")
_ORGANIZATION_RESOURCE = re.compile(
    r"^organizations/[1-9][0-9]{5,30}$"
)
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024


class OwnerGateTrustError(RuntimeError):
    """Stable, secret-free release trust failure."""


def _read_immutable(
    path: Path,
    *,
    maximum: int,
    expected_uid: int,
    allowed_modes: frozenset[int],
) -> bytes:
    if not path.is_absolute() or ".." in path.parts:
        raise OwnerGateTrustError("owner_gate_trust_path_invalid")
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        before = path.lstat()
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_nlink != 1
            or opened.st_uid != expected_uid
            or stat.S_IMODE(opened.st_mode) not in allowed_modes
            or opened.st_size < 1
            or opened.st_size > maximum
        ):
            raise OwnerGateTrustError("owner_gate_trust_file_identity_invalid")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise OwnerGateTrustError("owner_gate_trust_file_read_invalid")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise OwnerGateTrustError("owner_gate_trust_file_changed")
        return b"".join(chunks)
    except OSError as exc:
        raise OwnerGateTrustError("owner_gate_trust_file_unavailable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _b64url_decode(value: Any, *, expected_bytes: int) -> bytes:
    if not isinstance(value, str) or not value or "=" in value:
        raise OwnerGateTrustError("owner_gate_trust_signature_invalid")
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError) as exc:
        raise OwnerGateTrustError("owner_gate_trust_signature_invalid") from None
    if (
        len(raw) != expected_bytes
        or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != value
    ):
        raise OwnerGateTrustError("owner_gate_trust_signature_invalid")
    return raw


def _validate_unsigned(value: Mapping[str, Any]) -> None:
    if set(value) != {
        "schema",
        "approved_for_offline_install",
        "fork_repository",
        "release_revision",
        "source_tree_oid",
        "foundation_source_revision",
        "foundation_source_tree_oid",
        "package_inventory_sha256",
        "boot_image_self_link",
        "collector_public_key_ids",
        "credential_migration_envelope_sha256",
        "direct_iam_identity_authority_sha256",
        "pre_foundation_authority_sha256",
        "foundation_apply_receipt_sha256",
        "project_ancestry_evidence_sha256",
        "project_ancestry_chain_sha256",
        "resource_ancestor_chain",
        "interpreter_image",
        "release_attestation",
        "signer_key_id",
    }:
        raise OwnerGateTrustError("owner_gate_trust_manifest_invalid")
    image = value.get("interpreter_image")
    release_attestation = value.get("release_attestation")
    collectors = value.get("collector_public_key_ids")
    resource_ancestor_chain = value.get("resource_ancestor_chain")
    if (
        value.get("schema") != TRUST_SCHEMA
        or value.get("approved_for_offline_install") is not True
        or value.get("fork_repository") != FORK_REPOSITORY
        or _REVISION.fullmatch(str(value.get("release_revision", ""))) is None
        or _TREE_OID.fullmatch(str(value.get("source_tree_oid", ""))) is None
        or _REVISION.fullmatch(
            str(value.get("foundation_source_revision", ""))
        )
        is None
        or _TREE_OID.fullmatch(
            str(value.get("foundation_source_tree_oid", ""))
        )
        is None
        or value.get("foundation_source_revision")
        == value.get("release_revision")
        or _SHA256.fullmatch(
            str(value.get("package_inventory_sha256", ""))
        ) is None
        or not isinstance(value.get("boot_image_self_link"), str)
        or re.fullmatch(
            r"projects/debian-cloud/global/images/debian-12-bookworm-v[0-9]{8}",
            value["boot_image_self_link"],
        ) is None
        or not isinstance(collectors, Mapping)
        or set(collectors) != {"network", "cloud", "host"}
        or any(_SHA256.fullmatch(str(item)) is None for item in collectors.values())
        or len(set(collectors.values())) != 3
        or _SHA256.fullmatch(
            str(value.get("credential_migration_envelope_sha256", ""))
        ) is None
        or _SHA256.fullmatch(
            str(value.get("direct_iam_identity_authority_sha256", ""))
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
        or _SHA256.fullmatch(
            str(value.get("project_ancestry_evidence_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(value.get("project_ancestry_chain_sha256", ""))
        )
        is None
        or not isinstance(resource_ancestor_chain, list)
        or not resource_ancestor_chain
        or len(resource_ancestor_chain) > 31
        or any(not isinstance(item, str) for item in resource_ancestor_chain)
        or len(resource_ancestor_chain) != len(set(resource_ancestor_chain))
        or _ORGANIZATION_RESOURCE.fullmatch(
            str(resource_ancestor_chain[-1])
        )
        is None
        or any(
            _FOLDER_RESOURCE.fullmatch(str(item)) is None
            for item in resource_ancestor_chain[:-1]
        )
        or _SHA256.fullmatch(str(value.get("signer_key_id", ""))) is None
        or not isinstance(image, Mapping)
        or set(image) != {
            "project",
            "image_name",
            "image_numeric_id",
            "image_self_link",
            "python_version",
            "interpreter_sha256",
        }
        or image.get("project") != "debian-cloud"
        or not isinstance(image.get("image_name"), str)
        or not image["image_name"]
        or _NUMERIC_ID.fullmatch(str(image.get("image_numeric_id", ""))) is None
        or not isinstance(image.get("image_self_link"), str)
        or image["image_self_link"] != (
            "https://www.googleapis.com/compute/v1/"
            + value["boot_image_self_link"]
        )
        or image["image_name"]
        != value["boot_image_self_link"].rsplit("/", 1)[-1]
        or image.get("python_version") != "3.11.2"
        or _SHA256.fullmatch(str(image.get("interpreter_sha256", ""))) is None
        or not isinstance(release_attestation, Mapping)
        or set(release_attestation) != {"purpose", "attested_at_unix"}
        or release_attestation.get("purpose") != ATTESTATION_PURPOSE
        or type(release_attestation.get("attested_at_unix")) is not int
        or release_attestation["attested_at_unix"] <= 0
    ):
        raise OwnerGateTrustError("owner_gate_trust_manifest_invalid")


def load_pinned_release_trust(
    *,
    manifest_path: Path,
    public_key_path: Path,
    expected_uid: int,
) -> Mapping[str, Any]:
    """Load the exact pinned trust manifest; no caller may supply its pins."""

    if (
        _SHA256.fullmatch(PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256 or "") is None
    ):
        raise OwnerGateTrustError("owner_gate_trust_anchor_unconfigured")
    manifest_raw = _read_immutable(
        manifest_path,
        maximum=_MAX_MANIFEST_BYTES,
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
    )
    public_key_raw = _read_immutable(
        public_key_path,
        maximum=32,
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
    )
    return decode_pinned_release_trust(
        manifest_raw=manifest_raw,
        public_key_raw=public_key_raw,
    )


def decode_pinned_release_trust(
    *,
    manifest_raw: bytes,
    public_key_raw: bytes,
) -> Mapping[str, Any]:
    """Verify canonical trust bytes already protected by a fixed outer stream.

    Filesystem callers should keep using :func:`load_pinned_release_trust` so
    its immutable-file checks remain part of their boundary.  This decoder is
    for release transports which have already pinned the complete byte stream
    and therefore must not materialize caller-selected temporary files merely
    to reuse the same cryptographic trust contract.
    """

    if (
        type(manifest_raw) is not bytes
        or not 0 < len(manifest_raw) <= _MAX_MANIFEST_BYTES
        or type(public_key_raw) is not bytes
    ):
        raise OwnerGateTrustError("owner_gate_trust_manifest_invalid")
    if (
        _SHA256.fullmatch(PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256 or "")
        is None
    ):
        raise OwnerGateTrustError("owner_gate_trust_anchor_unconfigured")
    if (
        hashlib.sha256(public_key_raw).hexdigest()
        != PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256
        or len(public_key_raw) != 32
    ):
        raise OwnerGateTrustError("owner_gate_trust_anchor_mismatch")
    try:
        value = json.loads(manifest_raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise OwnerGateTrustError("owner_gate_trust_manifest_invalid") from None
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "approved_for_offline_install",
        "fork_repository",
        "release_revision",
        "source_tree_oid",
        "foundation_source_revision",
        "foundation_source_tree_oid",
        "package_inventory_sha256",
        "boot_image_self_link",
        "collector_public_key_ids",
        "credential_migration_envelope_sha256",
        "direct_iam_identity_authority_sha256",
        "pre_foundation_authority_sha256",
        "foundation_apply_receipt_sha256",
        "project_ancestry_evidence_sha256",
        "project_ancestry_chain_sha256",
        "resource_ancestor_chain",
        "interpreter_image",
        "release_attestation",
        "signer_key_id",
        "signature_ed25519_b64url",
    }:
        raise OwnerGateTrustError("owner_gate_trust_manifest_invalid")
    if foundation.canonical_json_bytes(value) != manifest_raw:
        raise OwnerGateTrustError("owner_gate_trust_manifest_not_canonical")
    unsigned = {
        key: item
        for key, item in value.items()
        if key != "signature_ed25519_b64url"
    }
    _validate_unsigned(unsigned)
    key_id = hashlib.sha256(public_key_raw).hexdigest()
    if unsigned["signer_key_id"] != key_id:
        raise OwnerGateTrustError("owner_gate_trust_signer_mismatch")
    signature = _b64url_decode(
        value["signature_ed25519_b64url"],
        expected_bytes=64,
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key_raw).verify(
            signature,
            foundation.canonical_json_bytes(unsigned),
        )
    except (InvalidSignature, ValueError) as exc:
        raise OwnerGateTrustError("owner_gate_trust_signature_invalid") from None
    return {
        **unsigned,
        "trust_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "trust_public_key_sha256": PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256,
    }


def verify_inventory_authority(
    trust: Mapping[str, Any],
    *,
    inventory: Mapping[str, Any],
) -> None:
    if (
        trust.get("release_revision") != inventory.get("release_revision")
        or trust.get("source_tree_oid") != inventory.get("source_tree_oid")
        or trust.get("foundation_source_revision")
        != inventory.get("foundation_source_revision")
        or trust.get("foundation_source_tree_oid")
        != inventory.get("foundation_source_tree_oid")
        or trust.get("package_inventory_sha256")
        != foundation.sha256_json(inventory)
        or trust.get("interpreter_image", {}).get("interpreter_sha256")
        != inventory.get("interpreter_sha256")
        or trust.get("direct_iam_identity_authority_sha256")
        != inventory.get("direct_iam_identity_authority_sha256")
        or trust.get("pre_foundation_authority_sha256")
        != inventory.get("pre_foundation_authority_sha256")
        or trust.get("foundation_apply_receipt_sha256")
        != inventory.get("foundation_apply_receipt_sha256")
        or trust.get("resource_ancestor_chain")
        != inventory.get("resource_ancestor_chain")
    ):
        raise OwnerGateTrustError("owner_gate_trust_inventory_mismatch")
