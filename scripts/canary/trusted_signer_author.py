#!/usr/bin/env python3
"""Owner-side key initialization and secret-free signer envelope authoring.

Release-bound Ed25519 keypairs are generated in Emil's fixed private authority
directory through one of two exact lifecycle modes.  The bootstrap mode creates
only the network signer; the final-release mode creates network, cloud, and
host signers.  Only public identities are returned or printed.  The cloud and
host private seeds may be materialized solely as a mutable canonical stdin
frame for the fixed IAP provisioning transports; callers must wipe that frame
immediately after the exchange.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_trust_author as release_author
from scripts.canary import trusted_signer_provisioning as provisioning


ROLES = ("network", "cloud", "host")
INITIALIZATION_MODES = {
    "network-bootstrap": ("network",),
    "full-release": ROLES,
}
PROVISIONED_ROLES = frozenset({"cloud", "host"})
OBSERVATION_ROOT = release_author.KEY_DIRECTORY / "observation-signers"
PRIVATE_KEY_BYTES = 32
PUBLIC_KEY_BYTES = 32

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TrustedSignerAuthorError(RuntimeError):
    """Stable, secret-free owner-side signer authoring failure."""


def _release_directory(release_revision: str) -> Path:
    if _REVISION.fullmatch(release_revision or "") is None:
        raise TrustedSignerAuthorError("trusted_signer_author_release_invalid")
    return OBSERVATION_ROOT / release_revision


def _private_path(release_revision: str, role: str) -> Path:
    if role not in ROLES:
        raise TrustedSignerAuthorError("trusted_signer_author_role_invalid")
    return _release_directory(release_revision) / f"{role}.key"


def _public_path(release_revision: str, role: str) -> Path:
    if role not in ROLES:
        raise TrustedSignerAuthorError("trusted_signer_author_role_invalid")
    return _release_directory(release_revision) / f"{role}.pub"


def _require_authority_directories(release_revision: str, *, create: bool) -> Path:
    try:
        release_author._require_owner_directory(
            release_author.KEY_DIRECTORY,
            expected=release_author.KEY_DIRECTORY,
            parent=release_author.AUTHORITY_PARENT,
            create=False,
        )
        release_author._require_owner_directory(
            OBSERVATION_ROOT,
            expected=OBSERVATION_ROOT,
            parent=release_author.KEY_DIRECTORY,
            create=create,
        )
        release = _release_directory(release_revision)
        release_author._require_owner_directory(
            release,
            expected=release,
            parent=OBSERVATION_ROOT,
            create=create,
        )
        return release
    except release_author.OwnerGateTrustAuthorError as exc:
        raise TrustedSignerAuthorError(
            "trusted_signer_author_directory_invalid"
        ) from None


def _public_raw(private_raw: bytes) -> bytes:
    try:
        return Ed25519PrivateKey.from_private_bytes(private_raw).public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    except ValueError as exc:
        raise TrustedSignerAuthorError(
            "trusted_signer_author_private_key_invalid"
        ) from None


def initialize_observation_keys(
    release_revision: str,
    *,
    mode: str,
    entropy: Callable[[int], bytes] = os.urandom,
) -> Mapping[str, Any]:
    """Create or verify the exact named lifecycle key subset."""

    roles = INITIALIZATION_MODES.get(mode)
    if roles is None:
        raise TrustedSignerAuthorError("trusted_signer_author_mode_invalid")
    if not callable(entropy):
        raise TrustedSignerAuthorError("trusted_signer_author_entropy_invalid")
    release_directory = _require_authority_directories(
        release_revision,
        create=True,
    )
    excluded_roles = set(ROLES) - set(roles)
    for role in excluded_roles:
        private_path = _private_path(release_revision, role)
        public_path = _public_path(release_revision, role)
        private_stage = private_path.with_name(f".{private_path.name}.stage")
        public_stage = public_path.with_name(f".{public_path.name}.stage")
        if any(
            path.exists() or path.is_symlink()
            for path in (
                private_path,
                public_path,
                private_stage,
                public_stage,
            )
        ):
            raise TrustedSignerAuthorError(
                "trusted_signer_author_role_scope_invalid"
            )
    if release_directory != _release_directory(release_revision):
        raise TrustedSignerAuthorError(
            "trusted_signer_author_directory_invalid"
        )
    public_keys: dict[str, Mapping[str, str]] = {}
    for role in roles:
        private_path = _private_path(release_revision, role)
        public_path = _public_path(release_revision, role)
        try:
            release_author._recover_private_stage(private_path)
            if not private_path.exists():
                seed = bytearray(entropy(PRIVATE_KEY_BYTES))
                try:
                    if len(seed) != PRIVATE_KEY_BYTES:
                        raise TrustedSignerAuthorError(
                            "trusted_signer_author_entropy_invalid"
                        )
                    release_author._publish_exclusive(
                        private_path,
                        bytes(seed),
                        mode=0o600,
                        code="trusted_signer_author_private_key_write_failed",
                    )
                finally:
                    for index in range(len(seed)):
                        seed[index] = 0
            private_raw = release_author._read_exact_regular(
                private_path,
                size=PRIVATE_KEY_BYTES,
                modes=frozenset({0o600}),
                code="trusted_signer_author_private_key_invalid",
            )
            public_raw = _public_raw(private_raw)
            release_author._publish_exclusive(
                public_path,
                public_raw,
                mode=0o444,
                code="trusted_signer_author_public_key_invalid",
            )
        except release_author.OwnerGateTrustAuthorError as exc:
            raise TrustedSignerAuthorError(str(exc)) from None
        public_keys[role] = {
            "path": str(public_path),
            "public_key_id": hashlib.sha256(public_raw).hexdigest(),
        }
    return {
        "schema": "muncho-trusted-observation-key-initialization.v1",
        "release_revision": release_revision,
        "initialization_mode": mode,
        "initialized_roles": list(roles),
        "keys_initialized": True,
        "public_keys": public_keys,
        "private_key_material_printed": False,
        "private_key_digest_printed": False,
        "provisioning_envelope_materialized": False,
    }


def build_provisioning_envelope(
    *,
    role: str,
    release_revision: str,
    package_sha256: str,
    owner_authorization_receipt_sha256: str,
) -> bytearray:
    """Return one mutable LF-terminated secret frame for a fixed transport."""

    if (
        role not in PROVISIONED_ROLES
        or _SHA256.fullmatch(package_sha256 or "") is None
        or _SHA256.fullmatch(owner_authorization_receipt_sha256 or "") is None
    ):
        raise TrustedSignerAuthorError("trusted_signer_author_envelope_invalid")
    _require_authority_directories(release_revision, create=False)
    private_path = _private_path(release_revision, role)
    public_path = _public_path(release_revision, role)
    try:
        private_raw = release_author._read_exact_regular(
            private_path,
            size=PRIVATE_KEY_BYTES,
            modes=frozenset({0o600}),
            code="trusted_signer_author_private_key_invalid",
        )
        public_raw = release_author._read_exact_regular(
            public_path,
            size=PUBLIC_KEY_BYTES,
            modes=frozenset({0o400, 0o440, 0o444}),
            code="trusted_signer_author_public_key_invalid",
        )
    except release_author.OwnerGateTrustAuthorError as exc:
        raise TrustedSignerAuthorError(str(exc)) from None
    if _public_raw(private_raw) != public_raw:
        raise TrustedSignerAuthorError("trusted_signer_author_keypair_mismatch")
    value = {
        "schema": provisioning.ENVELOPE_SCHEMAS[role],
        "role": role,
        "release_revision": release_revision,
        "package_sha256": package_sha256,
        "owner_discord_user_id": provisioning.OWNER_DISCORD_USER_ID,
        "owner_authorization_receipt_sha256": (
            owner_authorization_receipt_sha256
        ),
        "private_seed_ed25519_b64url": base64.urlsafe_b64encode(private_raw)
        .rstrip(b"=")
        .decode("ascii"),
        "public_key_id": hashlib.sha256(public_raw).hexdigest(),
    }
    frame = bytearray(foundation.canonical_json_bytes(value) + b"\n")
    try:
        decoded, decoded_seed = provisioning.decode_provisioning_envelope(
            bytes(frame),
            role=role,
            release_revision=release_revision,
            package_sha256=package_sha256,
        )
        if decoded != value or decoded_seed != private_raw:
            raise TrustedSignerAuthorError(
                "trusted_signer_author_envelope_invalid"
            )
    except provisioning.TrustedSignerProvisioningError as exc:
        wipe_secret_frame(frame)
        raise TrustedSignerAuthorError(
            "trusted_signer_author_envelope_invalid"
        ) from None
    return frame


def wipe_secret_frame(frame: bytearray) -> None:
    if not isinstance(frame, bytearray):
        raise TrustedSignerAuthorError("trusted_signer_author_frame_invalid")
    for index in range(len(frame)):
        frame[index] = 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-revision", required=True)
    parser.add_argument(
        "--mode",
        choices=tuple(INITIALIZATION_MODES),
        required=True,
    )
    arguments = parser.parse_args(argv)
    receipt = initialize_observation_keys(
        arguments.release_revision,
        mode=arguments.mode,
    )
    print(foundation.canonical_json_bytes(receipt).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
