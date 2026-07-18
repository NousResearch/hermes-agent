#!/usr/bin/env python3
"""Author and persist the fixed production-v1 passkey migration artifacts.

Artifacts are written only below the release-specific trusted signer directory.
The host private seed never enters argv, stdin, stdout, the environment, or a
receipt.  Publication is exact and crash-recoverable through the existing
owner release authority's no-clobber file primitive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import owner_gate_trust_author as release_author
from scripts.canary import owner_gate_v1_credential_migration as migration
from scripts.canary import trusted_signer_author as signer_author


AUTHOR_RECEIPT_SCHEMA = "muncho-owner-gate-v1-credential-author-receipt.v1"
SOURCE_FILENAME = "credential-migration-source-receipt.json"
ENVELOPE_FILENAME = "credential-migration-envelope.json"
RECEIPT_FILENAME = "credential-migration-author-receipt.json"
MAX_JSON = 4 * 1024 * 1024
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class V1CredentialAuthorError(RuntimeError):
    """Stable, secret-free owner-side authoring failure."""


def _error(code: str, exc: BaseException | None = None) -> None:
    del exc
    raise V1CredentialAuthorError(code) from None


def _paths(release_revision: str) -> tuple[Path, Path, Path, Path]:
    if _REVISION.fullmatch(release_revision or "") is None:
        _error("owner_gate_v1_credential_author_release_invalid")
    try:
        release = signer_author._require_authority_directories(
            release_revision,
            create=False,
        )
    except signer_author.TrustedSignerAuthorError as exc:
        _error("owner_gate_v1_credential_author_directory_invalid", exc)
    return (
        release,
        release / SOURCE_FILENAME,
        release / ENVELOPE_FILENAME,
        release / RECEIPT_FILENAME,
    )


def _host_private_key(release_revision: str) -> Ed25519PrivateKey:
    _release, _source, _envelope, _receipt = _paths(release_revision)
    private_path = signer_author._private_path(release_revision, "host")
    public_path = signer_author._public_path(release_revision, "host")
    try:
        private_raw = release_author._read_exact_regular(
            private_path,
            size=signer_author.PRIVATE_KEY_BYTES,
            modes=frozenset({0o600}),
            code="owner_gate_v1_credential_author_private_key_invalid",
        )
        public_raw = release_author._read_exact_regular(
            public_path,
            size=signer_author.PUBLIC_KEY_BYTES,
            modes=frozenset({0o400, 0o440, 0o444}),
            code="owner_gate_v1_credential_author_public_key_invalid",
        )
        private = Ed25519PrivateKey.from_private_bytes(private_raw)
    except (ValueError, release_author.OwnerGateTrustAuthorError) as exc:
        _error("owner_gate_v1_credential_author_key_invalid", exc)
    if private.public_key().public_bytes_raw() != public_raw:
        _error("owner_gate_v1_credential_author_key_invalid")
    return private


def _encode(value: Mapping[str, Any]) -> bytes:
    return migration._canonical(value) + b"\n"


def _read_mapping(path: Path, *, mode: int) -> Mapping[str, Any]:
    try:
        raw, _state = release_author._read_publish_file(
            path,
            mode=mode,
            allowed_nlinks=frozenset({1}),
            maximum=MAX_JSON,
            code="owner_gate_v1_credential_author_artifact_invalid",
        )
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            _error("owner_gate_v1_credential_author_artifact_invalid")
        value = json.loads(raw[:-1].decode("ascii", errors="strict"))
    except (
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
        release_author.OwnerGateTrustAuthorError,
    ) as exc:
        _error("owner_gate_v1_credential_author_artifact_invalid", exc)
    if not isinstance(value, Mapping) or _encode(value) != raw:
        _error("owner_gate_v1_credential_author_artifact_invalid")
    return dict(value)


def _publish(path: Path, value: Mapping[str, Any], *, mode: int) -> bytes:
    raw = _encode(value)
    try:
        release_author._publish_exclusive(
            path,
            raw,
            mode=mode,
            code="owner_gate_v1_credential_author_publish_failed",
        )
    except release_author.OwnerGateTrustAuthorError as exc:
        _error("owner_gate_v1_credential_author_publish_failed", exc)
    return raw


def _validate_author_receipt(
    value: Mapping[str, Any],
    *,
    release_revision: str,
    source_path: Path,
    envelope_path: Path,
    source_raw: bytes,
    envelope_raw: bytes,
    host_key_id: str,
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "release_revision",
        "source_receipt_path",
        "source_receipt_sha256",
        "source_receipt_file_sha256",
        "migration_envelope_path",
        "migration_envelope_sha256",
        "migration_envelope_file_sha256",
        "host_collector_public_key_id",
        "artifact_modes",
        "private_key_material_recorded",
        "private_key_digest_recorded",
        "production_mutation_performed",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        _error("owner_gate_v1_credential_author_receipt_invalid")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    try:
        source_value = json.loads(source_raw[:-1].decode("ascii"))
        envelope_value = json.loads(envelope_raw[:-1].decode("ascii"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error("owner_gate_v1_credential_author_receipt_invalid", exc)
    if (
        value.get("schema") != AUTHOR_RECEIPT_SCHEMA
        or value.get("release_revision") != release_revision
        or value.get("source_receipt_path") != str(source_path)
        or value.get("source_receipt_sha256")
        != source_value.get("receipt_sha256")
        or value.get("source_receipt_file_sha256") != hashlib.sha256(source_raw).hexdigest()
        or value.get("migration_envelope_path") != str(envelope_path)
        or value.get("migration_envelope_sha256")
        != envelope_value.get("envelope_sha256")
        or value.get("migration_envelope_file_sha256")
        != hashlib.sha256(envelope_raw).hexdigest()
        or value.get("host_collector_public_key_id") != host_key_id
        or value.get("artifact_modes") != {
            SOURCE_FILENAME: "0400",
            ENVELOPE_FILENAME: "0400",
        }
        or value.get("private_key_material_recorded") is not False
        or value.get("private_key_digest_recorded") is not False
        or value.get("production_mutation_performed") is not False
        or value.get("receipt_sha256") != migration._sha256_json(unsigned)
    ):
        _error("owner_gate_v1_credential_author_receipt_invalid")
    return dict(value)


def author_live_migration(release_revision: str) -> Mapping[str, Any]:
    """Collect once, then publish/replay the exact source, envelope, and receipt."""

    _release, source_path, envelope_path, receipt_path = _paths(release_revision)
    private = _host_private_key(release_revision)
    host_key_id = hashlib.sha256(private.public_key().public_bytes_raw()).hexdigest()
    if source_path.exists() or source_path.is_symlink():
        source = _read_mapping(source_path, mode=0o400)
        signed_at = source.get("signed_at_unix")
        if type(signed_at) is not int:
            _error("owner_gate_v1_credential_author_artifact_invalid")
        source = migration.validate_source_receipt(
            source,
            release_revision=release_revision,
            host_key_id=host_key_id,
            now_unix=signed_at,
        )
        envelope = migration.sign_migration_from_source_receipt(
            source,
            release_revision=release_revision,
            host_private_key=private,
        )
    else:
        if (
            envelope_path.exists()
            or envelope_path.is_symlink()
            or receipt_path.exists()
            or receipt_path.is_symlink()
        ):
            _error("owner_gate_v1_credential_author_manual_reconciliation_required")
        transport = migration.V1CredentialMigrationTransport(
            revision=release_revision,
        )
        envelope, source = migration.collect_and_sign_migration(
            transport,
            release_revision=release_revision,
            host_private_key=private,
        )
    source_raw = _publish(source_path, source, mode=0o400)
    envelope_raw = _publish(envelope_path, envelope, mode=0o400)
    unsigned = {
        "schema": AUTHOR_RECEIPT_SCHEMA,
        "release_revision": release_revision,
        "source_receipt_path": str(source_path),
        "source_receipt_sha256": source["receipt_sha256"],
        "source_receipt_file_sha256": hashlib.sha256(source_raw).hexdigest(),
        "migration_envelope_path": str(envelope_path),
        "migration_envelope_sha256": envelope["envelope_sha256"],
        "migration_envelope_file_sha256": hashlib.sha256(envelope_raw).hexdigest(),
        "host_collector_public_key_id": host_key_id,
        "artifact_modes": {
            SOURCE_FILENAME: "0400",
            ENVELOPE_FILENAME: "0400",
        },
        "private_key_material_recorded": False,
        "private_key_digest_recorded": False,
        "production_mutation_performed": False,
    }
    receipt = {**unsigned, "receipt_sha256": migration._sha256_json(unsigned)}
    receipt_raw = _publish(receipt_path, receipt, mode=0o444)
    checked = _validate_author_receipt(
        receipt,
        release_revision=release_revision,
        source_path=source_path,
        envelope_path=envelope_path,
        source_raw=source_raw,
        envelope_raw=envelope_raw,
        host_key_id=host_key_id,
    )
    if _encode(checked) != receipt_raw:
        _error("owner_gate_v1_credential_author_receipt_invalid")
    return checked


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-revision", required=True)
    arguments = parser.parse_args(argv)
    try:
        receipt = author_live_migration(arguments.release_revision)
    except V1CredentialAuthorError as exc:
        failure = {
            "schema": "muncho-owner-gate-v1-credential-author-failure.v1",
            "ok": False,
            "error_code": str(exc),
            "private_key_material_recorded": False,
            "private_key_digest_recorded": False,
            "production_mutation_performed": False,
        }
        print(migration._canonical(failure).decode("ascii"))
        return 1
    print(migration._canonical(receipt).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["V1CredentialAuthorError", "author_live_migration"]
