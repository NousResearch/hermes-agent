"""Trusted create-only staging for production verifier and signing authority.

This root-only boundary runs before an owner signs the final cutover plan.  It
never writes a reusable API secret into the target gateway credential path.
Instead it creates public verifier artifacts plus two distinct Ed25519 private
keys under the fixed evidence staging tree.  Clean retry revalidates every
existing byte against the still-root-owned source secret and never overwrites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.api_verifier_credentials import (
    api_approval_passkey_matches,
    api_bearer_matches,
    build_api_approval_scrypt_verifier,
    build_api_bearer_verifier,
    parse_api_approval_scrypt_verifier,
    parse_api_bearer_verifier,
)
from gateway.operational_edge_bootstrap import (
    stage_operational_edge_key_foundation_from_staged_writer_private,
)


EVIDENCE_ROOT = Path("/var/lib/muncho-production-legacy-cutover")
STAGED_ROOT = EVIDENCE_ROOT / "staged"
HOST_STAGING_ROOT = STAGED_ROOT / "host"
KEY_STAGING_ROOT = STAGED_ROOT / "keys"

LEGACY_API_BEARER_PATH = Path("/etc/muncho/keys/api-server-control.key")
STAGED_APPROVAL_PASSKEY_PATH = STAGED_ROOT / "api-approval-passkey"
STAGED_API_BEARER_VERIFIER_PATH = (
    HOST_STAGING_ROOT / "api-server-bearer-sha256.json"
)
STAGED_API_APPROVAL_VERIFIER_PATH = (
    HOST_STAGING_ROOT / "api-approval-passkey-scrypt.json"
)
STAGED_WRITER_PRIVATE_KEY_PATH = (
    KEY_STAGING_ROOT / "writer-capability-private.pem"
)
STAGED_EDGE_PRIVATE_KEY_PATH = (
    KEY_STAGING_ROOT / "discord-edge-receipt-private.pem"
)

STAGING_SCHEMA = "muncho-production-secret-staging.v2"
MAX_SECRET_BYTES = 8_192
MAX_ARTIFACT_BYTES = 8_192


class ProductionSecretStagingError(RuntimeError):
    """Stable, secret-free staging failure."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _file_identity(item: os.stat_result) -> tuple[int, ...]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_gid,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _read_root_secret(path: Path, *, uid: int, gid: int) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        before = os.lstat(path)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != uid
            or before.st_gid != gid
            or stat.S_IMODE(before.st_mode) != 0o400
            or not 32 <= before.st_size <= MAX_SECRET_BYTES
        ):
            raise ProductionSecretStagingError("staging_secret_provenance_invalid")
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise ProductionSecretStagingError("staging_secret_changed")
        chunks: list[bytes] = []
        remaining = MAX_SECRET_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 4096))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if _file_identity(after) != _file_identity(opened):
            raise ProductionSecretStagingError("staging_secret_changed")
    except ProductionSecretStagingError:
        raise
    except OSError as exc:
        raise ProductionSecretStagingError("staging_secret_unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    raw = b"".join(chunks)
    if len(raw) > MAX_SECRET_BYTES:
        raise ProductionSecretStagingError("staging_secret_invalid")
    if raw.endswith(b"\r\n"):
        raw = raw[:-2]
    elif raw.endswith(b"\n"):
        raw = raw[:-1]
    try:
        value = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ProductionSecretStagingError("staging_secret_invalid") from exc
    if (
        value != value.strip()
        or len(value.encode("utf-8")) < 32
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value)
    ):
        raise ProductionSecretStagingError("staging_secret_invalid")
    return value


def _validate_directory(path: Path, *, uid: int, gid: int) -> None:
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise ProductionSecretStagingError("staging_directory_invalid") from exc
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != uid
        or observed.st_gid != gid
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise ProductionSecretStagingError("staging_directory_invalid")


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_or_validate(
    path: Path,
    payload: bytes,
    *,
    uid: int,
    gid: int,
) -> bool:
    if not 1 <= len(payload) <= MAX_ARTIFACT_BYTES:
        raise ProductionSecretStagingError("staging_artifact_invalid")
    created = False
    if not os.path.lexists(path):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o400)
        try:
            os.fchmod(descriptor, 0o400)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short staging write")
                view = view[written:]
            os.fsync(descriptor)
        except BaseException:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        finally:
            os.close(descriptor)
        _fsync_parent(path)
        created = True
    try:
        observed = os.lstat(path)
        actual = path.read_bytes()
    except OSError as exc:
        raise ProductionSecretStagingError("staging_artifact_invalid") from exc
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_uid != uid
        or observed.st_gid != gid
        or stat.S_IMODE(observed.st_mode) != 0o400
        or actual != payload
    ):
        raise ProductionSecretStagingError("staging_artifact_conflict")
    return created


def _private_pem(key: Ed25519PrivateKey) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _load_private_key(payload: bytes) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(payload, password=None)
    except (TypeError, ValueError) as exc:
        raise ProductionSecretStagingError("staging_private_key_invalid") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ProductionSecretStagingError("staging_private_key_invalid")
    if _private_pem(key) != payload:
        raise ProductionSecretStagingError("staging_private_key_noncanonical")
    return key


def _key_id(key: Ed25519PrivateKey) -> str:
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(public).hexdigest()


def _create_or_validate_private_key(
    path: Path,
    *,
    uid: int,
    gid: int,
) -> tuple[bool, str]:
    if os.path.lexists(path):
        try:
            observed = os.lstat(path)
            payload = path.read_bytes()
        except OSError as exc:
            raise ProductionSecretStagingError("staging_private_key_invalid") from exc
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_uid != uid
            or observed.st_gid != gid
            or stat.S_IMODE(observed.st_mode) != 0o400
            or not 1 <= len(payload) <= MAX_ARTIFACT_BYTES
        ):
            raise ProductionSecretStagingError("staging_private_key_invalid")
        return False, _key_id(_load_private_key(payload))
    key = Ed25519PrivateKey.generate()
    payload = _private_pem(key)
    created = _create_or_validate(path, payload, uid=uid, gid=gid)
    return created, _key_id(key)


def stage_production_secret_foundation(
    *,
    bearer_source: Path = LEGACY_API_BEARER_PATH,
    approval_source: Path = STAGED_APPROVAL_PASSKEY_PATH,
    bearer_verifier_path: Path = STAGED_API_BEARER_VERIFIER_PATH,
    approval_verifier_path: Path = STAGED_API_APPROVAL_VERIFIER_PATH,
    writer_private_path: Path = STAGED_WRITER_PRIVATE_KEY_PATH,
    edge_private_path: Path = STAGED_EDGE_PRIVATE_KEY_PATH,
    require_root: bool = True,
) -> dict[str, Any]:
    """Create or cleanly re-observe the complete fixed staging foundation."""

    effective_uid = os.geteuid()  # windows-footgun: ok — Linux production/canary boundary
    effective_gid = os.getegid()  # windows-footgun: ok — Linux production/canary boundary
    if require_root and effective_uid != 0:
        raise ProductionSecretStagingError("staging_requires_root")
    expected_uid = 0 if require_root else effective_uid
    expected_gid = 0 if require_root else effective_gid
    parents = {
        bearer_verifier_path.parent,
        approval_verifier_path.parent,
        writer_private_path.parent,
        edge_private_path.parent,
    }
    for parent in parents:
        _validate_directory(parent, uid=expected_uid, gid=expected_gid)

    bearer = _read_root_secret(
        bearer_source,
        uid=expected_uid,
        gid=expected_gid,
    )
    approval = _read_root_secret(
        approval_source,
        uid=expected_uid,
        gid=expected_gid,
    )
    bearer_payload = build_api_bearer_verifier(bearer)

    bearer_created = _create_or_validate(
        bearer_verifier_path,
        bearer_payload,
        uid=expected_uid,
        gid=expected_gid,
    )
    if os.path.lexists(approval_verifier_path):
        try:
            approval_payload = approval_verifier_path.read_bytes()
            approval_verifier = parse_api_approval_scrypt_verifier(
                approval_payload
            )
        except (OSError, ValueError) as exc:
            raise ProductionSecretStagingError(
                "staging_approval_verifier_invalid"
            ) from exc
        if not api_approval_passkey_matches(approval_verifier, approval):
            raise ProductionSecretStagingError(
                "staging_approval_verifier_conflict"
            )
        approval_created = _create_or_validate(
            approval_verifier_path,
            approval_payload,
            uid=expected_uid,
            gid=expected_gid,
        )
    else:
        approval_payload = build_api_approval_scrypt_verifier(approval)
        approval_created = _create_or_validate(
            approval_verifier_path,
            approval_payload,
            uid=expected_uid,
            gid=expected_gid,
        )

    if not api_bearer_matches(
        parse_api_bearer_verifier(bearer_payload),
        bearer,
    ):
        raise ProductionSecretStagingError("staging_bearer_verifier_invalid")
    writer_created, writer_key_id = _create_or_validate_private_key(
        writer_private_path,
        uid=expected_uid,
        gid=expected_gid,
    )
    edge_created, edge_key_id = _create_or_validate_private_key(
        edge_private_path,
        uid=expected_uid,
        gid=expected_gid,
    )
    if writer_key_id == edge_key_id:
        raise ProductionSecretStagingError("staging_private_keys_not_distinct")
    try:
        operational_key_foundation = (
            stage_operational_edge_key_foundation_from_staged_writer_private(
                expected_writer_public_key_id=writer_key_id,
                staged_writer_private_path=writer_private_path,
                staging_root=writer_private_path.parent,
                require_root=require_root,
            )
        )
    except RuntimeError as exc:
        raise ProductionSecretStagingError(
            "staging_operational_edge_key_foundation_invalid"
        ) from exc
    operational_key_ids = {
        row["domain"]: row["public_key_id"]
        for row in operational_key_foundation["keys"]
    }

    unsigned = {
        "schema": STAGING_SCHEMA,
        "bearer_verifier_path": str(bearer_verifier_path),
        "bearer_verifier_sha256": hashlib.sha256(bearer_payload).hexdigest(),
        "approval_verifier_path": str(approval_verifier_path),
        "approval_verifier_sha256": hashlib.sha256(approval_payload).hexdigest(),
        "writer_private_path": str(writer_private_path),
        "writer_public_key_id": writer_key_id,
        "edge_private_path": str(edge_private_path),
        "edge_public_key_id": edge_key_id,
        "operational_edge_key_foundation": operational_key_foundation,
        "operational_edge_key_foundation_sha256": (
            operational_key_foundation["receipt_sha256"]
        ),
        "operational_edge_receipt_public_key_ids": operational_key_ids,
        "created": {
            "bearer_verifier": bearer_created,
            "approval_verifier": approval_created,
            "writer_private_key": writer_created,
            "edge_private_key": edge_created,
        },
        "source_secrets_retained_for_cutover": True,
        "private_content_or_digest_recorded": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {
        **unsigned,
        "receipt_sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    receipt = stage_production_secret_foundation()
    print(_canonical_bytes(receipt).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ProductionSecretStagingError",
    "main",
    "stage_production_secret_foundation",
]
