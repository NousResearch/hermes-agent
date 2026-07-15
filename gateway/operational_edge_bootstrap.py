"""Create-only key staging and sealed host artifacts for operational edges.

This module is a deterministic privileged boundary.  It creates or revalidates
one Ed25519 receipt key pair per credential domain, verifies every packaged
helper byte, and renders the exact root-owned unit/config artifacts.  It never
reads operational credential values and never records private key bytes or
private-key digests in a receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from gateway.discord_edge_protocol import ed25519_public_key_id
from gateway.operational_edge_assets import (
    validate_packaged_operational_asset_verification,
    verify_packaged_operational_assets,
)
from gateway.operational_edge_catalog import CREDENTIALS_BY_DOMAIN
from gateway.operational_edge_units import (
    CLIENT_CONFIG_PATH,
    CONFIG_ROOT,
    KEY_ROOT,
    TRUST_ROOT,
    OperationalEdgeUnitBundle,
    receipt_private_key_path,
    receipt_public_key_path,
    render_operational_edge_units,
)


KEY_FOUNDATION_SCHEMA = "muncho-operational-edge-key-foundation.v1"
FOUNDATION_SCHEMA = "muncho-operational-edge-foundation.v1"
MAX_KEY_BYTES = 16 * 1024
SYSTEMD_ROOT = Path("/etc/systemd/system")
PRE_OWNER_STAGING_ROOT = Path(
    "/var/lib/muncho-production-legacy-cutover/staged/keys"
)
STAGED_WRITER_PRIVATE_KEY = (
    PRE_OWNER_STAGING_ROOT / "writer-capability-private.pem"
)

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OperationalEdgeBootstrapError(RuntimeError):
    """Stable error from the privileged operational-edge bootstrap."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise OperationalEdgeBootstrapError(
            "operational_edge_bootstrap_json_invalid"
        ) from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity(item: os.stat_result) -> tuple[int, ...]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _stable_file(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int,
) -> bytes:
    descriptor = -1
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != uid
            or before.st_gid != gid
            or stat.S_IMODE(before.st_mode) != mode
            or not 0 < before.st_size <= MAX_KEY_BYTES
        ):
            raise OperationalEdgeBootstrapError(
                "operational_edge_key_file_invalid"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 4096))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    except OperationalEdgeBootstrapError:
        raise
    except OSError as exc:
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_file_unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or _identity(before) != _identity(opened)
        or _identity(before) != _identity(after)
    ):
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_file_changed"
        )
    return raw


def _directory(
    path: Path,
    *,
    uid: int,
    gid: int,
    create_mode: int | None = None,
    exact_mode: int | None = None,
) -> None:
    if create_mode is not None and not os.path.lexists(path):
        try:
            path.mkdir(parents=False, mode=create_mode)
            os.chmod(path, create_mode)
            os.chown(path, uid, gid)
        except OSError as exc:
            raise OperationalEdgeBootstrapError(
                "operational_edge_key_directory_unavailable"
            ) from exc
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_directory_unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_uid != uid
        or observed.st_gid != gid
        or stat.S_IMODE(observed.st_mode) & 0o022
        or exact_mode is not None
        and stat.S_IMODE(observed.st_mode) != exact_mode
    ):
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_directory_invalid"
        )


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_exclusive(
    path: Path,
    payload: bytes,
    *,
    uid: int,
    gid: int,
    mode: int,
) -> None:
    descriptor = -1
    created = False
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        created = True
        os.fchmod(descriptor, mode)
        os.fchown(descriptor, uid, gid)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short_write")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_create_failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_parent(path)


def _private_pem(key: Ed25519PrivateKey) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _public_pem(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _private_key(raw: bytes) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(raw, password=None)
    except (TypeError, ValueError) as exc:
        raise OperationalEdgeBootstrapError(
            "operational_edge_private_key_invalid"
        ) from exc
    if not isinstance(key, Ed25519PrivateKey) or _private_pem(key) != raw:
        raise OperationalEdgeBootstrapError(
            "operational_edge_private_key_invalid"
        )
    return key


def _public_key(raw: bytes) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(raw)
    except (TypeError, ValueError) as exc:
        raise OperationalEdgeBootstrapError(
            "operational_edge_public_key_invalid"
        ) from exc
    if not isinstance(key, Ed25519PublicKey) or _public_pem(key) != raw:
        raise OperationalEdgeBootstrapError(
            "operational_edge_public_key_invalid"
        )
    return key


def _pair(
    *,
    domain: str,
    private_path: Path,
    public_path: Path,
    uid: int,
    gid: int,
) -> dict[str, Any]:
    private_exists = os.path.lexists(private_path)
    public_exists = os.path.lexists(public_path)
    if private_exists != public_exists:
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_pair_partial"
        )
    created = False
    if not private_exists:
        key = Ed25519PrivateKey.generate()
        private_raw = _private_pem(key)
        public_raw = _public_pem(key.public_key())
        _create_exclusive(
            private_path,
            private_raw,
            uid=uid,
            gid=gid,
            mode=0o400,
        )
        try:
            _create_exclusive(
                public_path,
                public_raw,
                uid=uid,
                gid=gid,
                mode=0o444,
            )
        except BaseException:
            try:
                private_path.unlink()
                _fsync_parent(private_path)
            except OSError:
                pass
            raise
        created = True
    private_raw = _stable_file(
        private_path,
        uid=uid,
        gid=gid,
        mode=0o400,
    )
    public_raw = _stable_file(
        public_path,
        uid=uid,
        gid=gid,
        mode=0o444,
    )
    private = _private_key(private_raw)
    public = _public_key(public_raw)
    if _public_pem(private.public_key()) != public_raw:
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_pair_mismatch"
        )
    return {
        "domain": domain,
        "private_path": str(private_path),
        "private_uid": uid,
        "private_gid": gid,
        "private_mode": "0400",
        "public_path": str(public_path),
        "public_uid": uid,
        "public_gid": gid,
        "public_mode": "0444",
        "public_key_id": ed25519_public_key_id(public),
        "created": created,
    }


def validate_operational_edge_key_foundation(
    value: Any,
    *,
    expected_writer_public_key_id: str,
    key_root: Path = KEY_ROOT,
    trust_root: Path = TRUST_ROOT,
    expected_uid: int = 0,
    expected_gid: int = 0,
) -> dict[str, Any]:
    if (
        type(expected_uid) is not int
        or type(expected_gid) is not int
        or expected_uid < 0
        or expected_gid < 0
        or any(
            not path.is_absolute() or ".." in path.parts
            for path in (key_root, trust_root)
        )
    ):
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_foundation_invalid"
        )
    expected_fields = {
        "schema",
        "writer_public_key_id",
        "keys",
        "key_count",
        "keys_distinct",
        "retain_created_keys_on_rollback",
        "private_content_or_digest_recorded",
        "credential_values_read",
        "secret_material_recorded",
        "secret_digest_recorded",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_foundation_invalid"
        )
    unsigned = {
        key: item for key, item in value.items() if key != "receipt_sha256"
    }
    keys = value.get("keys")
    domains = sorted(CREDENTIALS_BY_DOMAIN)
    if (
        value.get("schema") != KEY_FOUNDATION_SCHEMA
        or value.get("writer_public_key_id") != expected_writer_public_key_id
        or _SHA256.fullmatch(expected_writer_public_key_id or "") is None
        or not isinstance(keys, list)
        or value.get("key_count") != len(domains)
        or len(keys) != len(domains)
        or value.get("keys_distinct") is not True
        or value.get("retain_created_keys_on_rollback") is not True
        or value.get("private_content_or_digest_recorded") is not False
        or value.get("credential_values_read") is not False
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or _SHA256.fullmatch(str(value.get("receipt_sha256") or "")) is None
        or value.get("receipt_sha256") != _sha256(_canonical(unsigned))
    ):
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_foundation_invalid"
        )
    expected_row_fields = {
        "domain",
        "private_path",
        "private_uid",
        "private_gid",
        "private_mode",
        "public_path",
        "public_uid",
        "public_gid",
        "public_mode",
        "public_key_id",
        "created",
    }
    ids: list[str] = []
    for index, row in enumerate(keys):
        domain = domains[index]
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_row_fields
            or row.get("domain") != domain
            or row.get("private_path")
            != str(key_root / f"operational-edge-{domain}-receipt-private.pem")
            or row.get("public_path")
            != str(trust_root / f"{domain}-receipt-public.pem")
            or row.get("private_uid") != row.get("public_uid")
            or row.get("private_gid") != row.get("public_gid")
            or row.get("private_uid") != expected_uid
            or row.get("private_gid") != expected_gid
            or row.get("private_mode") != "0400"
            or row.get("public_mode") != "0444"
            or type(row.get("created")) is not bool
            or _SHA256.fullmatch(str(row.get("public_key_id") or "")) is None
        ):
            raise OperationalEdgeBootstrapError(
                "operational_edge_key_foundation_invalid"
            )
        ids.append(row["public_key_id"])
    if len(set(ids)) != len(ids) or expected_writer_public_key_id in ids:
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_foundation_not_distinct"
        )
    return dict(value)


def stage_operational_edge_key_foundation(
    *,
    expected_writer_public_key_id: str,
    writer_public_key_gid: int,
    writer_public_key_path: Path = KEY_ROOT / "writer-capability-public.pem",
    key_root: Path = KEY_ROOT,
    trust_root: Path = TRUST_ROOT,
    require_root: bool = True,
) -> dict[str, Any]:
    """Create missing complete pairs or cleanly re-observe existing pairs."""

    effective_uid = os.geteuid()
    effective_gid = os.getegid()
    if (
        type(writer_public_key_gid) is not int
        or writer_public_key_gid < 0
        or any(
            not path.is_absolute() or ".." in path.parts
            for path in (
                writer_public_key_path,
                key_root,
                trust_root,
            )
        )
    ):
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_foundation_input_invalid"
        )
    if require_root and effective_uid != 0:
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_foundation_requires_root"
        )
    if require_root and (
        key_root != KEY_ROOT
        or trust_root != TRUST_ROOT
        or writer_public_key_path != KEY_ROOT / "writer-capability-public.pem"
    ):
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_foundation_path_invalid"
        )
    uid = 0 if require_root else effective_uid
    gid = 0 if require_root else effective_gid
    _directory(key_root, uid=uid, gid=gid)
    _directory(
        trust_root.parent,
        uid=uid,
        gid=gid,
        create_mode=0o755,
        exact_mode=0o755,
    )
    _directory(
        trust_root,
        uid=uid,
        gid=gid,
        create_mode=0o755,
        exact_mode=0o755,
    )
    writer_raw = _stable_file(
        writer_public_key_path,
        uid=uid,
        gid=writer_public_key_gid,
        mode=0o440,
    )
    writer_id = ed25519_public_key_id(_public_key(writer_raw))
    if (
        _SHA256.fullmatch(expected_writer_public_key_id or "") is None
        or writer_id != expected_writer_public_key_id
    ):
        raise OperationalEdgeBootstrapError(
            "operational_edge_writer_key_identity_invalid"
        )
    rows = [
        _pair(
            domain=domain,
            private_path=key_root
            / f"operational-edge-{domain}-receipt-private.pem",
            public_path=trust_root / f"{domain}-receipt-public.pem",
            uid=uid,
            gid=gid,
        )
        for domain in sorted(CREDENTIALS_BY_DOMAIN)
    ]
    ids = [row["public_key_id"] for row in rows]
    if len(set(ids)) != len(ids) or writer_id in ids:
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_foundation_not_distinct"
        )
    unsigned = {
        "schema": KEY_FOUNDATION_SCHEMA,
        "writer_public_key_id": writer_id,
        "keys": rows,
        "key_count": len(rows),
        "keys_distinct": True,
        "retain_created_keys_on_rollback": True,
        "private_content_or_digest_recorded": False,
        "credential_values_read": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    receipt = {
        **unsigned,
        "receipt_sha256": _sha256(_canonical(unsigned)),
    }
    return validate_operational_edge_key_foundation(
        receipt,
        expected_writer_public_key_id=expected_writer_public_key_id,
        key_root=key_root,
        trust_root=trust_root,
        expected_uid=uid,
        expected_gid=gid,
    )


def stage_operational_edge_key_foundation_from_staged_writer_private(
    *,
    expected_writer_public_key_id: str | None = None,
    staged_writer_private_path: Path = STAGED_WRITER_PRIVATE_KEY,
    staging_root: Path = PRE_OWNER_STAGING_ROOT,
    require_root: bool = True,
) -> dict[str, Any]:
    """Pre-owner key staging rooted in the trusted writer private staging key.

    Only the derived writer public-key ID enters the receipt.  Neither writer
    private bytes nor any private-key digest is returned or persisted here.
    """

    effective_uid = os.geteuid()
    effective_gid = os.getegid()
    if any(
        not path.is_absolute() or ".." in path.parts
        for path in (staged_writer_private_path, staging_root)
    ):
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_foundation_input_invalid"
        )
    if require_root and effective_uid != 0:
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_foundation_requires_root"
        )
    if require_root and (
        staged_writer_private_path != STAGED_WRITER_PRIVATE_KEY
        or staging_root != PRE_OWNER_STAGING_ROOT
    ):
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_foundation_path_invalid"
        )
    uid = 0 if require_root else effective_uid
    gid = 0 if require_root else effective_gid
    _directory(staging_root, uid=uid, gid=gid, exact_mode=0o700)
    writer_private_raw = _stable_file(
        staged_writer_private_path,
        uid=uid,
        gid=gid,
        mode=0o400,
    )
    writer_id = ed25519_public_key_id(
        _private_key(writer_private_raw).public_key()
    )
    if expected_writer_public_key_id is not None and (
        _SHA256.fullmatch(expected_writer_public_key_id) is None
        or expected_writer_public_key_id != writer_id
    ):
        raise OperationalEdgeBootstrapError(
            "operational_edge_writer_key_identity_invalid"
        )
    rows = [
        _pair(
            domain=domain,
            private_path=(
                staging_root
                / f"operational-edge-{domain}-receipt-private.pem"
            ),
            public_path=staging_root / f"{domain}-receipt-public.pem",
            uid=uid,
            gid=gid,
        )
        for domain in sorted(CREDENTIALS_BY_DOMAIN)
    ]
    ids = [row["public_key_id"] for row in rows]
    if len(set(ids)) != len(ids) or writer_id in ids:
        raise OperationalEdgeBootstrapError(
            "operational_edge_key_foundation_not_distinct"
        )
    unsigned = {
        "schema": KEY_FOUNDATION_SCHEMA,
        "writer_public_key_id": writer_id,
        "keys": rows,
        "key_count": len(rows),
        "keys_distinct": True,
        "retain_created_keys_on_rollback": True,
        "private_content_or_digest_recorded": False,
        "credential_values_read": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    receipt = {
        **unsigned,
        "receipt_sha256": _sha256(_canonical(unsigned)),
    }
    return validate_operational_edge_key_foundation(
        receipt,
        expected_writer_public_key_id=writer_id,
        key_root=staging_root,
        trust_root=staging_root,
        expected_uid=uid,
        expected_gid=gid,
    )


@dataclass(frozen=True)
class OperationalEdgeHostArtifact:
    path: Path
    payload: bytes
    uid: int
    gid: int
    mode: int


@dataclass(frozen=True)
class OperationalEdgeFoundation:
    unit_bundle: OperationalEdgeUnitBundle
    artifacts: tuple[OperationalEdgeHostArtifact, ...]
    key_foundation: Mapping[str, Any]
    asset_verification: Mapping[str, Any]
    manifest: Mapping[str, Any]


def build_operational_edge_foundation(
    *,
    revision: str,
    service_identities: Mapping[str, Mapping[str, Any]],
    socket_groups: Mapping[str, Mapping[str, Any]],
    read_peer_uids: Sequence[int],
    gateway_uid: int,
    gateway_gid: int,
    release_owner_uid: int,
    release_owner_gid: int,
    writer_public_key_id: str,
    key_foundation: Mapping[str, Any],
    asset_verification: Mapping[str, Any],
    key_root: Path = KEY_ROOT,
    trust_root: Path = TRUST_ROOT,
    key_uid: int = 0,
    key_gid: int = 0,
) -> OperationalEdgeFoundation:
    """Render all install bytes from already attested pre-owner inputs."""

    if (
        _REVISION.fullmatch(revision or "") is None
        or gateway_uid not in read_peer_uids
        or type(release_owner_uid) is not int
        or type(release_owner_gid) is not int
        or release_owner_uid < 1
        or release_owner_gid < 1
    ):
        raise OperationalEdgeBootstrapError(
            "operational_edge_foundation_input_invalid"
        )
    keys = validate_operational_edge_key_foundation(
        key_foundation,
        expected_writer_public_key_id=writer_public_key_id,
        key_root=key_root,
        trust_root=trust_root,
        expected_uid=key_uid,
        expected_gid=key_gid,
    )
    verification = validate_packaged_operational_asset_verification(
        asset_verification,
        revision=revision,
        expected_release_root=(
            Path("/opt/adventico-ai-platform/hermes-agent-releases")
            / f"hermes-agent-{revision[:12]}"
        ),
        expected_uid=release_owner_uid,
        expected_gid=release_owner_gid,
    )
    receipt_ids = {
        row["domain"]: row["public_key_id"] for row in keys["keys"]
    }
    units = render_operational_edge_units(
        revision=revision,
        service_identities=service_identities,
        socket_groups=socket_groups,
        release_owner_uid=release_owner_uid,
        release_owner_gid=release_owner_gid,
        read_peer_uids=read_peer_uids,
        mutation_peer_uid=gateway_uid,
        mutation_peer_gid=gateway_gid,
        receipt_public_key_ids=receipt_ids,
        writer_key_id=writer_public_key_id,
    )
    artifacts = tuple(
        [
            OperationalEdgeHostArtifact(
                path=SYSTEMD_ROOT / name,
                payload=payload,
                uid=0,
                gid=0,
                mode=0o644,
            )
            for name, payload in sorted(units.units.items())
        ]
        + [
            OperationalEdgeHostArtifact(
                path=Path(path),
                payload=payload,
                uid=0,
                gid=0,
                mode=0o400,
            )
            for path, payload in sorted(units.configs.items())
        ]
        + [
            OperationalEdgeHostArtifact(
                path=CLIENT_CONFIG_PATH,
                payload=units.client_config,
                uid=0,
                gid=0,
                mode=0o444,
            )
        ]
    )
    artifact_rows = [
        {
            "path": str(item.path),
            "uid": item.uid,
            "gid": item.gid,
            "mode": f"{item.mode:04o}",
            "size": len(item.payload),
            "sha256": _sha256(item.payload),
        }
        for item in artifacts
    ]
    unsigned = {
        "schema": FOUNDATION_SCHEMA,
        "release_revision": revision,
        "unit_bundle_sha256": units.manifest["bundle_sha256"],
        "key_foundation_sha256": keys["receipt_sha256"],
        "asset_manifest_sha256": verification["manifest_sha256"],
        "release_owner_uid": release_owner_uid,
        "release_owner_gid": release_owner_gid,
        "writer_public_key_id": writer_public_key_id,
        "receipt_public_key_ids": receipt_ids,
        "artifacts": artifact_rows,
        "artifact_count": len(artifact_rows),
        "credential_values_read": False,
        "private_content_or_digest_recorded": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    manifest = {
        **unsigned,
        "foundation_sha256": _sha256(_canonical(unsigned)),
    }
    return OperationalEdgeFoundation(
        unit_bundle=units,
        artifacts=artifacts,
        key_foundation=keys,
        asset_verification=verification,
        manifest=manifest,
    )


def prepare_operational_edge_foundation(
    *,
    revision: str,
    service_identities: Mapping[str, Mapping[str, Any]],
    socket_groups: Mapping[str, Mapping[str, Any]],
    read_peer_uids: Sequence[int],
    gateway_uid: int,
    gateway_gid: int,
    release_owner_uid: int,
    release_owner_gid: int,
    writer_public_key_id: str,
    writer_public_key_gid: int,
    release_root: Path | None = None,
    writer_public_key_path: Path = KEY_ROOT / "writer-capability-public.pem",
    key_root: Path = KEY_ROOT,
    trust_root: Path = TRUST_ROOT,
    require_root: bool = True,
) -> OperationalEdgeFoundation:
    """Pre-owner stager: create keys, verify assets, then seal artifacts."""

    expected_release = (
        Path("/opt/adventico-ai-platform/hermes-agent-releases")
        / f"hermes-agent-{revision[:12]}"
    )
    release = expected_release if release_root is None else release_root
    if (
        _REVISION.fullmatch(revision or "") is None
        or require_root and release != expected_release
    ):
        raise OperationalEdgeBootstrapError(
            "operational_edge_foundation_input_invalid"
        )
    keys = stage_operational_edge_key_foundation(
        expected_writer_public_key_id=writer_public_key_id,
        writer_public_key_gid=writer_public_key_gid,
        writer_public_key_path=writer_public_key_path,
        key_root=key_root,
        trust_root=trust_root,
        require_root=require_root,
    )
    verification = verify_packaged_operational_assets(
        release_root=release,
        revision=revision,
        expected_uid=release_owner_uid,
        expected_gid=release_owner_gid,
        reported_release_root=expected_release,
    )
    return build_operational_edge_foundation(
        revision=revision,
        service_identities=service_identities,
        socket_groups=socket_groups,
        read_peer_uids=read_peer_uids,
        gateway_uid=gateway_uid,
        gateway_gid=gateway_gid,
        release_owner_uid=release_owner_uid,
        release_owner_gid=release_owner_gid,
        writer_public_key_id=writer_public_key_id,
        key_foundation=keys,
        asset_verification=verification,
        key_root=key_root,
        trust_root=trust_root,
        key_uid=0 if require_root else os.geteuid(),
        key_gid=0 if require_root else os.getegid(),
    )


__all__ = [
    "FOUNDATION_SCHEMA",
    "KEY_FOUNDATION_SCHEMA",
    "OperationalEdgeBootstrapError",
    "OperationalEdgeFoundation",
    "OperationalEdgeHostArtifact",
    "PRE_OWNER_STAGING_ROOT",
    "STAGED_WRITER_PRIVATE_KEY",
    "build_operational_edge_foundation",
    "prepare_operational_edge_foundation",
    "stage_operational_edge_key_foundation",
    "stage_operational_edge_key_foundation_from_staged_writer_private",
    "validate_operational_edge_key_foundation",
]
