#!/usr/bin/env python3
"""Author and install the exact owner-gate storage activation seal.

This is a post-bootstrap, post-IAM boundary.  It has no provider client and no
Cloud mutation capability.  The only mutation it can perform is publishing the
fixed ``storage-executor-enabled`` file after revalidating the immutable release
authority and the fixed, root-owned activation evidence directory.  Its
executor-readable output is the complete canonical authorization record; the
second no-replace output is only a root-readable append-only audit mirror.

All values written into the seal are derived from validated artifacts.  The
entrypoint accepts no release, evidence, digest, resource, or destination
argument.  Before any mutation it requires the canonical release binding sent
by the fixed owner launcher on stdin and compares it with its own immutable
installed release.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from scripts.canary import direct_iam_identity_authority as direct_iam
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_owner_reauth as owner_reauth
from scripts.canary import owner_gate_package as package
from scripts.canary import owner_gate_preflight as preflight
from scripts.canary import owner_gate_production_ingress_contract as production_ingress
from scripts.canary import owner_gate_trust as trust
from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import passkey_v2_service as service
from scripts.canary import passkey_v2_storage_growth as storage
from scripts.canary import production_cutover_passkey as production_cutover


ACTIVATION_EVIDENCE_BASE = Path(
    "/var/lib/muncho-owner-gate/activation-evidence"
)
ACTIVATION_RECEIPT_BASE = Path(
    "/var/lib/muncho-owner-gate/activation-receipts"
)
ACTIVATION_SEAL_PATH = foundation.MUTATION_ENABLE_SEAL
ACTIVATION_LOCK_PATH = Path(
    "/run/muncho-owner-gate/storage-executor-activation.lock"
)
AUTHORITY_RECEIPT_PUBLIC_KEY_PATH = Path(
    "/etc/muncho-owner-gate/public/authority-receipt-public.pem"
)
RELEASE_BASE = foundation.RELEASE_BASE
ROOT_UID = 0
ROOT_GID = 0
EXECUTOR_GID = storage.OWNER_GATE_EXECUTOR_UID
EVIDENCE_DIRECTORY_MODE = 0o500
STATE_DIRECTORY_MODE = 0o700
CONFIG_DIRECTORY_MODE = 0o755
RELEASE_DIRECTORY_MODE = 0o555
EVIDENCE_FILE_MODE = 0o444
SEAL_FILE_MODE = 0o440
RECEIPT_FILE_MODE = 0o444
LOCK_FILE_MODE = 0o600
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_PAYLOAD_BYTES = 128 * 1024 * 1024
MAX_ACTIVATION_REQUEST_BYTES = 1024

ACTIVATION_RECEIPT_SCHEMA = "muncho-owner-gate-storage-activation-receipt.v1"
ACTIVATION_RESPONSE_SCHEMA = "muncho-owner-gate-storage-activation-response.v1"
ACTIVATION_REQUEST_SCHEMA = "muncho-owner-gate-storage-activation-request.v1"
ACTIVATION_EVIDENCE_VALIDATION_SCHEMA = (
    "muncho-owner-gate-activation-evidence-validation.v1"
)
RELEASE_LINEAGE_SCHEMA = "muncho-owner-gate-portable-release-lineage.v1"

NETWORK_EVIDENCE_NAME = "network-evidence.json"
INERT_PRODUCTION_INGRESS_OBSERVATION_NAME = (
    "inert-production-ingress-observation.json"
)
INERT_CLOUD_OBSERVATION_NAME = "inert-cloud-observation.json"
INERT_HOST_OBSERVATION_NAME = "inert-host-observation.json"
INERT_PREFLIGHT_NAME = "inert-preflight.json"
POST_IAM_CLOUD_OBSERVATION_NAME = "post-iam-cloud-observation.json"
POST_IAM_HOST_OBSERVATION_NAME = "post-iam-host-observation.json"
POST_IAM_PREFLIGHT_NAME = "post-iam-preflight.json"
POST_IAM_PRODUCTION_INGRESS_OBSERVATION_NAME = (
    "post-iam-production-ingress-observation.json"
)
ACTIVATION_OWNER_REAUTH_NAME = "activation-owner-reauthentication-receipt.json"

EVIDENCE_NAMES = (
    NETWORK_EVIDENCE_NAME,
    INERT_PRODUCTION_INGRESS_OBSERVATION_NAME,
    INERT_CLOUD_OBSERVATION_NAME,
    INERT_HOST_OBSERVATION_NAME,
    INERT_PREFLIGHT_NAME,
    POST_IAM_PRODUCTION_INGRESS_OBSERVATION_NAME,
    POST_IAM_CLOUD_OBSERVATION_NAME,
    POST_IAM_HOST_OBSERVATION_NAME,
    POST_IAM_PREFLIGHT_NAME,
    ACTIVATION_OWNER_REAUTH_NAME,
)

_REQUIRED_ACTIVATION_PAYLOADS = frozenset({
    "bin/muncho-owner-gate-activate-storage",
    "bin/muncho-owner-gate-stage-activation-evidence",
    "scripts/canary/owner_gate_activation_evidence_stager.py",
    "scripts/canary/owner_gate_activation_seal.py",
})
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_PROCESS_LOCK = threading.Lock()


class OwnerGateActivationSealError(RuntimeError):
    """Stable, secret-free activation author/installer failure."""


def _close_descriptor(descriptor: int, *, code: str) -> None:
    try:
        os.close(descriptor)
    except OSError:
        raise OwnerGateActivationSealError(code) from None


def _release_lock(descriptor: int) -> None:
    failed = False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError:
        failed = True
    try:
        os.close(descriptor)
    except OSError:
        failed = True
    if failed:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_lock_release_failed"
        ) from None


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    identity: tuple[int, int, int, int, int, int]
    sha256: str

    def require_unchanged(self) -> None:
        try:
            current = self.path.lstat()
            resolved = self.path.resolve(strict=True)
            opened = resolved.stat()
        except (OSError, RuntimeError):
            raise OwnerGateActivationSealError(
                "owner_gate_activation_evidence_changed"
            ) from None
        if (
            stat.S_ISLNK(current.st_mode)
            or resolved != self.path
            or (current.st_dev, current.st_ino)
            != (opened.st_dev, opened.st_ino)
            or (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
                current.st_nlink,
            )
            != self.identity
        ):
            raise OwnerGateActivationSealError(
                "owner_gate_activation_evidence_changed"
            )


@dataclass(frozen=True)
class _ReleaseAuthority:
    release_revision: str
    plan: foundation.OwnerGateFoundationPlan
    manifest: Mapping[str, Any]
    release_trust: Mapping[str, Any]
    direct_iam: Mapping[str, Any]
    direct_iam_sha256: str
    release_public_key: Ed25519PublicKey
    collector_keys: Mapping[str, Ed25519PublicKey]
    snapshots: tuple[_FileSnapshot, ...]


@dataclass(frozen=True)
class _DerivedActivation:
    seal: Mapping[str, Any]
    snapshots: tuple[_FileSnapshot, ...]
    fresh_through_unix: int


def _canonical(value: Any) -> bytes:
    try:
        return protocol.canonical_json_bytes(value)
    except (protocol.PasskeyV2ProtocolError, TypeError, ValueError):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_json_invalid"
        ) from None


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        os.fsync(descriptor)
    except OSError:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_directory_fsync_failed"
        ) from None
    finally:
        if descriptor is not None:
            _close_descriptor(
                descriptor,
                code="owner_gate_activation_directory_fsync_failed",
            )


def _require_directory(
    path: Path,
    *,
    parent: Path | None,
    uid: int,
    gid: int,
    mode: int,
) -> None:
    if (
        not path.is_absolute()
        or ".." in path.parts
        or (parent is not None and path.parent != parent)
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_directory_invalid"
        )
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
        after = resolved.stat()
    except (OSError, RuntimeError):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_directory_invalid"
        ) from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(after.st_mode)
        or resolved != path
        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or after.st_uid != uid
        or after.st_gid != gid
        or stat.S_IMODE(after.st_mode) != mode
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_directory_invalid"
        )


def _read_regular(
    path: Path,
    *,
    maximum: int,
    uid: int,
    gid: int,
    modes: frozenset[int],
    code: str,
    minimum: int = 1,
) -> tuple[bytes, _FileSnapshot]:
    if not path.is_absolute() or ".." in path.parts:
        raise OwnerGateActivationSealError(code)
    descriptor: int | None = None
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or resolved != path
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_uid != uid
            or opened.st_gid != gid
            or stat.S_IMODE(opened.st_mode) not in modes
            or opened.st_nlink != 1
            or opened.st_size < minimum
            or opened.st_size > maximum
        ):
            raise OwnerGateActivationSealError(code)
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise OwnerGateActivationSealError(code)
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
            opened.st_nlink,
        )
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        ):
            raise OwnerGateActivationSealError(code)
        raw = b"".join(chunks)
        return raw, _FileSnapshot(
            path=path,
            identity=identity,
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    except (OSError, RuntimeError):
        raise OwnerGateActivationSealError(code) from None
    finally:
        if descriptor is not None:
            _close_descriptor(descriptor, code=code)


def _decode_canonical(raw: bytes, *, code: str) -> Mapping[str, Any]:
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise OwnerGateActivationSealError(code)
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError):
        raise OwnerGateActivationSealError(code) from None
    if not isinstance(value, Mapping) or _canonical(value) != raw:
        raise OwnerGateActivationSealError(code)
    return dict(value)


def _safe_release_path(release: Path, relative_text: Any) -> Path:
    if not isinstance(relative_text, str):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_package_payload_invalid"
        )
    relative = Path(relative_text)
    if (
        not relative_text
        or relative.is_absolute()
        or ".." in relative.parts
        or str(relative) != relative_text
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_package_payload_invalid"
        )
    path = release / relative
    try:
        if path.resolve(strict=True) != path:
            raise OwnerGateActivationSealError(
                "owner_gate_activation_package_payload_invalid"
            )
    except (OSError, RuntimeError):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_package_payload_invalid"
        ) from None
    return path


def _validate_release_root(release: Path) -> str:
    if (
        not release.is_absolute()
        or release.parent != RELEASE_BASE
        or _REVISION.fullmatch(release.name) is None
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_release_invalid"
        )
    _require_directory(
        RELEASE_BASE.parent,
        parent=RELEASE_BASE.parent.parent,
        uid=ROOT_UID,
        gid=ROOT_GID,
        mode=CONFIG_DIRECTORY_MODE,
    )
    _require_directory(
        RELEASE_BASE,
        parent=RELEASE_BASE.parent,
        uid=ROOT_UID,
        gid=ROOT_GID,
        mode=CONFIG_DIRECTORY_MODE,
    )
    _require_directory(
        release,
        parent=RELEASE_BASE,
        uid=ROOT_UID,
        gid=ROOT_GID,
        mode=RELEASE_DIRECTORY_MODE,
    )
    return release.name


def _load_collector_key(
    release: Path,
    *,
    name: str,
    expected_key_id: str,
) -> tuple[Ed25519PublicKey, _FileSnapshot]:
    raw, snapshot = _read_regular(
        release / "trust" / f"{name}-observation-attestation.pub",
        maximum=32,
        uid=ROOT_UID,
        gid=ROOT_GID,
        modes=frozenset({EVIDENCE_FILE_MODE}),
        code="owner_gate_activation_collector_key_invalid",
    )
    if (
        len(raw) != 32
        or hashlib.sha256(raw).hexdigest() != expected_key_id
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_collector_key_invalid"
        )
    try:
        return Ed25519PublicKey.from_public_bytes(raw), snapshot
    except ValueError:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_collector_key_invalid"
        ) from None


def _load_release_authority(
    *,
    release: Path,
    evidence_root: Path,
) -> _ReleaseAuthority:
    revision = _validate_release_root(release)
    snapshots: list[_FileSnapshot] = []
    try:
        release_trust = trust.load_pinned_release_trust(
            manifest_path=release / "trust/release-trust.json",
            public_key_path=release / "trust/release-trust-signing.pub",
            expected_uid=ROOT_UID,
        )
    except trust.OwnerGateTrustError:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_release_trust_invalid"
        ) from None
    _trust_raw, trust_snapshot = _read_regular(
        release / "trust/release-trust.json",
        maximum=MAX_JSON_BYTES,
        uid=ROOT_UID,
        gid=ROOT_GID,
        modes=frozenset({EVIDENCE_FILE_MODE}),
        code="owner_gate_activation_release_trust_invalid",
    )
    public_raw, public_snapshot = _read_regular(
        release / "trust/release-trust-signing.pub",
        maximum=32,
        uid=ROOT_UID,
        gid=ROOT_GID,
        modes=frozenset({EVIDENCE_FILE_MODE}),
        code="owner_gate_activation_release_trust_invalid",
    )
    if (
        trust_snapshot.sha256
        != release_trust.get("trust_manifest_sha256")
        or public_snapshot.sha256
        != release_trust.get("trust_public_key_sha256")
        or len(public_raw) != 32
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_release_trust_invalid"
        )
    try:
        release_public_key = Ed25519PublicKey.from_public_bytes(public_raw)
    except ValueError:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_release_trust_invalid"
        ) from None
    snapshots.extend((trust_snapshot, public_snapshot))

    manifest_raw, manifest_snapshot = _read_regular(
        release / "package-manifest.json",
        maximum=MAX_JSON_BYTES,
        uid=ROOT_UID,
        gid=ROOT_GID,
        modes=frozenset({EVIDENCE_FILE_MODE}),
        code="owner_gate_activation_package_invalid",
    )
    manifest = _decode_canonical(
        manifest_raw,
        code="owner_gate_activation_package_invalid",
    )
    try:
        package.validate_authorized_manifest(
            manifest,
            authority=release_trust,
        )
    except package.OwnerGatePackageError:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_package_invalid"
        ) from None
    snapshots.append(manifest_snapshot)
    if (
        manifest.get("release_revision") != revision
        or release_trust.get("release_revision") != revision
        or manifest.get("foundation_source_revision")
        != release_trust.get("foundation_source_revision")
        or manifest.get("foundation_source_tree_oid")
        != release_trust.get("foundation_source_tree_oid")
        or manifest.get("foundation_source_revision") == revision
        or manifest.get("release_root") != str(release)
        or manifest.get("release_owner") != "root:root"
        or manifest.get("release_directory_mode") != "0555"
        or manifest.get("immutable_after_install") is not True
        or manifest.get("offline_bootstrap") is not True
        or manifest.get("network_install_required") is not False
        or manifest.get("activation_performed") is not False
        or manifest.get("cloud_mutation_performed") is not False
        or _SHA256.fullmatch(str(manifest.get("package_sha256", ""))) is None
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_package_invalid"
        )

    payload_relatives: set[str] = set()
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_package_invalid"
        )
    for item in payloads:
        if not isinstance(item, Mapping):
            raise OwnerGateActivationSealError(
                "owner_gate_activation_package_payload_invalid"
            )
        relative = item.get("release_relative")
        if (
            not isinstance(relative, str)
            or relative in payload_relatives
            or _SHA256.fullmatch(str(item.get("sha256", ""))) is None
            or type(item.get("size")) is not int
            or item["size"] < 0
            or item["size"] > MAX_PAYLOAD_BYTES
            or item.get("owner") != "root:root"
            or item.get("mode") not in {"0444", "0555"}
        ):
            raise OwnerGateActivationSealError(
                "owner_gate_activation_package_payload_invalid"
            )
        payload_relatives.add(relative)
        path = _safe_release_path(release, relative)
        raw, snapshot = _read_regular(
            path,
            maximum=MAX_PAYLOAD_BYTES,
            uid=ROOT_UID,
            gid=ROOT_GID,
            modes=frozenset({int(str(item["mode"]), 8)}),
            code="owner_gate_activation_package_payload_invalid",
            minimum=0,
        )
        if (
            len(raw) != item["size"]
            or snapshot.sha256 != item["sha256"]
        ):
            raise OwnerGateActivationSealError(
                "owner_gate_activation_package_payload_invalid"
            )
        snapshots.append(snapshot)
    if not _REQUIRED_ACTIVATION_PAYLOADS.issubset(payload_relatives):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_package_payload_invalid"
        )

    direct_raw, direct_snapshot = _read_regular(
        release / "trust/direct-iam-identity-authority.json",
        maximum=direct_iam.MAX_BYTES,
        uid=ROOT_UID,
        gid=ROOT_GID,
        modes=frozenset({EVIDENCE_FILE_MODE}),
        code="owner_gate_activation_direct_iam_invalid",
    )
    try:
        direct = direct_iam.decode_canonical(
            direct_raw,
            release_revision=str(
                release_trust["foundation_source_revision"]
            ),
        )
    except direct_iam.DirectIamIdentityAuthorityError:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_direct_iam_invalid"
        ) from None
    direct_digest = hashlib.sha256(direct_raw).hexdigest()
    if (
        direct_digest
        != manifest.get("direct_iam_identity_authority_sha256")
        or direct_digest
        != release_trust.get("direct_iam_identity_authority_sha256")
        or direct.get("release_revision")
        != manifest.get("foundation_source_revision")
        or direct.get("release_revision")
        != release_trust.get("foundation_source_revision")
        or direct.get("pre_foundation_authority_sha256")
        != manifest.get("pre_foundation_authority_sha256")
        or direct.get("pre_foundation_authority_sha256")
        != release_trust.get("pre_foundation_authority_sha256")
        or direct.get("foundation_apply_receipt_sha256")
        != manifest.get("foundation_apply_receipt_sha256")
        or direct.get("foundation_apply_receipt_sha256")
        != release_trust.get("foundation_apply_receipt_sha256")
        or direct.get("resource_ancestor_chain")
        != manifest.get("resource_ancestor_chain")
        or direct.get("resource_ancestor_chain")
        != release_trust.get("resource_ancestor_chain")
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_foundation_chain_invalid"
        )
    snapshots.append(direct_snapshot)

    collectors = release_trust.get("collector_public_key_ids")
    if not isinstance(collectors, Mapping) or set(collectors) != {
        "network",
        "cloud",
        "host",
    }:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_collector_key_invalid"
        )
    collector_keys: dict[str, Ed25519PublicKey] = {}
    for name in ("network", "cloud", "host"):
        key, snapshot = _load_collector_key(
            release,
            name=name,
            expected_key_id=str(collectors[name]),
        )
        collector_keys[name] = key
        snapshots.append(snapshot)

    network_raw, network_snapshot = _read_regular(
        evidence_root / NETWORK_EVIDENCE_NAME,
        maximum=MAX_JSON_BYTES,
        uid=ROOT_UID,
        gid=ROOT_GID,
        modes=frozenset({EVIDENCE_FILE_MODE}),
        code="owner_gate_activation_network_evidence_invalid",
    )
    network_mapping = _decode_canonical(
        network_raw,
        code="owner_gate_activation_network_evidence_invalid",
    )
    collected_at = network_mapping.get("collected_at_unix")
    if type(collected_at) is not int or collected_at <= 0:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_network_evidence_invalid"
        )
    try:
        network = foundation.ProductionNetworkEvidence.from_mapping(
            network_mapping,
            public_key=collector_keys["network"],
            expected_public_key_id=str(collectors["network"]),
            now_unix=collected_at,
        )
        ancestor_chain = direct["resource_ancestor_chain"]
        organization = str(ancestor_chain[-1])
        if not organization.startswith("organizations/"):
            raise ValueError
        spec = foundation.OwnerGateSpec(
            release_revision=revision,
            boot_image_self_link=str(release_trust["boot_image_self_link"]),
            package_inventory_sha256=str(
                manifest["package_inventory_sha256"]
            ),
            interpreter_sha256=str(manifest["interpreter_sha256"]),
            network_collector_public_key_id=str(collectors["network"]),
            organization_id=organization.split("/", 1)[1],
            ancestry_evidence_sha256=str(
                release_trust["project_ancestry_evidence_sha256"]
            ),
            cloud_collector_public_key_id=str(collectors["cloud"]),
            host_collector_public_key_id=str(collectors["host"]),
        )
        plan = foundation.build_plan(
            spec=spec,
            network_evidence=network,
            network_collector_public_key=collector_keys["network"],
            now_unix=collected_at,
        )
    except (
        ValueError,
        KeyError,
        foundation.OwnerGateFoundationError,
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_foundation_plan_invalid"
        ) from None
    snapshots.append(network_snapshot)
    return _ReleaseAuthority(
        release_revision=revision,
        plan=plan,
        manifest=manifest,
        release_trust=release_trust,
        direct_iam=direct,
        direct_iam_sha256=direct_digest,
        release_public_key=release_public_key,
        collector_keys=collector_keys,
        snapshots=tuple(snapshots),
    )


def _load_evidence_json(
    evidence_root: Path,
    name: str,
) -> tuple[Mapping[str, Any], _FileSnapshot]:
    raw, snapshot = _read_regular(
        evidence_root / name,
        maximum=MAX_JSON_BYTES,
        uid=ROOT_UID,
        gid=ROOT_GID,
        modes=frozenset({EVIDENCE_FILE_MODE}),
        code="owner_gate_activation_evidence_invalid",
    )
    return (
        _decode_canonical(
            raw,
            code="owner_gate_activation_evidence_invalid",
        ),
        snapshot,
    )


def _derive_activation(
    *,
    release: Path,
    evidence_root: Path,
    now_unix: int,
    enforce_fresh: bool,
) -> _DerivedActivation:
    if type(now_unix) is not int or now_unix <= 0:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_time_invalid"
        )
    authority = _load_release_authority(
        release=release,
        evidence_root=evidence_root,
    )
    values: dict[str, Mapping[str, Any]] = {}
    snapshots = list(authority.snapshots)
    for name in EVIDENCE_NAMES[1:]:
        value, snapshot = _load_evidence_json(evidence_root, name)
        values[name] = value
        snapshots.append(snapshot)

    inert_report = values[INERT_PREFLIGHT_NAME]
    post_report = values[POST_IAM_PREFLIGHT_NAME]
    inert_time = inert_report.get("observed_at_unix")
    post_time = post_report.get("observed_at_unix")
    if (
        type(inert_time) is not int
        or type(post_time) is not int
        or inert_time <= 0
        or post_time < inert_time
        or post_time > now_unix
        or (
            enforce_fresh
            and (
                now_unix - inert_time > foundation.PREFLIGHT_MAX_AGE_SECONDS
                or now_unix - post_time > foundation.PREFLIGHT_MAX_AGE_SECONDS
            )
        )
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_evidence_stale"
        )
    try:
        expected_inert = preflight.build_preflight_report(
            plan=authority.plan,
            production_ingress_observation=values[
                INERT_PRODUCTION_INGRESS_OBSERVATION_NAME
            ],
            release_public_key=authority.release_public_key,
            cloud_observation=values[INERT_CLOUD_OBSERVATION_NAME],
            host_observation=values[INERT_HOST_OBSERVATION_NAME],
            cloud_collector_public_key=authority.collector_keys["cloud"],
            host_collector_public_key=authority.collector_keys["host"],
            now_unix=inert_time,
        )
        expected_post = preflight.build_post_iam_preflight_report(
            plan=authority.plan,
            production_ingress_observation=values[
                POST_IAM_PRODUCTION_INGRESS_OBSERVATION_NAME
            ],
            release_public_key=authority.release_public_key,
            cloud_observation=values[POST_IAM_CLOUD_OBSERVATION_NAME],
            host_observation=values[POST_IAM_HOST_OBSERVATION_NAME],
            cloud_collector_public_key=authority.collector_keys["cloud"],
            host_collector_public_key=authority.collector_keys["host"],
            now_unix=post_time,
        )
    except preflight.OwnerGatePreflightError:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_evidence_invalid"
        ) from None
    if (
        _canonical(inert_report) != _canonical(expected_inert)
        or _canonical(post_report) != _canonical(expected_post)
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_evidence_invalid"
        )

    if enforce_fresh:
        try:
            production_ingress.validate_signed_production_ingress_observation(
                values[INERT_PRODUCTION_INGRESS_OBSERVATION_NAME],
                phase="inert",
                release_revision=authority.release_revision,
                plan_sha256=authority.plan.sha256,
                release_public_key=authority.release_public_key,
                now_unix=now_unix,
            )
            production_ingress.validate_signed_production_ingress_observation(
                values[POST_IAM_PRODUCTION_INGRESS_OBSERVATION_NAME],
                phase="post_iam",
                release_revision=authority.release_revision,
                plan_sha256=authority.plan.sha256,
                release_public_key=authority.release_public_key,
                now_unix=now_unix,
            )
        except production_ingress.ProductionIngressObservationError:
            raise OwnerGateActivationSealError(
                "owner_gate_activation_evidence_stale"
            ) from None

    try:
        activation_owner_reauth = owner_reauth.validate_owner_reauth_receipt(
            values[ACTIVATION_OWNER_REAUTH_NAME],
            public_key=authority.release_public_key,
            now_unix=now_unix if enforce_fresh else None,
        )
    except owner_reauth.OwnerGateOwnerReauthError:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_owner_reauth_invalid"
        ) from None
    owner_runtime = activation_owner_reauth.get(
        "trusted_runtime_identity"
    )
    owner_probe = activation_owner_reauth.get("authenticated_probe")
    owner_issued = activation_owner_reauth.get("issued_at_unix")
    owner_expires = activation_owner_reauth.get("expires_at_unix")
    if (
        not isinstance(owner_runtime, Mapping)
        or not isinstance(owner_probe, Mapping)
        or type(owner_issued) is not int
        or type(owner_expires) is not int
        or owner_issued <= post_time
        or owner_issued > now_unix
        or owner_expires < owner_issued
        or owner_runtime.get("release_revision")
        != authority.release_revision
        or owner_probe.get("project_number")
        != authority.direct_iam.get("project_number")
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_owner_reauth_invalid"
        )

    inert_cloud = values[INERT_CLOUD_OBSERVATION_NAME]
    inert_host = values[INERT_HOST_OBSERVATION_NAME]
    post_cloud = values[POST_IAM_CLOUD_OBSERVATION_NAME]
    post_host = values[POST_IAM_HOST_OBSERVATION_NAME]
    owner_vm_id = str(post_cloud.get("instance", {}).get("numeric_id", ""))
    package_sha256 = str(authority.manifest["package_sha256"])
    package_inventory_sha256 = str(
        authority.manifest["package_inventory_sha256"]
    )
    interpreter_sha256 = str(authority.manifest["interpreter_sha256"])
    for host in (inert_host, post_host):
        host_release = host.get("release")
        if (
            not isinstance(host_release, Mapping)
            or host_release.get("package_sha256") != package_sha256
            or host_release.get("package_inventory_sha256")
            != package_inventory_sha256
            or host_release.get("python_executable_sha256")
            != interpreter_sha256
        ):
            raise OwnerGateActivationSealError(
                "owner_gate_activation_host_release_invalid"
            )
    if (
        _NUMERIC_ID.fullmatch(owner_vm_id) is None
        or owner_vm_id
        != str(authority.direct_iam["owner_gate_vm_numeric_id"])
        or inert_cloud.get("instance", {}).get("numeric_id") != owner_vm_id
        or post_cloud.get("targets", {}).get("instance_numeric_id")
        != storage.VM_INSTANCE_ID
        or post_cloud.get("targets", {}).get("disk_numeric_id")
        != storage.DISK_ID
        or expected_post.get("target_instance_numeric_id")
        != storage.VM_INSTANCE_ID
        or expected_post.get("target_disk_numeric_id") != storage.DISK_ID
        or expected_inert.get("mutation_iam_binding_present") is not False
        or expected_inert.get("inert_noop_security_smoke_ready") is not True
        or expected_post.get("executor_activation_seal_present") is not False
        or expected_post.get("mutation_attempted") is not False
        or expected_post.get("topology_iam_readiness_seal_can_be_installed")
        is not True
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_identity_invalid"
        )

    lineage_unsigned = {
        "schema": RELEASE_LINEAGE_SCHEMA,
        "release_revision": authority.release_revision,
        "source_tree_oid": authority.manifest["source_tree_oid"],
        "package_inventory_sha256": package_inventory_sha256,
        "release_trust_manifest_sha256": authority.release_trust[
            "trust_manifest_sha256"
        ],
        "release_trust_public_key_sha256": authority.release_trust[
            "trust_public_key_sha256"
        ],
        "direct_iam_identity_authority_sha256": (
            authority.direct_iam_sha256
        ),
        "pre_foundation_authority_sha256": authority.direct_iam[
            "pre_foundation_authority_sha256"
        ],
        "foundation_apply_receipt_sha256": authority.direct_iam[
            "foundation_apply_receipt_sha256"
        ],
        "foundation_owner_reauthentication_receipt_sha256": (
            authority.direct_iam[
                "owner_reauthentication_receipt_sha256"
            ]
        ),
        "activation_owner_reauthentication_receipt_sha256": (
            activation_owner_reauth[
                "owner_reauthentication_receipt_sha256"
            ]
        ),
        "project_ancestry_evidence_sha256": authority.release_trust[
            "project_ancestry_evidence_sha256"
        ],
        "project_ancestry_chain_sha256": authority.release_trust[
            "project_ancestry_chain_sha256"
        ],
        "resource_ancestor_chain": list(
            authority.direct_iam["resource_ancestor_chain"]
        ),
        "inert_preflight_receipt_sha256": expected_inert[
            "report_sha256"
        ],
        "post_iam_preflight_receipt_sha256": expected_post[
            "report_sha256"
        ],
    }
    release_lineage = {
        **lineage_unsigned,
        "lineage_sha256": protocol.sha256_json(lineage_unsigned),
    }
    evidence_digests = {
        snapshot.path.name: snapshot.sha256
        for snapshot in snapshots
        if snapshot.path.parent == evidence_root
    }
    if set(evidence_digests) != set(EVIDENCE_NAMES):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_evidence_invalid"
        )
    unsigned = {
        "schema": service.ACTIVATION_SEAL_SCHEMA,
        "release_revision": authority.release_revision,
        "foundation_plan_sha256": authority.plan.sha256,
        "package_sha256": package_sha256,
        "cloud_topology_receipt_sha256": str(post_cloud["report_sha256"]),
        "host_security_smoke_receipt_sha256": str(
            inert_host["report_sha256"]
        ),
        "iam_repreflight_receipt_sha256": str(expected_post["report_sha256"]),
        "owner_gate_vm_numeric_id": owner_vm_id,
        "target_instance_numeric_id": storage.VM_INSTANCE_ID,
        "target_disk_numeric_id": storage.DISK_ID,
        "created_at_unix": post_time,
        # The operative executor-readable file is itself the complete
        # canonical authorization record.  The receipt published below is an
        # append-only audit mirror, never a second half required for truth.
        "authorization_record_complete": True,
        "verified_release_lineage": release_lineage,
        "evidence_file_sha256": evidence_digests,
        "activation_installed": True,
        "cloud_mutation_performed": False,
    }
    seal = {
        **unsigned,
        "seal_sha256": protocol.sha256_json(unsigned),
    }
    try:
        checked = service.validate_activation_seal(
            seal,
            expected_release_revision=authority.release_revision,
            now_unix=now_unix,
        )
    except service.PasskeyV2ServiceError:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_service_contract_invalid"
        ) from None
    return _DerivedActivation(
        seal=checked,
        snapshots=tuple(snapshots),
        fresh_through_unix=min(
            inert_time + foundation.PREFLIGHT_MAX_AGE_SECONDS,
            post_time + foundation.PREFLIGHT_MAX_AGE_SECONDS,
            int(
                values[INERT_PRODUCTION_INGRESS_OBSERVATION_NAME][
                    "fresh_through_unix"
                ]
            ),
            int(
                values[POST_IAM_PRODUCTION_INGRESS_OBSERVATION_NAME][
                    "fresh_through_unix"
                ]
            ),
            owner_expires,
        ),
    )


def validate_activation_evidence_strict(
    *,
    release: Path,
    evidence_root: Path,
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate an evidence bundle through the activation derivation boundary.

    This public wrapper deliberately has no freshness waiver.  It derives the
    exact prospective activation record, rechecks every release and evidence
    snapshot, and returns only a non-authorizing validation summary.  It never
    publishes the activation seal or any other file.
    """

    try:
        derived = _derive_activation(
            release=release,
            evidence_root=evidence_root,
            now_unix=now_unix,
            enforce_fresh=True,
        )
        for snapshot in derived.snapshots:
            snapshot.require_unchanged()
        unsigned = {
            "schema": ACTIVATION_EVIDENCE_VALIDATION_SCHEMA,
            "release_revision": release.name,
            "evidence_file_sha256": dict(
                derived.seal["evidence_file_sha256"]
            ),
            "prospective_activation_seal_sha256": derived.seal[
                "seal_sha256"
            ],
            "freshness_enforced": True,
            "fresh_through_unix": derived.fresh_through_unix,
            "activation_seal_published": False,
            "cloud_mutation_performed": False,
        }
        result = {
            **unsigned,
            "validation_sha256": protocol.sha256_json(unsigned),
        }
        for snapshot in derived.snapshots:
            snapshot.require_unchanged()
        return result
    except OwnerGateActivationSealError:
        raise
    except Exception:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_internal_failure"
        ) from None


def _require_exact_file(
    path: Path,
    *,
    raw: bytes,
    uid: int,
    gid: int,
    mode: int,
    allow_link_count: frozenset[int] = frozenset({1}),
) -> os.stat_result:
    descriptor: int | None = None
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        data = bytearray()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise OwnerGateActivationSealError(
                    "owner_gate_activation_publication_drift"
                )
            data.extend(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or resolved != path
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_uid != uid
            or opened.st_gid != gid
            or stat.S_IMODE(opened.st_mode) != mode
            or opened.st_nlink not in allow_link_count
            or bytes(data) != raw
            or (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_nlink,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_nlink,
            )
        ):
            raise OwnerGateActivationSealError(
                "owner_gate_activation_publication_drift"
            )
        return opened
    except (OSError, RuntimeError):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_publication_drift"
        ) from None
    finally:
        if descriptor is not None:
            _close_descriptor(
                descriptor,
                code="owner_gate_activation_publication_drift",
            )


def _write_scratch(
    path: Path,
    *,
    raw: bytes,
    uid: int,
    gid: int,
    mode: int,
) -> None:
    descriptor: int | None = None
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
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, mode)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    except FileExistsError:
        _require_exact_file(
            path,
            raw=raw,
            uid=uid,
            gid=gid,
            mode=mode,
            allow_link_count=frozenset({1, 2}),
        )
        return
    except OSError:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_scratch_write_failed"
        ) from None
    finally:
        if descriptor is not None:
            _close_descriptor(
                descriptor,
                code="owner_gate_activation_scratch_write_failed",
            )
    _fsync_directory(path.parent)


def _publish_no_replace(
    path: Path,
    *,
    raw: bytes,
    uid: int,
    gid: int,
    mode: int,
) -> bool:
    scratch = path.with_name(f".{path.name}.staged")
    final_exists = os.path.lexists(path)
    scratch_exists = os.path.lexists(scratch)
    if final_exists:
        final_state = _require_exact_file(
            path,
            raw=raw,
            uid=uid,
            gid=gid,
            mode=mode,
            allow_link_count=frozenset({1, 2}),
        )
        if scratch_exists:
            scratch_state = _require_exact_file(
                scratch,
                raw=raw,
                uid=uid,
                gid=gid,
                mode=mode,
                allow_link_count=frozenset({2}),
            )
            if (
                final_state.st_dev,
                final_state.st_ino,
            ) != (
                scratch_state.st_dev,
                scratch_state.st_ino,
            ):
                raise OwnerGateActivationSealError(
                    "owner_gate_activation_publication_drift"
                )
            try:
                os.unlink(scratch)
            except OSError:
                raise OwnerGateActivationSealError(
                    "owner_gate_activation_publication_failed"
                ) from None
            _fsync_directory(path.parent)
        _require_exact_file(
            path,
            raw=raw,
            uid=uid,
            gid=gid,
            mode=mode,
        )
        return False

    _write_scratch(
        scratch,
        raw=raw,
        uid=uid,
        gid=gid,
        mode=mode,
    )
    try:
        os.link(scratch, path, follow_symlinks=False)
    except FileExistsError:
        return _publish_no_replace(
            path,
            raw=raw,
            uid=uid,
            gid=gid,
            mode=mode,
        )
    except OSError:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_publication_failed"
        ) from None
    _fsync_directory(path.parent)
    _require_exact_file(
        path,
        raw=raw,
        uid=uid,
        gid=gid,
        mode=mode,
        allow_link_count=frozenset({2}),
    )
    try:
        os.unlink(scratch)
    except OSError:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_publication_failed"
        ) from None
    _fsync_directory(path.parent)
    _require_exact_file(
        path,
        raw=raw,
        uid=uid,
        gid=gid,
        mode=mode,
    )
    return True


def _preflight_no_replace(
    path: Path,
    *,
    raw: bytes,
    uid: int,
    gid: int,
    mode: int,
) -> None:
    """Reject namespace drift without publishing or cleaning recovery state."""

    scratch = path.with_name(f".{path.name}.staged")
    final_exists = os.path.lexists(path)
    scratch_exists = os.path.lexists(scratch)
    if final_exists:
        final_state = _require_exact_file(
            path,
            raw=raw,
            uid=uid,
            gid=gid,
            mode=mode,
            allow_link_count=frozenset({1, 2}),
        )
        if scratch_exists:
            scratch_state = _require_exact_file(
                scratch,
                raw=raw,
                uid=uid,
                gid=gid,
                mode=mode,
                allow_link_count=frozenset({2}),
            )
            if (final_state.st_dev, final_state.st_ino) != (
                scratch_state.st_dev,
                scratch_state.st_ino,
            ):
                raise OwnerGateActivationSealError(
                    "owner_gate_activation_publication_drift"
                )
        return
    if scratch_exists:
        _require_exact_file(
            scratch,
            raw=raw,
            uid=uid,
            gid=gid,
            mode=mode,
        )


def _open_lock(path: Path) -> int:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            LOCK_FILE_MODE,
        )
        os.fchown(descriptor, ROOT_UID, ROOT_GID)
        os.fchmod(descriptor, LOCK_FILE_MODE)
        state = os.fstat(descriptor)
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or (before.st_dev, before.st_ino) != (state.st_dev, state.st_ino)
            or state.st_uid != ROOT_UID
            or state.st_gid != ROOT_GID
            or stat.S_IMODE(state.st_mode) != LOCK_FILE_MODE
            or state.st_nlink != 1
        ):
            raise OwnerGateActivationSealError(
                "owner_gate_activation_lock_invalid"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor
    except OwnerGateActivationSealError:
        if descriptor is not None:
            _close_descriptor(
                descriptor,
                code="owner_gate_activation_lock_invalid",
            )
        raise
    except OSError:
        if descriptor is not None:
            _close_descriptor(
                descriptor,
                code="owner_gate_activation_lock_invalid",
            )
        raise OwnerGateActivationSealError(
            "owner_gate_activation_lock_invalid"
        ) from None


def _receipt(
    derived: _DerivedActivation,
    *,
    release_revision: str,
) -> Mapping[str, Any]:
    seal = derived.seal
    unsigned = {
        "schema": ACTIVATION_RECEIPT_SCHEMA,
        "release_revision": release_revision,
        "activation_seal_path": str(ACTIVATION_SEAL_PATH),
        "activation_seal_sha256": seal["seal_sha256"],
        "foundation_plan_sha256": seal["foundation_plan_sha256"],
        "package_sha256": seal["package_sha256"],
        "cloud_topology_receipt_sha256": seal[
            "cloud_topology_receipt_sha256"
        ],
        "host_security_smoke_receipt_sha256": seal[
            "host_security_smoke_receipt_sha256"
        ],
        "iam_repreflight_receipt_sha256": seal[
            "iam_repreflight_receipt_sha256"
        ],
        "owner_gate_vm_numeric_id": seal["owner_gate_vm_numeric_id"],
        "target_instance_numeric_id": seal["target_instance_numeric_id"],
        "target_disk_numeric_id": seal["target_disk_numeric_id"],
        "verified_release_lineage": dict(seal["verified_release_lineage"]),
        "evidence_file_sha256": dict(seal["evidence_file_sha256"]),
        "activation_record_is_canonical_authorization": True,
        "seal_uid": ROOT_UID,
        "seal_gid": EXECUTOR_GID,
        "seal_mode": "0440",
        "activation_installed": True,
        "cloud_mutation_performed": False,
        "activated_at_unix": seal["created_at_unix"],
    }
    return {
        **unsigned,
        "receipt_sha256": protocol.sha256_json(unsigned),
    }


def _install_activation_seal(
    *,
    release: Path,
    evidence_root: Path,
    receipt_path: Path,
    seal_path: Path,
    lock_path: Path,
    now_unix: int,
) -> Mapping[str, Any]:
    """Install or replay one exact seal using fixed-path caller wrappers."""

    if os.geteuid() != ROOT_UID:  # windows-footgun: ok — Debian root boundary
        raise OwnerGateActivationSealError(
            "owner_gate_activation_root_required"
        )
    revision = _validate_release_root(release)
    expected_evidence = ACTIVATION_EVIDENCE_BASE / revision
    expected_receipt = ACTIVATION_RECEIPT_BASE / f"{revision}.json"
    if (
        evidence_root != expected_evidence
        or receipt_path != expected_receipt
        or seal_path != ACTIVATION_SEAL_PATH
        or lock_path != ACTIVATION_LOCK_PATH
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_fixed_path_required"
        )
    _require_directory(
        ACTIVATION_EVIDENCE_BASE,
        parent=ACTIVATION_EVIDENCE_BASE.parent,
        uid=ROOT_UID,
        gid=ROOT_GID,
        mode=STATE_DIRECTORY_MODE,
    )
    _require_directory(
        evidence_root,
        parent=ACTIVATION_EVIDENCE_BASE,
        uid=ROOT_UID,
        gid=ROOT_GID,
        mode=EVIDENCE_DIRECTORY_MODE,
    )
    _require_directory(
        ACTIVATION_RECEIPT_BASE,
        parent=ACTIVATION_RECEIPT_BASE.parent,
        uid=ROOT_UID,
        gid=ROOT_GID,
        mode=STATE_DIRECTORY_MODE,
    )
    _require_directory(
        seal_path.parent,
        parent=seal_path.parent.parent,
        uid=ROOT_UID,
        gid=ROOT_GID,
        mode=CONFIG_DIRECTORY_MODE,
    )
    _require_directory(
        lock_path.parent,
        parent=lock_path.parent.parent,
        uid=ROOT_UID,
        gid=ROOT_GID,
        mode=CONFIG_DIRECTORY_MODE,
    )

    with _PROCESS_LOCK:
        lock_descriptor = _open_lock(lock_path)
        try:
            seal_scratch = seal_path.with_name(f".{seal_path.name}.staged")
            receipt_scratch = receipt_path.with_name(
                f".{receipt_path.name}.staged"
            )
            if (
                os.path.lexists(receipt_path)
                or os.path.lexists(receipt_scratch)
            ) and not (
                os.path.lexists(seal_path)
                or os.path.lexists(seal_scratch)
            ):
                raise OwnerGateActivationSealError(
                    "owner_gate_activation_publication_drift"
                )
            # Freshness may be waived only for a fully-published exact replay.
            # A seal-only recovery is already safe because the seal is the
            # complete canonical record, but it must not turn stale evidence
            # into a fresh audit receipt after an interrupted first install.
            complete_replay = os.path.lexists(
                seal_path
            ) and os.path.lexists(receipt_path)
            derived = _derive_activation(
                release=release,
                evidence_root=evidence_root,
                now_unix=now_unix,
                enforce_fresh=not complete_replay,
            )
            for snapshot in derived.snapshots:
                snapshot.require_unchanged()
            seal_raw = _canonical(derived.seal)
            stored_receipt = _receipt(
                derived,
                release_revision=revision,
            )
            receipt_raw = _canonical(stored_receipt)
            package_raw, _package_snapshot = _read_regular(
                release / "package-manifest.json",
                maximum=MAX_JSON_BYTES,
                uid=ROOT_UID,
                gid=ROOT_GID,
                modes=frozenset({EVIDENCE_FILE_MODE}),
                code="owner_gate_activation_package_invalid",
            )
            packaged = _decode_canonical(
                package_raw, code="owner_gate_activation_package_invalid"
            )
            cutover_capable = any(
                isinstance(item, Mapping)
                and item.get("release_relative")
                == "scripts/canary/production_cutover_passkey.py"
                for item in packaged.get("payloads", [])
            )
            cutover_trust_raw: bytes | None = None
            cutover_trust_snapshots: tuple[_FileSnapshot, ...] = ()
            if cutover_capable:
                trust_manifest_raw, trust_manifest_snapshot = _read_regular(
                    release / "trust/release-trust.json",
                    maximum=MAX_JSON_BYTES,
                    uid=ROOT_UID,
                    gid=ROOT_GID,
                    modes=frozenset({EVIDENCE_FILE_MODE}),
                    code="owner_gate_activation_cutover_trust_invalid",
                )
                trust_public_raw, trust_public_snapshot = _read_regular(
                    release / "trust/release-trust-signing.pub",
                    maximum=32,
                    uid=ROOT_UID,
                    gid=ROOT_GID,
                    modes=frozenset({EVIDENCE_FILE_MODE}),
                    code="owner_gate_activation_cutover_trust_invalid",
                )
                host_public_raw, host_public_snapshot = _read_regular(
                    release / "trust/host-observation-attestation.pub",
                    maximum=32,
                    uid=ROOT_UID,
                    gid=ROOT_GID,
                    modes=frozenset({EVIDENCE_FILE_MODE}),
                    code="owner_gate_activation_cutover_trust_invalid",
                )
                receipt_public_raw, receipt_public_snapshot = _read_regular(
                    AUTHORITY_RECEIPT_PUBLIC_KEY_PATH,
                    maximum=16 * 1024,
                    uid=ROOT_UID,
                    gid=ROOT_GID,
                    modes=frozenset({EVIDENCE_FILE_MODE}),
                    code="owner_gate_activation_cutover_trust_invalid",
                )
                post_host_raw, post_host_snapshot = _read_regular(
                    evidence_root / POST_IAM_HOST_OBSERVATION_NAME,
                    maximum=MAX_JSON_BYTES,
                    uid=ROOT_UID,
                    gid=ROOT_GID,
                    modes=frozenset({EVIDENCE_FILE_MODE}),
                    code="owner_gate_activation_cutover_trust_invalid",
                )
                try:
                    post_host = _decode_canonical(
                        post_host_raw,
                        code="owner_gate_activation_cutover_trust_invalid",
                    )
                    portable_cutover_trust = (
                        production_cutover.build_trust_bundle(
                            authority_release_sha=revision,
                            release_trust_manifest_raw=trust_manifest_raw,
                            release_trust_public_key_raw=trust_public_raw,
                            host_observation_public_key_raw=host_public_raw,
                            post_iam_host_observation=post_host,
                            authority_receipt_public_key_pem=(
                                receipt_public_raw
                            ),
                        )
                    )
                except production_cutover.ProductionCutoverPasskeyError:
                    raise OwnerGateActivationSealError(
                        "owner_gate_activation_cutover_trust_invalid"
                    ) from None
                cutover_trust_raw = _canonical(portable_cutover_trust)
                cutover_trust_snapshots = (
                    trust_manifest_snapshot,
                    trust_public_snapshot,
                    host_public_snapshot,
                    receipt_public_snapshot,
                    post_host_snapshot,
                )
            try:
                accepted = service.validate_activation_seal(
                    protocol.decode_canonical_json(seal_raw),
                    expected_release_revision=revision,
                    now_unix=now_unix,
                )
            except (
                protocol.PasskeyV2ProtocolError,
                service.PasskeyV2ServiceError,
            ):
                raise OwnerGateActivationSealError(
                    "owner_gate_activation_service_contract_invalid"
                ) from None
            if accepted["seal_sha256"] != derived.seal["seal_sha256"]:
                raise OwnerGateActivationSealError(
                    "owner_gate_activation_service_contract_invalid"
                )
            _preflight_no_replace(
                seal_path,
                raw=seal_raw,
                uid=ROOT_UID,
                gid=EXECUTOR_GID,
                mode=SEAL_FILE_MODE,
            )
            _preflight_no_replace(
                receipt_path,
                raw=receipt_raw,
                uid=ROOT_UID,
                gid=ROOT_GID,
                mode=RECEIPT_FILE_MODE,
            )
            if cutover_trust_raw is not None:
                _preflight_no_replace(
                    production_cutover.CUTOVER_TRUST_BUNDLE_PATH,
                    raw=cutover_trust_raw,
                    uid=ROOT_UID,
                    gid=ROOT_GID,
                    mode=EVIDENCE_FILE_MODE,
                )
            for snapshot in derived.snapshots:
                snapshot.require_unchanged()
            for snapshot in cutover_trust_snapshots:
                snapshot.require_unchanged()
            if cutover_trust_raw is not None:
                _publish_no_replace(
                    production_cutover.CUTOVER_TRUST_BUNDLE_PATH,
                    raw=cutover_trust_raw,
                    uid=ROOT_UID,
                    gid=ROOT_GID,
                    mode=EVIDENCE_FILE_MODE,
                )
            installed = _publish_no_replace(
                seal_path,
                raw=seal_raw,
                uid=ROOT_UID,
                gid=EXECUTOR_GID,
                mode=SEAL_FILE_MODE,
            )
            _publish_no_replace(
                receipt_path,
                raw=receipt_raw,
                uid=ROOT_UID,
                gid=ROOT_GID,
                mode=RECEIPT_FILE_MODE,
            )
            for snapshot in cutover_trust_snapshots:
                snapshot.require_unchanged()
            response_unsigned = {
                "schema": ACTIVATION_RESPONSE_SCHEMA,
                "release_revision": revision,
                "disposition": "installed" if installed else "exact_replay",
                "activation_seal_path": str(seal_path),
                "activation_seal_sha256": derived.seal["seal_sha256"],
                "activation_receipt_path": str(receipt_path),
                "activation_receipt_sha256": stored_receipt[
                    "receipt_sha256"
                ],
                "service_contract_accepted": True,
                "cloud_mutation_performed": False,
            }
            return {
                **response_unsigned,
                "response_sha256": protocol.sha256_json(response_unsigned),
            }
        finally:
            _release_lock(lock_descriptor)


def install_activation_seal(
    *,
    release: Path,
    evidence_root: Path,
    receipt_path: Path,
    seal_path: Path,
    lock_path: Path,
    now_unix: int,
) -> Mapping[str, Any]:
    """Install/replay with one stable, secret-free public error boundary."""

    try:
        return _install_activation_seal(
            release=release,
            evidence_root=evidence_root,
            receipt_path=receipt_path,
            seal_path=seal_path,
            lock_path=lock_path,
            now_unix=now_unix,
        )
    except OwnerGateActivationSealError:
        raise
    except Exception:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_internal_failure"
        ) from None


def _installed_release() -> Path:
    try:
        release = Path(__file__).resolve(strict=True).parents[2]
    except (OSError, RuntimeError, IndexError):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_release_invalid"
        ) from None
    _validate_release_root(release)
    return release


def validate_activation_install_request(
    raw: bytes,
    *,
    expected_release_revision: str,
) -> Mapping[str, Any]:
    """Bind the fixed action to the installed release before any mutation."""

    if (
        _REVISION.fullmatch(expected_release_revision or "") is None
        or type(raw) is not bytes
        or not raw
        or len(raw) > MAX_ACTIVATION_REQUEST_BYTES
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_request_invalid"
        )
    value = _decode_canonical(
        raw,
        code="owner_gate_activation_request_invalid",
    )
    if (
        set(value) != {"schema", "release_revision"}
        or value.get("schema") != ACTIVATION_REQUEST_SCHEMA
        or value.get("release_revision") != expected_release_revision
    ):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_request_invalid"
        )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(argv or ())
    if arguments != ("install",):
        raise OwnerGateActivationSealError(
            "owner_gate_activation_command_invalid"
        )
    release = _installed_release()
    revision = release.name
    try:
        request_raw = sys.stdin.buffer.read(MAX_ACTIVATION_REQUEST_BYTES + 1)
    except OSError:
        raise OwnerGateActivationSealError(
            "owner_gate_activation_request_invalid"
        ) from None
    validate_activation_install_request(
        request_raw,
        expected_release_revision=revision,
    )
    response = install_activation_seal(
        release=release,
        evidence_root=ACTIVATION_EVIDENCE_BASE / revision,
        receipt_path=ACTIVATION_RECEIPT_BASE / f"{revision}.json",
        seal_path=ACTIVATION_SEAL_PATH,
        lock_path=ACTIVATION_LOCK_PATH,
        now_unix=int(time.time()),
    )
    print(_canonical(response).decode("ascii"))
    return 0


__all__ = [
    "ACTIVATION_EVIDENCE_BASE",
    "ACTIVATION_EVIDENCE_VALIDATION_SCHEMA",
    "ACTIVATION_LOCK_PATH",
    "ACTIVATION_RECEIPT_BASE",
    "ACTIVATION_RECEIPT_SCHEMA",
    "ACTIVATION_REQUEST_SCHEMA",
    "ACTIVATION_RESPONSE_SCHEMA",
    "ACTIVATION_SEAL_PATH",
    "EVIDENCE_NAMES",
    "OwnerGateActivationSealError",
    "install_activation_seal",
    "main",
    "validate_activation_install_request",
    "validate_activation_evidence_strict",
]


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
