#!/usr/bin/env python3
"""Role-owned evidence producers for the production capability canary.

The capability canary must never turn a root observer into a universal
attestation oracle.  This module provides the deliberately small evidence
boundary used by the live driver:

* each non-owner role has its own AF_UNIX endpoint and Ed25519 key;
* a role producer obtains native evidence through a role-local collector and
  signs only after that collector has verified the evidence it owns;
* the owner grant is an already-staged SSHSIG artifact verified against the
  pinned Emil owner key; and
* a root readiness collector authenticates every endpoint with SO_PEERCRED and
  publishes only public key and service-identity metadata.

No model tool, prompt middleware, task-text classifier, keyword router, or
success synthesizer is implemented here.  GPT/Hermes remains the sole
semantic authority.  Producers enforce only fixed schemas, identities,
cryptographic bindings, no-replace publication, and receipt ownership.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import signal
import socket
import stat
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Protocol, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


PRODUCER_CONFIG_SCHEMA = "muncho-production-capability-producer-config.v2"
PRODUCER_REQUEST_SCHEMA = "muncho-production-capability-producer-request.v1"
PRODUCER_RESPONSE_SCHEMA = "muncho-production-capability-producer-response.v1"
PRODUCER_ENDPOINT_READINESS_SCHEMA = (
    "muncho-production-capability-producer-endpoint-readiness.v1"
)
PRODUCER_ENDPOINT_ACTIVATION_SCHEMA = (
    "muncho-production-capability-producer-endpoint-activation.v1"
)
PRODUCER_FOUNDATION_SCHEMA = (
    "muncho-production-capability-canary-producer-foundation.v1"
)
PRODUCER_ACTIVATION_SCHEMA = (
    "muncho-production-capability-canary-producer-activation.v1"
)
# Compatibility name used by the offline evidence verifier.  A readiness
# artifact is now explicitly the per-run activation, never the foundation.
PRODUCER_FLEET_READINESS_SCHEMA = PRODUCER_ACTIVATION_SCHEMA
NATIVE_EVIDENCE_SCHEMA = "muncho-production-capability-native-evidence.v1"
SIGNED_RECEIPT_SCHEMA = "muncho-production-capability-canary-signed-receipt.v1"
OWNER_SSHSIG_NAMESPACE = "muncho-production-capability-canary-owner-v1"
PRODUCER_FOUNDATION_SSHSIG_NAMESPACE = (
    "muncho-production-capability-canary-producer-foundation-v1"
)

DEFAULT_FOUNDATION_PATH = Path(
    "/etc/muncho/capability-canary/producer-foundation.json"
)
DEFAULT_READINESS_PATH = Path(
    "/run/muncho-capability-canary/producer-activation.json"
)
DEFAULT_RECEIPT_ROOT = Path("/var/lib/muncho-capability-canary-evidence")
DEFAULT_RUNTIME_ROOT = Path("/run/muncho-capability-canary-producers")
DEFAULT_CONFIG_ROOT = Path("/etc/muncho/capability-canary/producers")
DEFAULT_KEY_ROOT = Path("/etc/muncho/capability-canary/producer-keys")
DEFAULT_OWNER_GRANT_PATH = Path(
    "/etc/muncho/capability-canary/owner-grant.sshsig.json"
)
DEFAULT_PROBE_CATALOG_PATH = Path(
    "/etc/muncho/capability-canary/probe-catalog.json"
)
DEFAULT_OWNER_PUBLIC_KEY_HEX_PIN_PATH = Path(
    "/etc/muncho/capability-canary/owner-public-key-ed25519.hex"
)
DEFAULT_OWNER_PUBLIC_KEY_SOURCE_SHA256_PIN_PATH = Path(
    "/etc/muncho/capability-canary/owner-public-key-source.sha256"
)

PRODUCTION_OWNER_ID = "1279454038731264061"
PRODUCTION_BOT_USER_ID = "1501976597455044801"

BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT = (
    "muncho-operational-edge-bitrix.service"
)
BITRIX_OPERATIONAL_EDGE_ASSET_NAMES = (
    "bitrix_skyvision_crm.py",
    "bitrix_voucher_ops.py",
    "muncho_step_up_verify",
    "dangerous_action_guard",
)
BITRIX_OPERATIONAL_EDGE_STAGING_PROTOCOL = (
    "sealed_nonsecret_assets_before_service_activation.v1"
)
BITRIX_OPERATIONAL_EDGE_SERVICE_USER = "muncho-edge-bitrix"
BITRIX_OPERATIONAL_EDGE_SERVICE_GROUP = "muncho-edge-bitrix"
BITRIX_OPERATIONAL_EDGE_SOCKET_GROUP = "muncho-edge-bitrix-c"
BITRIX_OPERATIONAL_EDGE_UNIT_PATH = (
    "/etc/systemd/system/muncho-operational-edge-bitrix.service"
)
BITRIX_OPERATIONAL_EDGE_CONFIG_PATH = (
    "/etc/muncho/operational-edge/bitrix.json"
)
BITRIX_OPERATIONAL_EDGE_TRUST_PATH = (
    "/etc/muncho/operational-edge/trust/bitrix-receipt-public.pem"
)
BITRIX_CANARY_READ_ARGUMENTS = {"entity_id": "STATUS"}
BITRIX_CANARY_MUTATION_ARGUMENTS = {
    "title": "Muncho capability canary dry-run",
    "requester": "capability-canary",
    "reason": "Verify pre-dispatch denial without mutation",
    "execute": False,
}

AUTHORITY_ROLES = (
    "business_edge",
    "canonical_writer",
    "discord_edge",
    "gateway_observer",
    "owner",
)
ENDPOINT_ROLES = tuple(role for role in AUTHORITY_ROLES if role != "owner")
PRODUCER_SERVICE_UNITS = {
    role: f"muncho-capability-producer-{role.replace('_', '-')}.service"
    for role in ENDPOINT_ROLES
}
AUTHORITY_ALGORITHMS = {
    "business_edge": "ed25519",
    "canonical_writer": "ed25519",
    "discord_edge": "ed25519",
    "gateway_observer": "ed25519",
    "owner": "sshsig-ed25519-sha512",
}

SLOT_ROLE = {
    "runtime": "gateway_observer",
    "workspace_gateway": "gateway_observer",
    "workspace_writer": "canonical_writer",
    "workspace_owner": "owner",
    "worker_restart_checkpoint": "canonical_writer",
    "capability_denials": "canonical_writer",
    "database_reconciliation": "canonical_writer",
    "bitrix_edge": "business_edge",
    "bitrix_writer": "canonical_writer",
    "discord_edge": "discord_edge",
    "discord_writer": "canonical_writer",
    "failure_gateway": "gateway_observer",
    "failure_writer": "canonical_writer",
    "cleanup": "gateway_observer",
}
RECEIPT_SLOTS = tuple(SLOT_ROLE)
SLOT_FILENAME = {
    "runtime": "runtime.json",
    "workspace_gateway": "workspace-gateway.json",
    "workspace_writer": "workspace-writer.json",
    "workspace_owner": "workspace-owner.json",
    "worker_restart_checkpoint": "worker-restart-checkpoint.json",
    "capability_denials": "capability-denials.json",
    "database_reconciliation": "database-reconciliation.json",
    "bitrix_edge": "bitrix-edge.json",
    "bitrix_writer": "bitrix-writer.json",
    "discord_edge": "discord-edge.json",
    "discord_writer": "discord-writer.json",
    "failure_gateway": "failure-gateway.json",
    "failure_writer": "failure-writer.json",
    "cleanup": "cleanup.json",
}
DENIAL_KINDS = (
    "unapproved_command",
    "expired_capability",
    "changed_command_bytes",
    "wrong_owner",
    "wrong_session_epoch",
    "stale_plan_revision",
)
FAILURE_COMPONENTS = ("tool", "browser", "database", "writer", "egress")

# These are native fact classes, not semantic routes.  A producer selects the
# row by the exact slot named in its sealed config; it never inspects task text.
SLOT_NATIVE_BINDING_KINDS = {
    "runtime": (
        "gateway_runtime_readiness",
        "discord_connector_readiness",
        "routeback_bot_identity",
    ),
    "workspace_gateway": (
        "gateway_observer_frame_chain",
        "authenticated_api_terminal_event",
        "isolated_worker_restart_receipt",
    ),
    "workspace_writer": (
        "canonical_writer_resume_bundle",
        "canonical_writer_projection_events",
    ),
    "workspace_owner": (
        "owner_staged_sshsig_grant",
        "owner_public_key_source",
    ),
    "worker_restart_checkpoint": ("canonical_writer_checkpoint_event",),
    "capability_denials": ("canonical_writer_capability_events",),
    "database_reconciliation": (
        "canonical_writer_database_events",
        "database_live_readback",
    ),
    "bitrix_edge": (
        "operational_edge_bitrix_signed_receipt",
        "operational_edge_bitrix_authenticated_live_readback",
    ),
    "bitrix_writer": (
        "canonical_writer_handoff_events",
        "operational_edge_bitrix_mutation_predispatch_denial",
    ),
    "discord_edge": (
        "discord_edge_signed_receipt",
        "discord_edge_journal_readback",
        "discord_public_readback",
        "discord_private_predispatch_denial",
        "routeback_bot_identity",
    ),
    "discord_writer": ("canonical_writer_routeback_events",),
    "failure_gateway": (
        "gateway_observer_frame_chain",
        "authenticated_api_terminal_event",
        "failure_probe_receipts",
    ),
    "failure_writer": ("canonical_writer_failure_events",),
    "cleanup": (
        "systemd_non_observer_services_stopped_state",
        "gateway_observer_cleanup_signer_live_identity",
        "api_control_credential_retirement_journal",
        "routeback_credential_retirement_journal",
        "connector_credential_retirement_journal",
        "codex_credential_retirement_journal",
        "mac_ops_credential_retirement_journal",
        "bitrix_operational_edge_credential_retirement_journal",
        "all_six_credentials_absent_readback",
        "bitrix_receipt_key_pair_retirement_journal",
        "bitrix_receipt_key_pair_absence_readback",
        "browser_session_retirement",
        "isolated_worker_lease_cleanup",
        "production_diff_observation",
    ),
}

MAX_CONFIG_BYTES = 256 * 1024
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_READINESS_BYTES = 512 * 1024
MAX_SSHSIG_BYTES = 4096
MAX_BINDINGS = 16
MAX_CLOCK_SKEW_MS = 30_000
_HEADER = struct.Struct("!I")
_PEER = struct.Struct("3i")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
_SNOWFLAKE_RE = re.compile(r"^[1-9][0-9]{5,24}$")
_SSHSIG_BEGIN = "-----BEGIN SSH SIGNATURE-----"
_SSHSIG_END = "-----END SSH SIGNATURE-----"


class CapabilityProducerError(RuntimeError):
    """Stable, non-secret producer failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise CapabilityProducerError(code)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CapabilityProducerError("non_canonical_json") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(code)
    return value


def _safe_id(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        _fail(code)
    return value


def _strict(value: Any, fields: Sequence[str], code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(code)
    result = dict(value)
    if set(result) != set(fields):
        _fail(code)
    return result


def _strict_json(raw: bytes, code: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _token: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CapabilityProducerError(code) from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        _fail(code)
    return value


def _ssh_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def _read_ssh_string(
    value: bytes,
    offset: int,
    *,
    code: str,
) -> tuple[bytes, int]:
    if offset + 4 > len(value):
        _fail(code)
    size = struct.unpack(">I", value[offset : offset + 4])[0]
    start = offset + 4
    end = start + size
    if size > MAX_SSHSIG_BYTES or end > len(value):
        _fail(code)
    return value[start:end], end


def _verify_sshsig(
    signature: Any,
    *,
    message: bytes,
    public_key_hex: str,
    namespace: str,
    code: str,
) -> None:
    """Verify one exact OpenSSH Ed25519 SSHSIG envelope."""

    if (
        not isinstance(signature, str)
        or len(signature.encode("ascii", errors="ignore")) > MAX_SSHSIG_BYTES
        or not signature.startswith(_SSHSIG_BEGIN + "\n")
        or not signature.endswith("\n" + _SSHSIG_END + "\n")
    ):
        _fail(code)
    lines = signature.splitlines()
    if (
        len(lines) < 3
        or lines[0] != _SSHSIG_BEGIN
        or lines[-1] != _SSHSIG_END
        or any(
            re.fullmatch(r"[A-Za-z0-9+/=]{1,70}", line) is None
            for line in lines[1:-1]
        )
        or any(len(line) != 70 for line in lines[1:-2])
    ):
        _fail(code)
    try:
        envelope = base64.b64decode("".join(lines[1:-1]), validate=True)
    except (ValueError, UnicodeError):
        _fail(code)
    if not envelope.startswith(b"SSHSIG"):
        _fail(code)
    offset = 6
    if (
        offset + 4 > len(envelope)
        or struct.unpack(">I", envelope[offset : offset + 4])[0] != 1
    ):
        _fail(code)
    offset += 4
    public_blob, offset = _read_ssh_string(envelope, offset, code=code)
    observed_namespace, offset = _read_ssh_string(envelope, offset, code=code)
    reserved, offset = _read_ssh_string(envelope, offset, code=code)
    hash_algorithm, offset = _read_ssh_string(envelope, offset, code=code)
    signature_blob, offset = _read_ssh_string(envelope, offset, code=code)
    if offset != len(envelope):
        _fail(code)
    key_type, key_offset = _read_ssh_string(public_blob, 0, code=code)
    public_key, key_offset = _read_ssh_string(public_blob, key_offset, code=code)
    signature_type, signature_offset = _read_ssh_string(
        signature_blob, 0, code=code
    )
    raw_signature, signature_offset = _read_ssh_string(
        signature_blob, signature_offset, code=code
    )
    try:
        expected_public = bytes.fromhex(public_key_hex)
    except (TypeError, ValueError):
        _fail(code)
    namespace_bytes = namespace.encode("ascii")
    if (
        key_offset != len(public_blob)
        or signature_offset != len(signature_blob)
        or key_type != b"ssh-ed25519"
        or signature_type != b"ssh-ed25519"
        or public_key != expected_public
        or len(public_key) != 32
        or len(raw_signature) != 64
        or observed_namespace != namespace_bytes
        or reserved != b""
        or hash_algorithm != b"sha512"
    ):
        _fail(code)
    signed = (
        b"SSHSIG"
        + _ssh_string(namespace_bytes)
        + _ssh_string(reserved)
        + _ssh_string(hash_algorithm)
        + _ssh_string(hashlib.sha512(message).digest())
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            raw_signature,
            signed,
        )
    except (InvalidSignature, ValueError):
        _fail(code)


def _identity(item: os.stat_result) -> tuple[int, ...]:
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


def _directory_identity(item: os.stat_result) -> tuple[int, ...]:
    return (
        item.st_dev,
        item.st_ino,
        stat.S_IFMT(item.st_mode),
        stat.S_IMODE(item.st_mode),
        item.st_uid,
        item.st_gid,
    )


def _stable_read(
    path: Path,
    *,
    maximum: int,
    uid: int,
    gid: int,
    mode: int,
) -> tuple[bytes, os.stat_result]:
    if not path.is_absolute() or ".." in path.parts:
        _fail("artifact_path_invalid")
    try:
        before = path.lstat()
    except OSError as exc:
        raise CapabilityProducerError("artifact_unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != uid
        or before.st_gid != gid
        or stat.S_IMODE(before.st_mode) != mode
        or not 1 < before.st_size <= maximum
    ):
        _fail("artifact_identity_invalid")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        reachable = path.lstat()
    except OSError as exc:
        raise CapabilityProducerError("artifact_replaced") from exc
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or len(raw) > maximum
        or _identity(before) != _identity(opened)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(reachable)
    ):
        _fail("artifact_replaced")
    return raw, before


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_directory(path: Path, *, uid: int, gid: int, mode: int) -> None:
    try:
        item = path.lstat()
    except OSError as exc:
        raise CapabilityProducerError("directory_unavailable") from exc
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not stat.S_ISDIR(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != uid
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) != mode
    ):
        _fail("directory_identity_invalid")


def _read_expected_publication(
    path: Path,
    payload: bytes,
    *,
    uid: int,
    gid: int,
    mode: int,
) -> bool:
    """Return true only for an existing byte-identical immutable artifact."""

    if not os.path.lexists(path):
        return False
    observed: bytes | None = None
    for attempt in range(500):
        # The final name is created as a hard link to a fully-written private
        # temporary inode.  A concurrent reader may therefore see nlink=2
        # until the winning publisher removes that private name.  Inspect the
        # exact final inode before the strict read so that this one mechanical
        # transition cannot be misclassified as an unsafe artifact identity.
        # Every other type/owner/group/mode/size/link-count drift remains an
        # immediate hard failure.
        try:
            candidate = path.lstat()
        except OSError as exc:
            raise CapabilityProducerError("artifact_unavailable") from exc
        if (
            not stat.S_ISREG(candidate.st_mode)
            or stat.S_ISLNK(candidate.st_mode)
            or candidate.st_uid != uid
            or candidate.st_gid != gid
            or stat.S_IMODE(candidate.st_mode) != mode
            or not 1 < candidate.st_size <= max(len(payload), 2)
            or candidate.st_nlink not in {1, 2}
        ):
            _fail("artifact_identity_invalid")
        if candidate.st_nlink == 2:
            if attempt == 499:
                _fail("artifact_identity_invalid")
            time.sleep(0.001)
            continue
        try:
            observed, _item = _stable_read(
                path,
                maximum=max(len(payload), 2),
                uid=uid,
                gid=gid,
                mode=mode,
            )
            break
        except CapabilityProducerError:
            # Once nlink=1 was observed, there is no legitimate publication
            # metadata transition left.  Preserve the exact strict-read error
            # instead of converting replacement or identity drift into an
            # ordinary divergent collision.
            raise
    if observed is None:  # pragma: no cover - loop invariant
        _fail("artifact_identity_invalid")
    if observed != payload:
        _fail("publication_collision_diverged")
    return True


def _publish_no_replace(
    path: Path,
    payload: bytes,
    *,
    uid: int,
    gid: int,
    mode: int = 0o400,
    parent_mode: int = 0o700,
    parent_uid: int | None = None,
    parent_gid: int | None = None,
) -> None:
    if not payload or len(payload) > MAX_RECEIPT_BYTES:
        _fail("receipt_size_invalid")
    directory_uid = uid if parent_uid is None else parent_uid
    directory_gid = gid if parent_gid is None else parent_gid
    _require_directory(
        path.parent,
        uid=directory_uid,
        gid=directory_gid,
        mode=parent_mode,
    )
    if _read_expected_publication(
        path,
        payload,
        uid=uid,
        gid=gid,
        mode=mode,
    ):
        return
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(path.parent, directory_flags)
    directory_before = os.fstat(directory_fd)
    reachable_before = path.parent.lstat()
    if _directory_identity(directory_before) != _directory_identity(reachable_before):
        os.close(directory_fd)
        _fail("directory_replaced")
    temporary_name = f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(temporary_name, flags, mode, dir_fd=directory_fd)
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail("receipt_write_stalled")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            if not _read_expected_publication(
                path,
                payload,
                uid=uid,
                gid=gid,
                mode=mode,
            ):
                _fail("publication_collision_diverged")
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.fsync(directory_fd)
        directory_after = os.fstat(directory_fd)
        try:
            reachable_after = path.parent.lstat()
        finally:
            os.close(directory_fd)
        if (
            _directory_identity(directory_before) != _directory_identity(directory_after)
            or _directory_identity(directory_before)
            != _directory_identity(reachable_after)
        ):
            _fail("directory_replaced")
    observed, _item = _stable_read(
        path,
        maximum=MAX_RECEIPT_BYTES,
        uid=uid,
        gid=gid,
        mode=mode,
    )
    if observed != payload:
        _fail("receipt_readback_invalid")


def _load_private_key(
    path: Path,
    *,
    uid: int,
    gid: int,
) -> tuple[Ed25519PrivateKey, str]:
    raw, _item = _stable_read(path, maximum=16 * 1024, uid=uid, gid=gid, mode=0o400)
    try:
        key = serialization.load_pem_private_key(raw, password=None)
    except (TypeError, ValueError) as exc:
        raise CapabilityProducerError("producer_private_key_invalid") from exc
    if not isinstance(key, Ed25519PrivateKey):
        _fail("producer_private_key_invalid")
    return key, _sha256_bytes(raw)


def _load_public_key(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int = 0o440,
) -> tuple[Ed25519PublicKey, str, str]:
    raw, _item = _stable_read(path, maximum=16 * 1024, uid=uid, gid=gid, mode=mode)
    try:
        key = serialization.load_pem_public_key(raw)
    except (TypeError, ValueError) as exc:
        raise CapabilityProducerError("producer_public_key_invalid") from exc
    if not isinstance(key, Ed25519PublicKey):
        _fail("producer_public_key_invalid")
    public_raw = key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return key, public_raw.hex(), _sha256_bytes(raw)


@dataclass(frozen=True)
class NativeEvidenceBinding:
    kind: str
    source_identity_sha256: str
    artifact_sha256: str
    verification_receipt_sha256: str

    @classmethod
    def from_mapping(cls, value: Any) -> "NativeEvidenceBinding":
        raw = _strict(
            value,
            (
                "kind",
                "source_identity_sha256",
                "artifact_sha256",
                "verification_receipt_sha256",
            ),
            "native_evidence_invalid",
        )
        return cls(
            kind=_safe_id(raw["kind"], "native_evidence_invalid"),
            source_identity_sha256=_digest(
                raw["source_identity_sha256"], "native_evidence_invalid"
            ),
            artifact_sha256=_digest(raw["artifact_sha256"], "native_evidence_invalid"),
            verification_receipt_sha256=_digest(
                raw["verification_receipt_sha256"], "native_evidence_invalid"
            ),
        )

    def to_mapping(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "source_identity_sha256": self.source_identity_sha256,
            "artifact_sha256": self.artifact_sha256,
            "verification_receipt_sha256": self.verification_receipt_sha256,
        }


class NativeEvidenceCollector(Protocol):
    """Role-local collector; callers cannot supply its verification output."""

    def collect(
        self,
        *,
        slot: str,
        payload: Mapping[str, Any],
    ) -> Sequence[NativeEvidenceBinding]: ...


class BitrixOperationalEdgeNativeCollector:
    """Collect real Bitrix facts through the signed operational-edge client."""

    def __init__(
        self,
        client: Any,
        *,
        release_revision: str,
        receipt_key_id: str,
    ) -> None:
        if (
            not isinstance(release_revision, str)
            or _GIT_SHA_RE.fullmatch(release_revision) is None
        ):
            _fail("bitrix_native_collector_config_invalid")
        self.client = client
        self.release_revision = release_revision
        self.receipt_key_id = _digest(
            receipt_key_id,
            "bitrix_native_collector_config_invalid",
        )
        config = getattr(client, "config", None)
        receipt_public_key = getattr(client, "receipt_public_key", None)
        if (
            config is None
            or getattr(config, "domain", None) != "bitrix"
            or getattr(config, "receipt_key_id", None) != self.receipt_key_id
            or getattr(config, "service_unit", None)
            != BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT
            or not isinstance(receipt_public_key, Ed25519PublicKey)
            or _sha256_bytes(
                receipt_public_key.public_bytes(
                    serialization.Encoding.Raw,
                    serialization.PublicFormat.Raw,
                )
            )
            != self.receipt_key_id
        ):
            _fail("bitrix_native_collector_config_invalid")
        self.receipt_public_key = receipt_public_key

    def _evidence(
        self,
        value: Any,
        *,
        operation_id: str,
        code: str,
    ) -> dict[str, Any]:
        raw = _strict(
            value,
            (
                "schema",
                "payload",
                "signed_envelope",
                "signed_envelope_sha256",
                "request_sha256",
                "peer",
            ),
            code,
        )
        payload = raw["payload"]
        peer = raw["peer"]
        try:
            from gateway.operational_edge_protocol import (
                OperationalProtocolError,
                verify_envelope,
            )

            verified_payload = verify_envelope(
                raw["signed_envelope"],
                key_id=self.receipt_key_id,
                public_key=self.receipt_public_key,
                code=code,
            )
        except OperationalProtocolError as exc:
            raise CapabilityProducerError(code) from exc
        if (
            raw["schema"] != "muncho-operational-edge-verified-evidence.v1"
            or not isinstance(payload, Mapping)
            or payload.get("operation_id") != operation_id
            or payload.get("domain") != "bitrix"
            or payload.get("release_revision") is None
            or not isinstance(raw["signed_envelope"], Mapping)
            or not isinstance(verified_payload, Mapping)
            or _canonical_bytes(verified_payload) != _canonical_bytes(payload)
            or _sha256_json(raw["signed_envelope"])
            != _digest(raw["signed_envelope_sha256"], code)
            or payload.get("request_sha256")
            != _digest(raw["request_sha256"], code)
            or not isinstance(peer, Mapping)
            or set(peer) != {"pid", "uid", "gid", "service_unit"}
            or peer.get("service_unit")
            != "muncho-operational-edge-bitrix.service"
            or payload.get("service_pid") != peer.get("pid")
        ):
            _fail(code)
        return raw

    @staticmethod
    def _normalized_readback(value: Any, *, code: str) -> Any:
        def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("duplicate")
                result[key] = item
            return result

        try:
            raw = base64.b64decode(value, validate=True)
            decoded = json.loads(
                raw.decode("utf-8", errors="strict"),
                object_pairs_hook=unique_pairs,
                parse_constant=lambda _token: (_ for _ in ()).throw(
                    ValueError()
                ),
            )
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
            _fail(code)
        if not isinstance(decoded, Mapping):
            _fail(code)

        def normalize(item: Any) -> Any:
            if isinstance(item, Mapping):
                return {
                    key: normalize(child)
                    for key, child in item.items()
                    if key != "generated_at_utc"
                }
            if isinstance(item, list):
                return [normalize(child) for child in item]
            return item

        return normalize(decoded)

    def collect(
        self,
        *,
        slot: str,
        payload: Mapping[str, Any],
    ) -> Sequence[NativeEvidenceBinding]:
        code = "bitrix_native_evidence_invalid"
        if slot != "bitrix_edge":
            _fail(code)
        probe = _strict(
            payload.get("read_probe"),
            (
                "selected_edge_id",
                "read_operation_id",
                "read_arguments",
                "initial_read_probe_id",
                "readback_probe_id",
                "normalized_equality_excluded_fields",
                "stable_normalized_equality",
            ),
            code,
        )
        if (
            probe["selected_edge_id"] != "operational-edge:bitrix"
            or probe["read_operation_id"] != "bitrix.crm.status_list"
            or probe["read_arguments"] != BITRIX_CANARY_READ_ARGUMENTS
            or probe["normalized_equality_excluded_fields"]
            != ["generated_at_utc"]
            or probe["stable_normalized_equality"] is not True
        ):
            _fail(code)
        initial_read_probe_id = _safe_id(
            probe["initial_read_probe_id"], code
        )
        readback_probe_id = _safe_id(probe["readback_probe_id"], code)
        if initial_read_probe_id == readback_probe_id:
            _fail(code)
        read = self._evidence(
            self.client.invoke_verified_evidence(
                "bitrix.crm.status_list",
                dict(BITRIX_CANARY_READ_ARGUMENTS),
                idempotency_key=initial_read_probe_id,
                expected_release_revision=self.release_revision,
            ),
            operation_id="bitrix.crm.status_list",
            code=code,
        )
        readback = self._evidence(
            self.client.invoke_verified_evidence(
                "bitrix.crm.status_list",
                dict(BITRIX_CANARY_READ_ARGUMENTS),
                idempotency_key=readback_probe_id,
                expected_release_revision=self.release_revision,
            ),
            operation_id="bitrix.crm.status_list",
            code=code,
        )
        for value in (read, readback):
            native_payload = value["payload"]
            if (
                native_payload.get("release_revision") != self.release_revision
                or native_payload.get("access") != "read"
                or native_payload.get("outcome") != "succeeded"
                or native_payload.get("readback_verified") is not True
                or native_payload.get("secret_material_recorded") is not False
            ):
                _fail(code)
        normalized_initial = self._normalized_readback(
            read["payload"].get("stdout_b64"), code=code
        )
        normalized_readback = self._normalized_readback(
            readback["payload"].get("stdout_b64"), code=code
        )
        if normalized_initial != normalized_readback:
            _fail(code)
        normalized_sha256 = _sha256_json(normalized_readback)
        source_identity = _sha256_json(
            {
                "service_unit": "muncho-operational-edge-bitrix.service",
                "release_revision": self.release_revision,
                "receipt_key_id": self.receipt_key_id,
                "peer": read["peer"],
            }
        )
        return (
            NativeEvidenceBinding(
                kind="operational_edge_bitrix_signed_receipt",
                source_identity_sha256=source_identity,
                artifact_sha256=read["signed_envelope_sha256"],
                verification_receipt_sha256=_sha256_json(read),
            ),
            NativeEvidenceBinding(
                kind="operational_edge_bitrix_authenticated_live_readback",
                source_identity_sha256=source_identity,
                artifact_sha256=normalized_sha256,
                verification_receipt_sha256=_sha256_json(
                    {
                        "initial_signed_envelope_sha256": read[
                            "signed_envelope_sha256"
                        ],
                        "readback_signed_envelope_sha256": readback[
                            "signed_envelope_sha256"
                        ],
                        "normalized_sha256": normalized_sha256,
                        "excluded_fields": ["generated_at_utc"],
                    }
                ),
            ),
        )


class BitrixWriterNativeCollector(BitrixOperationalEdgeNativeCollector):
    """Add the writer-owned no-capability denial to native writer evidence."""

    def __init__(
        self,
        client: Any,
        *,
        release_revision: str,
        receipt_key_id: str,
        canonical_writer_collector: NativeEvidenceCollector,
    ) -> None:
        super().__init__(
            client,
            release_revision=release_revision,
            receipt_key_id=receipt_key_id,
        )
        self.canonical_writer_collector = canonical_writer_collector

    def collect(
        self,
        *,
        slot: str,
        payload: Mapping[str, Any],
    ) -> Sequence[NativeEvidenceBinding]:
        code = "bitrix_writer_native_evidence_invalid"
        if slot != "bitrix_writer":
            _fail(code)
        handoff = tuple(
            self.canonical_writer_collector.collect(
                slot=slot,
                payload=payload,
            )
        )
        if (
            len(handoff) != 1
            or not isinstance(handoff[0], NativeEvidenceBinding)
            or handoff[0].kind != "canonical_writer_handoff_events"
        ):
            _fail(code)
        probe = _strict(
            payload.get("mutation_probe"),
            (
                "selected_edge_id",
                "mutation_operation_id",
                "mutation_arguments",
                "mutation_probe_id",
            ),
            code,
        )
        if (
            probe["selected_edge_id"] != "operational-edge:bitrix"
            or probe["mutation_operation_id"] != "bitrix.crm.lead_add"
            or probe["mutation_arguments"] != BITRIX_CANARY_MUTATION_ARGUMENTS
        ):
            _fail(code)
        mutation_probe_id = _safe_id(probe["mutation_probe_id"], code)
        denial = self._evidence(
            self.client.invoke_verified_evidence(
                "bitrix.crm.lead_add",
                dict(BITRIX_CANARY_MUTATION_ARGUMENTS),
                idempotency_key=mutation_probe_id,
                expected_release_revision=self.release_revision,
                capability=None,
            ),
            operation_id="bitrix.crm.lead_add",
            code=code,
        )
        denied = denial["payload"]
        if (
            denied.get("release_revision") != self.release_revision
            or denied.get("access") != "mutation"
            or denied.get("outcome") != "blocked"
            or denied.get("blocker_code") != "mutation_capability_required"
            or denied.get("dispatched") is not False
            or denied.get("executable_started") is not False
            or denied.get("mutation_performed") is not False
            or denied.get("readback_verified") is not False
            or denied.get("secret_material_recorded") is not False
        ):
            _fail(code)
        source_identity = _sha256_json(
            {
                "service_unit": BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT,
                "release_revision": self.release_revision,
                "receipt_key_id": self.receipt_key_id,
                "peer": denial["peer"],
            }
        )
        return (
            handoff[0],
            NativeEvidenceBinding(
                kind="operational_edge_bitrix_mutation_predispatch_denial",
                source_identity_sha256=source_identity,
                artifact_sha256=denial["signed_envelope_sha256"],
                verification_receipt_sha256=_sha256_json(denial),
            ),
        )


def project_pinned_owner_public_key_source(
    authority: Mapping[str, Any],
    *,
    expected_comment: str,
) -> Mapping[str, Any]:
    """Project the already-inspected local owner authority into foundation form."""

    code = "producer_owner_public_key_source_invalid"
    raw = _strict(
        authority,
        (
            "public_key_ed25519_hex",
            "key_id",
            "public_key_file_sha256",
            "public_fingerprint",
            "public_key_source",
        ),
        code,
    )
    source = _strict(
        raw["public_key_source"],
        ("path", "file_sha256", "device", "inode", "uid", "gid", "mode", "size"),
        code,
    )
    public_hex = raw["public_key_ed25519_hex"]
    try:
        mode = int(str(source["mode"]), 8)
    except (TypeError, ValueError):
        _fail(code)
    if (
        not isinstance(public_hex, str)
        or re.fullmatch(r"[0-9a-f]{64}", public_hex) is None
        or raw["key_id"] != _sha256_bytes(bytes.fromhex(public_hex))
        or source["file_sha256"] != raw["public_key_file_sha256"]
        or not isinstance(expected_comment, str)
        or not expected_comment
        or not isinstance(raw["public_fingerprint"], str)
        or not raw["public_fingerprint"].startswith("SHA256:")
    ):
        _fail(code)
    projected = {
        "kind": "skyvision_mac_ops_ed25519",
        "path": source["path"],
        "comment": expected_comment,
        "fingerprint": raw["public_fingerprint"],
        "file_sha256": source["file_sha256"],
        "uid": source["uid"],
        "gid": source["gid"],
        "mode": mode,
        "size": source["size"],
    }
    # Device/inode are intentionally excluded: the foundation is installed on
    # Cloud while the pinned source remains the owner's local Mac key file.
    return {
        "public_key_ed25519_hex": public_hex,
        "key_id": raw["key_id"],
        "public_key_source": projected,
        "public_key_source_sha256": _sha256_json(projected),
    }


_PRODUCER_FOUNDATION_UNSIGNED_FIELDS = (
    "schema",
    "release_sha",
    "capability_plan_sha256",
    "full_canary_plan_sha256",
    "owner_id",
    "owner_authority",
    "authority_keys",
    "endpoints",
    "bitrix_operational_edge_contract",
    "discord_edge_evidence_contract",
    "receipt_contract",
    "producer_protocol",
    "root_can_sign_non_observer_roles",
    "token_or_token_digest_recorded",
    "signature_namespace",
    "signature_algorithm",
)


def validate_discord_edge_evidence_contract(value: Any) -> dict[str, Any]:
    """Validate the owner-pinned public Discord evidence trust path."""

    code = "discord_edge_evidence_contract_invalid"
    raw = _strict(
        value,
        (
            "edge_service_unit",
            "edge_socket_path",
            "edge_service_uid",
            "edge_service_gid",
            "receipt_public_key_path",
            "receipt_public_key_id",
            "receipt_public_key_file_sha256",
            "connector_service_unit",
            "connector_socket_path",
            "connector_service_uid",
            "connector_service_gid",
            "public_history_operation",
            "direct_message_allowed",
            "token_or_token_digest_recorded",
        ),
        code,
    )
    if (
        raw["edge_service_unit"] != "muncho-discord-egress.service"
        or raw["edge_socket_path"] != "/run/muncho-discord-egress/edge.sock"
        or raw["receipt_public_key_path"]
        != "/etc/muncho/keys/discord-edge-receipt-public.pem"
        or raw["connector_service_unit"]
        != "muncho-discord-connector.service"
        or raw["connector_socket_path"]
        != "/run/muncho-discord-connector/connector.sock"
        or raw["public_history_operation"] != "public.history.fetch"
        or raw["direct_message_allowed"] is not False
        or raw["token_or_token_digest_recorded"] is not False
        or any(
            type(raw[name]) is not int or raw[name] <= 0
            for name in (
                "edge_service_uid",
                "edge_service_gid",
                "connector_service_uid",
                "connector_service_gid",
            )
        )
    ):
        _fail(code)
    _digest(raw["receipt_public_key_id"], code)
    _digest(raw["receipt_public_key_file_sha256"], code)
    return raw


def validate_bitrix_operational_edge_contract(
    value: Any,
    *,
    release_sha: str,
) -> dict[str, Any]:
    """Validate the immutable Bitrix edge prerequisite without credentials.

    This contract is owner-signed in the run-independent producer foundation.
    The reviewed fixture must copy it byte-for-byte.  It proves which packaged
    non-secret helper bytes and host artifacts may be activated; it never
    contains a webhook value or a digest of one.
    """

    code = "bitrix_operational_edge_contract_invalid"
    if not isinstance(release_sha, str) or _GIT_SHA_RE.fullmatch(release_sha) is None:
        _fail(code)
    raw = _strict(
        value,
        (
            "revision",
            "service_unit",
            "service_identity_sha256",
            "asset_manifest_sha256",
            "asset_names",
            "asset_manifest_path",
            "rendered_unit_sha256",
            "rendered_unit_path",
            "rendered_config_sha256",
            "rendered_config_path",
            "rendered_trust_sha256",
            "rendered_trust_path",
            "identity_bootstrap",
            "credential_projection",
            "receipt_key_contract",
            "expected_active_service_state",
            "expected_cleanup_service_state",
            "credential_binding",
            "staging_protocol",
            "secret_material_recorded",
            "secret_digest_recorded",
        ),
        code,
    )
    expected_release = f"/opt/muncho-canary-releases/{release_sha}"
    identity = _strict(
        raw["identity_bootstrap"],
        (
            "service_user",
            "service_group",
            "service_uid",
            "service_gid",
            "socket_client_group",
            "socket_client_gid",
            "receipt_sha256",
        ),
        code,
    )
    active = _strict(
        raw["expected_active_service_state"],
        ("load_state", "active_state", "sub_state", "unit_file_state"),
        code,
    )
    cleanup = _strict(
        raw["expected_cleanup_service_state"],
        ("active_state", "sub_state", "overlay_retired_or_prior_restored"),
        code,
    )
    credential_projection = _strict(
        raw["credential_projection"],
        (
            "name",
            "source_path",
            "projected_path",
            "bind_target_path",
            "source_owner_uid",
            "source_owner_gid",
            "source_mode",
            "service_reads_projection",
            "original_source_inaccessible",
            "value_or_digest_recorded",
        ),
        code,
    )
    receipt_key = _strict(
        raw["receipt_key_contract"],
        (
            "private_credential_name",
            "private_source_path",
            "private_projection_path",
            "private_owner_uid",
            "private_owner_gid",
            "private_mode",
            "public_path",
            "public_key_id",
            "public_trust_sha256",
            "writer_public_key_credential_name",
            "writer_public_key_source_path",
            "writer_public_key_projection_path",
            "key_bootstrap_receipt_sha256",
            "create_only",
            "retire_private_on_stop",
            "retire_public_on_stop",
            "private_content_or_digest_recorded",
        ),
        code,
    )
    credential_source = (
        "/opt/adventico-ai-platform/hermes-home/secrets/"
        "bitrix_skyvision_crm_webhook.url"
    )
    credential_runtime = (
        f"/run/credentials/{BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT}"
    )
    if (
        raw["revision"] != release_sha
        or raw["service_unit"] != BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT
        or raw["asset_names"] != list(BITRIX_OPERATIONAL_EDGE_ASSET_NAMES)
        or raw["asset_manifest_path"]
        != (
            f"{expected_release}/ops/muncho/runtime/operational-assets/"
            "manifest.json"
        )
        or raw["rendered_unit_path"] != BITRIX_OPERATIONAL_EDGE_UNIT_PATH
        or raw["rendered_config_path"] != BITRIX_OPERATIONAL_EDGE_CONFIG_PATH
        or raw["rendered_trust_path"] != BITRIX_OPERATIONAL_EDGE_TRUST_PATH
        or raw["credential_binding"] != "bitrix_operational_edge_webhook"
        or raw["staging_protocol"] != BITRIX_OPERATIONAL_EDGE_STAGING_PROTOCOL
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or identity["service_user"] != BITRIX_OPERATIONAL_EDGE_SERVICE_USER
        or identity["service_group"] != BITRIX_OPERATIONAL_EDGE_SERVICE_GROUP
        or identity["socket_client_group"]
        != BITRIX_OPERATIONAL_EDGE_SOCKET_GROUP
        or any(
            type(identity[field]) is not int or identity[field] <= 0
            for field in ("service_uid", "service_gid", "socket_client_gid")
        )
        or identity["service_gid"] == identity["socket_client_gid"]
        or credential_projection
        != {
            "name": "bitrix-webhook-url",
            "source_path": credential_source,
            "projected_path": f"{credential_runtime}/bitrix-webhook-url",
            "bind_target_path": credential_source,
            "source_owner_uid": 0,
            "source_owner_gid": 0,
            "source_mode": "0400",
            "service_reads_projection": True,
            "original_source_inaccessible": True,
            "value_or_digest_recorded": False,
        }
        or receipt_key["private_credential_name"] != "receipt-private-key"
        or receipt_key["private_source_path"]
        != "/etc/muncho/keys/operational-edge-bitrix-receipt-private.pem"
        or receipt_key["private_projection_path"]
        != f"{credential_runtime}/receipt-private-key"
        or receipt_key["private_owner_uid"] != 0
        or receipt_key["private_owner_gid"] != 0
        or receipt_key["private_mode"] != "0400"
        or receipt_key["public_path"] != BITRIX_OPERATIONAL_EDGE_TRUST_PATH
        or receipt_key["public_trust_sha256"]
        != raw["rendered_trust_sha256"]
        or receipt_key["writer_public_key_credential_name"]
        != "writer-public-key"
        or receipt_key["writer_public_key_source_path"]
        != "/etc/muncho/keys/writer-capability-public.pem"
        or receipt_key["writer_public_key_projection_path"]
        != f"{credential_runtime}/writer-public-key"
        or receipt_key["create_only"] is not True
        or receipt_key["retire_private_on_stop"] is not True
        or receipt_key["retire_public_on_stop"] is not True
        or receipt_key["private_content_or_digest_recorded"] is not False
        or active
        != {
            "load_state": "loaded",
            "active_state": "active",
            "sub_state": "running",
            "unit_file_state": "disabled",
        }
        or cleanup
        != {
            "active_state": "inactive",
            "sub_state": "dead",
            "overlay_retired_or_prior_restored": True,
        }
    ):
        _fail(code)
    for field in (
        "service_identity_sha256",
        "asset_manifest_sha256",
        "rendered_unit_sha256",
        "rendered_config_sha256",
        "rendered_trust_sha256",
    ):
        _digest(raw[field], code)
    _digest(identity["receipt_sha256"], code)
    _digest(receipt_key["public_key_id"], code)
    _digest(receipt_key["public_trust_sha256"], code)
    _digest(receipt_key["key_bootstrap_receipt_sha256"], code)
    return raw


def producer_foundation_signature_payload(value: Mapping[str, Any]) -> bytes:
    """Return the exact bytes passed to the external OpenSSH signer."""

    raw = dict(value)
    if "owner_signature" in raw:
        raw.pop("owner_signature")
    unsigned = _strict(
        raw,
        _PRODUCER_FOUNDATION_UNSIGNED_FIELDS,
        "producer_foundation_invalid",
    )
    return _canonical_bytes(unsigned)


def seal_producer_foundation(
    unsigned_value: Mapping[str, Any],
    *,
    owner_signature: str,
    pinned_owner_public_key_ed25519_hex: str,
    pinned_owner_public_key_source_sha256: str,
) -> Mapping[str, Any]:
    """Attach an external SSHSIG and validate the complete trust root."""

    unsigned = _strict(
        unsigned_value,
        _PRODUCER_FOUNDATION_UNSIGNED_FIELDS,
        "producer_foundation_invalid",
    )
    return validate_producer_foundation(
        {**unsigned, "owner_signature": owner_signature},
        pinned_owner_public_key_ed25519_hex=(
            pinned_owner_public_key_ed25519_hex
        ),
        pinned_owner_public_key_source_sha256=(
            pinned_owner_public_key_source_sha256
        ),
    )


def validate_producer_foundation(
    value: Any,
    *,
    pinned_owner_public_key_ed25519_hex: str,
    pinned_owner_public_key_source_sha256: str,
) -> dict[str, Any]:
    """Validate the run-independent owner-signed producer trust root.

    The owner key is a required external input.  It is deliberately never
    learned from this artifact or from a fixture signed by that artifact.
    """

    code = "producer_foundation_invalid"
    raw = _strict(
        value,
        (*_PRODUCER_FOUNDATION_UNSIGNED_FIELDS, "owner_signature"),
        code,
    )
    public_hex = pinned_owner_public_key_ed25519_hex
    source_sha256 = _digest(pinned_owner_public_key_source_sha256, code)
    if (
        raw["schema"] != PRODUCER_FOUNDATION_SCHEMA
        or not isinstance(raw["release_sha"], str)
        or _GIT_SHA_RE.fullmatch(raw["release_sha"]) is None
        or raw["owner_id"] != PRODUCTION_OWNER_ID
        or not isinstance(public_hex, str)
        or re.fullmatch(r"[0-9a-f]{64}", public_hex) is None
        or raw["producer_protocol"] != "role_local_native_evidence_v1"
        or raw["root_can_sign_non_observer_roles"] is not False
        or raw["token_or_token_digest_recorded"] is not False
        or raw["signature_namespace"] != PRODUCER_FOUNDATION_SSHSIG_NAMESPACE
        or raw["signature_algorithm"] != "sshsig-ed25519-sha512"
    ):
        _fail(code)
    _digest(raw["capability_plan_sha256"], code)
    _digest(raw["full_canary_plan_sha256"], code)
    owner = _strict(
        raw["owner_authority"],
        (
            "owner_id",
            "key_id",
            "algorithm",
            "public_key_ed25519_hex",
            "public_key_source",
            "public_key_source_sha256",
        ),
        code,
    )
    source = _strict(
        owner["public_key_source"],
        (
            "kind",
            "path",
            "comment",
            "fingerprint",
            "file_sha256",
            "uid",
            "gid",
            "mode",
            "size",
        ),
        code,
    )
    if (
        owner["owner_id"] != PRODUCTION_OWNER_ID
        or owner["algorithm"] != "sshsig-ed25519-sha512"
        or owner["public_key_ed25519_hex"] != public_hex
        or owner["key_id"] != _sha256_bytes(bytes.fromhex(public_hex))
        or owner["public_key_source_sha256"] != source_sha256
        or _sha256_json(source) != source_sha256
        or source["kind"] != "skyvision_mac_ops_ed25519"
        or not isinstance(source["path"], str)
        or not Path(source["path"]).is_absolute()
        or ".." in Path(source["path"]).parts
        or not isinstance(source["comment"], str)
        or not source["comment"]
        or not isinstance(source["fingerprint"], str)
        or not source["fingerprint"].startswith("SHA256:")
        or type(source["uid"]) is not int
        or type(source["gid"]) is not int
        or source["uid"] < 0
        or source["gid"] < 0
        or source["mode"] not in {0o400, 0o440, 0o444, 0o600, 0o640, 0o644}
        or type(source["size"]) is not int
        or source["size"] <= 0
    ):
        _fail(code)
    _digest(source["file_sha256"], code)

    authorities = _strict(raw["authority_keys"], AUTHORITY_ROLES, code)
    endpoints = _strict(raw["endpoints"], ENDPOINT_ROLES, code)
    public_keys: list[str] = []
    for role in AUTHORITY_ROLES:
        authority = _strict(
            authorities[role],
            ("key_id", "algorithm", "public_key_ed25519_hex"),
            code,
        )
        role_public = authority["public_key_ed25519_hex"]
        if (
            authority["algorithm"] != AUTHORITY_ALGORITHMS[role]
            or not isinstance(role_public, str)
            or re.fullmatch(r"[0-9a-f]{64}", role_public) is None
            or authority["key_id"]
            != _sha256_bytes(bytes.fromhex(role_public))
        ):
            _fail(code)
        public_keys.append(role_public)
    if len(set(public_keys)) != len(AUTHORITY_ROLES):
        _fail("producer_authority_keys_not_separated")
    if authorities["owner"] != {
        "key_id": owner["key_id"],
        "algorithm": owner["algorithm"],
        "public_key_ed25519_hex": public_hex,
    }:
        _fail(code)

    validate_bitrix_operational_edge_contract(
        raw["bitrix_operational_edge_contract"],
        release_sha=raw["release_sha"],
    )
    validate_discord_edge_evidence_contract(
        raw["discord_edge_evidence_contract"]
    )

    receipt = _strict(
        raw["receipt_contract"],
        (
            "base_root",
            "run_directory_uid",
            "run_directory_gid",
            "run_directory_mode",
            "slot_filenames",
            "slot_roles",
            "slot_native_binding_kinds",
        ),
        code,
    )
    base_root = Path(receipt["base_root"]) if isinstance(receipt["base_root"], str) else Path()
    if (
        not base_root.is_absolute()
        or ".." in base_root.parts
        or type(receipt["run_directory_uid"]) is not int
        or type(receipt["run_directory_gid"]) is not int
        or receipt["run_directory_uid"] < 0
        or receipt["run_directory_gid"] < 0
        or receipt["run_directory_mode"] not in {0o730, 0o770, 0o1730, 0o1770, 0o3730, 0o3770}
        or receipt["slot_filenames"] != SLOT_FILENAME
        or receipt["slot_roles"] != SLOT_ROLE
        or receipt["slot_native_binding_kinds"]
        != {slot: list(SLOT_NATIVE_BINDING_KINDS[slot]) for slot in RECEIPT_SLOTS}
    ):
        _fail(code)

    endpoint_uids: list[int] = []
    endpoint_gids: list[int] = []
    for role in ENDPOINT_ROLES:
        endpoint = _strict(
            endpoints[role],
            (
                "service_unit",
                "service_identity_sha256",
                "uid",
                "gid",
                "socket_path",
                "private_key_path",
                "public_key_path",
                "public_key_file_sha256",
                "allowed_slots",
                "key_id",
                "algorithm",
                "public_key_ed25519_hex",
            ),
            code,
        )
        paths = (
            endpoint["socket_path"],
            endpoint["private_key_path"],
            endpoint["public_key_path"],
        )
        expected_slots = [
            slot for slot in RECEIPT_SLOTS if SLOT_ROLE[slot] == role
        ]
        if (
            not isinstance(endpoint["service_unit"], str)
            or not endpoint["service_unit"].endswith(".service")
            or type(endpoint["uid"]) is not int
            or type(endpoint["gid"]) is not int
            or endpoint["uid"] < 0
            or endpoint["gid"] < 0
            or endpoint["uid"] == 0
            or endpoint["gid"] == 0
            or endpoint["allowed_slots"] != expected_slots
            or any(
                not isinstance(item, str)
                or not Path(item).is_absolute()
                or ".." in Path(item).parts
                for item in paths
            )
            or endpoint["key_id"] != authorities[role]["key_id"]
            or endpoint["algorithm"] != "ed25519"
            or endpoint["public_key_ed25519_hex"]
            != authorities[role]["public_key_ed25519_hex"]
        ):
            _fail(code)
        _digest(endpoint["service_identity_sha256"], code)
        _digest(endpoint["public_key_file_sha256"], code)
        endpoint_uids.append(endpoint["uid"])
        endpoint_gids.append(endpoint["gid"])

    if (
        len(endpoint_uids) != len(set(endpoint_uids))
        or len(endpoint_gids) != len(set(endpoint_gids))
    ):
        _fail("producer_endpoint_identities_not_separated")

    unsigned = {key: item for key, item in raw.items() if key != "owner_signature"}
    _verify_sshsig(
        raw["owner_signature"],
        message=_canonical_bytes(unsigned),
        public_key_hex=public_hex,
        namespace=PRODUCER_FOUNDATION_SSHSIG_NAMESPACE,
        code=code,
    )
    return raw


def producer_foundation_sha256(value: Mapping[str, Any]) -> str:
    """Digest the complete signed foundation, including its owner signature."""

    return _sha256_json(value)


def load_producer_foundation(
    path: Path = DEFAULT_FOUNDATION_PATH,
    *,
    pinned_owner_public_key_ed25519_hex: str,
    pinned_owner_public_key_source_sha256: str,
    uid: int = 0,
    gid: int = 0,
) -> dict[str, Any]:
    raw, _item = _stable_read(
        path,
        maximum=MAX_READINESS_BYTES,
        uid=uid,
        gid=gid,
        mode=0o400,
    )
    return validate_producer_foundation(
        _strict_json(raw, "producer_foundation_invalid"),
        pinned_owner_public_key_ed25519_hex=pinned_owner_public_key_ed25519_hex,
        pinned_owner_public_key_source_sha256=pinned_owner_public_key_source_sha256,
    )


def publish_producer_foundation(
    value: Mapping[str, Any],
    *,
    pinned_owner_public_key_ed25519_hex: str,
    pinned_owner_public_key_source_sha256: str,
    path: Path = DEFAULT_FOUNDATION_PATH,
    uid: int = 0,
    gid: int = 0,
) -> Mapping[str, Any]:
    validated = validate_producer_foundation(
        value,
        pinned_owner_public_key_ed25519_hex=pinned_owner_public_key_ed25519_hex,
        pinned_owner_public_key_source_sha256=pinned_owner_public_key_source_sha256,
    )
    _publish_no_replace(
        path,
        _canonical_bytes(validated),
        uid=uid,
        gid=gid,
        mode=0o400,
    )
    return validated


@dataclass(frozen=True)
class InstalledProducerFoundation:
    """One installed foundation plus its two independent owner-key pins."""

    value: Mapping[str, Any]
    pinned_owner_public_key_ed25519_hex: str
    pinned_owner_public_key_source_sha256: str

    @property
    def sha256(self) -> str:
        return producer_foundation_sha256(self.value)


def _load_root_pin(path: Path, *, pattern: re.Pattern[str], code: str) -> str:
    raw, _item = _stable_read(
        path,
        maximum=4096,
        uid=0,
        gid=0,
        mode=0o400,
    )
    try:
        text = raw.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise CapabilityProducerError(code) from exc
    if not text.endswith("\n") or text.count("\n") != 1:
        _fail(code)
    value = text[:-1]
    if pattern.fullmatch(value) is None:
        _fail(code)
    return value


def load_installed_producer_foundation(
    *,
    foundation_path: Path = DEFAULT_FOUNDATION_PATH,
    owner_public_key_hex_path: Path = DEFAULT_OWNER_PUBLIC_KEY_HEX_PIN_PATH,
    owner_public_key_source_sha256_path: Path = (
        DEFAULT_OWNER_PUBLIC_KEY_SOURCE_SHA256_PIN_PATH
    ),
) -> InstalledProducerFoundation:
    """Load the root-owned trust root without learning either pin from it."""

    public_hex = _load_root_pin(
        owner_public_key_hex_path,
        pattern=re.compile(r"^[0-9a-f]{64}$"),
        code="producer_owner_public_key_pin_invalid",
    )
    source_sha256 = _load_root_pin(
        owner_public_key_source_sha256_path,
        pattern=_SHA256_RE,
        code="producer_owner_public_key_pin_invalid",
    )
    value = load_producer_foundation(
        foundation_path,
        pinned_owner_public_key_ed25519_hex=public_hex,
        pinned_owner_public_key_source_sha256=source_sha256,
        uid=0,
        gid=0,
    )
    return InstalledProducerFoundation(
        value=copy.deepcopy(value),
        pinned_owner_public_key_ed25519_hex=public_hex,
        pinned_owner_public_key_source_sha256=source_sha256,
    )


class ProducerServiceStateReader(Protocol):
    def state(self, unit: str) -> Mapping[str, Any]: ...


class SystemctlProducerServiceStateReader:
    """Read exact service MainPID and kernel process ownership, without shell."""

    def __init__(
        self,
        *,
        systemctl_path: Path = Path("/usr/bin/systemctl"),
        proc_root: Path = Path("/proc"),
        timeout_seconds: float = 3.0,
    ) -> None:
        if (
            not systemctl_path.is_absolute()
            or not proc_root.is_absolute()
            or not 0.1 <= timeout_seconds <= 10.0
        ):
            _fail("producer_service_state_reader_invalid")
        self.systemctl_path = systemctl_path
        self.proc_root = proc_root
        self.timeout_seconds = timeout_seconds

    def _show(self, unit: str) -> dict[str, str]:
        if unit not in PRODUCER_SERVICE_UNITS.values():
            _fail("producer_service_unit_invalid")
        try:
            completed = subprocess.run(
                [
                    str(self.systemctl_path),
                    "show",
                    "--no-pager",
                    "--property=MainPID",
                    "--property=LoadState",
                    "--property=ActiveState",
                    "--property=SubState",
                    "--",
                    unit,
                ],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise CapabilityProducerError(
                "producer_service_state_unavailable"
            ) from exc
        rows: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            name, separator, value = line.partition("=")
            if not separator or name in rows:
                _fail("producer_service_state_invalid")
            rows[name] = value
        if completed.returncode != 0 or set(rows) != {
            "MainPID",
            "LoadState",
            "ActiveState",
            "SubState",
        }:
            _fail("producer_service_state_invalid")
        return rows

    def _process_ids(self, pid: int) -> tuple[int, int]:
        path = self.proc_root / str(pid) / "status"
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CapabilityProducerError(
                "producer_service_process_unavailable"
            ) from exc
        if not raw or len(raw) > 256 * 1024 or b"\x00" in raw:
            _fail("producer_service_process_invalid")
        uid_row = gid_row = None
        for line in raw.splitlines():
            if line.startswith(b"Uid:"):
                uid_row = line.split()
            elif line.startswith(b"Gid:"):
                gid_row = line.split()
        try:
            if uid_row is None or gid_row is None or len(uid_row) != 5 or len(gid_row) != 5:
                raise ValueError
            uids = tuple(int(item) for item in uid_row[1:])
            gids = tuple(int(item) for item in gid_row[1:])
        except (TypeError, ValueError) as exc:
            raise CapabilityProducerError(
                "producer_service_process_invalid"
            ) from exc
        if len(set(uids)) != 1 or len(set(gids)) != 1:
            _fail("producer_service_process_identity_ambiguous")
        return uids[0], gids[0]

    def state(self, unit: str) -> Mapping[str, Any]:
        rows = self._show(unit)
        try:
            pid = int(rows["MainPID"])
        except ValueError as exc:
            raise CapabilityProducerError(
                "producer_service_state_invalid"
            ) from exc
        if pid <= 1:
            _fail("producer_service_not_running")
        uid, gid = self._process_ids(pid)
        return {
            "unit": unit,
            "main_pid": pid,
            "uid": uid,
            "gid": gid,
            "load_state": rows["LoadState"],
            "active_state": rows["ActiveState"],
            "sub_state": rows["SubState"],
        }


def production_endpoint_clients(
    foundation: Mapping[str, Any],
    *,
    state_reader: ProducerServiceStateReader | None = None,
) -> Mapping[str, "ProducerEndpointClient"]:
    """Bind root clients to exact active units, processes, and socket inodes."""

    endpoints = foundation.get("endpoints")
    if not isinstance(endpoints, Mapping) or set(endpoints) != set(ENDPOINT_ROLES):
        _fail("producer_endpoint_foundation_mismatch")
    reader = state_reader or SystemctlProducerServiceStateReader()
    clients: dict[str, ProducerEndpointClient] = {}
    for role in ENDPOINT_ROLES:
        endpoint = endpoints[role]
        if not isinstance(endpoint, Mapping):
            _fail("producer_endpoint_foundation_mismatch")
        unit = endpoint.get("service_unit")
        expected_unit = PRODUCER_SERVICE_UNITS[role]
        state = reader.state(expected_unit)
        if (
            unit != expected_unit
            or state
            != {
                "unit": expected_unit,
                "main_pid": state.get("main_pid"),
                "uid": endpoint.get("uid"),
                "gid": endpoint.get("gid"),
                "load_state": "loaded",
                "active_state": "active",
                "sub_state": "running",
            }
            or type(state.get("main_pid")) is not int
            or state["main_pid"] <= 1
        ):
            _fail("producer_service_identity_invalid")
        path = Path(str(endpoint.get("socket_path") or ""))
        try:
            parent = path.parent.lstat()
            item = path.lstat()
        except OSError as exc:
            raise CapabilityProducerError(
                "producer_endpoint_unavailable"
            ) from exc
        if (
            not path.is_absolute()
            or ".." in path.parts
            or not stat.S_ISDIR(parent.st_mode)
            or stat.S_ISLNK(parent.st_mode)
            or parent.st_uid != endpoint["uid"]
            or parent.st_gid != endpoint["gid"]
            or stat.S_IMODE(parent.st_mode) != 0o700
            or not stat.S_ISSOCK(item.st_mode)
            or item.st_uid != endpoint["uid"]
            or item.st_gid != endpoint["gid"]
            or stat.S_IMODE(item.st_mode) != 0o600
        ):
            _fail("producer_endpoint_identity_invalid")
        clients[role] = ProducerEndpointClient(
            path,
            expected_peer=PeerIdentity(
                state["main_pid"], endpoint["uid"], endpoint["gid"]
            ),
        )
    return clients


@dataclass(frozen=True)
class ProductionFleetActivation:
    readiness: Mapping[str, Any]
    endpoint_activation_receipts: Mapping[str, Mapping[str, Any]]


def activate_production_fleet(
    *,
    plan: Any,
    full_plan: Any,
    installed_foundation: InstalledProducerFoundation,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
    endpoint_clients: Mapping[str, "ProducerEndpointClient"] | None = None,
    state_reader: ProducerServiceStateReader | None = None,
    readiness_path: Path = DEFAULT_READINESS_PATH,
    owner_grant_path: Path = DEFAULT_OWNER_GRANT_PATH,
    probe_catalog_path: Path = DEFAULT_PROBE_CATALOG_PATH,
    probe_catalog_gid: int | None = None,
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    routeback_attester: Callable[[Any, Any], Mapping[str, Any]] | None = None,
    routeback_binding_checker: Callable[
        [Any, Any, Mapping[str, Any]], Mapping[str, Any]
    ]
    | None = None,
) -> ProductionFleetActivation:
    """Build, barrier-activate, and atomically publish one live fleet."""

    if not isinstance(installed_foundation, InstalledProducerFoundation):
        _fail("producer_foundation_invalid")
    foundation = validate_producer_foundation(
        installed_foundation.value,
        pinned_owner_public_key_ed25519_hex=(
            installed_foundation.pinned_owner_public_key_ed25519_hex
        ),
        pinned_owner_public_key_source_sha256=(
            installed_foundation.pinned_owner_public_key_source_sha256
        ),
    )
    owner_grant_raw, _item = _stable_read(
        owner_grant_path,
        maximum=MAX_READINESS_BYTES,
        uid=0,
        gid=0,
        mode=0o400,
    )
    writer_gid = (
        probe_catalog_gid
        if probe_catalog_gid is not None
        else getattr(getattr(full_plan, "identities", None), "writer_gid", None)
    )
    if type(writer_gid) is not int or writer_gid < 0:
        _fail("probe_catalog_identity_invalid")
    catalog_raw, _catalog_item = _stable_read(
        probe_catalog_path,
        maximum=MAX_READINESS_BYTES,
        uid=0,
        gid=writer_gid,
        mode=0o440,
    )
    catalog = validate_probe_catalog(
        _strict_json(catalog_raw, "probe_catalog_invalid")
    )
    pregrant = validate_owner_pregrant(
        _strict_json(owner_grant_raw, "owner_pregrant_invalid"),
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        catalog=catalog,
    )
    clients = (
        production_endpoint_clients(foundation, state_reader=state_reader)
        if endpoint_clients is None
        else endpoint_clients
    )
    activation_now = now_ms()
    readiness = build_fleet_readiness_for_plans(
        plan=plan,
        full_plan=full_plan,
        foundation=foundation,
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        pinned_owner_public_key_ed25519_hex=(
            installed_foundation.pinned_owner_public_key_ed25519_hex
        ),
        pinned_owner_public_key_source_sha256=(
            installed_foundation.pinned_owner_public_key_source_sha256
        ),
        owner_grant_sha256=pregrant["grant_sha256"],
        endpoint_clients=clients,
        now_ms=lambda: activation_now,
        routeback_attester=routeback_attester,
        routeback_binding_checker=routeback_binding_checker,
    )
    activation_receipts = activate_fleet_readiness(
        readiness,
        endpoint_clients=clients,
        now_ms=activation_now,
    )
    publish_fleet_readiness(
        readiness,
        path=readiness_path,
        expected_foundation_sha256=installed_foundation.sha256,
        now_ms=activation_now,
    )
    return ProductionFleetActivation(
        readiness=copy.deepcopy(readiness),
        endpoint_activation_receipts=copy.deepcopy(activation_receipts),
    )


@dataclass(frozen=True)
class ProducerConfig:
    role: str
    foundation_sha256: str
    release_sha: str
    capability_plan_sha256: str
    full_canary_plan_sha256: str
    service_unit: str
    service_identity_sha256: str
    service_uid: int
    service_gid: int
    root_client_uid: int
    socket_path: Path
    receipt_base_root: Path
    receipt_directory_uid: int
    receipt_directory_gid: int
    receipt_directory_mode: int
    private_key_path: Path
    public_key_path: Path
    allowed_slots: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "ProducerConfig":
        raw = _strict(
            value,
            (
                "schema",
                "role",
                "foundation_sha256",
                "release_sha",
                "capability_plan_sha256",
                "full_canary_plan_sha256",
                "service_unit",
                "service_identity_sha256",
                "service_uid",
                "service_gid",
                "root_client_uid",
                "socket_path",
                "receipt_base_root",
                "receipt_directory_uid",
                "receipt_directory_gid",
                "receipt_directory_mode",
                "private_key_path",
                "public_key_path",
                "allowed_slots",
            ),
            "producer_config_invalid",
        )
        role = raw["role"]
        slots = raw["allowed_slots"]
        if (
            raw["schema"] != PRODUCER_CONFIG_SCHEMA
            or role not in ENDPOINT_ROLES
            or not isinstance(raw["release_sha"], str)
            or _GIT_SHA_RE.fullmatch(raw["release_sha"]) is None
            or not isinstance(slots, list)
            or not slots
            or len(slots) != len(set(slots))
            or tuple(slots) != tuple(slot for slot in RECEIPT_SLOTS if SLOT_ROLE[slot] == role)
            or any(type(raw[name]) is not int or raw[name] < 0 for name in (
                "service_uid", "service_gid", "root_client_uid",
                "receipt_directory_uid", "receipt_directory_gid",
            ))
            or raw["service_uid"] == 0
            or raw["service_gid"] == 0
            or raw["root_client_uid"] != 0
            or raw["receipt_directory_mode"]
            not in {0o730, 0o770, 0o1730, 0o1770, 0o3730, 0o3770}
        ):
            _fail("producer_config_invalid")
        service_unit = _safe_id(raw["service_unit"], "producer_config_invalid")
        if not service_unit.endswith(".service"):
            _fail("producer_config_invalid")
        path_values: dict[str, Path] = {}
        for name in (
            "socket_path",
            "receipt_base_root",
            "private_key_path",
            "public_key_path",
        ):
            item = raw[name]
            if not isinstance(item, str):
                _fail("producer_config_invalid")
            path = Path(item)
            if not path.is_absolute() or str(path) != item or ".." in path.parts:
                _fail("producer_config_invalid")
            path_values[name] = path
        return cls(
            role=role,
            foundation_sha256=_digest(
                raw["foundation_sha256"], "producer_config_invalid"
            ),
            release_sha=raw["release_sha"],
            capability_plan_sha256=_digest(raw["capability_plan_sha256"], "producer_config_invalid"),
            full_canary_plan_sha256=_digest(raw["full_canary_plan_sha256"], "producer_config_invalid"),
            service_unit=service_unit,
            service_identity_sha256=_digest(raw["service_identity_sha256"], "producer_config_invalid"),
            service_uid=raw["service_uid"],
            service_gid=raw["service_gid"],
            root_client_uid=raw["root_client_uid"],
            socket_path=path_values["socket_path"],
            receipt_base_root=path_values["receipt_base_root"],
            receipt_directory_uid=raw["receipt_directory_uid"],
            receipt_directory_gid=raw["receipt_directory_gid"],
            receipt_directory_mode=raw["receipt_directory_mode"],
            private_key_path=path_values["private_key_path"],
            public_key_path=path_values["public_key_path"],
            allowed_slots=tuple(slots),
        )


def load_producer_config(path: Path) -> ProducerConfig:
    if os.geteuid() < 0:  # pragma: no cover - defensive platform guard
        _fail("producer_identity_invalid")
    raw, _item = _stable_read(
        path,
        maximum=MAX_CONFIG_BYTES,
        uid=0,
        gid=os.getegid(),
        mode=0o440,
    )
    return ProducerConfig.from_mapping(_strict_json(raw, "producer_config_invalid"))


def validate_producer_config_binding(
    config: ProducerConfig,
    foundation: Mapping[str, Any],
) -> None:
    """Require one packaged role config to be an exact foundation projection."""

    code = "producer_config_foundation_mismatch"
    endpoint = foundation.get("endpoints", {}).get(config.role)
    receipt = foundation.get("receipt_contract")
    if not isinstance(endpoint, Mapping) or not isinstance(receipt, Mapping):
        _fail(code)
    expected = {
        "foundation_sha256": producer_foundation_sha256(foundation),
        "release_sha": foundation.get("release_sha"),
        "capability_plan_sha256": foundation.get("capability_plan_sha256"),
        "full_canary_plan_sha256": foundation.get("full_canary_plan_sha256"),
        "service_unit": endpoint.get("service_unit"),
        "service_identity_sha256": endpoint.get("service_identity_sha256"),
        "service_uid": endpoint.get("uid"),
        "service_gid": endpoint.get("gid"),
        "socket_path": endpoint.get("socket_path"),
        "receipt_base_root": receipt.get("base_root"),
        "receipt_directory_uid": receipt.get("run_directory_uid"),
        "receipt_directory_gid": receipt.get("run_directory_gid"),
        "receipt_directory_mode": receipt.get("run_directory_mode"),
        "private_key_path": endpoint.get("private_key_path"),
        "public_key_path": endpoint.get("public_key_path"),
        "allowed_slots": tuple(endpoint.get("allowed_slots") or ()),
    }
    observed = {
        "foundation_sha256": config.foundation_sha256,
        "release_sha": config.release_sha,
        "capability_plan_sha256": config.capability_plan_sha256,
        "full_canary_plan_sha256": config.full_canary_plan_sha256,
        "service_unit": config.service_unit,
        "service_identity_sha256": config.service_identity_sha256,
        "service_uid": config.service_uid,
        "service_gid": config.service_gid,
        "socket_path": str(config.socket_path),
        "receipt_base_root": str(config.receipt_base_root),
        "receipt_directory_uid": config.receipt_directory_uid,
        "receipt_directory_gid": config.receipt_directory_gid,
        "receipt_directory_mode": config.receipt_directory_mode,
        "private_key_path": str(config.private_key_path),
        "public_key_path": str(config.public_key_path),
        "allowed_slots": config.allowed_slots,
    }
    if observed != expected:
        _fail(code)


def _load_projected_private_key(
    config: ProducerConfig,
) -> tuple[Ed25519PrivateKey, str]:
    last: CapabilityProducerError | None = None
    for uid, gid in (
        (config.service_uid, config.service_gid),
        (0, 0),
    ):
        try:
            return _load_private_key(config.private_key_path, uid=uid, gid=gid)
        except CapabilityProducerError as exc:
            last = exc
            if exc.code != "artifact_identity_invalid":
                raise
    assert last is not None
    raise last


def _load_projected_public_key(
    config: ProducerConfig,
) -> tuple[Ed25519PublicKey, str, str]:
    last: CapabilityProducerError | None = None
    for uid, gid in (
        (config.service_uid, config.service_gid),
        (0, 0),
    ):
        try:
            return _load_public_key(
                config.public_key_path,
                uid=uid,
                gid=gid,
                mode=0o400,
            )
        except CapabilityProducerError as exc:
            last = exc
            if exc.code != "artifact_identity_invalid":
                raise
    assert last is not None
    raise last


class RoleReceiptProducer:
    """Sign one fixed slot only after role-local native evidence collection."""

    def __init__(
        self,
        config: ProducerConfig,
        *,
        native_collector: NativeEvidenceCollector,
        private_key: Ed25519PrivateKey | None = None,
        public_key: Ed25519PublicKey | None = None,
        now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
        publisher: Callable[..., None] = _publish_no_replace,
    ) -> None:
        if os.geteuid() != config.service_uid or os.getegid() != config.service_gid:
            _fail("producer_process_identity_invalid")
        loaded_private, _private_file_sha256 = (
            _load_projected_private_key(config)
            if private_key is None
            else (private_key, "")
        )
        loaded_public, public_hex, public_file_sha256 = (
            _load_projected_public_key(config)
            if public_key is None
            else (
                public_key,
                public_key.public_bytes(
                    serialization.Encoding.Raw,
                    serialization.PublicFormat.Raw,
                ).hex(),
                "",
            )
        )
        private_public = loaded_private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        if private_public != public_hex:
            _fail("producer_key_pair_mismatch")
        self.config = config
        self.native_collector = native_collector
        self.private_key = loaded_private
        self.public_key = loaded_public
        self.public_key_hex = public_hex
        self.public_key_file_sha256 = public_file_sha256
        self.key_id = _sha256_bytes(bytes.fromhex(public_hex))
        self.now_ms = now_ms
        self.publisher = publisher
        self._active_readiness_sha256: str | None = None
        self._active_run_id: str | None = None
        self._active_fixture_sha256: str | None = None
        self._active_receipt_root: Path | None = None

    def readiness(self, *, main_pid: int | None = None) -> dict[str, Any]:
        pid = os.getpid() if main_pid is None else main_pid
        if type(pid) is not int or pid <= 1:
            _fail("producer_readiness_invalid")
        unsigned = {
            "schema": PRODUCER_ENDPOINT_READINESS_SCHEMA,
            "role": self.config.role,
            "foundation_sha256": self.config.foundation_sha256,
            "release_sha": self.config.release_sha,
            "capability_plan_sha256": self.config.capability_plan_sha256,
            "full_canary_plan_sha256": self.config.full_canary_plan_sha256,
            "service_unit": self.config.service_unit,
            "service_identity_sha256": self.config.service_identity_sha256,
            "main_pid": pid,
            "uid": os.geteuid(),
            "gid": os.getegid(),
            "socket_path": str(self.config.socket_path),
            "allowed_slots": list(self.config.allowed_slots),
            "key_id": self.key_id,
            "algorithm": AUTHORITY_ALGORITHMS[self.config.role],
            "public_key_ed25519_hex": self.public_key_hex,
            "public_key_file_sha256": self.public_key_file_sha256,
            "private_key_or_digest_present": False,
            "observed_at_unix_ms": self.now_ms(),
        }
        return {**unsigned, "readiness_sha256": _sha256_json(unsigned)}

    def activate(self, readiness_value: Any) -> Mapping[str, Any]:
        """Bind this live process to the final fleet readiness artifact."""

        readiness = validate_fleet_readiness(
            readiness_value,
            now_ms=self.now_ms(),
            expected_foundation_sha256=self.config.foundation_sha256,
        )
        endpoint = readiness["endpoint_readiness"].get(self.config.role)
        authority = readiness["authority_keys"].get(self.config.role)
        if (
            readiness["foundation_sha256"] != self.config.foundation_sha256
            or readiness["release_sha"] != self.config.release_sha
            or readiness["capability_plan_sha256"]
            != self.config.capability_plan_sha256
            or readiness["full_canary_plan_sha256"]
            != self.config.full_canary_plan_sha256
            or not isinstance(endpoint, Mapping)
            or endpoint.get("main_pid") != os.getpid()
            or endpoint.get("uid") != os.geteuid()
            or endpoint.get("gid") != os.getegid()
            or endpoint.get("service_identity_sha256")
            != self.config.service_identity_sha256
            or endpoint.get("key_id") != self.key_id
            or not isinstance(authority, Mapping)
            or authority.get("key_id") != self.key_id
            or authority.get("public_key_ed25519_hex") != self.public_key_hex
        ):
            _fail("producer_activation_invalid")
        digest = readiness["readiness_sha256"]
        if self._active_readiness_sha256 not in {None, digest}:
            _fail("producer_activation_rotated")
        self._active_readiness_sha256 = digest
        self._active_run_id = readiness["run_id"]
        self._active_fixture_sha256 = readiness["fixture_sha256"]
        run_root = Path(readiness["run_receipt_root"])
        expected_root = self.config.receipt_base_root / readiness["run_id"]
        if run_root != expected_root:
            _fail("producer_activation_invalid")
        _require_directory(
            run_root,
            uid=self.config.receipt_directory_uid,
            gid=self.config.receipt_directory_gid,
            mode=self.config.receipt_directory_mode,
        )
        self._active_receipt_root = run_root
        unsigned = {
            "schema": PRODUCER_ENDPOINT_ACTIVATION_SCHEMA,
            "role": self.config.role,
            "readiness_sha256": digest,
            "main_pid": os.getpid(),
            "activated_at_unix_ms": self.now_ms(),
        }
        return {**unsigned, "activation_sha256": _sha256_json(unsigned)}

    def produce(self, request_value: Any) -> dict[str, Any]:
        request = _strict(
            request_value,
            (
                "schema",
                "slot",
                "role",
                "run_id",
                "release_sha",
                "fixture_sha256",
                "producer_readiness_sha256",
                "payload",
            ),
            "producer_request_invalid",
        )
        slot = request["slot"]
        if (
            request["schema"] != PRODUCER_REQUEST_SCHEMA
            or slot not in self.config.allowed_slots
            or request["role"] != self.config.role
            or request["run_id"] != self._active_run_id
            or SLOT_ROLE.get(slot) != self.config.role
            or request["release_sha"] != self.config.release_sha
            or request["fixture_sha256"] != self._active_fixture_sha256
            or self._active_readiness_sha256 is None
            or request["producer_readiness_sha256"]
            != self._active_readiness_sha256
            or not isinstance(request["payload"], Mapping)
        ):
            _fail("producer_request_invalid")
        payload = copy.deepcopy(dict(request["payload"]))
        # The collector is called with only the slot and proposed public payload.
        # It independently reads/queries native state; the requester cannot pass
        # a claimed verification result into the signing boundary.
        collected = tuple(
            self.native_collector.collect(slot=slot, payload=payload)
        )
        if not 1 <= len(collected) <= MAX_BINDINGS or any(
            not isinstance(item, NativeEvidenceBinding) for item in collected
        ):
            _fail("native_evidence_invalid")
        expected_kinds = SLOT_NATIVE_BINDING_KINDS[slot]
        if tuple(item.kind for item in collected) != expected_kinds:
            _fail("native_evidence_incomplete")
        bindings = [item.to_mapping() for item in collected]
        native = {
            "schema": NATIVE_EVIDENCE_SCHEMA,
            "producer_readiness_sha256": self._active_readiness_sha256,
            "bindings": bindings,
        }
        unsigned = {
            "schema": SIGNED_RECEIPT_SCHEMA,
            "authority_role": self.config.role,
            "key_id": self.key_id,
            "signature_algorithm": "ed25519",
            "payload": payload,
            "native_evidence": native,
        }
        receipt = {
            **unsigned,
            "signature": self.private_key.sign(_canonical_bytes(unsigned)).hex(),
        }
        raw = _canonical_bytes(receipt)
        if self._active_receipt_root is None:
            _fail("producer_not_activated")
        output = self._active_receipt_root / SLOT_FILENAME[slot]
        self.publisher(
            output,
            raw,
            uid=self.config.service_uid,
            gid=self.config.service_gid,
            mode=0o400,
            parent_uid=self.config.receipt_directory_uid,
            parent_gid=self.config.receipt_directory_gid,
            parent_mode=self.config.receipt_directory_mode,
        )
        return receipt


def verify_role_receipt(
    value: Any,
    *,
    role: str,
    slot: str,
    public_key: Ed25519PublicKey,
    producer_readiness_sha256: str,
) -> Mapping[str, Any]:
    receipt = _strict(
        value,
        (
            "schema",
            "authority_role",
            "key_id",
            "signature_algorithm",
            "payload",
            "native_evidence",
            "signature",
        ),
        "signed_receipt_invalid",
    )
    if role not in ENDPOINT_ROLES or SLOT_ROLE.get(slot) != role:
        _fail("signed_receipt_invalid")
    public_raw = public_key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    unsigned = {key: value for key, value in receipt.items() if key != "signature"}
    signature = receipt["signature"]
    if (
        receipt["schema"] != SIGNED_RECEIPT_SCHEMA
        or receipt["authority_role"] != role
        or receipt["key_id"] != _sha256_bytes(public_raw)
        or receipt["signature_algorithm"] != "ed25519"
        or not isinstance(signature, str)
        or re.fullmatch(r"[0-9a-f]{128}", signature) is None
    ):
        _fail("signed_receipt_invalid")
    native = _strict(
        receipt["native_evidence"],
        ("schema", "producer_readiness_sha256", "bindings"),
        "native_evidence_invalid",
    )
    if (
        native["schema"] != NATIVE_EVIDENCE_SCHEMA
        or native["producer_readiness_sha256"]
        != _digest(producer_readiness_sha256, "native_evidence_invalid")
        or not isinstance(native["bindings"], list)
    ):
        _fail("native_evidence_invalid")
    bindings = [NativeEvidenceBinding.from_mapping(item) for item in native["bindings"]]
    if tuple(item.kind for item in bindings) != SLOT_NATIVE_BINDING_KINDS[slot]:
        _fail("native_evidence_incomplete")
    try:
        public_key.verify(bytes.fromhex(signature), _canonical_bytes(unsigned))
    except (InvalidSignature, ValueError):
        _fail("signed_receipt_invalid")
    return receipt["payload"]


@dataclass(frozen=True)
class PeerIdentity:
    pid: int
    uid: int
    gid: int


def linux_peer_identity(connection: socket.socket) -> PeerIdentity:
    peercred = getattr(socket, "SO_PEERCRED", None)
    if peercred is None:
        _fail("producer_peer_credentials_unavailable")
    try:
        raw = connection.getsockopt(socket.SOL_SOCKET, peercred, _PEER.size)
    except OSError as exc:
        raise CapabilityProducerError("producer_peer_credentials_unavailable") from exc
    if len(raw) != _PEER.size:
        _fail("producer_peer_credentials_unavailable")
    return PeerIdentity(*_PEER.unpack(raw))


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            _fail("producer_frame_truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(connection: socket.socket, *, maximum: int) -> dict[str, Any]:
    (size,) = _HEADER.unpack(_recv_exact(connection, _HEADER.size))
    if not 1 < size <= maximum:
        _fail("producer_frame_invalid")
    return _strict_json(_recv_exact(connection, size), "producer_frame_invalid")


def _send_frame(connection: socket.socket, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value)
    if len(raw) > MAX_RESPONSE_BYTES:
        _fail("producer_response_too_large")
    connection.sendall(_HEADER.pack(len(raw)) + raw)


class CapabilityProducerServer:
    """One role endpoint; only root may request readiness or publication."""

    def __init__(
        self,
        producer: RoleReceiptProducer,
        *,
        peer_getter: Callable[[socket.socket], PeerIdentity] = linux_peer_identity,
    ) -> None:
        self.producer = producer
        self.peer_getter = peer_getter

    def handle(self, connection: socket.socket) -> None:
        peer = self.peer_getter(connection)
        if peer.uid != self.producer.config.root_client_uid or peer.pid <= 1:
            _fail("producer_peer_unauthorized")
        request = _recv_frame(connection, maximum=MAX_REQUEST_BYTES)
        action = request.get("action")
        if action == "readiness" and set(request) == {"action"}:
            result = self.producer.readiness()
        elif action == "activate" and set(request) == {"action", "readiness"}:
            result = self.producer.activate(request["readiness"])
        elif action == "produce" and set(request) == {"action", "request"}:
            result = self.producer.produce(request["request"])
        else:
            _fail("producer_action_invalid")
        _send_frame(
            connection,
            {
                "schema": PRODUCER_RESPONSE_SCHEMA,
                "ok": True,
                "result": result,
            },
        )

    def serve_forever(
        self,
        *,
        should_stop: Callable[[], bool] = lambda: False,
        poll_seconds: float = 0.25,
    ) -> None:
        """Own one exact AF_UNIX endpoint until the service is stopped."""

        path = self.producer.config.socket_path
        if not 0.01 <= poll_seconds <= 1.0:
            _fail("producer_server_config_invalid")
        _require_directory(
            path.parent,
            uid=self.producer.config.service_uid,
            gid=self.producer.config.service_gid,
            mode=0o700,
        )
        if os.path.lexists(path):
            # RuntimeDirectory must be empty at unit start.  Never unlink an
            # unproven path merely because a service restarted.
            _fail("producer_socket_path_occupied")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.set_inheritable(False)
        bound_identity: tuple[int, ...] | None = None
        try:
            listener.bind(str(path))
            os.chmod(path, 0o600, follow_symlinks=False)
            item = path.lstat()
            if (
                not stat.S_ISSOCK(item.st_mode)
                or item.st_uid != self.producer.config.service_uid
                or item.st_gid != self.producer.config.service_gid
                or stat.S_IMODE(item.st_mode) != 0o600
            ):
                _fail("producer_socket_identity_invalid")
            bound_identity = _identity(item)
            listener.listen(8)
            listener.settimeout(poll_seconds)
            while not should_stop():
                try:
                    connection, _address = listener.accept()
                except TimeoutError:
                    continue
                with connection:
                    connection.settimeout(30.0)
                    try:
                        self.handle(connection)
                    except CapabilityProducerError as exc:
                        try:
                            _send_frame(
                                connection,
                                {
                                    "schema": PRODUCER_RESPONSE_SCHEMA,
                                    "ok": False,
                                    "failure_code": exc.code,
                                },
                            )
                        except (CapabilityProducerError, OSError):
                            pass
        finally:
            listener.close()
            if bound_identity is not None:
                try:
                    current = path.lstat()
                except FileNotFoundError:
                    current = None
                if current is not None:
                    if _identity(current) != bound_identity:
                        _fail("producer_socket_replaced")
                    path.unlink()
                    _fsync_directory(path.parent)


class ProducerEndpointClient:
    """Root peer-authenticated client for one exact role producer endpoint."""

    def __init__(
        self,
        socket_path: Path,
        *,
        expected_peer: PeerIdentity,
        peer_getter: Callable[[socket.socket], PeerIdentity] = linux_peer_identity,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not socket_path.is_absolute() or ".." in socket_path.parts:
            _fail("producer_endpoint_invalid")
        if not 0 < timeout_seconds <= 30:
            _fail("producer_endpoint_invalid")
        self.socket_path = socket_path
        self.expected_peer = expected_peer
        self.peer_getter = peer_getter
        self.timeout_seconds = timeout_seconds

    def call(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.set_inheritable(False)
        try:
            connection.settimeout(self.timeout_seconds)
            connection.connect(str(self.socket_path))
            if self.peer_getter(connection) != self.expected_peer:
                _fail("producer_endpoint_peer_invalid")
            _send_frame(connection, value)
            response = _recv_frame(connection, maximum=MAX_RESPONSE_BYTES)
            if response.get("schema") != PRODUCER_RESPONSE_SCHEMA:
                _fail("producer_response_invalid")
            if response.get("ok") is False:
                if (
                    set(response) != {"schema", "ok", "failure_code"}
                    or not isinstance(response["failure_code"], str)
                ):
                    _fail("producer_response_invalid")
                _fail(f"producer_remote_{response['failure_code']}")
            if (
                set(response) != {"schema", "ok", "result"}
                or response["ok"] is not True
                or not isinstance(response["result"], Mapping)
            ):
                _fail("producer_response_invalid")
            return dict(response["result"])
        finally:
            connection.close()


PRODUCTION_PUMP_SLOTS = tuple(
    slot for slot in RECEIPT_SLOTS if SLOT_ROLE[slot] in ENDPOINT_ROLES
)
PRODUCTION_PRE_CLEANUP_PUMP_SLOTS = tuple(
    slot for slot in PRODUCTION_PUMP_SLOTS if slot != "cleanup"
)


class ProductionReceiptPump:
    """Mechanical fixed-slot caller for an already activated producer fleet."""

    def __init__(
        self,
        *,
        installed_foundation: InstalledProducerFoundation,
        readiness: Mapping[str, Any],
        endpoint_clients: Mapping[str, ProducerEndpointClient],
    ) -> None:
        foundation = validate_producer_foundation(
            installed_foundation.value,
            pinned_owner_public_key_ed25519_hex=(
                installed_foundation.pinned_owner_public_key_ed25519_hex
            ),
            pinned_owner_public_key_source_sha256=(
                installed_foundation.pinned_owner_public_key_source_sha256
            ),
        )
        activation = validate_fleet_readiness_for_retirement(
            readiness,
            expected_foundation_sha256=producer_foundation_sha256(foundation),
            expected_capability_plan_sha256=foundation[
                "capability_plan_sha256"
            ],
            expected_full_canary_plan_sha256=foundation[
                "full_canary_plan_sha256"
            ],
        )
        if set(endpoint_clients) != set(ENDPOINT_ROLES):
            _fail("producer_pump_invalid")
        for role in ENDPOINT_ROLES:
            endpoint = foundation["endpoints"][role]
            client = endpoint_clients[role]
            if (
                client.socket_path != Path(endpoint["socket_path"])
                or client.expected_peer.uid != endpoint["uid"]
                or client.expected_peer.gid != endpoint["gid"]
                or client.expected_peer.pid
                != activation["endpoint_readiness"][role]["main_pid"]
            ):
                _fail("producer_pump_endpoint_mismatch")
        self.foundation = copy.deepcopy(foundation)
        self.readiness = copy.deepcopy(activation)
        self.endpoint_clients = dict(endpoint_clients)

    def produce(self, *, slot: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        role = SLOT_ROLE.get(slot)
        if slot not in PRODUCTION_PUMP_SLOTS or role not in ENDPOINT_ROLES:
            _fail("producer_pump_slot_invalid")
        if not isinstance(payload, Mapping):
            _fail("producer_pump_payload_invalid")
        value = copy.deepcopy(dict(payload))
        if (
            value.get("run_id") != self.readiness["run_id"]
            or value.get("release_sha") != self.readiness["release_sha"]
            or value.get("fixture_sha256")
            != self.readiness["fixture_sha256"]
        ):
            _fail("producer_pump_payload_invalid")
        request = {
            "schema": PRODUCER_REQUEST_SCHEMA,
            "slot": slot,
            "role": role,
            "run_id": self.readiness["run_id"],
            "release_sha": self.readiness["release_sha"],
            "fixture_sha256": self.readiness["fixture_sha256"],
            "producer_readiness_sha256": self.readiness[
                "readiness_sha256"
            ],
            "payload": value,
        }
        receipt = self.endpoint_clients[role].call(
            {"action": "produce", "request": request}
        )
        public_key = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(
                self.foundation["authority_keys"][role][
                    "public_key_ed25519_hex"
                ]
            )
        )
        if verify_role_receipt(
            receipt,
            role=role,
            slot=slot,
            public_key=public_key,
            producer_readiness_sha256=self.readiness["readiness_sha256"],
        ) != value:
            _fail("producer_pump_receipt_invalid")
        endpoint = self.foundation["endpoints"][role]
        path = Path(self.readiness["run_receipt_root"]) / SLOT_FILENAME[slot]
        raw, _item = _stable_read(
            path,
            maximum=MAX_RECEIPT_BYTES,
            uid=endpoint["uid"],
            gid=endpoint["gid"],
            mode=0o400,
        )
        if _strict_json(raw, "producer_pump_receipt_invalid") != receipt:
            _fail("producer_pump_receipt_invalid")
        return copy.deepcopy(receipt)

    def produce_pre_cleanup(
        self,
        payloads: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, Mapping[str, Any]]:
        if set(payloads) != set(PRODUCTION_PRE_CLEANUP_PUMP_SLOTS):
            _fail("producer_pump_payload_set_invalid")
        return {
            slot: self.produce(slot=slot, payload=payloads[slot])
            for slot in PRODUCTION_PRE_CLEANUP_PUMP_SLOTS
        }

    def produce_cleanup(
        self, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return self.produce(slot="cleanup", payload=payload)


def _validate_endpoint_readiness(
    value: Any,
    *,
    role: str,
    foundation_sha256: str,
    release_sha: str,
    capability_plan_sha256: str,
    full_canary_plan_sha256: str,
    expected_peer: PeerIdentity,
    socket_path: Path,
    now_ms: int,
) -> dict[str, Any]:
    raw = _strict(
        value,
        (
            "schema",
            "role",
            "foundation_sha256",
            "release_sha",
            "capability_plan_sha256",
            "full_canary_plan_sha256",
            "service_unit",
            "service_identity_sha256",
            "main_pid",
            "uid",
            "gid",
            "socket_path",
            "allowed_slots",
            "key_id",
            "algorithm",
            "public_key_ed25519_hex",
            "public_key_file_sha256",
            "private_key_or_digest_present",
            "observed_at_unix_ms",
            "readiness_sha256",
        ),
        "producer_endpoint_readiness_invalid",
    )
    unsigned = {key: value for key, value in raw.items() if key != "readiness_sha256"}
    public_hex = raw["public_key_ed25519_hex"]
    observed = raw["observed_at_unix_ms"]
    if (
        raw["schema"] != PRODUCER_ENDPOINT_READINESS_SCHEMA
        or raw["role"] != role
        or raw["foundation_sha256"] != foundation_sha256
        or raw["release_sha"] != release_sha
        or raw["capability_plan_sha256"] != capability_plan_sha256
        or raw["full_canary_plan_sha256"] != full_canary_plan_sha256
        or raw["main_pid"] != expected_peer.pid
        or raw["uid"] != expected_peer.uid
        or raw["gid"] != expected_peer.gid
        or raw["socket_path"] != str(socket_path)
        or raw["allowed_slots"]
        != [slot for slot in RECEIPT_SLOTS if SLOT_ROLE[slot] == role]
        or raw["algorithm"] != "ed25519"
        or not isinstance(public_hex, str)
        or re.fullmatch(r"[0-9a-f]{64}", public_hex) is None
        or raw["key_id"] != _sha256_bytes(bytes.fromhex(public_hex))
        or raw["private_key_or_digest_present"] is not False
        or type(observed) is not int
        or abs(now_ms - observed) > MAX_CLOCK_SKEW_MS
        or raw["readiness_sha256"] != _sha256_json(unsigned)
    ):
        _fail("producer_endpoint_readiness_invalid")
    _digest(raw["service_identity_sha256"], "producer_endpoint_readiness_invalid")
    if raw["public_key_file_sha256"]:
        _digest(raw["public_key_file_sha256"], "producer_endpoint_readiness_invalid")
    return raw


def _validate_routeback_identity(
    value: Any,
    *,
    capability_plan_sha256: str,
    full_canary_plan_sha256: str,
    connector_bot_user_id: str,
    routeback_bot_user_id: str,
) -> dict[str, Any]:
    """Validate the security-review attestation without importing unstable code."""

    if not isinstance(value, Mapping):
        _fail("routeback_identity_invalid")
    raw = dict(value)
    required = {
        "schema",
        "plan_sha256",
        "full_canary_plan_sha256",
        "live_bot_user_id",
        "planned_routeback_bot_user_id",
        "connector_bot_user_id",
        "production_bot_user_id",
        "provenance",
        "pairwise_distinct",
        "observed_at_unix",
        "secret_material_recorded",
        "secret_digest_recorded",
        "credential_file_metadata_sha256",
        "attestation_sha256",
    }
    if set(raw) != required:
        _fail("routeback_identity_invalid")
    unsigned = {key: value for key, value in raw.items() if key != "attestation_sha256"}
    ids = (
        raw["live_bot_user_id"],
        raw["connector_bot_user_id"],
        raw["production_bot_user_id"],
    )
    from gateway.canonical_capability_canary_runtime import (
        CAPABILITY_ROUTEBACK_BOT_IDENTITY_SCHEMA,
    )

    if (
        raw["schema"] != CAPABILITY_ROUTEBACK_BOT_IDENTITY_SCHEMA
        or raw["plan_sha256"] != capability_plan_sha256
        or raw["full_canary_plan_sha256"] != full_canary_plan_sha256
        or raw["live_bot_user_id"] != routeback_bot_user_id
        or raw["planned_routeback_bot_user_id"] != routeback_bot_user_id
        or raw["connector_bot_user_id"] != connector_bot_user_id
        or raw["production_bot_user_id"] != PRODUCTION_BOT_USER_ID
        or any(not isinstance(item, str) or _SNOWFLAKE_RE.fullmatch(item) is None for item in ids)
        or len(set(ids)) != 3
        or raw["pairwise_distinct"] is not True
        or type(raw["observed_at_unix"]) is not int
        or raw["observed_at_unix"] < 1
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["attestation_sha256"] != _sha256_json(unsigned)
    ):
        _fail("routeback_identity_invalid")
    _digest(raw["credential_file_metadata_sha256"], "routeback_identity_invalid")
    provenance = raw["provenance"]
    if provenance != {
        "source": "discord_rest_api_v10_current_user",
        "http_method": "GET",
        "resource": "/users/@me",
        "credential_boundary": "sealed_routeback_credential_file",
    }:
        _fail("routeback_identity_invalid")
    return raw


def build_fleet_readiness_for_plans(
    *,
    plan: Any,
    full_plan: Any,
    foundation: Mapping[str, Any],
    fixture: Mapping[str, Any],
    fixture_sha256: str,
    pinned_owner_public_key_ed25519_hex: str,
    pinned_owner_public_key_source_sha256: str,
    owner_grant_sha256: str,
    endpoint_clients: Mapping[str, ProducerEndpointClient],
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    routeback_attester: Callable[[Any, Any], Mapping[str, Any]] | None = None,
    routeback_binding_checker: Callable[
        [Any, Any, Mapping[str, Any]], Mapping[str, Any]
    ]
    | None = None,
) -> dict[str, Any]:
    """Production wrapper that performs live route-back TOCTOU revalidation."""

    from gateway.canonical_capability_canary_runtime import (
        _attest_live_routeback_bot_identity,
        _require_routeback_credential_binding,
    )

    attest = _attest_live_routeback_bot_identity if routeback_attester is None else routeback_attester
    require_binding = (
        _require_routeback_credential_binding
        if routeback_binding_checker is None
        else routeback_binding_checker
    )
    identity = attest(plan, full_plan)
    metadata = require_binding(plan, full_plan, identity)
    if (
        not isinstance(metadata, Mapping)
        or _sha256_json(metadata)
        != identity.get("credential_file_metadata_sha256")
    ):
        _fail("routeback_identity_invalid")
    return build_fleet_activation(
        foundation=foundation,
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        pinned_owner_public_key_ed25519_hex=(
            pinned_owner_public_key_ed25519_hex
        ),
        pinned_owner_public_key_source_sha256=(
            pinned_owner_public_key_source_sha256
        ),
        owner_grant_sha256=owner_grant_sha256,
        routeback_bot_identity=identity,
        endpoint_clients=endpoint_clients,
        now_ms=now_ms,
)


def build_fleet_activation(
    *,
    foundation: Mapping[str, Any],
    fixture: Mapping[str, Any],
    fixture_sha256: str,
    pinned_owner_public_key_ed25519_hex: str,
    pinned_owner_public_key_source_sha256: str,
    owner_grant_sha256: str,
    routeback_bot_identity: Mapping[str, Any],
    endpoint_clients: Mapping[str, ProducerEndpointClient],
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
) -> dict[str, Any]:
    """Build the per-run activation after fixture and live endpoint readiness."""

    current = now_ms()
    foundation_value = validate_producer_foundation(
        foundation,
        pinned_owner_public_key_ed25519_hex=(
            pinned_owner_public_key_ed25519_hex
        ),
        pinned_owner_public_key_source_sha256=(
            pinned_owner_public_key_source_sha256
        ),
    )
    foundation_sha256 = producer_foundation_sha256(foundation_value)
    if not isinstance(fixture, Mapping):
        _fail("producer_fleet_readiness_invalid")
    fixture_value = dict(fixture)
    fixture_sha256 = _digest(fixture_sha256, "producer_fleet_readiness_invalid")
    if _sha256_json(fixture_value) != fixture_sha256:
        _fail("producer_fleet_readiness_invalid")
    release_sha = fixture_value.get("release_sha")
    capability_plan_sha256 = foundation_value["capability_plan_sha256"]
    full_canary_plan_sha256 = foundation_value["full_canary_plan_sha256"]
    valid_from_unix_ms = fixture_value.get("valid_from_unix_ms")
    valid_until_unix_ms = fixture_value.get("valid_until_unix_ms")
    run_id = fixture_value.get("run_id")
    bot_identities = fixture_value.get("discord_bot_identities")
    authorities = fixture_value.get("authority_keys")
    if (
        not isinstance(release_sha, str)
        or _GIT_SHA_RE.fullmatch(release_sha) is None
        or type(valid_from_unix_ms) is not int
        or type(valid_until_unix_ms) is not int
        or not valid_from_unix_ms <= current < valid_until_unix_ms
        or set(endpoint_clients) != set(ENDPOINT_ROLES)
        or not isinstance(run_id, str)
        or _SAFE_ID_RE.fullmatch(run_id) is None
        or fixture_value.get("producer_foundation_sha256")
        != foundation_sha256
        or release_sha != foundation_value["release_sha"]
        or fixture_value.get("owner_id") != PRODUCTION_OWNER_ID
        or authorities != foundation_value["authority_keys"]
        or not isinstance(bot_identities, Mapping)
        or set(bot_identities)
        != {
            "production_bot_user_id",
            "connector_bot_user_id",
            "routeback_bot_user_id",
        }
    ):
        _fail("producer_fleet_readiness_invalid")
    owner_grant_sha256 = _digest(owner_grant_sha256, "producer_fleet_readiness_invalid")
    connector_bot_user_id = bot_identities["connector_bot_user_id"]
    routeback_bot_user_id = bot_identities["routeback_bot_user_id"]
    routeback = _validate_routeback_identity(
        routeback_bot_identity,
        capability_plan_sha256=capability_plan_sha256,
        full_canary_plan_sha256=full_canary_plan_sha256,
        connector_bot_user_id=connector_bot_user_id,
        routeback_bot_user_id=routeback_bot_user_id,
    )
    observed_routeback = routeback["observed_at_unix"] * 1000
    if not valid_from_unix_ms <= observed_routeback <= current:
        _fail("routeback_identity_invalid")

    endpoints: dict[str, Any] = {}
    for role in ENDPOINT_ROLES:
        client = endpoint_clients[role]
        result = client.call({"action": "readiness"})
        endpoints[role] = _validate_endpoint_readiness(
            result,
            role=role,
            foundation_sha256=foundation_sha256,
            release_sha=release_sha,
            capability_plan_sha256=capability_plan_sha256,
            full_canary_plan_sha256=full_canary_plan_sha256,
            expected_peer=client.expected_peer,
            socket_path=client.socket_path,
            now_ms=current,
        )
        expected_endpoint = foundation_value["endpoints"][role]
        if (
            endpoints[role]["service_unit"]
            != expected_endpoint["service_unit"]
            or endpoints[role]["service_identity_sha256"]
            != expected_endpoint["service_identity_sha256"]
            or endpoints[role]["uid"] != expected_endpoint["uid"]
            or endpoints[role]["gid"] != expected_endpoint["gid"]
            or endpoints[role]["key_id"] != expected_endpoint["key_id"]
            or endpoints[role]["public_key_ed25519_hex"]
            != expected_endpoint["public_key_ed25519_hex"]
            or endpoints[role]["public_key_file_sha256"]
            != expected_endpoint["public_key_file_sha256"]
        ):
            _fail("producer_endpoint_foundation_mismatch")
    receipt_contract = foundation_value["receipt_contract"]
    run_receipt_root = str(Path(receipt_contract["base_root"]) / run_id)
    _require_directory(
        Path(run_receipt_root),
        uid=receipt_contract["run_directory_uid"],
        gid=receipt_contract["run_directory_gid"],
        mode=receipt_contract["run_directory_mode"],
    )
    owner = foundation_value["owner_authority"]
    unsigned = {
        "schema": PRODUCER_FLEET_READINESS_SCHEMA,
        "foundation_sha256": foundation_sha256,
        "release_sha": release_sha,
        "capability_plan_sha256": capability_plan_sha256,
        "full_canary_plan_sha256": full_canary_plan_sha256,
        "fixture_sha256": fixture_sha256,
        "run_id": run_id,
        "run_receipt_root": run_receipt_root,
        "owner_id": PRODUCTION_OWNER_ID,
        "valid_from_unix_ms": valid_from_unix_ms,
        "valid_until_unix_ms": valid_until_unix_ms,
        "receipt_slots": list(RECEIPT_SLOTS),
        "slot_filenames": dict(SLOT_FILENAME),
        "slot_roles": dict(SLOT_ROLE),
        "slot_native_binding_kinds": {
            slot: list(SLOT_NATIVE_BINDING_KINDS[slot]) for slot in RECEIPT_SLOTS
        },
        "authority_keys": copy.deepcopy(foundation_value["authority_keys"]),
        "endpoint_readiness": endpoints,
        "owner_authority": {
            "producer_kind": "pre_staged_sshsig_grant",
            "owner_id": PRODUCTION_OWNER_ID,
            "key_id": owner["key_id"],
            "public_key_source_sha256": owner["public_key_source_sha256"],
            "grant_sha256": owner_grant_sha256,
            "grant_path": str(DEFAULT_OWNER_GRANT_PATH),
        },
        "discord_bot_identities": {
            "production_bot_user_id": PRODUCTION_BOT_USER_ID,
            "connector_bot_user_id": connector_bot_user_id,
            "routeback_bot_user_id": routeback_bot_user_id,
            "routeback_identity_attestation_sha256": routeback["attestation_sha256"],
            "routeback_credential_file_metadata_sha256": routeback[
                "credential_file_metadata_sha256"
            ],
        },
        "producer_protocol": "role_local_native_evidence_v1",
        "root_can_sign_non_observer_roles": False,
        "token_or_token_digest_recorded": False,
        "observed_at_unix_ms": current,
    }
    return {**unsigned, "readiness_sha256": _sha256_json(unsigned)}


# Kept as the public compatibility spelling while callers migrate to the more
# accurate activation name.
build_fleet_readiness = build_fleet_activation


def activate_fleet_readiness(
    readiness_value: Mapping[str, Any],
    *,
    endpoint_clients: Mapping[str, ProducerEndpointClient],
    now_ms: int | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Two-phase barrier: every endpoint accepts the same final readiness."""

    readiness = validate_fleet_readiness(readiness_value, now_ms=now_ms)
    if set(endpoint_clients) != set(ENDPOINT_ROLES):
        _fail("producer_activation_invalid")
    receipts: dict[str, Mapping[str, Any]] = {}
    for role in ENDPOINT_ROLES:
        value = endpoint_clients[role].call(
            {"action": "activate", "readiness": readiness}
        )
        raw = _strict(
            value,
            (
                "schema",
                "role",
                "readiness_sha256",
                "main_pid",
                "activated_at_unix_ms",
                "activation_sha256",
            ),
            "producer_activation_invalid",
        )
        unsigned = {
            key: item for key, item in raw.items() if key != "activation_sha256"
        }
        if (
            raw["schema"] != PRODUCER_ENDPOINT_ACTIVATION_SCHEMA
            or raw["role"] != role
            or raw["readiness_sha256"] != readiness["readiness_sha256"]
            or raw["main_pid"] != endpoint_clients[role].expected_peer.pid
            or type(raw["activated_at_unix_ms"]) is not int
            or raw["activation_sha256"] != _sha256_json(unsigned)
        ):
            _fail("producer_activation_invalid")
        receipts[role] = raw
    return receipts


def validate_fleet_readiness(
    value: Any,
    *,
    now_ms: int | None = None,
    expected_foundation_sha256: str | None = None,
) -> dict[str, Any]:
    raw = _strict(
        value,
        (
            "schema",
            "foundation_sha256",
            "release_sha",
            "capability_plan_sha256",
            "full_canary_plan_sha256",
            "fixture_sha256",
            "run_id",
            "run_receipt_root",
            "owner_id",
            "valid_from_unix_ms",
            "valid_until_unix_ms",
            "receipt_slots",
            "slot_filenames",
            "slot_roles",
            "slot_native_binding_kinds",
            "authority_keys",
            "endpoint_readiness",
            "owner_authority",
            "discord_bot_identities",
            "producer_protocol",
            "root_can_sign_non_observer_roles",
            "token_or_token_digest_recorded",
            "observed_at_unix_ms",
            "readiness_sha256",
        ),
        "producer_fleet_readiness_invalid",
    )
    unsigned = {key: value for key, value in raw.items() if key != "readiness_sha256"}
    current = int(time.time() * 1000) if now_ms is None else now_ms
    foundation_sha256 = _digest(
        raw["foundation_sha256"], "producer_fleet_readiness_invalid"
    )
    run_id = raw["run_id"]
    run_root = (
        Path(raw["run_receipt_root"])
        if isinstance(raw["run_receipt_root"], str)
        else Path()
    )
    if (
        raw["schema"] != PRODUCER_FLEET_READINESS_SCHEMA
        or (
            expected_foundation_sha256 is not None
            and foundation_sha256
            != _digest(
                expected_foundation_sha256,
                "producer_fleet_readiness_invalid",
            )
        )
        or raw["owner_id"] != PRODUCTION_OWNER_ID
        or not isinstance(raw["release_sha"], str)
        or _GIT_SHA_RE.fullmatch(raw["release_sha"]) is None
        or not isinstance(run_id, str)
        or _SAFE_ID_RE.fullmatch(run_id) is None
        or not run_root.is_absolute()
        or ".." in run_root.parts
        or run_root.name != run_id
        or raw["receipt_slots"] != list(RECEIPT_SLOTS)
        or raw["slot_filenames"] != SLOT_FILENAME
        or raw["slot_roles"] != SLOT_ROLE
        or raw["slot_native_binding_kinds"]
        != {slot: list(SLOT_NATIVE_BINDING_KINDS[slot]) for slot in RECEIPT_SLOTS}
        or raw["producer_protocol"] != "role_local_native_evidence_v1"
        or raw["root_can_sign_non_observer_roles"] is not False
        or raw["token_or_token_digest_recorded"] is not False
        or type(raw["valid_from_unix_ms"]) is not int
        or type(raw["valid_until_unix_ms"]) is not int
        or not raw["valid_from_unix_ms"] <= current < raw["valid_until_unix_ms"]
        or type(raw["observed_at_unix_ms"]) is not int
        or not raw["valid_from_unix_ms"]
        <= raw["observed_at_unix_ms"]
        <= current
        or raw["readiness_sha256"] != _sha256_json(unsigned)
    ):
        _fail("producer_fleet_readiness_invalid")
    for field in (
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "fixture_sha256",
    ):
        _digest(raw[field], "producer_fleet_readiness_invalid")
    authorities = _strict(
        raw["authority_keys"],
        AUTHORITY_ROLES,
        "producer_fleet_readiness_invalid",
    )
    public_keys: list[str] = []
    for role in AUTHORITY_ROLES:
        authority = _strict(
            authorities[role],
            ("key_id", "algorithm", "public_key_ed25519_hex"),
            "producer_fleet_readiness_invalid",
        )
        public_hex = authority["public_key_ed25519_hex"]
        if (
            authority["algorithm"] != AUTHORITY_ALGORITHMS[role]
            or not isinstance(public_hex, str)
            or re.fullmatch(r"[0-9a-f]{64}", public_hex) is None
            or authority["key_id"] != _sha256_bytes(bytes.fromhex(public_hex))
        ):
            _fail("producer_fleet_readiness_invalid")
        public_keys.append(public_hex)
    if len(set(public_keys)) != len(public_keys):
        _fail("producer_authority_keys_not_separated")
    endpoints = _strict(
        raw["endpoint_readiness"],
        ENDPOINT_ROLES,
        "producer_fleet_readiness_invalid",
    )
    if any(
        endpoints[role].get("foundation_sha256") != foundation_sha256
        or endpoints[role].get("release_sha") != raw["release_sha"]
        or endpoints[role].get("capability_plan_sha256")
        != raw["capability_plan_sha256"]
        or endpoints[role].get("full_canary_plan_sha256")
        != raw["full_canary_plan_sha256"]
        or endpoints[role].get("key_id") != authorities[role]["key_id"]
        or endpoints[role].get("public_key_ed25519_hex")
        != authorities[role]["public_key_ed25519_hex"]
        for role in ENDPOINT_ROLES
    ):
        _fail("producer_fleet_readiness_invalid")
    owner = _strict(
        raw["owner_authority"],
        (
            "producer_kind",
            "owner_id",
            "key_id",
            "public_key_source_sha256",
            "grant_sha256",
            "grant_path",
        ),
        "producer_fleet_readiness_invalid",
    )
    if (
        owner["producer_kind"] != "pre_staged_sshsig_grant"
        or owner["owner_id"] != PRODUCTION_OWNER_ID
        or owner["key_id"] != authorities["owner"]["key_id"]
        or owner["grant_path"] != str(DEFAULT_OWNER_GRANT_PATH)
    ):
        _fail("producer_fleet_readiness_invalid")
    _digest(owner["public_key_source_sha256"], "producer_fleet_readiness_invalid")
    _digest(owner["grant_sha256"], "producer_fleet_readiness_invalid")
    bots = _strict(
        raw["discord_bot_identities"],
        (
            "production_bot_user_id",
            "connector_bot_user_id",
            "routeback_bot_user_id",
            "routeback_identity_attestation_sha256",
            "routeback_credential_file_metadata_sha256",
        ),
        "producer_fleet_readiness_invalid",
    )
    bot_ids = (
        bots["production_bot_user_id"],
        bots["connector_bot_user_id"],
        bots["routeback_bot_user_id"],
    )
    if (
        bots["production_bot_user_id"] != PRODUCTION_BOT_USER_ID
        or any(
            not isinstance(item, str)
            or _SNOWFLAKE_RE.fullmatch(item) is None
            for item in bot_ids
        )
        or len(set(bot_ids)) != 3
    ):
        _fail("producer_fleet_readiness_invalid")
    _digest(
        bots["routeback_identity_attestation_sha256"],
        "producer_fleet_readiness_invalid",
    )
    _digest(
        bots["routeback_credential_file_metadata_sha256"],
        "producer_fleet_readiness_invalid",
    )
    return raw


def load_fleet_readiness(
    path: Path = DEFAULT_READINESS_PATH,
    *,
    uid: int = 0,
    gid: int = 0,
    now_ms: int | None = None,
    expected_foundation_sha256: str | None = None,
) -> dict[str, Any]:
    raw, _item = _stable_read(
        path,
        maximum=MAX_READINESS_BYTES,
        uid=uid,
        gid=gid,
        mode=0o400,
    )
    return validate_fleet_readiness(
        _strict_json(raw, "producer_fleet_readiness_invalid"),
        now_ms=now_ms,
        expected_foundation_sha256=expected_foundation_sha256,
    )


def validate_fleet_readiness_for_retirement(
    value: Any,
    *,
    expected_foundation_sha256: str | None = None,
    expected_capability_plan_sha256: str | None = None,
    expected_full_canary_plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an activation's immutable truth even after its live window."""

    if not isinstance(value, Mapping):
        _fail("producer_fleet_readiness_retirement_invalid")
    observed_at = value.get("observed_at_unix_ms")
    if type(observed_at) is not int:
        _fail("producer_fleet_readiness_retirement_invalid")
    readiness = validate_fleet_readiness(
        value,
        now_ms=observed_at,
        expected_foundation_sha256=expected_foundation_sha256,
    )
    expected = (
        ("capability_plan_sha256", expected_capability_plan_sha256),
        ("full_canary_plan_sha256", expected_full_canary_plan_sha256),
    )
    if any(
        digest is not None
        and readiness[field]
        != _digest(digest, "producer_fleet_readiness_retirement_invalid")
        for field, digest in expected
    ):
        _fail("producer_fleet_readiness_retirement_invalid")
    return readiness


def publish_fleet_readiness(
    value: Mapping[str, Any],
    *,
    path: Path = DEFAULT_READINESS_PATH,
    uid: int = 0,
    gid: int = 0,
    expected_foundation_sha256: str | None = None,
    now_ms: int | None = None,
) -> None:
    validated = validate_fleet_readiness(
        value,
        now_ms=now_ms,
        expected_foundation_sha256=expected_foundation_sha256,
    )
    _publish_no_replace(
        path,
        _canonical_bytes(validated),
        uid=uid,
        gid=gid,
        mode=0o400,
    )


def retire_fleet_readiness(
    *,
    expected_readiness_sha256: str,
    path: Path = DEFAULT_READINESS_PATH,
    uid: int = 0,
    gid: int = 0,
    expected_foundation_sha256: str | None = None,
    expected_capability_plan_sha256: str | None = None,
    expected_full_canary_plan_sha256: str | None = None,
    retired_at_unix_ms: int | None = None,
) -> Mapping[str, Any]:
    """Retire only the exact installed activation after stable readback."""

    expected = _digest(
        expected_readiness_sha256,
        "producer_fleet_readiness_retirement_invalid",
    )
    raw, before = _stable_read(
        path,
        maximum=MAX_READINESS_BYTES,
        uid=uid,
        gid=gid,
        mode=0o400,
    )
    value = validate_fleet_readiness_for_retirement(
        _strict_json(raw, "producer_fleet_readiness_retirement_invalid"),
        expected_foundation_sha256=expected_foundation_sha256,
        expected_capability_plan_sha256=expected_capability_plan_sha256,
        expected_full_canary_plan_sha256=expected_full_canary_plan_sha256,
    )
    if value["readiness_sha256"] != expected:
        _fail("producer_fleet_readiness_retirement_mismatch")
    retired_at = (
        int(time.time() * 1000)
        if retired_at_unix_ms is None
        else retired_at_unix_ms
    )
    if type(retired_at) is not int or retired_at < value["observed_at_unix_ms"]:
        _fail("producer_fleet_readiness_retirement_invalid")
    try:
        reachable = path.lstat()
        if _identity(reachable) != _identity(before):
            _fail("producer_fleet_readiness_retirement_replaced")
        path.unlink()
        _fsync_directory(path.parent)
    except CapabilityProducerError:
        raise
    except OSError as exc:
        raise CapabilityProducerError(
            "producer_fleet_readiness_retirement_failed"
        ) from exc
    if os.path.lexists(path):
        _fail("producer_fleet_readiness_retirement_failed")
    unsigned = {
        "schema": "muncho-production-capability-fleet-retirement.v1",
        "readiness_sha256": expected,
        "foundation_sha256": value["foundation_sha256"],
        "release_sha": value["release_sha"],
        "capability_plan_sha256": value["capability_plan_sha256"],
        "full_canary_plan_sha256": value["full_canary_plan_sha256"],
        "fixture_sha256": value["fixture_sha256"],
        "run_id": value["run_id"],
        "path": str(path),
        "retired": True,
        "absence_verified": True,
        "retired_at_unix_ms": retired_at,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


PROBE_CATALOG_SCHEMA = "muncho-production-capability-probe-catalog.v1"
OWNER_PREGRANT_INSTALL_SCHEMA = (
    "muncho-production-capability-owner-pregrant-install.v1"
)


def _command_projection(value: Any, *, code: str) -> dict[str, Any]:
    raw = _strict(
        value,
        ("command_id", "command_b64", "command_sha256", "max_uses"),
        code,
    )
    command_id = _safe_id(raw["command_id"], code)
    encoded = raw["command_b64"]
    if not isinstance(encoded, str) or not 1 <= len(encoded) <= 64 * 1024:
        _fail(code)
    try:
        command = base64.b64decode(encoded, validate=True)
    except (ValueError, UnicodeError) as exc:
        raise CapabilityProducerError(code) from exc
    if (
        not command
        or len(command) > 32 * 1024
        or b"\x00" in command
        or _sha256_bytes(command) != _digest(raw["command_sha256"], code)
        or type(raw["max_uses"]) is not int
        or not 1 <= raw["max_uses"] <= 64
    ):
        _fail(code)
    return {
        "command_id": command_id,
        "command_b64": encoded,
        "command_sha256": raw["command_sha256"],
        "max_uses": raw["max_uses"],
    }


def validate_probe_catalog(value: Any) -> dict[str, Any]:
    """Validate fixed canary facts without choosing task meaning or routes."""

    code = "probe_catalog_invalid"
    raw = _strict(
        value,
        (
            "schema",
            "release_sha",
            "capability_plan_sha256",
            "full_canary_plan_sha256",
            "fixture_sha256",
            "run_id",
            "session_id",
            "capability_epoch_sha256",
            "case_ids",
            "workspace",
            "commands",
            "database",
            "bitrix",
            "discord",
            "failure",
            "catalog_sha256",
        ),
        code,
    )
    unsigned = {key: item for key, item in raw.items() if key != "catalog_sha256"}
    if (
        raw["schema"] != PROBE_CATALOG_SCHEMA
        or not isinstance(raw["release_sha"], str)
        or _GIT_SHA_RE.fullmatch(raw["release_sha"]) is None
        or raw["catalog_sha256"] != _sha256_json(unsigned)
    ):
        _fail(code)
    for name in (
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "fixture_sha256",
        "capability_epoch_sha256",
    ):
        _digest(raw[name], code)
    _safe_id(raw["run_id"], code)
    _safe_id(raw["session_id"], code)

    objective_ids = (
        "workspace_continuation",
        "capability_denials",
        "database_reconciliation",
        "bitrix_boundary",
        "discord_routeback",
        "failure_recovery",
    )
    case_ids = _strict(raw["case_ids"], objective_ids, code)
    normalized_cases = {name: _safe_id(case_ids[name], code) for name in objective_ids}
    if len(set(normalized_cases.values())) != len(normalized_cases):
        _fail(code)

    workspace = _strict(
        raw["workspace"],
        (
            "first_path_probe_id",
            "alternate_path_probe_id",
            "worker_restart_checkpoint_step_id",
        ),
        code,
    )
    for value_id in workspace.values():
        _safe_id(value_id, code)

    commands = _strict(raw["commands"], ("allowed", "denied"), code)
    allowed_raw = commands["allowed"]
    denied_raw = commands["denied"]
    if (
        not isinstance(allowed_raw, list)
        or not 1 <= len(allowed_raw) <= 64
        or not isinstance(denied_raw, list)
        or len(denied_raw) != len(DENIAL_KINDS)
    ):
        _fail(code)
    allowed = [_command_projection(item, code=code) for item in allowed_raw]
    denied: list[dict[str, Any]] = []
    for expected_kind, item in zip(DENIAL_KINDS, denied_raw, strict=True):
        row = _strict(item, ("kind", "command"), code)
        if row["kind"] != expected_kind:
            _fail(code)
        denied.append(
            {
                "kind": expected_kind,
                "command": _command_projection(row["command"], code=code),
            }
        )
    hashes = [item["command_sha256"] for item in allowed]
    hashes.extend(item["command"]["command_sha256"] for item in denied)
    if len(set(hashes)) != len(hashes):
        _fail(code)

    database = _strict(
        raw["database"],
        (
            "row_key",
            "idempotency_key",
            "read_probe_id",
            "write_probe_id",
            "lost_response_probe_id",
        ),
        code,
    )
    for value_id in database.values():
        _safe_id(value_id, code)

    bitrix = _strict(
        raw["bitrix"],
        (
            "handoff_id",
            "selected_edge_id",
            "read_operation_id",
            "read_arguments",
            "initial_read_probe_id",
            "readback_probe_id",
            "normalized_equality_excluded_fields",
            "mutation_operation_id",
            "mutation_arguments",
            "mutation_probe_id",
        ),
        code,
    )
    if (
        bitrix["selected_edge_id"] != "operational-edge:bitrix"
        or bitrix["read_operation_id"] != "bitrix.crm.status_list"
        or bitrix["read_arguments"] != BITRIX_CANARY_READ_ARGUMENTS
        or bitrix["normalized_equality_excluded_fields"]
        != ["generated_at_utc"]
        or bitrix["mutation_operation_id"] != "bitrix.crm.lead_add"
        or bitrix["mutation_arguments"] != BITRIX_CANARY_MUTATION_ARGUMENTS
    ):
        _fail(code)
    for name in (
        "handoff_id",
        "selected_edge_id",
        "initial_read_probe_id",
        "readback_probe_id",
        "mutation_probe_id",
    ):
        _safe_id(bitrix[name], code)

    discord = _strict(
        raw["discord"],
        (
            "public_target",
            "public_idempotency_key",
            "private_target_kind",
            "private_probe_id",
        ),
        code,
    )
    target = _strict(
        discord["public_target"],
        ("target_type", "guild_id", "channel_id"),
        code,
    )
    if (
        target["target_type"] not in {"public_channel", "public_thread"}
        or any(
            not isinstance(target[name], str)
            or _SNOWFLAKE_RE.fullmatch(target[name]) is None
            for name in ("guild_id", "channel_id")
        )
        or discord["private_target_kind"] != "dm"
    ):
        _fail(code)
    _safe_id(discord["public_idempotency_key"], code)
    _safe_id(discord["private_probe_id"], code)

    failure = _strict(raw["failure"], ("probes",), code)
    probes = failure["probes"]
    if not isinstance(probes, list) or len(probes) != len(FAILURE_COMPONENTS):
        _fail(code)
    normalized_probes: list[dict[str, Any]] = []
    for expected_component, item in zip(FAILURE_COMPONENTS, probes, strict=True):
        row = _strict(
            item,
            ("component", "failure_id", "alternative_available", "alternative_id"),
            code,
        )
        if (
            row["component"] != expected_component
            or type(row["alternative_available"]) is not bool
            or (
                row["alternative_available"]
                and not isinstance(row["alternative_id"], str)
            )
            or (
                not row["alternative_available"]
                and row["alternative_id"] is not None
            )
        ):
            _fail(code)
        _safe_id(row["failure_id"], code)
        if row["alternative_id"] is not None:
            _safe_id(row["alternative_id"], code)
        normalized_probes.append(dict(row))

    return {
        **unsigned,
        "case_ids": normalized_cases,
        "commands": {"allowed": allowed, "denied": denied},
        "failure": {"probes": normalized_probes},
        "catalog_sha256": raw["catalog_sha256"],
    }


def build_probe_catalog(
    *,
    release_sha: str,
    capability_plan_sha256: str,
    full_canary_plan_sha256: str,
    fixture_sha256: str,
    run_id: str,
    session_id: str,
    capability_epoch_sha256: str,
    case_ids: Mapping[str, Any],
    workspace: Mapping[str, Any],
    commands: Mapping[str, Any],
    database: Mapping[str, Any],
    bitrix: Mapping[str, Any],
    discord: Mapping[str, Any],
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema": PROBE_CATALOG_SCHEMA,
        "release_sha": release_sha,
        "capability_plan_sha256": capability_plan_sha256,
        "full_canary_plan_sha256": full_canary_plan_sha256,
        "fixture_sha256": fixture_sha256,
        "run_id": run_id,
        "session_id": session_id,
        "capability_epoch_sha256": capability_epoch_sha256,
        "case_ids": copy.deepcopy(dict(case_ids)),
        "workspace": copy.deepcopy(dict(workspace)),
        "commands": copy.deepcopy(dict(commands)),
        "database": copy.deepcopy(dict(database)),
        "bitrix": copy.deepcopy(dict(bitrix)),
        "discord": copy.deepcopy(dict(discord)),
        "failure": copy.deepcopy(dict(failure)),
    }
    return validate_probe_catalog(
        {**unsigned, "catalog_sha256": _sha256_json(unsigned)}
    )


def publish_probe_catalog(
    value: Mapping[str, Any],
    *,
    path: Path = DEFAULT_PROBE_CATALOG_PATH,
    uid: int = 0,
    gid: int,
) -> None:
    catalog = validate_probe_catalog(value)
    _publish_no_replace(
        path,
        _canonical_bytes(catalog),
        uid=uid,
        gid=gid,
        mode=0o440,
        parent_mode=0o750,
    )


def validate_owner_pregrant(
    value: Any,
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the real pre-run SSHSIG grant; never invent future plan fields."""

    from gateway import canonical_capability_canary_e2e as e2e

    catalog_value = validate_probe_catalog(catalog)
    if (
        catalog_value["fixture_sha256"] != fixture_sha256
        or catalog_value["run_id"] != fixture.get("run_id")
        or catalog_value["release_sha"] != fixture.get("release_sha")
    ):
        _fail("owner_pregrant_invalid")
    try:
        payload = e2e._signed_payload(
            value,
            slot="workspace_owner",
            role="owner",
            payload_schema=e2e.PLAN_APPROVAL_SCHEMA,
            fields=(
                "approval_id",
                "owner_id",
                "session_id",
                "capability_epoch_sha256",
                "command_sha256s",
                "ttl_seconds",
                "max_uses",
            ),
            fixture=fixture,
            fixture_sha256=fixture_sha256,
            code="owner_pregrant_invalid",
        )
    except e2e.CapabilityCanaryEvidenceError as exc:
        raise CapabilityProducerError("owner_pregrant_invalid") from exc
    commands = sorted(
        item["command_sha256"]
        for item in catalog_value["commands"]["allowed"]
    )
    supplied = payload["command_sha256s"]
    if (
        payload["owner_id"] != PRODUCTION_OWNER_ID
        or payload["session_id"] != catalog_value["session_id"]
        or payload["capability_epoch_sha256"]
        != catalog_value["capability_epoch_sha256"]
        or not isinstance(supplied, list)
        or supplied != commands
        or len(set(supplied)) != len(supplied)
        or type(payload["ttl_seconds"]) is not int
        or not 1 <= payload["ttl_seconds"] <= 8 * 60 * 60
        or type(payload["max_uses"]) is not int
        or not len(commands) <= payload["max_uses"] <= 64
    ):
        _fail("owner_pregrant_invalid")
    return {
        "receipt": copy.deepcopy(dict(value)),
        "payload": payload,
        "grant_sha256": _sha256_bytes(_canonical_bytes(value)),
        "catalog_sha256": catalog_value["catalog_sha256"],
    }


def install_owner_pregrant(
    value: Mapping[str, Any],
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
    catalog: Mapping[str, Any],
    path: Path = DEFAULT_OWNER_GRANT_PATH,
    uid: int = 0,
    gid: int = 0,
    installed_at_unix_ms: int | None = None,
) -> Mapping[str, Any]:
    validated = validate_owner_pregrant(
        value,
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        catalog=catalog,
    )
    raw = _canonical_bytes(validated["receipt"])
    _publish_no_replace(path, raw, uid=uid, gid=gid, mode=0o400)
    unsigned = {
        "schema": OWNER_PREGRANT_INSTALL_SCHEMA,
        "owner_id": PRODUCTION_OWNER_ID,
        "grant_path": str(path),
        "grant_sha256": validated["grant_sha256"],
        "catalog_sha256": validated["catalog_sha256"],
        "fixture_sha256": fixture_sha256,
        "installed_at_unix_ms": (
            int(time.time() * 1000)
            if installed_at_unix_ms is None
            else installed_at_unix_ms
        ),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def bind_owner_pregrant_to_active_plan(
    *,
    validated_pregrant: Mapping[str, Any],
    catalog: Mapping[str, Any],
    active_plan: Mapping[str, Any],
    session_id: str,
    capability_epoch_sha256: str,
    consumed_command_sha256s: Sequence[str],
) -> Mapping[str, Any]:
    """Mechanical writer-side binding after GPT has authored an active plan."""

    code = "owner_pregrant_plan_binding_invalid"
    if set(validated_pregrant) != {
        "receipt",
        "payload",
        "grant_sha256",
        "catalog_sha256",
    }:
        _fail(code)
    catalog_value = validate_probe_catalog(catalog)
    payload = validated_pregrant["payload"]
    plan = _strict(
        active_plan,
        ("case_id", "plan_id", "revision", "state", "session_id", "capability_epoch_sha256"),
        code,
    )
    allowed = sorted(
        item["command_sha256"]
        for item in catalog_value["commands"]["allowed"]
    )
    consumed = list(consumed_command_sha256s)
    if (
        not isinstance(payload, Mapping)
        or validated_pregrant["catalog_sha256"] != catalog_value["catalog_sha256"]
        or payload.get("session_id") != session_id
        or payload.get("capability_epoch_sha256") != capability_epoch_sha256
        or plan["session_id"] != session_id
        or plan["capability_epoch_sha256"] != capability_epoch_sha256
        or plan["case_id"] != catalog_value["case_ids"]["workspace_continuation"]
        or plan["state"] not in {"active", "in_progress"}
        or type(plan["revision"]) is not int
        or plan["revision"] < 1
        or sorted(payload.get("command_sha256s") or []) != allowed
        or sorted(consumed) != allowed
        or len(set(consumed)) != len(consumed)
    ):
        _fail(code)
    _safe_id(plan["plan_id"], code)
    _digest(capability_epoch_sha256, code)
    unsigned = {
        "schema": "muncho-production-capability-owner-pregrant-plan-binding.v1",
        "approval_id": payload["approval_id"],
        "owner_id": PRODUCTION_OWNER_ID,
        "grant_sha256": validated_pregrant["grant_sha256"],
        "catalog_sha256": catalog_value["catalog_sha256"],
        "session_id": session_id,
        "capability_epoch_sha256": capability_epoch_sha256,
        "case_id": plan["case_id"],
        "terminal_plan_id": plan["plan_id"],
        "terminal_plan_revision": plan["revision"],
        "allowed_command_sha256s": allowed,
        "consumed_command_sha256s": sorted(consumed),
        "owner_identity_source": "pre_staged_sshsig_not_runtime_user",
    }
    return {**unsigned, "binding_sha256": _sha256_json(unsigned)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate production capability-canary producer readiness"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-readiness")
    validate.add_argument("--path", type=Path, default=DEFAULT_READINESS_PATH)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--config", type=Path, required=True)
    serve.add_argument("--foundation", type=Path, required=True)
    serve.add_argument(
        "--owner-public-key-hex-file",
        type=Path,
        required=True,
    )
    serve.add_argument(
        "--owner-public-key-source-sha256-file",
        type=Path,
        required=True,
    )
    return parser


def _serve_role(args: argparse.Namespace) -> None:
    from gateway.canonical_capability_canary_producer_units import (
        build_role_native_collector,
        producer_private_key_projection_path,
        producer_public_key_path,
        producer_socket_path,
    )

    config = load_producer_config(args.config)
    unit = PRODUCER_SERVICE_UNITS[config.role]
    credential_root = Path("/run/credentials") / unit
    expected = {
        "config": DEFAULT_CONFIG_ROOT / f"{config.role}.json",
        "foundation": credential_root / "producer-foundation",
        "owner_public_key_hex_file": credential_root / "owner-public-key-hex",
        "owner_public_key_source_sha256_file": (
            credential_root / "owner-public-key-source-sha256"
        ),
    }
    if (
        args.config != expected["config"]
        or args.foundation != expected["foundation"]
        or args.owner_public_key_hex_file
        != expected["owner_public_key_hex_file"]
        or args.owner_public_key_source_sha256_file
        != expected["owner_public_key_source_sha256_file"]
        or config.service_unit != unit
        or config.socket_path != producer_socket_path(config.role)
        or config.private_key_path
        != producer_private_key_projection_path(config.role)
        or config.public_key_path != producer_public_key_path(config.role)
    ):
        _fail("producer_service_projection_invalid")
    public_hex = _load_root_pin(
        args.owner_public_key_hex_file,
        pattern=re.compile(r"^[0-9a-f]{64}$"),
        code="producer_owner_public_key_pin_invalid",
    )
    source_sha256 = _load_root_pin(
        args.owner_public_key_source_sha256_file,
        pattern=_SHA256_RE,
        code="producer_owner_public_key_pin_invalid",
    )
    foundation = load_producer_foundation(
        args.foundation,
        pinned_owner_public_key_ed25519_hex=public_hex,
        pinned_owner_public_key_source_sha256=source_sha256,
        uid=0,
        gid=0,
    )
    validate_producer_config_binding(config, foundation)
    collector = build_role_native_collector(config, foundation)
    producer = RoleReceiptProducer(config, native_collector=collector)
    server = CapabilityProducerServer(producer)
    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    server.serve_forever(should_stop=lambda: stopping)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "serve":
            _serve_role(args)
            return 0
        readiness = load_fleet_readiness(args.path)
    except CapabilityProducerError as exc:
        print(
            json.dumps(
                {
                    "schema": PRODUCER_FLEET_READINESS_SCHEMA,
                    "ok": False,
                    "failure_code": exc.code,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(
        json.dumps(
            {
                "schema": PRODUCER_FLEET_READINESS_SCHEMA,
                "ok": True,
                "readiness_sha256": readiness["readiness_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


__all__ = [
    "AUTHORITY_ALGORITHMS",
    "AUTHORITY_ROLES",
    "BITRIX_CANARY_MUTATION_ARGUMENTS",
    "BITRIX_CANARY_READ_ARGUMENTS",
    "BITRIX_OPERATIONAL_EDGE_ASSET_NAMES",
    "BITRIX_OPERATIONAL_EDGE_CONFIG_PATH",
    "BITRIX_OPERATIONAL_EDGE_SERVICE_GROUP",
    "BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT",
    "BITRIX_OPERATIONAL_EDGE_SERVICE_USER",
    "BITRIX_OPERATIONAL_EDGE_SOCKET_GROUP",
    "BITRIX_OPERATIONAL_EDGE_STAGING_PROTOCOL",
    "BITRIX_OPERATIONAL_EDGE_TRUST_PATH",
    "BITRIX_OPERATIONAL_EDGE_UNIT_PATH",
    "CapabilityProducerError",
    "CapabilityProducerServer",
    "BitrixOperationalEdgeNativeCollector",
    "BitrixWriterNativeCollector",
    "DEFAULT_FOUNDATION_PATH",
    "DEFAULT_OWNER_PUBLIC_KEY_HEX_PIN_PATH",
    "DEFAULT_OWNER_PUBLIC_KEY_SOURCE_SHA256_PIN_PATH",
    "DEFAULT_OWNER_GRANT_PATH",
    "DEFAULT_READINESS_PATH",
    "ENDPOINT_ROLES",
    "InstalledProducerFoundation",
    "NATIVE_EVIDENCE_SCHEMA",
    "NativeEvidenceBinding",
    "PeerIdentity",
    "PRODUCER_ENDPOINT_READINESS_SCHEMA",
    "PRODUCER_ENDPOINT_ACTIVATION_SCHEMA",
    "PRODUCER_FOUNDATION_SCHEMA",
    "PRODUCER_FOUNDATION_SSHSIG_NAMESPACE",
    "PRODUCER_ACTIVATION_SCHEMA",
    "PRODUCER_FLEET_READINESS_SCHEMA",
    "PRODUCER_REQUEST_SCHEMA",
    "PRODUCER_CONFIG_SCHEMA",
    "PRODUCER_SERVICE_UNITS",
    "ProductionFleetActivation",
    "ProductionReceiptPump",
    "PRODUCTION_PUMP_SLOTS",
    "PRODUCTION_PRE_CLEANUP_PUMP_SLOTS",
    "ProducerConfig",
    "ProducerEndpointClient",
    "RECEIPT_SLOTS",
    "RoleReceiptProducer",
    "SLOT_NATIVE_BINDING_KINDS",
    "SLOT_FILENAME",
    "SLOT_ROLE",
    "SystemctlProducerServiceStateReader",
    "activate_fleet_readiness",
    "activate_production_fleet",
    "build_fleet_activation",
    "build_fleet_readiness_for_plans",
    "build_fleet_readiness",
    "build_probe_catalog",
    "bind_owner_pregrant_to_active_plan",
    "install_owner_pregrant",
    "load_fleet_readiness",
    "load_installed_producer_foundation",
    "load_producer_foundation",
    "load_producer_config",
    "producer_foundation_sha256",
    "producer_foundation_signature_payload",
    "project_pinned_owner_public_key_source",
    "publish_producer_foundation",
    "publish_fleet_readiness",
    "publish_probe_catalog",
    "production_endpoint_clients",
    "retire_fleet_readiness",
    "seal_producer_foundation",
    "validate_owner_pregrant",
    "validate_bitrix_operational_edge_contract",
    "validate_discord_edge_evidence_contract",
    "validate_probe_catalog",
    "validate_producer_config_binding",
    "validate_producer_foundation",
    "validate_fleet_readiness",
    "validate_fleet_readiness_for_retirement",
    "verify_role_receipt",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
