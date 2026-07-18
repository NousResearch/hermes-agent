#!/usr/bin/env python3
"""Operational split-UID services for the Muncho passkey-v2 owner gate.

The public web process can only render an existing exact request, return its
WebAuthn options, and forward one assertion.  The authority process is the
only WebAuthn verifier and receipt signer.  The executor process is the only
process with Compute network access and it accepts a mutation only from the
authority UID, after validating both a signed single-use receipt and the
separate root-owned topology/IAM activation seal.

Operation names below are a fixed wire protocol, not semantic routing.  No
message, keyword, task, or intent classifier exists in this module.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import http.client
import json
import math
import os
import re
import secrets
import socket
import ssl
import stat
import struct
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import passkey_v2_storage_growth as storage
from scripts.canary import production_cutover_passkey as production_cutover
from scripts.canary import owner_gate_firewall_readiness as firewall
from scripts.canary import storage_growth_evidence as growth_evidence
from scripts.canary import storage_growth_contract as growth_contract
from scripts.canary import storage_growth_trusted_collector as trusted_collector
from scripts.canary import trusted_signer_provisioning as signer_provisioning
from scripts.canary.passkey_v2_signer import ReceiptSigner
from scripts.canary.passkey_v2_sqlite import (
    PasskeyV2AuthorityDatabase,
    PasskeyV2ExecutorDatabase,
    PasskeyV2SqliteDenied,
)


WEB_CONFIG = Path("/etc/muncho-owner-gate/web.json")
AUTHORITY_CONFIG = Path("/etc/muncho-owner-gate/authority.json")
EXECUTOR_CONFIG = Path("/etc/muncho-owner-gate/executor.json")
ACTIVATION_SEAL = Path(
    "/etc/muncho-owner-gate/storage-executor-enabled"
)
AUTHORITY_SOCKET = Path(storage.AUTHORITY_SOCKET)
EXECUTOR_SOCKET = Path(storage.EXECUTOR_SOCKET)
AUTHORITY_DB = Path(storage.AUTHORITY_DB)
EXECUTOR_DB = Path(storage.EXECUTOR_DB)
FIREWALL_RULES = Path("/etc/muncho-owner-gate/metadata-firewall.rules")
FIREWALL_READINESS_RECEIPT = Path(
    "/run/muncho-owner-gate/metadata-firewall-ready.json"
)
BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")
CLOUD_OBSERVATION_PUBLIC_KEY = Path(
    "/etc/muncho-owner-gate/public/cloud-observation-attestation.pub"
)
HOST_OBSERVATION_PUBLIC_KEY = Path(
    "/etc/muncho-owner-gate/public/host-observation-attestation.pub"
)
CUTOVER_TRUST_BUNDLE = production_cutover.CUTOVER_TRUST_BUNDLE_PATH

# The public web worker is the only expected caller of the authority socket,
# and the authority worker is the only expected caller of the executor socket.
# Keep the framing deadline short and the in-process fan-out bounded so one
# slow local client cannot serialize either security service indefinitely.
SERVICE_FRAME_TIMEOUT_SECONDS = 2.0
SERVICE_CONNECT_TIMEOUT_SECONDS = 5.0
# These deadlines are fixed by the operation, never selected by an untrusted
# caller.  The executor's mutation path may legitimately include three
# Compute reconciliation polls (90 * 2 seconds each, one stage per call), so
# its response window must outlive one complete stage.  Read-only and local
# SQLite operations remain tightly bounded.
SERVICE_OPERATION_RESPONSE_TIMEOUT_SECONDS = {
    "health": 10.0,
    "render": 10.0,
    "options": 10.0,
    "verify": 30.0,
    "create_request": 30.0,
    "consume": 30.0,
    "preflight": 30.0,
    "execute": 240.0,
    "terminal": 10.0,
    "context": 10.0,
    "verify_observation": 30.0,
    "observation_request": 10.0,
    "reconcile_read_only": 30.0,
    "attest_cloud_observation": 60.0,
}
WEB_READINESS_REFRESH_SECONDS = 5.0
WEB_READINESS_MAX_STALE_SECONDS = 15.0
SERVICE_MAX_CONNECTIONS = 8
SERVICE_MAX_CONNECTIONS_PER_UID = 4
SERVICE_GLOBAL_RATE_PER_SECOND = 16.0
SERVICE_GLOBAL_BURST = 16.0
SERVICE_UID_RATE_PER_SECOND = 8.0
SERVICE_UID_BURST = 8.0

_DIRECT_IAM_PROJECT_PERMISSIONS = (
    "iam.roles.get",
    "iam.serviceAccountKeys.list",
    "iam.serviceAccounts.get",
    "iam.serviceAccounts.getIamPolicy",
    "resourcemanager.projects.get",
    "resourcemanager.projects.getIamPolicy",
)
_DIRECT_IAM_ANCESTOR_PERMISSIONS = (
    "iam.roles.get",
    "resourcemanager.folders.get",
    "resourcemanager.folders.getIamPolicy",
    "resourcemanager.organizations.get",
    "resourcemanager.organizations.getIamPolicy",
)
_EXECUTOR_CONFIG_FIELDS = frozenset({
    "api_host",
    "api_private_vip_range",
    "cloud_observation_public_key",
    "cloud_observation_public_key_id",
    "cloud_observation_public_key_sha256",
    "direct_iam_allowed_owner_gate_impersonators",
    "direct_iam_ancestor_read_permissions",
    "direct_iam_ancestor_read_role",
    "direct_iam_ancestor_read_role_description",
    "direct_iam_ancestor_read_role_title",
    "direct_iam_external_gcp_admin_trust_root",
    "direct_iam_fixed_api_hosts",
    "direct_iam_metadata_oauth_scopes",
    "direct_iam_mutation_activation_seal",
    "direct_iam_mutation_activation_seal_present",
    "direct_iam_mutation_binding_member",
    "direct_iam_mutation_binding_present",
    "direct_iam_mutation_condition",
    "direct_iam_mutation_role",
    "direct_iam_mutation_role_description",
    "direct_iam_mutation_role_title",
    "direct_iam_owner_gate_user_managed_key_inventory",
    "direct_iam_project_number",
    "direct_iam_project_read_permissions",
    "direct_iam_project_read_role",
    "direct_iam_project_read_role_description",
    "direct_iam_project_read_role_title",
    "direct_iam_resource_ancestor_chain",
    "direct_iam_runtime_instance_numeric_id",
    "direct_iam_runtime_service_account_email",
    "direct_iam_runtime_service_account_unique_id",
    "direct_iam_signed_readiness_required",
    "direct_iam_target_service_account_email",
    "direct_iam_target_service_account_unique_id",
    "executor_database",
    "expected_disk_id",
    "expected_instance_id",
    "firewall_readiness_max_age_seconds",
    "firewall_readiness_receipt",
    "firewall_readiness_receipt_gid",
    "firewall_readiness_receipt_mode",
    "firewall_readiness_receipt_uid",
    "firewall_readiness_requires_current_boot_id",
    "firewall_readiness_requires_rules_source_sha256",
    "host_observation_public_key",
    "host_observation_public_key_id",
    "host_observation_public_key_sha256",
    "journal_root",
    "metadata_host",
    "mutation_enable_seal",
    "mutation_enable_seal_gid",
    "mutation_enable_seal_mode",
    "mutation_enable_seal_uid",
    "project",
    "receipt_public_key",
    "receipt_public_key_mode",
    "receipt_public_key_owner",
    "receipt_public_key_sha256",
    "schema",
    "signed_authorization_receipt_required",
    "target_boot_device",
    "target_disk",
    "target_instance",
    "topology_iam_readiness_seal_required_for_mutation_only",
    "zone",
})

WEB_UID = storage.OWNER_GATE_WEB_UID
AUTHORITY_UID = storage.OWNER_GATE_AUTHORITY_UID
EXECUTOR_UID = storage.OWNER_GATE_EXECUTOR_UID
EXECUTOR_GID = storage.OWNER_GATE_EXECUTOR_UID

SERVICE_FRAME_SCHEMA = "muncho-passkey-v2-service-frame.v1"
SERVICE_RESPONSE_SCHEMA = "muncho-passkey-v2-service-response.v1"
ACTIVATION_SEAL_SCHEMA = (
    "muncho-owner-gate-topology-iam-readiness-seal.v2"
)
ACTIVATION_RELEASE_LINEAGE_SCHEMA = (
    "muncho-owner-gate-portable-release-lineage.v1"
)
WEB_VERIFY_SCHEMA = "muncho-passkey-v2-web-verify.v1"
MAX_FRAME_BYTES = 1024 * 1024
MAX_HTTP_BODY_BYTES = 256 * 1024
MAX_CONFIG_BYTES = 64 * 1024
MAX_SEAL_BYTES = 16 * 1024
MAX_FUTURE_SKEW_SECONDS = 300
MAX_HTTP_RESPONSE_BYTES = 1024 * 1024
COMPUTE_POLL_ATTEMPTS = 90

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_CSRF = re.compile(r"^[A-Za-z0-9_-]{43}$")
_APPROVAL_PATH = re.compile(r"^/approve/([A-Za-z0-9_-]{32,64})$")
_VIEW_PATH = re.compile(r"^/approve/([A-Za-z0-9_-]{32,64})/view$")
_OPTIONS_PATH = re.compile(
    r"^/approve/([A-Za-z0-9_-]{32,64})/options$"
)
_VERIFY_PATH = re.compile(
    r"^/approve/([A-Za-z0-9_-]{32,64})/verify$"
)

_SEAL_FIELDS = frozenset({
    "schema",
    "release_revision",
    "foundation_plan_sha256",
    "package_sha256",
    "cloud_topology_receipt_sha256",
    "host_security_smoke_receipt_sha256",
    "iam_repreflight_receipt_sha256",
    "owner_gate_vm_numeric_id",
    "target_instance_numeric_id",
    "target_disk_numeric_id",
    "created_at_unix",
    "authorization_record_complete",
    "verified_release_lineage",
    "evidence_file_sha256",
    "activation_installed",
    "cloud_mutation_performed",
    "seal_sha256",
})

_ACTIVATION_RELEASE_LINEAGE_FIELDS = frozenset({
    "schema",
    "release_revision",
    "source_tree_oid",
    "package_inventory_sha256",
    "release_trust_manifest_sha256",
    "release_trust_public_key_sha256",
    "direct_iam_identity_authority_sha256",
    "pre_foundation_authority_sha256",
    "foundation_apply_receipt_sha256",
    "foundation_owner_reauthentication_receipt_sha256",
    "activation_owner_reauthentication_receipt_sha256",
    "project_ancestry_evidence_sha256",
    "project_ancestry_chain_sha256",
    "resource_ancestor_chain",
    "inert_preflight_receipt_sha256",
    "post_iam_preflight_receipt_sha256",
    "lineage_sha256",
})

_ACTIVATION_EVIDENCE_NAMES = frozenset({
    "network-evidence.json",
    "inert-production-ingress-observation.json",
    "inert-cloud-observation.json",
    "inert-host-observation.json",
    "inert-preflight.json",
    "post-iam-production-ingress-observation.json",
    "post-iam-cloud-observation.json",
    "post-iam-host-observation.json",
    "post-iam-preflight.json",
    "activation-owner-reauthentication-receipt.json",
})


class PasskeyV2ServiceError(RuntimeError):
    """Stable, secret-free owner-gate service failure."""


def _strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, item in items:
        if name in result:
            raise PasskeyV2ServiceError("passkey_v2_service_duplicate_json_key")
        result[name] = item
    return result


def _reject_number(_value: str) -> None:
    raise PasskeyV2ServiceError("passkey_v2_service_number_invalid")


def decode_strict_json(raw: bytes, *, maximum: int) -> Mapping[str, Any]:
    """Decode browser/config JSON without accepting ambiguous JSON syntax."""

    if not isinstance(raw, bytes) or not raw or len(raw) > maximum:
        raise PasskeyV2ServiceError("passkey_v2_service_json_size_invalid")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_number,
            parse_float=_reject_number,
        )
        protocol.canonical_json_bytes(value)
    except PasskeyV2ServiceError:
        raise
    except (
        protocol.PasskeyV2ProtocolError,
        UnicodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        raise PasskeyV2ServiceError("passkey_v2_service_json_invalid") from None
    if not isinstance(value, Mapping):
        raise PasskeyV2ServiceError("passkey_v2_service_json_object_required")
    return dict(value)


def _read_regular_file(
    path: Path,
    *,
    maximum: int,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    expected_mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
    if not isinstance(path, Path) or not path.is_absolute():
        raise PasskeyV2ServiceError("passkey_v2_service_file_path_invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        before = path.lstat()
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_nlink != 1
            or opened.st_size < 1
            or opened.st_size > maximum
            or (expected_uid is not None and opened.st_uid != expected_uid)
            or (expected_gid is not None and opened.st_gid != expected_gid)
            or (
                expected_mode is not None
                and stat.S_IMODE(opened.st_mode) != expected_mode
            )
        ):
            raise PasskeyV2ServiceError(
                "passkey_v2_service_file_identity_invalid"
            )
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise PasskeyV2ServiceError(
                    "passkey_v2_service_file_read_invalid"
                )
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
            raise PasskeyV2ServiceError("passkey_v2_service_file_changed")
        return b"".join(chunks), opened
    except OSError as exc:
        raise PasskeyV2ServiceError(
            "passkey_v2_service_file_unavailable"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _load_config(
    path: Path,
    *,
    exact_path: Path,
    schema: str,
    fields: frozenset[str],
) -> Mapping[str, Any]:
    if path != exact_path:
        raise PasskeyV2ServiceError("passkey_v2_service_config_path_invalid")
    raw, metadata = _read_regular_file(path, maximum=MAX_CONFIG_BYTES)
    if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) not in {0o444, 0o640}:
        raise PasskeyV2ServiceError("passkey_v2_service_config_identity_invalid")
    value = decode_strict_json(raw, maximum=MAX_CONFIG_BYTES)
    if set(value) != fields or value.get("schema") != schema:
        raise PasskeyV2ServiceError("passkey_v2_service_config_invalid")
    return value


def validate_activation_seal(
    value: Any,
    *,
    expected_release_revision: str,
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate the root-authored, non-WebAuthn activation authority."""

    if (
        not isinstance(value, Mapping)
        or set(value) != _SEAL_FIELDS
        or value.get("schema") != ACTIVATION_SEAL_SCHEMA
    ):
        raise PasskeyV2ServiceError("passkey_v2_activation_seal_invalid")
    seal = dict(value)
    if (
        _REVISION.fullmatch(str(expected_release_revision)) is None
        or seal.get("release_revision") != expected_release_revision
    ):
        raise PasskeyV2ServiceError("passkey_v2_activation_seal_stale")
    for name in (
        "foundation_plan_sha256",
        "package_sha256",
        "cloud_topology_receipt_sha256",
        "host_security_smoke_receipt_sha256",
        "iam_repreflight_receipt_sha256",
    ):
        if _SHA256.fullmatch(str(seal.get(name))) is None:
            raise PasskeyV2ServiceError("passkey_v2_activation_seal_invalid")
    lineage = seal.get("verified_release_lineage")
    evidence = seal.get("evidence_file_sha256")
    if (
        seal.get("authorization_record_complete") is not True
        or seal.get("activation_installed") is not True
        or seal.get("cloud_mutation_performed") is not False
        or not isinstance(lineage, Mapping)
        or set(lineage) != _ACTIVATION_RELEASE_LINEAGE_FIELDS
        or lineage.get("schema") != ACTIVATION_RELEASE_LINEAGE_SCHEMA
        or lineage.get("release_revision") != expected_release_revision
        or _REVISION.fullmatch(str(lineage.get("source_tree_oid"))) is None
        or not isinstance(evidence, Mapping)
        or set(evidence) != _ACTIVATION_EVIDENCE_NAMES
    ):
        raise PasskeyV2ServiceError("passkey_v2_activation_seal_invalid")
    for name in (
        "package_inventory_sha256",
        "release_trust_manifest_sha256",
        "release_trust_public_key_sha256",
        "direct_iam_identity_authority_sha256",
        "pre_foundation_authority_sha256",
        "foundation_apply_receipt_sha256",
        "foundation_owner_reauthentication_receipt_sha256",
        "activation_owner_reauthentication_receipt_sha256",
        "project_ancestry_evidence_sha256",
        "project_ancestry_chain_sha256",
        "inert_preflight_receipt_sha256",
        "post_iam_preflight_receipt_sha256",
    ):
        if _SHA256.fullmatch(str(lineage.get(name))) is None:
            raise PasskeyV2ServiceError("passkey_v2_activation_seal_invalid")
    ancestors = lineage.get("resource_ancestor_chain")
    if (
        not isinstance(ancestors, list)
        or not ancestors
        or any(
            not isinstance(item, str) or not item or item.strip() != item
            for item in ancestors
        )
        or len(set(ancestors)) != len(ancestors)
        or lineage.get("post_iam_preflight_receipt_sha256")
        != seal.get("iam_repreflight_receipt_sha256")
    ):
        raise PasskeyV2ServiceError("passkey_v2_activation_seal_invalid")
    lineage_unsigned = {
        key: item for key, item in lineage.items() if key != "lineage_sha256"
    }
    if lineage.get("lineage_sha256") != protocol.sha256_json(
        lineage_unsigned
    ):
        raise PasskeyV2ServiceError("passkey_v2_activation_seal_tampered")
    if any(_SHA256.fullmatch(str(item)) is None for item in evidence.values()):
        raise PasskeyV2ServiceError("passkey_v2_activation_seal_invalid")
    if (
        _NUMERIC_ID.fullmatch(str(seal.get("owner_gate_vm_numeric_id"))) is None
        or seal.get("target_instance_numeric_id") != storage.VM_INSTANCE_ID
        or seal.get("target_disk_numeric_id") != storage.DISK_ID
        or type(seal.get("created_at_unix")) is not int
        or seal["created_at_unix"] < 1
        or seal["created_at_unix"] > now_unix + MAX_FUTURE_SKEW_SECONDS
    ):
        raise PasskeyV2ServiceError("passkey_v2_activation_seal_invalid")
    unsigned = {key: item for key, item in seal.items() if key != "seal_sha256"}
    if seal.get("seal_sha256") != protocol.sha256_json(unsigned):
        raise PasskeyV2ServiceError("passkey_v2_activation_seal_tampered")
    return seal


def read_activation_seal(
    *,
    expected_release_revision: str,
    now_unix: int,
    path: Path = ACTIVATION_SEAL,
) -> Mapping[str, Any]:
    """Read the optional source file only at mutation/recovery time.

    The file stays absent during inert service smoke.  Once activated, it is
    root-owned and group-readable only by the executor identity.
    """

    raw, _metadata = _read_regular_file(
        path,
        maximum=MAX_SEAL_BYTES,
        expected_uid=0,
        expected_gid=EXECUTOR_GID,
        expected_mode=0o440,
    )
    value = protocol.decode_canonical_json(raw)
    return validate_activation_seal(
        value,
        expected_release_revision=expected_release_revision,
        now_unix=now_unix,
    )


def activation_seal_status(
    *, expected_release_revision: str, now_unix: int
) -> Mapping[str, Any]:
    try:
        seal = read_activation_seal(
            expected_release_revision=expected_release_revision,
            now_unix=now_unix,
        )
    except PasskeyV2ServiceError as exc:
        return {
            "active": False,
            "status": str(exc),
            "seal_sha256": None,
        }
    return {
        "active": True,
        "status": "ready",
        "seal_sha256": seal["seal_sha256"],
    }


def _read_current_boot_id(path: Path = BOOT_ID_FILE) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        state = os.fstat(descriptor)
        if not stat.S_ISREG(state.st_mode) or state.st_uid != 0:
            raise PasskeyV2ServiceError("passkey_v2_firewall_boot_id_invalid")
        raw = os.read(descriptor, 128)
        if len(raw) >= 128 or os.read(descriptor, 1):
            raise PasskeyV2ServiceError("passkey_v2_firewall_boot_id_invalid")
        value = raw.decode("ascii", errors="strict").strip()
    except (OSError, UnicodeError) as exc:
        raise PasskeyV2ServiceError("passkey_v2_firewall_boot_id_invalid") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    parts = value.split("-")
    if [len(item) for item in parts] != [8, 4, 4, 4, 12] or any(
        character not in "0123456789abcdef"
        for item in parts
        for character in item
    ):
        raise PasskeyV2ServiceError("passkey_v2_firewall_boot_id_invalid")
    return value


def validate_firewall_readiness(
    config: Mapping[str, Any],
    *,
    now_unix: int,
    receipt_path: Path = FIREWALL_READINESS_RECEIPT,
    rules_path: Path = FIREWALL_RULES,
    boot_id_path: Path = BOOT_ID_FILE,
) -> Mapping[str, Any]:
    """Validate root-authored current-boot metadata isolation evidence."""

    if (
        config.get("firewall_readiness_receipt") != str(receipt_path)
        or config.get("firewall_readiness_receipt_uid") != 0
        or config.get("firewall_readiness_receipt_gid") != EXECUTOR_GID
        or config.get("firewall_readiness_receipt_mode") != "0440"
        or config.get("firewall_readiness_max_age_seconds") != 60
        or config.get("firewall_readiness_requires_current_boot_id") is not True
        or config.get("firewall_readiness_requires_rules_source_sha256") is not True
    ):
        raise PasskeyV2ServiceError("passkey_v2_firewall_config_invalid")
    rules_raw, rules_state = _read_regular_file(rules_path, maximum=64 * 1024)
    if rules_state.st_uid != 0 or stat.S_IMODE(rules_state.st_mode) & 0o022:
        raise PasskeyV2ServiceError("passkey_v2_firewall_rules_invalid")
    expected_rules_sha256 = hashlib.sha256(rules_raw).hexdigest()
    receipt_raw, _state = _read_regular_file(
        receipt_path,
        maximum=MAX_SEAL_BYTES,
        expected_uid=0,
        expected_gid=EXECUTOR_GID,
        expected_mode=0o440,
    )
    receipt = protocol.decode_canonical_json(receipt_raw)
    fields = {
        "schema", "backend", "boot_id", "rules_source_sha256",
        "live_projection_sha256", "executor_uid", "root_admin_metadata_allowed",
        "other_unprivileged_uids_blocked", "web_uid_blocked",
        "authority_uid_blocked", "observed_at_unix", "ready", "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != fields:
        raise PasskeyV2ServiceError("passkey_v2_firewall_receipt_invalid")
    value = dict(receipt)
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    observed = value.get("observed_at_unix")
    if (
        value.get("schema") != firewall.READINESS_SCHEMA
        or value.get("backend") != "iptables-nft"
        or value.get("boot_id") != _read_current_boot_id(boot_id_path)
        or value.get("rules_source_sha256") != expected_rules_sha256
        or _SHA256.fullmatch(str(value.get("live_projection_sha256"))) is None
        or value.get("executor_uid") != EXECUTOR_UID
        or value.get("root_admin_metadata_allowed") is not True
        or value.get("other_unprivileged_uids_blocked") is not True
        or value.get("web_uid_blocked") is not True
        or value.get("authority_uid_blocked") is not True
        or type(observed) is not int
        or not 0 <= now_unix - observed <= 60
        or value.get("ready") is not True
        or value.get("receipt_sha256") != protocol.sha256_json(unsigned)
    ):
        raise PasskeyV2ServiceError("passkey_v2_firewall_receipt_invalid")
    return value


def _release_revision() -> str:
    release = Path(__file__).resolve(strict=True).parents[2]
    if (
        release.parent != Path("/opt/muncho-owner-gate/releases")
        or _REVISION.fullmatch(release.name) is None
    ):
        raise PasskeyV2ServiceError("passkey_v2_release_path_invalid")
    return release.name


def _local_runtime_binding(release_revision: str) -> Mapping[str, Any]:
    """Derive authorization binding only from installed root-owned bytes."""

    release = Path(__file__).resolve(strict=True).parents[2]
    if release.name != release_revision:
        raise PasskeyV2ServiceError("passkey_v2_runtime_release_invalid")
    manifest_path = release / "package-manifest.json"
    manifest_raw, manifest_state = _read_regular_file(
        manifest_path,
        maximum=MAX_FRAME_BYTES,
        expected_uid=0,
        expected_gid=0,
        expected_mode=0o444,
    )
    manifest = protocol.decode_canonical_json(manifest_raw)
    if not isinstance(manifest, Mapping):
        raise PasskeyV2ServiceError("passkey_v2_runtime_manifest_invalid")
    manifest_unsigned = {
        key: item for key, item in manifest.items()
        if key != "package_sha256"
    }
    if (
        manifest.get("release_revision") != release_revision
        or manifest.get("release_root") != str(release)
        or manifest.get("package_sha256")
        != protocol.sha256_json(manifest_unsigned)
        or manifest_state.st_nlink != 1
        or not isinstance(manifest.get("payloads"), list)
    ):
        raise PasskeyV2ServiceError("passkey_v2_runtime_manifest_invalid")

    def payload_digest(path: Path) -> str:
        try:
            relative = str(path.resolve(strict=True).relative_to(release))
        except (OSError, ValueError) as exc:
            raise PasskeyV2ServiceError(
                "passkey_v2_runtime_payload_invalid"
            ) from None
        matches = [
            item for item in manifest["payloads"]
            if isinstance(item, Mapping)
            and item.get("release_relative") == relative
        ]
        if len(matches) != 1:
            raise PasskeyV2ServiceError("passkey_v2_runtime_payload_invalid")
        item = matches[0]
        raw, state = _read_regular_file(
            path,
            maximum=MAX_FRAME_BYTES,
            expected_uid=0,
            expected_gid=0,
            expected_mode=0o555 if relative.startswith("bin/") else 0o444,
        )
        digest = hashlib.sha256(raw).hexdigest()
        if item.get("sha256") != digest or item.get("size") != state.st_size:
            raise PasskeyV2ServiceError("passkey_v2_runtime_payload_invalid")
        return digest

    return protocol.build_runtime_binding(
        executor_release_sha=release_revision,
        executor_plan_sha256=storage.exact_storage_plan()["plan_sha256"],
        executor_binary_sha256=payload_digest(Path(__file__)),
        mutation_wrapper_sha256=payload_digest(Path(storage.__file__)),
        remote_transport_sha256=payload_digest(
            release / "bin/muncho-owner-gate-intake"
        ),
    )


def _local_action_authority_binding(
    release_revision: str,
) -> tuple[Mapping[str, Any], str, str]:
    runtime = _local_runtime_binding(release_revision)
    release = Path(__file__).resolve(strict=True).parents[2]
    raw, _state = _read_regular_file(
        release / "package-manifest.json",
        maximum=MAX_FRAME_BYTES,
        expected_uid=0,
        expected_gid=0,
        expected_mode=0o444,
    )
    manifest = protocol.decode_canonical_json(raw)
    if (
        not isinstance(manifest, Mapping)
        or _SHA256.fullmatch(str(manifest.get("package_sha256"))) is None
    ):
        raise PasskeyV2ServiceError("passkey_v2_runtime_manifest_invalid")
    return (
        runtime,
        str(manifest["package_sha256"]),
        str(runtime["runtime_binding_sha256"]),
    )


def _local_cutover_authority_binding(
    release_revision: str,
    freeze_plan_sha256: str,
) -> tuple[Mapping[str, Any], str, str, Mapping[str, Any]]:
    """Bind a cutover receipt to installed bytes and portable signed trust."""

    if _SHA256.fullmatch(freeze_plan_sha256 or "") is None:
        raise PasskeyV2ServiceError(
            "passkey_v2_cutover_plan_binding_invalid"
        )
    release = Path(__file__).resolve(strict=True).parents[2]
    if release.name != release_revision:
        raise PasskeyV2ServiceError("passkey_v2_runtime_release_invalid")
    manifest_raw, manifest_state = _read_regular_file(
        release / "package-manifest.json",
        maximum=MAX_FRAME_BYTES,
        expected_uid=0,
        expected_gid=0,
        expected_mode=0o444,
    )
    manifest = protocol.decode_canonical_json(manifest_raw)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("release_revision") != release_revision
        or manifest.get("release_root") != str(release)
        or manifest_state.st_nlink != 1
        or not isinstance(manifest.get("payloads"), list)
        or _SHA256.fullmatch(str(manifest.get("package_sha256"))) is None
    ):
        raise PasskeyV2ServiceError("passkey_v2_runtime_manifest_invalid")

    def payload_digest(path: Path) -> str:
        try:
            relative = str(path.resolve(strict=True).relative_to(release))
        except (OSError, ValueError):
            raise PasskeyV2ServiceError(
                "passkey_v2_runtime_payload_invalid"
            ) from None
        matches = [
            item for item in manifest["payloads"]
            if isinstance(item, Mapping)
            and item.get("release_relative") == relative
        ]
        if len(matches) != 1:
            raise PasskeyV2ServiceError(
                "passkey_v2_runtime_payload_invalid"
            )
        raw, state = _read_regular_file(
            path,
            maximum=MAX_FRAME_BYTES,
            expected_uid=0,
            expected_gid=0,
            expected_mode=(
                0o555 if relative.startswith("bin/") else 0o444
            ),
        )
        digest = hashlib.sha256(raw).hexdigest()
        if matches[0].get("sha256") != digest or matches[0].get(
            "size"
        ) != state.st_size:
            raise PasskeyV2ServiceError(
                "passkey_v2_runtime_payload_invalid"
            )
        return digest

    runtime = protocol.build_runtime_binding(
        executor_release_sha=release_revision,
        executor_plan_sha256=freeze_plan_sha256,
        executor_binary_sha256=payload_digest(Path(__file__)),
        mutation_wrapper_sha256=payload_digest(
            Path(production_cutover.__file__)
        ),
        remote_transport_sha256=payload_digest(
            release / "bin/muncho-owner-gate-intake"
        ),
    )
    trust_raw, _trust_state = _read_regular_file(
        CUTOVER_TRUST_BUNDLE,
        maximum=MAX_FRAME_BYTES,
        expected_uid=0,
        expected_gid=0,
        expected_mode=0o444,
    )
    trust_bundle = protocol.decode_canonical_json(trust_raw)
    try:
        checked_trust, _receipt_key = (
            production_cutover.validate_trust_bundle(trust_bundle)
        )
    except production_cutover.ProductionCutoverPasskeyError:
        raise PasskeyV2ServiceError(
            "passkey_v2_cutover_trust_invalid"
        ) from None
    host_observation = checked_trust["post_iam_host_observation"]
    if checked_trust["authority_release_sha"] != release_revision:
        raise PasskeyV2ServiceError(
            "passkey_v2_cutover_trust_invalid"
        )
    return (
        runtime,
        str(manifest["package_sha256"]),
        str(host_observation["report_sha256"]),
        checked_trust,
    )


def build_service_frame(
    operation: str, document: Mapping[str, Any]
) -> Mapping[str, Any]:
    if operation not in {
        "health",
        "render",
        "options",
        "verify",
        "create_request",
        "consume",
        "preflight",
        "execute",
        "terminal",
        "context",
        "verify_observation",
        "observation_request",
        "reconcile_read_only",
        "attest_cloud_observation",
    }:
        raise PasskeyV2ServiceError("passkey_v2_service_operation_invalid")
    unsigned = {
        "schema": SERVICE_FRAME_SCHEMA,
        "operation": operation,
        "document": dict(document),
    }
    return {**unsigned, "frame_sha256": protocol.sha256_json(unsigned)}


def validate_service_frame(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "operation",
        "document",
        "frame_sha256",
    }:
        raise PasskeyV2ServiceError("passkey_v2_service_frame_invalid")
    frame = dict(value)
    unsigned = {key: item for key, item in frame.items() if key != "frame_sha256"}
    if (
        frame.get("schema") != SERVICE_FRAME_SCHEMA
        or not isinstance(frame.get("operation"), str)
        or not isinstance(frame.get("document"), Mapping)
        or frame.get("frame_sha256") != protocol.sha256_json(unsigned)
    ):
        raise PasskeyV2ServiceError("passkey_v2_service_frame_invalid")
    return frame


def build_service_response(
    operation: str, document: Mapping[str, Any]
) -> Mapping[str, Any]:
    unsigned = {
        "schema": SERVICE_RESPONSE_SCHEMA,
        "operation": operation,
        "ok": True,
        "document": dict(document),
    }
    return {**unsigned, "response_sha256": protocol.sha256_json(unsigned)}


def build_service_error(operation: str = "rejected") -> Mapping[str, Any]:
    """Return a bounded secret-free error without exposing validation detail."""

    unsigned = {
        "schema": SERVICE_RESPONSE_SCHEMA,
        "operation": operation if isinstance(operation, str) else "rejected",
        "ok": False,
        "document": {"error": "request_rejected"},
    }
    return {**unsigned, "response_sha256": protocol.sha256_json(unsigned)}


def validate_service_response(
    value: Any, *, expected_operation: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "operation",
        "ok",
        "document",
        "response_sha256",
    }:
        raise PasskeyV2ServiceError("passkey_v2_service_response_invalid")
    response = dict(value)
    unsigned = {
        key: item for key, item in response.items() if key != "response_sha256"
    }
    if (
        response.get("schema") != SERVICE_RESPONSE_SCHEMA
        or response.get("operation") != expected_operation
        or response.get("ok") is not True
        or not isinstance(response.get("document"), Mapping)
        or response.get("response_sha256") != protocol.sha256_json(unsigned)
    ):
        raise PasskeyV2ServiceError("passkey_v2_service_response_invalid")
    return dict(response["document"])


def _canonical_observation_request(
    executor: PasskeyV2ExecutorDatabase,
    *,
    transaction_id: str,
    release_revision: str,
    now_unix: int,
) -> Mapping[str, Any]:
    """Derive one self-digested observation request from canonical truth."""

    if _SHA256.fullmatch(transaction_id) is None:
        raise PasskeyV2ServiceError("passkey_v2_transaction_id_invalid")
    try:
        transaction_state = executor.read_transaction_state(transaction_id)
    except PasskeyV2SqliteDenied:
        transaction_state = None
    if transaction_state is None:
        checkpoint = "source"
        canonical_state = "await_source"
        prior_event_head = protocol.GENESIS_JOURNAL_HEAD_SHA256
    else:
        events = tuple(transaction_state["events"])
        prior_event_head = events[-1]["event_head_sha256"]
        kinds = {event["event_kind"] for event in events}
        if transaction_state["state"] == "terminal":
            checkpoint = None
            canonical_state = "terminal"
        elif kinds & {
            "start_intent", "start_complete",
            "post_start_observation_required",
            "post_start_observation_accepted",
        }:
            checkpoint = "post_start"
            canonical_state = "await_post_start"
        elif kinds & {
            "stop_intent", "stop_complete",
            "post_stop_observation_required",
            "post_stop_observation_accepted",
        }:
            checkpoint = "post_stop"
            canonical_state = "await_post_stop"
        elif kinds & {
            "resize_intent", "resize_complete",
            "post_resize_observation_required",
            "post_resize_observation_accepted",
        }:
            checkpoint = "post_resize"
            canonical_state = "await_post_resize"
        else:
            checkpoint = "source"
            canonical_state = "await_source"
    plan_sha256 = storage.exact_storage_plan()["plan_sha256"]
    collection_attempt = (
        None
        if checkpoint is None
        else executor.issue_observation_collection_attempt(
            transaction_id=transaction_id,
            checkpoint=checkpoint,
            prior_event_head_sha256=prior_event_head,
            release_sha=release_revision,
            plan_sha256=plan_sha256,
            now_unix=now_unix,
            ttl_seconds=growth_evidence.OBSERVATION_BUNDLE_TTL_SECONDS,
        )
    )
    if checkpoint is None:
        request_binding = None
        observation_nonce = None
    else:
        request_binding = growth_evidence.observation_request_binding_sha256(
            transaction_id=transaction_id,
            checkpoint=checkpoint,
            prior_event_head_sha256=prior_event_head,
            release_sha=release_revision,
            plan_sha256=plan_sha256,
        )
        observation_nonce = growth_evidence.observation_nonce_sha256(
            request_binding_sha256=request_binding,
            transaction_id=transaction_id,
            checkpoint=checkpoint,
        )
    unsigned = {
        "schema": "muncho-storage-growth-observation-request.v1",
        "transaction_id": transaction_id,
        "checkpoint": checkpoint,
        "canonical_state": canonical_state,
        "prior_event_head_sha256": prior_event_head,
        "request_binding_sha256": request_binding,
        "observation_nonce_sha256": observation_nonce,
        "collection_attempt_id": (
            None
            if collection_attempt is None
            else collection_attempt["collection_attempt_id"]
        ),
        "collection_attempt_sequence": (
            None
            if collection_attempt is None
            else collection_attempt["context_sequence"]
        ),
        "collection_attempt_issued_at_unix": (
            None
            if collection_attempt is None
            else collection_attempt["issued_at_unix"]
        ),
        "collection_attempt_expires_at_unix": (
            None
            if collection_attempt is None
            else collection_attempt["expires_at_unix"]
        ),
        "release_sha": release_revision,
        "plan_sha256": plan_sha256,
    }
    return {
        **unsigned,
        "observation_request_sha256": protocol.sha256_json(unsigned),
    }


_OBSERVATION_EXPECTED_BINDING_FIELDS = frozenset({
    "transaction_id",
    "checkpoint",
    "request_binding_sha256",
    "prior_event_head_sha256",
    "observation_request_sha256",
    "collection_attempt_id",
    "collection_attempt_sequence",
    "collection_attempt_issued_at_unix",
    "collection_attempt_expires_at_unix",
})


def _expected_observation_binding(
    request: Mapping[str, Any],
) -> Mapping[str, Any]:
    if (
        request.get("canonical_state") == "terminal"
        or request.get("checkpoint") is None
    ):
        raise PasskeyV2ServiceError(
            "passkey_v2_observation_request_invalid"
        )
    return {
        name: request[name]
        for name in _OBSERVATION_EXPECTED_BINDING_FIELDS
    }


def _authority_options(
    authority: PasskeyV2AuthorityDatabase,
    request_id: str,
) -> Mapping[str, Any]:
    state = authority.read_request_state(request_id)
    challenge = state["challenge_record"]
    if challenge is None or state["grant_record"] is not None:
        raise PasskeyV2ServiceError("passkey_v2_request_not_approvable")
    credentials = authority.read_active_credentials()
    if not credentials:
        raise PasskeyV2ServiceError("passkey_v2_credential_unavailable")
    return {
        "request_id": request_id,
        "publicKey": {
            "challenge": challenge["challenge_b64url"],
            "rpId": protocol.PRODUCTION_RP_ID,
            "timeout": 300_000,
            "userVerification": "required",
            "allowCredentials": [
                {
                    "id": credential["credential_id_b64url"],
                    "type": "public-key",
                }
                for credential in credentials
            ],
        },
    }


def _validate_authority_action(value: Any) -> Mapping[str, Any]:
    """Closed action-schema dispatch; unknown capabilities stay absent."""

    if not isinstance(value, Mapping):
        raise PasskeyV2ServiceError("passkey_v2_action_invalid")
    payload = value.get("action_payload")
    schema = payload.get("schema") if isinstance(payload, Mapping) else None
    try:
        if schema == storage.STORAGE_ACTION_SCHEMA:
            return storage.validate_storage_growth_envelope(value)
        if schema == production_cutover.CUTOVER_ACTION_SCHEMA:
            return production_cutover.validate_cutover_action_envelope(value)
    except (
        storage.PasskeyV2StorageBoundaryError,
        production_cutover.ProductionCutoverPasskeyError,
    ):
        raise PasskeyV2ServiceError("passkey_v2_action_invalid") from None
    raise PasskeyV2ServiceError("passkey_v2_action_schema_forbidden")


def _mechanical_authority_facts(
    action: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = action["action_payload"]
    if payload["schema"] == storage.STORAGE_ACTION_SCHEMA:
        return storage.mechanical_approval_facts(action)
    if payload["schema"] == production_cutover.CUTOVER_ACTION_SCHEMA:
        return production_cutover.mechanical_approval_facts(action)
    raise PasskeyV2ServiceError("passkey_v2_action_schema_forbidden")


def handle_authority_frame(
    value: Any,
    *,
    authority: PasskeyV2AuthorityDatabase,
    signer: ReceiptSigner,
    peer_uid: int,
    now_unix: int,
) -> Mapping[str, Any]:
    frame = validate_service_frame(value)
    operation = frame["operation"]
    document = dict(frame["document"])
    if peer_uid not in {WEB_UID, AUTHORITY_UID}:
        raise PasskeyV2ServiceError("passkey_v2_authority_peer_forbidden")
    if operation == "health":
        if document:
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        result = {"healthy": True, "preflight": authority.preflight()}
    elif operation in {"render", "options"}:
        if set(document) != {"request_id"}:
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        request_id = str(document["request_id"])
        if operation == "render":
            state = authority.read_request_state(request_id)
            action = _validate_authority_action(state["action_envelope"])
            result = {
                **protocol.build_ui_view(action),
                "mechanical_facts": _mechanical_authority_facts(action),
            }
        else:
            result = _authority_options(authority, request_id)
    elif operation == "verify":
        if set(document) != {"request_id", "assertion"}:
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        request_id = str(document["request_id"])
        state = authority.read_request_state(request_id)
        if state["challenge_record"] is None or state["grant_record"] is not None:
            raise PasskeyV2ServiceError("passkey_v2_request_not_approvable")
        grant = authority.verify_assertion_and_record_grant(
            assertion=document["assertion"],
            envelope=state["action_envelope"],
            challenge=state["challenge_record"],
            grant_id=base64.urlsafe_b64encode(secrets.token_bytes(24))
            .rstrip(b"=")
            .decode("ascii"),
            now_unix=now_unix,
        )
        result = {
            "request_id": request_id,
            "state": "granted",
            "grant_sha256": grant["grant_sha256"],
            "expires_at_unix": grant["expires_at_unix"],
        }
    elif operation in {"create_request", "consume"}:
        if peer_uid != AUTHORITY_UID:
            raise PasskeyV2ServiceError(
                "passkey_v2_authority_privileged_peer_required"
            )
        if operation == "create_request":
            if set(document) != {"action_envelope"}:
                raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
            checked = _validate_authority_action(
                document["action_envelope"]
            )
            action = authority.create_request(checked)
            challenge_bytes = secrets.token_bytes(32)
            challenge = protocol.build_challenge_record(
                envelope=action,
                challenge_id=base64.urlsafe_b64encode(secrets.token_bytes(24))
                .rstrip(b"=")
                .decode("ascii"),
                challenge_b64url=base64.urlsafe_b64encode(challenge_bytes)
                .rstrip(b"=")
                .decode("ascii"),
                rp_id=protocol.PRODUCTION_RP_ID,
                origin=protocol.PRODUCTION_ORIGIN,
                created_at_unix=now_unix,
            )
            authority.create_challenge(challenge, envelope=action)
            result = {
                "request_id": action["request_id"],
                "action_envelope_sha256": action["envelope_sha256"],
                "challenge_record_sha256": challenge[
                    "challenge_record_sha256"
                ],
                "expires_at_unix": action["expires_at_unix"],
            }
        else:
            if set(document) != {
                "request_id",
                "runtime_binding",
                "consume_attempt_id",
            }:
                raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
            state = authority.read_request_state(str(document["request_id"]))
            _validate_authority_action(state["action_envelope"])
            if state["grant_record"] is None or state["challenge_record"] is None:
                raise PasskeyV2ServiceError("passkey_v2_request_not_granted")
            consumed = authority.consume_or_replay(
                envelope=state["action_envelope"],
                runtime_binding=document["runtime_binding"],
                consume_attempt_id=str(document["consume_attempt_id"]),
                signer=signer,
                now_unix=now_unix,
            )
            result = {
                "disposition": consumed.disposition,
                "authorization_receipt": consumed.receipt,
                "action_envelope": state["action_envelope"],
                "challenge_record": state["challenge_record"],
                "grant_record": state["grant_record"],
            }
    else:
        raise PasskeyV2ServiceError("passkey_v2_authority_operation_forbidden")
    return build_service_response(operation, result)


def handle_executor_frame(
    value: Any,
    *,
    executor: PasskeyV2ExecutorDatabase,
    peer_uid: int,
    release_revision: str,
    now_unix: int,
    mutation_handler: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
        Mapping[str, Any],
    ],
    readiness_handler: Callable[[], Mapping[str, Any]],
    observation_verifier: Callable[
        [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
    ],
    cloud_attestor: Callable[..., Mapping[str, Any]] | None = None,
    observation_attestors: Mapping[str, Any] | None = None,
    cloud_signer_runtime_readiness: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    frame = validate_service_frame(value)
    operation = frame["operation"]
    document = dict(frame["document"])
    if peer_uid != AUTHORITY_UID:
        raise PasskeyV2ServiceError("passkey_v2_executor_peer_forbidden")
    if operation == "preflight":
        if document:
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        if (
            not isinstance(observation_attestors, Mapping)
            or set(observation_attestors) != {"cloud", "host"}
        ):
            raise PasskeyV2ServiceError(
                "passkey_v2_observation_attestor_trust_invalid"
            )
        result = {
            "healthy": True,
            "database": executor.preflight(),
            "firewall_readiness": dict(readiness_handler()),
            "activation_seal": activation_seal_status(
                expected_release_revision=release_revision,
                now_unix=now_unix,
            ),
            "observation_attestors": dict(observation_attestors),
            "cloud_signer_runtime_readiness": (
                None
                if cloud_signer_runtime_readiness is None
                else dict(cloud_signer_runtime_readiness)
            ),
        }
    elif operation == "terminal":
        if set(document) != {"transaction_id"}:
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        try:
            receipt = executor.read_terminal_receipt(
                str(document["transaction_id"])
            )
        except PasskeyV2SqliteDenied:
            receipt = None
        result = {
            "transaction_id": str(document["transaction_id"]),
            "terminal": receipt is not None,
            "terminal_receipt": receipt,
        }
    elif operation == "context":
        if set(document) != {"transaction_id"}:
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        state = executor.read_transaction_state(str(document["transaction_id"]))
        events = tuple(state["events"])
        latest_authorization = state["authorization"]
        initial_authorization = state["authorizations"][0]
        result = {
            "transaction_id": state["transaction"]["transaction_id"],
            "state": state["state"],
            "prior_event_head_sha256": events[-1]["event_head_sha256"],
            "prior_authoritative_receipt_sha256": latest_authorization[
                "authorization_receipt"
            ]["receipt_sha256"],
            "initial_source_preflight": initial_authorization[
                "action_envelope"
            ]["action_payload"]["source_preflight"],
            "latest_request_id": latest_authorization[
                "action_envelope"
            ]["request_id"],
            "milestone_event_kinds": [
                event["event_kind"] for event in events
                if event["event_kind"] not in {
                    "provider_pending", "attempt_failed",
                    "reconciliation_observed",
                }
            ],
        }
    elif operation == "verify_observation":
        if set(document) != {"observation_bundle", "expected_binding"}:
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        expected = document["expected_binding"]
        if (
            not isinstance(expected, Mapping)
            or set(expected) != _OBSERVATION_EXPECTED_BINDING_FIELDS
        ):
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        observation = observation_verifier(
            document["observation_bundle"], expected
        )
        result = {
            "observation": observation,
            "observation_projection": growth_evidence.observation_projection(
                observation
            ),
        }
    elif operation == "observation_request":
        if set(document) != {"transaction_id"}:
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        result = _canonical_observation_request(
            executor,
            transaction_id=str(document["transaction_id"]),
            release_revision=release_revision,
            now_unix=now_unix,
        )
    elif operation == "attest_cloud_observation":
        if set(document) != {
            "transaction_id", "attestation_request"
        }:
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        transaction_id = str(document["transaction_id"])
        canonical_before = _canonical_observation_request(
            executor,
            transaction_id=transaction_id,
            release_revision=release_revision,
            now_unix=now_unix,
        )
        if canonical_before["canonical_state"] == "terminal":
            terminal = executor.read_terminal_receipt(transaction_id)
            if terminal is None:
                raise PasskeyV2ServiceError(
                    "passkey_v2_terminal_receipt_invalid"
                )
            result = {
                "terminal": True,
                "terminal_receipt": terminal,
                "observation_request": canonical_before,
                "attestation_response": None,
            }
        else:
            attestation_request = document["attestation_request"]
            if (
                not isinstance(attestation_request, Mapping)
                or attestation_request.get("role") != "cloud"
                or protocol.canonical_json_bytes(
                    attestation_request.get("observation_request")
                )
                != protocol.canonical_json_bytes(canonical_before)
                or not callable(cloud_attestor)
            ):
                raise PasskeyV2ServiceError(
                    "passkey_v2_cloud_attestation_request_invalid"
                )
            if canonical_before["checkpoint"] == "post_stop":
                transaction_state = executor.read_transaction_state(
                    transaction_id
                )
                initial_bundle = transaction_state["authorizations"][0][
                    "action_envelope"
                ]["action_payload"]["source_preflight"]
                initial_observation = initial_bundle.get("observation")
                stopped_observation = attestation_request.get(
                    "candidate_observation"
                )
                receipt_bindings = (
                    "current_stopped_release_sha",
                    "current_host_receipt_file_sha256",
                    "current_host_receipt_sha256",
                    "current_stopped_release_receipt_file_sha256",
                    "current_stopped_release_receipt_sha256",
                )
                if (
                    not isinstance(initial_observation, Mapping)
                    or not isinstance(stopped_observation, Mapping)
                    or not isinstance(
                        initial_bundle.get("cloud_attestation"), Mapping
                    )
                    or not isinstance(
                        initial_bundle.get("host_attestation"), Mapping
                    )
                    or stopped_observation.get("canonical_receipt_source")
                    != "durable_signed_source_snapshot_for_stopped_vm"
                    or any(
                        stopped_observation.get(name)
                        != initial_observation.get(name)
                        for name in receipt_bindings
                    )
                ):
                    raise PasskeyV2ServiceError(
                        "passkey_v2_stopped_snapshot_binding_invalid"
                    )
            try:
                attestation_response = dict(cloud_attestor(
                    attestation_request,
                    now_unix=now_unix,
                ))
            except trusted_collector.TrustedObservationError as exc:
                raise PasskeyV2ServiceError(
                    "passkey_v2_cloud_attestation_failed"
                ) from None
            canonical_after = _canonical_observation_request(
                executor,
                transaction_id=transaction_id,
                release_revision=release_revision,
                now_unix=now_unix,
            )
            if (
                protocol.canonical_json_bytes(canonical_after)
                != protocol.canonical_json_bytes(canonical_before)
                or attestation_response.get("role") != "cloud"
                or attestation_response.get("observation_request_sha256")
                != canonical_before["observation_request_sha256"]
            ):
                raise PasskeyV2ServiceError(
                    "passkey_v2_cloud_attestation_binding_invalid"
                )
            result = {
                "terminal": False,
                "terminal_receipt": None,
                "observation_request": canonical_after,
                "attestation_response": attestation_response,
            }
    elif operation == "reconcile_read_only":
        if set(document) != {
            "transaction_id", "continuation_observation"
        }:
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        transaction_id = str(document["transaction_id"])
        canonical_request = _canonical_observation_request(
            executor,
            transaction_id=transaction_id,
            release_revision=release_revision,
            now_unix=now_unix,
        )
        if canonical_request["canonical_state"] == "terminal":
            terminal = executor.read_terminal_receipt(transaction_id)
            if terminal is None:
                raise PasskeyV2ServiceError(
                    "passkey_v2_terminal_receipt_invalid"
                )
            result = {
                "schema": "muncho-passkey-v2-read-only-reconciliation.v1",
                "terminal": True,
                "state": "target_ready",
                "terminal_receipt": terminal,
            }
        else:
            raw_observation = document["continuation_observation"]
            if not isinstance(raw_observation, Mapping):
                raise PasskeyV2ServiceError(
                    "passkey_v2_continuation_observation_invalid"
                )
            expected = _expected_observation_binding(canonical_request)
            observation = dict(observation_verifier(
                raw_observation, expected
            ))
            try:
                executor.read_transaction_state(transaction_id)
            except PasskeyV2SqliteDenied:
                unsigned = {
                    "schema": (
                        "muncho-passkey-v2-read-only-reconciliation.v1"
                    ),
                    "terminal": False,
                    "state": "not_started",
                    "remaining_stage": "intent",
                    "transaction_id": transaction_id,
                    "observation_request": canonical_request,
                }
                result = {
                    **unsigned,
                    "reconciliation_sha256": protocol.sha256_json(unsigned),
                }
            else:
                reconcile = getattr(
                    mutation_handler, "reconcile_read_only", None
                )
                if not callable(reconcile):
                    raise PasskeyV2ServiceError(
                        "passkey_v2_read_only_reconciler_unavailable"
                    )
                result = dict(reconcile(
                    transaction_id=transaction_id,
                    observation=observation,
                    observation_bundle=raw_observation,
                ))
    elif operation == "execute":
        if set(document) != {
            "authorization_receipt",
            "action_envelope",
            "challenge_record",
            "grant_record",
            "continuation_observation",
        }:
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        action = storage.validate_storage_growth_envelope(
            document["action_envelope"]
        )
        if (
            action["executor_release_sha"] != release_revision
            or action["authority_release_sha"] != release_revision
        ):
            raise PasskeyV2ServiceError("passkey_v2_action_release_invalid")
        # Terminal truth is looked up before time, readiness, activation,
        # observation, metadata, or Compute access.  A lost completion response
        # therefore remains byte-identically recoverable after authorization
        # expiry or infrastructure drift.
        try:
            terminal_receipt = executor.read_terminal_receipt(
                action["transaction_id"]
            )
        except PasskeyV2SqliteDenied:
            terminal_receipt = None
        if terminal_receipt is not None:
            return build_service_response(
                operation,
                {
                    "claim_disposition": "terminal_replay",
                    "authorization": None,
                    "mutation_receipt": terminal_receipt,
                    "completion_event": None,
                    "terminal_replay": True,
                },
            )
        # Mutation readiness is checked only after terminal replay is ruled
        # out and immediately before the durable authorization claim.
        readiness = dict(readiness_handler())
        seal = read_activation_seal(
            expected_release_revision=release_revision,
            now_unix=now_unix,
        )
        raw_observation = document["continuation_observation"]
        if raw_observation is None:
            raw_observation = action["action_payload"]["source_preflight"]
        if not isinstance(raw_observation, Mapping):
            raise PasskeyV2ServiceError(
                "passkey_v2_continuation_observation_invalid"
            )
        canonical_request = _canonical_observation_request(
            executor,
            transaction_id=action["transaction_id"],
            release_revision=release_revision,
            now_unix=now_unix,
        )
        checkpoint = canonical_request["checkpoint"]
        prior_event_head = canonical_request["prior_event_head_sha256"]
        request_binding_sha256 = canonical_request[
            "request_binding_sha256"
        ]
        if checkpoint is None or request_binding_sha256 is None:
            raise PasskeyV2ServiceError(
                "passkey_v2_observation_request_invalid"
            )
        expected_observation = _expected_observation_binding(
            canonical_request
        )
        observation = dict(observation_verifier(
            raw_observation, expected_observation
        ))
        claim = executor.claim_execution(
            receipt=document["authorization_receipt"],
            envelope=action,
            grant=document["grant_record"],
            challenge=document["challenge_record"],
            now_unix=now_unix,
        )
        try:
            mutation = dict(mutation_handler(
                action,
                observation,
                {
                    "authorization_request_id": action["request_id"],
                    "observation_bundle": dict(raw_observation),
                    "observation_bundle_sha256": raw_observation[
                        "bundle_sha256"
                    ],
                    "observation_nonce_sha256": raw_observation[
                        "observation_nonce_sha256"
                    ],
                    "activation_seal_sha256": seal["seal_sha256"],
                    "firewall_readiness_receipt_sha256": readiness[
                        "receipt_sha256"
                    ],
                },
            ))
        except BaseException as exc:
            raise PasskeyV2ServiceError(
                "passkey_v2_storage_mutation_failed"
            ) from None
        if mutation.get("terminal") is not True:
            if (
                mutation.get("state") != "observation_required"
                or mutation.get("schema")
                != "muncho-passkey-v2-storage-progress.v1"
            ):
                raise PasskeyV2ServiceError(
                    "passkey_v2_storage_progress_invalid"
                )
            result = {
                "claim_disposition": claim.disposition,
                "authorization": claim.intent,
                "mutation_receipt": mutation,
                "completion_event": None,
                "terminal_replay": False,
            }
            return build_service_response(operation, result)
        completed = executor.append_execution_event(
            request_id=action["request_id"],
            event_kind="completed",
            event_payload={"terminal_receipt": mutation},
            now_unix=int(time.time()),
        )
        result = {
            "claim_disposition": claim.disposition,
            "authorization": claim.intent,
            "mutation_receipt": mutation,
            "completion_event": completed,
            "terminal_replay": False,
        }
    else:
        raise PasskeyV2ServiceError("passkey_v2_executor_operation_forbidden")
    return build_service_response(operation, result)


class UnixServiceClient:
    def __init__(self, path: Path) -> None:
        self.path = path

    def call(self, operation: str, document: Mapping[str, Any]) -> Mapping[str, Any]:
        frame = build_service_frame(operation, document)
        raw = protocol.canonical_json_bytes(frame) + b"\n"
        if len(raw) > MAX_FRAME_BYTES:
            raise PasskeyV2ServiceError("passkey_v2_service_frame_oversized")
        response_timeout = SERVICE_OPERATION_RESPONSE_TIMEOUT_SECONDS.get(
            operation
        )
        if (
            type(response_timeout) is not float
            or not math.isfinite(response_timeout)
            or response_timeout <= 0.0
        ):
            raise PasskeyV2ServiceError(
                "passkey_v2_service_operation_timeout_invalid"
            )
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        request_may_have_been_received = False
        connection.settimeout(SERVICE_CONNECT_TIMEOUT_SECONDS)
        try:
            connection.connect(str(self.path))
            # Set the uncertainty marker before sendall: an interrupted local
            # write can have delivered a complete frame even when the caller
            # receives an OSError.  Retrying must therefore reconcile or
            # replay terminal truth instead of assuming that no mutation ran.
            request_may_have_been_received = True
            connection.sendall(raw)
            connection.shutdown(socket.SHUT_WR)
            connection.settimeout(response_timeout)
            response = bytearray()
            while True:
                chunk = connection.recv(64 * 1024)
                if not chunk:
                    break
                response.extend(chunk)
                if len(response) > MAX_FRAME_BYTES + 1:
                    raise PasskeyV2ServiceError(
                        "passkey_v2_service_response_oversized"
                    )
        except OSError as exc:
            if request_may_have_been_received:
                raise PasskeyV2ServiceError(
                    "passkey_v2_service_operation_outcome_unknown"
                ) from None
            raise PasskeyV2ServiceError(
                "passkey_v2_service_socket_unavailable"
            ) from None
        finally:
            connection.close()
        if not response.endswith(b"\n") or b"\n" in response[:-1]:
            raise PasskeyV2ServiceError("passkey_v2_service_response_invalid")
        value = protocol.decode_canonical_json(bytes(response[:-1]))
        return validate_service_response(value, expected_operation=operation)


class _CachedAuthorityReadiness:
    """Non-blocking, single-flight authority/socket/DB readiness cache."""

    def __init__(
        self,
        client: UnixServiceClient,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(clock):
            raise PasskeyV2ServiceError(
                "passkey_v2_web_readiness_clock_invalid"
            )
        self._client = client
        self._clock = clock
        self._lock = threading.Lock()
        self._last_completed_at: float | None = None
        self._ready = False
        self._refreshing = False

    def _refresh(self) -> None:
        ready = False
        try:
            response = self._client.call("health", {})
            ready = (
                response.get("healthy") is True
                and isinstance(response.get("preflight"), Mapping)
            )
        except Exception:
            ready = False
        completed_at = self._clock()
        with self._lock:
            self._ready = ready
            self._last_completed_at = completed_at
            self._refreshing = False

    def status(self) -> bool:
        """Return recent cached truth and start at most one bounded refresh."""

        now = self._clock()
        start_refresh = False
        with self._lock:
            completed_at = self._last_completed_at
            age = (
                math.inf
                if completed_at is None
                else max(0.0, now - completed_at)
            )
            ready = self._ready and age <= WEB_READINESS_MAX_STALE_SECONDS
            if age >= WEB_READINESS_REFRESH_SECONDS and not self._refreshing:
                self._refreshing = True
                start_refresh = True
        if start_refresh:
            threading.Thread(
                target=self._refresh,
                name="muncho-passkey-v2-readiness",
                daemon=True,
            ).start()
        return ready


def _peer_uid(connection: socket.socket) -> int:
    if hasattr(socket, "SO_PEERCRED"):
        raw = connection.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            struct.calcsize("3i"),
        )
        _pid, uid, _gid = struct.unpack("3i", raw)
        return int(uid)
    # Darwin exposes xucred through LOCAL_PEERCRED.  Only the fixed version
    # and uid prefix are consumed; supplementary groups are irrelevant.
    if hasattr(socket, "LOCAL_PEERCRED"):
        raw = connection.getsockopt(
            getattr(socket, "SOL_LOCAL", 0),
            socket.LOCAL_PEERCRED,
            12,
        )
        if len(raw) < 8:
            raise PasskeyV2ServiceError(
                "passkey_v2_peer_credentials_unavailable"
            )
        version, uid = struct.unpack("=II", raw[:8])
        if version != 0:
            raise PasskeyV2ServiceError(
                "passkey_v2_peer_credentials_unavailable"
            )
        return int(uid)
    raise PasskeyV2ServiceError("passkey_v2_peer_credentials_unavailable")


def _handle_service_connection(
    connection: socket.socket,
    handler: Callable[[Mapping[str, Any], int], Mapping[str, Any]],
    *,
    peer_uid: int | None = None,
) -> None:
    """Handle one bounded frame; failure is isolated to this connection."""

    connection.settimeout(SERVICE_FRAME_TIMEOUT_SECONDS)
    try:
        selected_uid = _peer_uid(connection) if peer_uid is None else peer_uid
        if type(selected_uid) is not int or selected_uid < 0:
            raise PasskeyV2ServiceError(
                "passkey_v2_peer_credentials_unavailable"
            )
        raw = bytearray()
        while True:
            chunk = connection.recv(64 * 1024)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > MAX_FRAME_BYTES + 1:
                raise PasskeyV2ServiceError(
                    "passkey_v2_service_frame_oversized"
                )
        if not raw.endswith(b"\n") or b"\n" in raw[:-1]:
            raise PasskeyV2ServiceError(
                "passkey_v2_service_frame_invalid"
            )
        value = protocol.decode_canonical_json(bytes(raw[:-1]))
        response = handler(value, selected_uid)
    except Exception:
        response = build_service_error()
    try:
        connection.sendall(protocol.canonical_json_bytes(response) + b"\n")
    except OSError:
        # A disconnected client cannot terminate the socket service.
        pass


class _ServiceConnectionGate:
    """Bound local socket concurrency and request rate, globally and per UID."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        if not callable(clock):
            raise PasskeyV2ServiceError(
                "passkey_v2_service_connection_gate_invalid"
            )
        self._clock = clock
        self._lock = threading.Lock()
        self._active_total = 0
        self._active_by_uid: dict[int, int] = {}
        now = float(clock())
        self._global_tokens = SERVICE_GLOBAL_BURST
        self._global_updated = now
        self._uid_tokens: dict[int, tuple[float, float]] = {}

    @staticmethod
    def _refill(
        tokens: float,
        updated: float,
        *,
        now: float,
        rate: float,
        burst: float,
    ) -> tuple[float, float]:
        if now < updated:
            raise PasskeyV2ServiceError(
                "passkey_v2_service_connection_gate_invalid"
            )
        return min(burst, tokens + ((now - updated) * rate)), now

    def try_acquire(self, peer_uid: int) -> bool:
        if type(peer_uid) is not int or peer_uid < 0:
            return False
        with self._lock:
            now = float(self._clock())
            global_tokens, global_updated = self._refill(
                self._global_tokens,
                self._global_updated,
                now=now,
                rate=SERVICE_GLOBAL_RATE_PER_SECOND,
                burst=SERVICE_GLOBAL_BURST,
            )
            uid_tokens, uid_updated = self._uid_tokens.get(
                peer_uid,
                (SERVICE_UID_BURST, now),
            )
            uid_tokens, uid_updated = self._refill(
                uid_tokens,
                uid_updated,
                now=now,
                rate=SERVICE_UID_RATE_PER_SECOND,
                burst=SERVICE_UID_BURST,
            )
            active_for_uid = self._active_by_uid.get(peer_uid, 0)
            allowed = (
                self._active_total < SERVICE_MAX_CONNECTIONS
                and active_for_uid < SERVICE_MAX_CONNECTIONS_PER_UID
                and global_tokens >= 1.0
                and uid_tokens >= 1.0
            )
            if allowed:
                global_tokens -= 1.0
                uid_tokens -= 1.0
                self._active_total += 1
                self._active_by_uid[peer_uid] = active_for_uid + 1
            self._global_tokens = global_tokens
            self._global_updated = global_updated
            self._uid_tokens[peer_uid] = (uid_tokens, uid_updated)
            return allowed

    def release(self, peer_uid: int) -> None:
        if type(peer_uid) is not int or peer_uid < 0:
            raise PasskeyV2ServiceError(
                "passkey_v2_service_connection_gate_invalid"
            )
        with self._lock:
            active_for_uid = self._active_by_uid.get(peer_uid, 0)
            if self._active_total <= 0 or active_for_uid <= 0:
                raise PasskeyV2ServiceError(
                    "passkey_v2_service_connection_gate_invalid"
                )
            self._active_total -= 1
            if active_for_uid == 1:
                del self._active_by_uid[peer_uid]
            else:
                self._active_by_uid[peer_uid] = active_for_uid - 1


def _reject_service_connection(connection: socket.socket) -> None:
    try:
        connection.settimeout(0.1)
        connection.sendall(
            protocol.canonical_json_bytes(build_service_error()) + b"\n"
        )
    except OSError:
        pass


def _dispatch_service_connection(
    connection: socket.socket,
    handler: Callable[[Mapping[str, Any], int], Mapping[str, Any]],
    gate: _ServiceConnectionGate,
) -> threading.Thread | None:
    """Admit and start one bounded worker, or reject it synchronously."""

    acquired = False
    peer_uid = -1
    try:
        peer_uid = _peer_uid(connection)
        if not gate.try_acquire(peer_uid):
            _reject_service_connection(connection)
            connection.close()
            return None
        acquired = True

        def serve_one() -> None:
            try:
                with connection:
                    _handle_service_connection(
                        connection,
                        handler,
                        peer_uid=peer_uid,
                    )
            finally:
                gate.release(peer_uid)

        worker = threading.Thread(
            target=serve_one,
            name="muncho-passkey-v2-connection",
            daemon=True,
        )
        worker.start()
        return worker
    except Exception:
        if acquired:
            gate.release(peer_uid)
        _reject_service_connection(connection)
        connection.close()
        return None


def _validate_activated_listener_descriptor(
    descriptor: int, *, expected_path: Path
) -> socket.socket:
    """Duplicate and validate one already-open systemd AF_UNIX listener."""

    if (
        type(descriptor) is not int
        or descriptor < 0
        or not isinstance(expected_path, Path)
        or not expected_path.is_absolute()
    ):
        raise PasskeyV2ServiceError(
            "passkey_v2_socket_activation_descriptor_invalid"
        )
    listener: socket.socket | None = None
    try:
        descriptor_state = os.fstat(descriptor)
        if not stat.S_ISSOCK(descriptor_state.st_mode):
            raise PasskeyV2ServiceError(
                "passkey_v2_socket_activation_descriptor_invalid"
            )
        listener = socket.fromfd(
            descriptor, socket.AF_UNIX, socket.SOCK_STREAM
        )
        accepting = True
        if sys.platform.startswith("linux"):
            accepting = (
                listener.getsockopt(
                    socket.SOL_SOCKET, socket.SO_ACCEPTCONN
                )
                == 1
            )
        if (
            listener.family != socket.AF_UNIX
            or listener.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
            != socket.SOCK_STREAM
            or not accepting
            or listener.getsockname() != str(expected_path)
        ):
            raise PasskeyV2ServiceError(
                "passkey_v2_socket_activation_descriptor_invalid"
            )
        return listener
    except PasskeyV2ServiceError:
        if listener is not None:
            listener.close()
        raise
    except OSError as exc:
        if listener is not None:
            listener.close()
        raise PasskeyV2ServiceError(
            "passkey_v2_socket_activation_descriptor_invalid"
        ) from None


def _validated_activated_listener(
    *, expected_path: Path, expected_name: str
) -> socket.socket:
    if (
        os.environ.get("LISTEN_FDS") != "1"
        or os.environ.get("LISTEN_PID") != str(os.getpid())
        or os.environ.get("LISTEN_FDNAMES") != expected_name
    ):
        raise PasskeyV2ServiceError("passkey_v2_socket_activation_required")
    return _validate_activated_listener_descriptor(
        3, expected_path=expected_path
    )


def _serve_activated_socket(
    handler: Callable[[Mapping[str, Any], int], Mapping[str, Any]],
    *,
    expected_path: Path,
    expected_name: str,
) -> int:
    listener = _validated_activated_listener(
        expected_path=expected_path,
        expected_name=expected_name,
    )
    gate = _ServiceConnectionGate()
    try:
        while True:
            connection, _address = listener.accept()
            _dispatch_service_connection(connection, handler, gate)
    finally:
        listener.close()


def validate_web_request(
    *,
    method: str,
    path: str,
    headers: Mapping[str, str],
    body: bytes,
    csrf_cookie: str | None,
) -> tuple[str, str | None, Mapping[str, Any] | None]:
    """Validate the complete public HTTP boundary before any socket call."""

    normalized = {str(key).lower(): str(item) for key, item in headers.items()}
    if normalized.get("host") != "auth.lomliev.com":
        raise PasskeyV2ServiceError("passkey_v2_web_host_invalid")
    route: str
    request_id: str | None
    if method == "GET" and path == "/healthz":
        route, request_id = "health", None
    elif method == "GET" and path == "/readyz":
        route, request_id = "readiness", None
    elif method == "GET" and (match := _APPROVAL_PATH.fullmatch(path)):
        route, request_id = "render", match.group(1)
    elif method == "GET" and (match := _VIEW_PATH.fullmatch(path)):
        route, request_id = "view", match.group(1)
    elif method == "GET" and (match := _OPTIONS_PATH.fullmatch(path)):
        route, request_id = "options", match.group(1)
    elif method == "GET" and path == "/static/approve.js":
        route, request_id = "javascript", None
    elif method == "POST" and (match := _VERIFY_PATH.fullmatch(path)):
        route, request_id = "verify", match.group(1)
    else:
        raise PasskeyV2ServiceError("passkey_v2_web_route_forbidden")
    if method == "GET":
        if body:
            raise PasskeyV2ServiceError("passkey_v2_web_get_body_forbidden")
        return route, request_id, None
    if (
        normalized.get("origin") != protocol.PRODUCTION_ORIGIN
        or normalized.get("content-type") != "application/json"
        or len(body) > MAX_HTTP_BODY_BYTES
    ):
        raise PasskeyV2ServiceError("passkey_v2_web_post_headers_invalid")
    header_token = normalized.get("x-muncho-csrf")
    if (
        csrf_cookie is None
        or header_token is None
        or _CSRF.fullmatch(csrf_cookie) is None
        or not hmac.compare_digest(header_token, csrf_cookie)
    ):
        raise PasskeyV2ServiceError("passkey_v2_web_csrf_invalid")
    parsed = decode_strict_json(body, maximum=MAX_HTTP_BODY_BYTES)
    if set(parsed) != {"schema", "assertion"} or parsed.get("schema") != WEB_VERIFY_SCHEMA:
        raise PasskeyV2ServiceError("passkey_v2_web_body_invalid")
    return route, request_id, parsed


_APPROVAL_HTML = b"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Muncho owner approval</title></head>
<body><main><h1>Exact owner approval</h1><h2>Mechanical facts</h2><pre id="facts"></pre><h2>Full signed request</h2><pre id="action"></pre><button id="approve" type="button" disabled>Approve with passkey</button><p id="status" role="status">Loading exact request...</p></main><script src="/static/approve.js" defer></script></body></html>"""

_APPROVAL_JS = rb"""'use strict';
const b64ToBytes = value => Uint8Array.from(atob(value.replace(/-/g,'+').replace(/_/g,'/') + '='.repeat((4-value.length%4)%4)), c => c.charCodeAt(0));
const bytesToB64 = value => btoa(String.fromCharCode(...new Uint8Array(value))).replace(/\+/g,'-').replace(/\//g,'_').replace(/=+$/,'');
const base = location.pathname;
const csrf = () => document.cookie.split('; ').find(v => v.startsWith('muncho_csrf='))?.split('=')[1];
const approve = document.getElementById('approve');
async function load() { const response=await fetch(base + '/view',{cache:'no-store'}); if(!response.ok) throw new Error('view'); const view=await response.json(); const schemas=new Set(['muncho-passkey-v2-storage-growth-facts.v1','muncho-passkey-v2-production-cutover-facts.v1']); if(!schemas.has(view.mechanical_facts?.schema)||view.values_are_complete_and_untruncated!==true) throw new Error('facts'); document.getElementById('facts').textContent=JSON.stringify(view.mechanical_facts,null,2); document.getElementById('action').textContent=view.exact_action_envelope_canonical_json; document.getElementById('status').textContent='Review all facts before approval.'; approve.disabled=false; }
approve.addEventListener('click', async () => { if(approve.disabled) return; approve.disabled=true; try { const optionsResponse=await fetch(base + '/options',{cache:'no-store'}); if(!optionsResponse.ok) throw new Error('options'); const options=await optionsResponse.json(); options.publicKey.challenge=b64ToBytes(options.publicKey.challenge); options.publicKey.allowCredentials=options.publicKey.allowCredentials.map(v=>({...v,id:b64ToBytes(v.id)})); const value=await navigator.credentials.get(options); const c=value; const assertion={id:c.id,rawId:bytesToB64(c.rawId),type:c.type,authenticatorAttachment:c.authenticatorAttachment,clientExtensionResults:c.getClientExtensionResults(),response:{clientDataJSON:bytesToB64(c.response.clientDataJSON),authenticatorData:bytesToB64(c.response.authenticatorData),signature:bytesToB64(c.response.signature),userHandle:c.response.userHandle===null?null:bytesToB64(c.response.userHandle)}}; const response=await fetch(base+'/verify',{method:'POST',headers:{'Content-Type':'application/json','X-Muncho-CSRF':csrf()},body:JSON.stringify({schema:'muncho-passkey-v2-web-verify.v1',assertion:{schema:'muncho-passkey-v2-assertion.v1',credential:assertion}})}); if(!response.ok) throw new Error('verify'); const result=await response.json(); document.getElementById('status').textContent=result.state==='granted'?'Approved. You may return to Muncho.':'Approval failed.'; } catch (_error) { document.getElementById('status').textContent='Approval failed safely.'; approve.disabled=false; }});
load().catch(()=>{approve.disabled=true;document.getElementById('status').textContent='Request unavailable.';});
"""


def _web_headers() -> dict[str, str]:
    return {
        **protocol.UI_SECURITY_HEADERS,
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    }


def create_web_app(config: Mapping[str, Any]) -> Any:
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse, Response
        from starlette.concurrency import run_in_threadpool
    except ImportError as exc:
        raise PasskeyV2ServiceError("passkey_v2_web_runtime_unavailable") from None
    client = UnixServiceClient(Path(str(config["authority_socket"])))
    readiness = _CachedAuthorityReadiness(client)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[..., Any]) -> Any:
        response = await call_next(request)
        for name, item in _web_headers().items():
            response.headers[name] = item
        return response

    @app.exception_handler(PasskeyV2ServiceError)
    async def safe_boundary_error(_request: Request, _exc: PasskeyV2ServiceError) -> Any:
        return JSONResponse(
            {"ok": False, "error": "request_rejected"},
            status_code=400,
            headers=_web_headers(),
        )

    async def bounded_body(request: Request) -> bytes:
        length = request.headers.get("content-length")
        if length is not None:
            try:
                parsed_length = int(length, 10)
            except ValueError as exc:
                raise PasskeyV2ServiceError("passkey_v2_web_body_size_invalid") from None
            if parsed_length < 0 or parsed_length > MAX_HTTP_BODY_BYTES:
                raise PasskeyV2ServiceError("passkey_v2_web_body_size_invalid")
        chunks = bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks) > MAX_HTTP_BODY_BYTES:
                raise PasskeyV2ServiceError("passkey_v2_web_body_size_invalid")
        return bytes(chunks)

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def boundary(request: Request, path: str) -> Any:
        raw = await bounded_body(request)
        route, request_id, parsed = validate_web_request(
            method=request.method,
            path="/" + path,
            headers=dict(request.headers),
            body=raw,
            csrf_cookie=request.cookies.get("muncho_csrf"),
        )
        headers = _web_headers()
        if route == "health":
            # Caddy probes only this process's liveness.  Keeping the probe
            # local prevents a public health flood from occupying the serial
            # WebAuthn authority or its database.
            return JSONResponse(
                {
                    "ok": True,
                    "service": "muncho-passkey-v2-web",
                    "authority_checked": False,
                },
                headers=headers,
            )
        if route == "readiness":
            ready = readiness.status()
            return JSONResponse(
                {
                    "ok": ready,
                    "service": "muncho-passkey-v2-web",
                    "authority_ready": ready,
                },
                status_code=200 if ready else 503,
                headers=headers,
            )
        if route == "javascript":
            return Response(
                _APPROVAL_JS,
                media_type="application/javascript",
                headers=headers,
            )
        assert request_id is not None
        if route == "render":
            await run_in_threadpool(client.call, "render", {"request_id": request_id})
            token = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
            response = HTMLResponse(_APPROVAL_HTML, headers=headers)
            response.set_cookie(
                "muncho_csrf",
                token,
                secure=True,
                httponly=False,
                samesite="strict",
                path=f"/approve/{request_id}",
                max_age=300,
            )
            return response
        if route == "view":
            document = await run_in_threadpool(
                client.call, "render", {"request_id": request_id}
            )
        elif route == "options":
            document = await run_in_threadpool(
                client.call, "options", {"request_id": request_id}
            )
        else:
            assert parsed is not None
            document = await run_in_threadpool(
                client.call,
                "verify",
                {"request_id": request_id, "assertion": parsed["assertion"]},
            )
        return JSONResponse(document, headers=headers)

    return app


def _load_receipt_signer() -> ReceiptSigner:
    root = os.environ.get("CREDENTIALS_DIRECTORY")
    if not root or not os.path.isabs(root):
        raise PasskeyV2ServiceError("passkey_v2_signing_credential_unavailable")
    raw, _metadata = _read_regular_file(
        Path(root) / "receipt-signing-key",
        maximum=16 * 1024,
    )
    try:
        key = serialization.load_pem_private_key(raw, password=None)
    except (TypeError, ValueError) as exc:
        raise PasskeyV2ServiceError("passkey_v2_signing_key_invalid") from None
    if not isinstance(key, Ed25519PrivateKey):
        raise PasskeyV2ServiceError("passkey_v2_signing_key_invalid")
    return ReceiptSigner(key)


def _load_executor_public_key(config: Mapping[str, Any]) -> Ed25519PublicKey:
    path = Path(str(config["receipt_public_key"]))
    raw, metadata = _read_regular_file(path, maximum=16 * 1024)
    if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise PasskeyV2ServiceError("passkey_v2_receipt_public_key_invalid")
    if hashlib.sha256(raw).hexdigest() != config["receipt_public_key_sha256"]:
        raise PasskeyV2ServiceError("passkey_v2_receipt_public_key_invalid")
    try:
        key = serialization.load_pem_public_key(raw)
    except (TypeError, ValueError) as exc:
        raise PasskeyV2ServiceError("passkey_v2_receipt_public_key_invalid") from None
    if not isinstance(key, Ed25519PublicKey):
        raise PasskeyV2ServiceError("passkey_v2_receipt_public_key_invalid")
    return key


def _load_raw_observation_public_key(
    config: Mapping[str, Any], *, role: str, expected_path: Path
) -> tuple[Ed25519PublicKey, str]:
    path_name = f"{role}_observation_public_key"
    digest_name = f"{role}_observation_public_key_sha256"
    key_id_name = f"{role}_observation_public_key_id"
    if config.get(path_name) != str(expected_path):
        raise PasskeyV2ServiceError("passkey_v2_observation_key_path_invalid")
    raw, _state = _read_regular_file(
        expected_path,
        maximum=32,
        expected_uid=0,
        expected_gid=0,
        expected_mode=0o444,
    )
    digest = hashlib.sha256(raw).hexdigest()
    if (
        len(raw) != 32
        or config.get(digest_name) != digest
        or config.get(key_id_name) != digest
    ):
        raise PasskeyV2ServiceError("passkey_v2_observation_key_invalid")
    try:
        return Ed25519PublicKey.from_public_bytes(raw), digest
    except ValueError as exc:
        raise PasskeyV2ServiceError("passkey_v2_observation_key_invalid") from None


def _direct_iam_pins(config: Mapping[str, Any]) -> Mapping[str, Any]:
    ancestors = config.get("direct_iam_resource_ancestor_chain")
    collector_email = (
        "muncho-owner-gate-executor@adventico-ai-platform."
        "iam.gserviceaccount.com"
    )
    if (
        not isinstance(ancestors, list)
        or not ancestors
        or len(ancestors) != len(set(ancestors))
        or re.fullmatch(
            r"organizations/[1-9][0-9]{5,31}",
            str(ancestors[-1]) if ancestors else "",
        )
        is None
        or any(
            re.fullmatch(r"folders/[1-9][0-9]{5,31}", str(item))
            is None
            for item in ancestors[:-1]
        )
    ):
        raise PasskeyV2ServiceError(
            "passkey_v2_direct_iam_config_invalid"
        )
    expected_ancestor_role = (
        trusted_collector._owner_gate_ancestor_read_role(ancestors)
    )
    try:
        external_gcp_admin_trust_root = (
            trusted_collector.validate_external_gcp_admin_trust_root(
                config.get("direct_iam_external_gcp_admin_trust_root"),
                ancestor_chain=ancestors,
            )
        )
    except trusted_collector.TrustedObservationError as exc:
        raise PasskeyV2ServiceError(
            "passkey_v2_direct_iam_config_invalid"
        ) from None
    if (
        config.get("direct_iam_fixed_api_hosts")
        != [
            trusted_collector.COMPUTE_HOST,
            trusted_collector.CLOUD_RESOURCE_MANAGER_HOST,
            trusted_collector.IAM_HOST,
        ]
        or config.get("direct_iam_metadata_oauth_scopes")
        != list(trusted_collector.OWNER_GATE_METADATA_SCOPES)
        or config.get("direct_iam_project_number")
        != growth_contract.PROJECT_NUMBER
        or config.get("direct_iam_project_read_permissions")
        != list(_DIRECT_IAM_PROJECT_PERMISSIONS)
        or config.get("direct_iam_ancestor_read_permissions")
        != list(_DIRECT_IAM_ANCESTOR_PERMISSIONS)
        or config.get("direct_iam_project_read_role")
        != trusted_collector.OWNER_GATE_PROJECT_READ_ROLE
        or config.get("direct_iam_project_read_role_title")
        != trusted_collector.OWNER_GATE_PROJECT_READ_ROLE_TITLE
        or config.get("direct_iam_project_read_role_description")
        != trusted_collector.OWNER_GATE_PROJECT_READ_ROLE_DESCRIPTION
        or config.get("direct_iam_ancestor_read_role")
        != expected_ancestor_role
        or config.get("direct_iam_ancestor_read_role_title")
        != trusted_collector.OWNER_GATE_ANCESTOR_READ_ROLE_TITLE
        or config.get("direct_iam_ancestor_read_role_description")
        != trusted_collector.OWNER_GATE_ANCESTOR_READ_ROLE_DESCRIPTION
        or config.get("direct_iam_mutation_role")
        != trusted_collector.OWNER_GATE_MUTATION_ROLE
        or config.get("direct_iam_mutation_role_title")
        != trusted_collector.OWNER_GATE_MUTATION_ROLE_TITLE
        or config.get("direct_iam_mutation_role_description")
        != trusted_collector.OWNER_GATE_MUTATION_ROLE_DESCRIPTION
        or config.get("direct_iam_mutation_condition")
        != trusted_collector.OWNER_GATE_MUTATION_CONDITION
        or config.get("direct_iam_mutation_binding_member")
        != f"serviceAccount:{collector_email}"
        or config.get("direct_iam_mutation_binding_present") is not False
        or config.get("direct_iam_mutation_activation_seal")
        != str(ACTIVATION_SEAL)
        or config.get("direct_iam_mutation_activation_seal_present")
        is not False
        or config.get("direct_iam_allowed_owner_gate_impersonators")
        != []
        or config.get("direct_iam_owner_gate_user_managed_key_inventory")
        != {
            "requested_key_types": ["USER_MANAGED"],
            "allowed_key_names": [],
        }
        or config.get("direct_iam_external_gcp_admin_trust_root")
        != external_gcp_admin_trust_root
        or config.get("direct_iam_runtime_service_account_email")
        != collector_email
        or _NUMERIC_ID.fullmatch(str(config.get(
            "direct_iam_runtime_instance_numeric_id"
        )))
        is None
        or _NUMERIC_ID.fullmatch(str(config.get(
            "direct_iam_runtime_service_account_unique_id"
        )))
        is None
        or config.get("direct_iam_target_service_account_email")
        != growth_contract.RUNTIME_SERVICE_ACCOUNT
        or _NUMERIC_ID.fullmatch(str(config.get(
            "direct_iam_target_service_account_unique_id"
        )))
        is None
        or config.get("direct_iam_signed_readiness_required") is not True
    ):
        raise PasskeyV2ServiceError(
            "passkey_v2_direct_iam_config_invalid"
        )
    return {
        "expected_project_number": config[
            "direct_iam_project_number"
        ],
        "expected_ancestor_chain": list(ancestors),
        "expected_runtime_instance_numeric_id": config[
            "direct_iam_runtime_instance_numeric_id"
        ],
        "expected_runtime_service_account_email": config[
            "direct_iam_runtime_service_account_email"
        ],
        "expected_runtime_service_account_unique_id": config[
            "direct_iam_runtime_service_account_unique_id"
        ],
        "expected_target_service_account_unique_id": config[
            "direct_iam_target_service_account_unique_id"
        ],
        "expected_metadata_scopes": list(
            trusted_collector.OWNER_GATE_METADATA_SCOPES
        ),
        "expected_external_gcp_admin_trust_root": (
            external_gcp_admin_trust_root
        ),
    }


def _verify_attested_observation_with_direct_iam(
    bundle: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    release_revision: str,
    cloud_public_key: Ed25519PublicKey,
    cloud_public_key_id: str,
    host_public_key: Ed25519PublicKey,
    host_public_key_id: str,
    direct_iam_pins: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    activation = activation_seal_status(
        expected_release_revision=release_revision,
        now_unix=now_unix,
    )
    observation = growth_evidence.validate_attested_observation(
        bundle,
        cloud_public_key=cloud_public_key,
        cloud_public_key_id=cloud_public_key_id,
        host_public_key=host_public_key,
        host_public_key_id=host_public_key_id,
        now_unix=now_unix,
        allowed_states=frozenset({
            "source_ready",
            "resize_complete_boot_required",
            "terminated_after_growth_intent",
            "target_ready",
        }),
        expected_transaction_id=expected["transaction_id"],
        expected_checkpoint=expected["checkpoint"],
        expected_request_binding_sha256=expected[
            "request_binding_sha256"
        ],
        expected_prior_event_head_sha256=expected[
            "prior_event_head_sha256"
        ],
        expected_observation_request_sha256=expected[
            "observation_request_sha256"
        ],
        expected_collection_attempt_id=expected[
            "collection_attempt_id"
        ],
        expected_collection_attempt_sequence=expected[
            "collection_attempt_sequence"
        ],
        expected_collection_attempt_issued_at_unix=expected[
            "collection_attempt_issued_at_unix"
        ],
        expected_collection_attempt_expires_at_unix=expected[
            "collection_attempt_expires_at_unix"
        ],
    )
    canonical_state = {
        "source": "await_source",
        "post_resize": "await_post_resize",
        "post_stop": "await_post_stop",
        "post_start": "await_post_start",
    }.get(str(expected["checkpoint"]))
    unsigned_request = {
        "schema": "muncho-storage-growth-observation-request.v1",
        "transaction_id": expected["transaction_id"],
        "checkpoint": expected["checkpoint"],
        "canonical_state": canonical_state,
        "prior_event_head_sha256": expected[
            "prior_event_head_sha256"
        ],
        "request_binding_sha256": expected[
            "request_binding_sha256"
        ],
        "observation_nonce_sha256": bundle[
            "observation_nonce_sha256"
        ],
        "collection_attempt_id": expected["collection_attempt_id"],
        "collection_attempt_sequence": expected[
            "collection_attempt_sequence"
        ],
        "collection_attempt_issued_at_unix": expected[
            "collection_attempt_issued_at_unix"
        ],
        "collection_attempt_expires_at_unix": expected[
            "collection_attempt_expires_at_unix"
        ],
        "release_sha": release_revision,
        "plan_sha256": storage.exact_storage_plan()["plan_sha256"],
    }
    request = trusted_collector.validate_observation_request({
        **unsigned_request,
        "observation_request_sha256": protocol.sha256_json(
            unsigned_request
        ),
    })
    if (
        request["observation_request_sha256"]
        != expected["observation_request_sha256"]
    ):
        raise PasskeyV2ServiceError(
            "passkey_v2_observation_request_invalid"
        )
    trusted_collector.validate_trusted_iam_projection(
        bundle.get("trusted_iam_projection"),
        observation_request=request,
        candidate_observation=observation,
        now_unix=now_unix,
        expected_project_number=str(
            direct_iam_pins["expected_project_number"]
        ),
        expected_ancestor_chain=direct_iam_pins[
            "expected_ancestor_chain"
        ],
        expected_runtime_instance_numeric_id=str(
            direct_iam_pins[
                "expected_runtime_instance_numeric_id"
            ]
        ),
        expected_runtime_service_account_email=str(
            direct_iam_pins[
                "expected_runtime_service_account_email"
            ]
        ),
        expected_runtime_service_account_unique_id=str(
            direct_iam_pins[
                "expected_runtime_service_account_unique_id"
            ]
        ),
        expected_target_service_account_unique_id=str(
            direct_iam_pins[
                "expected_target_service_account_unique_id"
            ]
        ),
        expected_mutation_binding_present=bool(activation["active"]),
        expected_activation_seal_sha256=activation["seal_sha256"],
        expected_external_gcp_admin_trust_root=direct_iam_pins[
            "expected_external_gcp_admin_trust_root"
        ],
    )
    return observation


def _require_cloud_signer_runtime_ready(
    release_revision: str,
) -> Mapping[str, Any]:
    try:
        readiness = signer_provisioning.verify_cloud_signer_runtime_readiness(
            release_revision
        )
    except signer_provisioning.TrustedSignerProvisioningError as exc:
        raise PasskeyV2ServiceError(
            "passkey_v2_cloud_signer_not_ready"
        ) from None
    if (
        not isinstance(readiness, Mapping)
        or readiness.get("schema")
        != "muncho-cloud-trusted-signer-runtime-readiness.v1"
        or readiness.get("release_revision") != release_revision
        or readiness.get("private_public_identity_matched") is not True
        or readiness.get("config_exact") is not True
        or readiness.get("replay_directory_exact") is not True
        or readiness.get("historical_root_inert_receipt_verified")
        is not True
        or _SHA256.fullmatch(
            str(readiness.get("readiness_sha256", ""))
        )
        is None
    ):
        raise PasskeyV2ServiceError(
            "passkey_v2_cloud_signer_not_ready"
        )
    return dict(readiness)


def _parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    parser.add_argument("operation", choices=[command])
    parser.add_argument("--config", required=True)
    if command in {"serve-authority", "serve-executor"}:
        parser.add_argument("--socket-activation", action="store_true", required=True)
    if command == "preflight":
        parser.add_argument(
            "--require-firewall-readiness-receipt",
            required=True,
        )
    return parser


def web_main(argv: Sequence[str]) -> int:
    args = _parser("serve-web").parse_args(tuple(argv))
    config = _load_config(
        Path(args.config),
        exact_path=WEB_CONFIG,
        schema="muncho-owner-gate-web-config.v1",
        fields=frozenset({
            "schema", "authority_socket", "listen_host", "listen_port",
            "origin", "owner_discord_user_id", "rp_id"
        }),
    )
    if (
        config["authority_socket"] != str(AUTHORITY_SOCKET)
        or config["listen_host"] != "0.0.0.0"
        or config["listen_port"] != 8080
        or config["origin"] != protocol.PRODUCTION_ORIGIN
        or config["rp_id"] != protocol.PRODUCTION_RP_ID
        or config["owner_discord_user_id"] != storage.OWNER_DISCORD_USER_ID
    ):
        raise PasskeyV2ServiceError("passkey_v2_web_config_invalid")
    try:
        import uvicorn
    except ImportError as exc:
        raise PasskeyV2ServiceError("passkey_v2_web_runtime_unavailable") from None
    uvicorn.run(create_web_app(config), host="0.0.0.0", port=8080, access_log=False)
    return 0


def authority_main(argv: Sequence[str]) -> int:
    args = _parser("serve-authority").parse_args(tuple(argv))
    config = _load_config(
        Path(args.config),
        exact_path=AUTHORITY_CONFIG,
        schema="muncho-owner-gate-authority-config.v1",
        fields=frozenset({
            "schema", "database", "executor_socket", "origin", "owner_discord_user_id", "rp_id", "sqlite_journal_mode", "sqlite_synchronous", "totp_dangerous_actions_enabled"
        }),
    )
    if (
        config["database"] != str(AUTHORITY_DB)
        or config["executor_socket"] != str(EXECUTOR_SOCKET)
        or config["origin"] != protocol.PRODUCTION_ORIGIN
        or config["owner_discord_user_id"] != storage.OWNER_DISCORD_USER_ID
        or config["rp_id"] != protocol.PRODUCTION_RP_ID
        or config["sqlite_journal_mode"] != "DELETE"
        or config["sqlite_synchronous"] != "FULL"
        or config["totp_dangerous_actions_enabled"] is not False
    ):
        raise PasskeyV2ServiceError("passkey_v2_authority_config_invalid")
    authority = PasskeyV2AuthorityDatabase(
        AUTHORITY_DB,
        authority_uid=AUTHORITY_UID,
        authority_gid=AUTHORITY_UID,
    )
    signer = _load_receipt_signer()
    return _serve_activated_socket(
        lambda value, peer: handle_authority_frame(
            value,
            authority=authority,
            signer=signer,
            peer_uid=peer,
            now_unix=int(time.time()),
        ),
        expected_path=AUTHORITY_SOCKET,
        expected_name="passkey-authority",
    )


class FixedComputeRestClient:
    """Exact metadata+Compute REST client; no generic URL/operation surface."""

    def __init__(
        self,
        *,
        requester: Callable[
            [str, str, str, Mapping[str, str], bytes | None],
            tuple[int, bytes, Mapping[str, str]],
        ] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._requester = requester or self._request
        self._sleep = sleeper

    @staticmethod
    def _request(
        host: str,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes | None,
    ) -> tuple[int, bytes, Mapping[str, str]]:
        if host == "169.254.169.254":
            if method != "GET" or path != (
                "/computeMetadata/v1/instance/service-accounts/default/token"
            ) or body is not None:
                raise PasskeyV2ServiceError("passkey_v2_metadata_request_forbidden")
            connection: http.client.HTTPConnection = http.client.HTTPConnection(
                host, 80, timeout=10
            )
        elif host == "compute.googleapis.com":
            prefix = f"/compute/v1/projects/{storage.PROJECT}/zones/{storage.ZONE}/"
            if not path.startswith(prefix):
                raise PasskeyV2ServiceError("passkey_v2_compute_request_forbidden")
            try:
                tls_context = trusted_collector.fixed_debian_tls_context()
            except trusted_collector.TrustedObservationError as exc:
                raise PasskeyV2ServiceError(
                    "passkey_v2_compute_tls_invalid"
                ) from None
            connection = http.client.HTTPSConnection(
                host,
                443,
                timeout=30,
                context=tls_context,
            )
        else:
            raise PasskeyV2ServiceError("passkey_v2_http_host_forbidden")
        try:
            connection.request(method, path, body=body, headers=dict(headers))
            response = connection.getresponse()
            raw = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
            if len(raw) > MAX_HTTP_RESPONSE_BYTES:
                raise PasskeyV2ServiceError("passkey_v2_http_response_oversized")
            response_headers = {
                name.lower(): value for name, value in response.getheaders()
            }
            return int(response.status), raw, response_headers
        except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
            raise PasskeyV2ServiceError("passkey_v2_http_request_failed") from None
        finally:
            connection.close()

    def _token(self) -> str:
        status, raw, response_headers = self._requester(
            "169.254.169.254",
            "GET",
            "/computeMetadata/v1/instance/service-accounts/default/token",
            {"Metadata-Flavor": "Google", "Accept": "application/json"},
            None,
        )
        if (
            status != 200
            or response_headers.get("metadata-flavor") != "Google"
        ):
            raise PasskeyV2ServiceError("passkey_v2_metadata_token_unavailable")
        value = decode_strict_json(raw, maximum=MAX_HTTP_RESPONSE_BYTES)
        if (
            set(value) != {"access_token", "expires_in", "token_type"}
            or value.get("token_type") != "Bearer"
            or not isinstance(value.get("access_token"), str)
            or not 20 <= len(value["access_token"]) <= 8192
            or any(character in value["access_token"] for character in "\x00\r\n ")
            or type(value.get("expires_in")) is not int
            or not 60 <= value["expires_in"] <= 3600
        ):
            raise PasskeyV2ServiceError("passkey_v2_metadata_token_invalid")
        return value["access_token"]

    def _compute(
        self,
        method: str,
        resource: str,
        *,
        body: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> Mapping[str, Any]:
        allowed = {
            ("GET", f"disks/{storage.DISK_NAME}"),
            ("POST", f"disks/{storage.DISK_NAME}/resize"),
            ("GET", f"instances/{storage.VM_NAME}"),
            ("POST", f"instances/{storage.VM_NAME}/stop"),
            ("POST", f"instances/{storage.VM_NAME}/start"),
        }
        if (method, resource) not in allowed:
            raise PasskeyV2ServiceError("passkey_v2_compute_request_forbidden")
        if (
            (method == "POST" and (
                not isinstance(request_id, str)
                or re.fullmatch(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
                    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                    request_id,
                ) is None
            ))
            or (method == "GET" and request_id is not None)
        ):
            raise PasskeyV2ServiceError("passkey_v2_compute_request_id_invalid")
        raw_body = None if body is None else protocol.canonical_json_bytes(body)
        headers = {
            "Authorization": f"Bearer {self._token()}",
            "Accept": "application/json",
        }
        if raw_body is not None:
            headers["Content-Type"] = "application/json"
        path = (
            f"/compute/v1/projects/{storage.PROJECT}/zones/{storage.ZONE}/"
            f"{resource}"
        )
        if request_id is not None:
            path = f"{path}?requestId={request_id}"
        status, raw, _response_headers = self._requester(
            "compute.googleapis.com", method, path, headers, raw_body
        )
        if not 200 <= status < 300 or not raw:
            raise PasskeyV2ServiceError("passkey_v2_compute_request_rejected")
        return decode_strict_json(raw, maximum=MAX_HTTP_RESPONSE_BYTES)

    @staticmethod
    def _disk_projection(value: Mapping[str, Any]) -> Mapping[str, Any]:
        users = value.get("users")
        if (
            value.get("id") != storage.DISK_ID
            or value.get("name") != storage.DISK_NAME
            or str(value.get("sizeGb")) not in {"40", "80"}
            or value.get("status") != "READY"
            or not str(value.get("zone", "")).endswith(
                f"/zones/{storage.ZONE}"
            )
            or not str(value.get("type", "")).endswith(
                "/diskTypes/pd-balanced"
            )
            or not isinstance(users, list)
            or users != [
                f"https://www.googleapis.com/compute/v1/projects/{storage.PROJECT}/"
                f"zones/{storage.ZONE}/instances/{storage.VM_NAME}"
            ]
        ):
            raise PasskeyV2ServiceError("passkey_v2_compute_disk_identity_invalid")
        return {
            "id": value["id"],
            "name": value["name"],
            "size_gb": int(str(value["sizeGb"])),
            "status": value["status"],
            "zone": storage.ZONE,
            "type": "pd-balanced",
        }

    @staticmethod
    def _instance_projection(value: Mapping[str, Any]) -> Mapping[str, Any]:
        disks = value.get("disks")
        attachment = (
            disks[0]
            if isinstance(disks, list)
            and len(disks) == 1
            and isinstance(disks[0], Mapping)
            else None
        )
        allowed_status = {
            "RUNNING", "TERMINATED", "STOPPING", "STARTING",
            "PROVISIONING", "STAGING",
        }
        if (
            value.get("id") != storage.VM_INSTANCE_ID
            or value.get("name") != storage.VM_NAME
            or value.get("status") not in allowed_status
            or not str(value.get("zone", "")).endswith(f"/zones/{storage.ZONE}")
            or not isinstance(attachment, Mapping)
            or attachment.get("boot") is not True
            or attachment.get("deviceName") != storage.BOOT_DEVICE_NAME
            or attachment.get("mode") != "READ_WRITE"
            or attachment.get("type") != "PERSISTENT"
            or not str(attachment.get("source", "")).endswith(
                f"/zones/{storage.ZONE}/disks/{storage.DISK_NAME}"
            )
        ):
            raise PasskeyV2ServiceError(
                "passkey_v2_compute_instance_identity_invalid"
            )
        return {
            "id": value["id"],
            "name": value["name"],
            "status": value["status"],
            "zone": storage.ZONE,
            "boot_device_name": storage.BOOT_DEVICE_NAME,
            "disk_name": storage.DISK_NAME,
        }

    def observe(self) -> Mapping[str, Any]:
        disk = self._disk_projection(
            self._compute("GET", f"disks/{storage.DISK_NAME}")
        )
        instance = self._instance_projection(
            self._compute("GET", f"instances/{storage.VM_NAME}")
        )
        unsigned = {
            "schema": "muncho-passkey-v2-compute-live-projection.v1",
            "project": storage.PROJECT,
            "zone": storage.ZONE,
            "disk": disk,
            "instance": instance,
        }
        return {**unsigned, "projection_sha256": protocol.sha256_json(unsigned)}

    def _wait(self, *, instance_status: str | None = None, disk_size: int | None = None) -> Mapping[str, Any]:
        for _attempt in range(COMPUTE_POLL_ATTEMPTS):
            live = self.observe()
            if (
                (instance_status is None or live["instance"]["status"] == instance_status)
                and (disk_size is None or live["disk"]["size_gb"] == disk_size)
            ):
                return live
            self._sleep(2.0)
        raise PasskeyV2ServiceError("passkey_v2_compute_reconciliation_timeout")

    def resize(self, request_id: str) -> None:
        self._compute(
            "POST",
            f"disks/{storage.DISK_NAME}/resize",
            body={"sizeGb": str(storage.TARGET_SIZE_GB)},
            request_id=request_id,
        )

    def stop(self, request_id: str) -> None:
        self._compute(
            "POST",
            f"instances/{storage.VM_NAME}/stop",
            body={},
            request_id=request_id,
        )

    def start(self, request_id: str) -> None:
        self._compute(
            "POST",
            f"instances/{storage.VM_NAME}/start",
            body={},
            request_id=request_id,
        )

    def wait_disk_80(self) -> Mapping[str, Any]:
        return self._wait(disk_size=storage.TARGET_SIZE_GB)

    def wait_terminated(self) -> Mapping[str, Any]:
        return self._wait(instance_status="TERMINATED", disk_size=storage.TARGET_SIZE_GB)

    def wait_running(self) -> Mapping[str, Any]:
        return self._wait(instance_status="RUNNING", disk_size=storage.TARGET_SIZE_GB)


class StorageGrowthComputeExecutor:
    """Mechanical exact-stage executor with durable before/after events."""

    def __init__(
        self,
        database: PasskeyV2ExecutorDatabase,
        client: FixedComputeRestClient,
        *,
        clock: Callable[[], int] = lambda: int(time.time()),
    ) -> None:
        self.database = database
        self.client = client
        self.clock = clock

    def _event(
        self, request_id: str, event_kind: str, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return self.database.append_execution_event(
            request_id=request_id,
            event_kind=event_kind,
            event_payload=payload,
            now_unix=self.clock(),
        )

    @staticmethod
    def _gce_request_id(transaction_id: str, stage: str) -> str:
        raw = bytearray(hashlib.sha256(protocol.canonical_json_bytes({
            "schema": "muncho-passkey-v2-gce-request-id.v1",
            "transaction_id": transaction_id,
            "stage": stage,
        })).digest()[:16])
        raw[6] = (raw[6] & 0x0F) | 0x50
        raw[8] = (raw[8] & 0x3F) | 0x80
        return str(uuid.UUID(bytes=bytes(raw)))

    @staticmethod
    def _attempt_binding(
        transaction_id: str,
        stage: str,
        events: Sequence[Mapping[str, Any]],
    ) -> tuple[str, str]:
        anchor = events[-1]["event_head_sha256"]
        attempt_id = protocol.sha256_json({
            "schema": "muncho-passkey-v2-stage-attempt-id.v1",
            "transaction_id": transaction_id,
            "stage": stage,
            "attempt_anchor_event_head_sha256": anchor,
        })
        return attempt_id, anchor

    def _record_intent(
        self,
        *,
        request_id: str,
        transaction_id: str,
        stage: str,
        operation: str,
        observation_bundle_sha256: str,
        live: Mapping[str, Any],
        context: Mapping[str, Any],
        kinds: set[str],
    ) -> str:
        gce_request_id = self._gce_request_id(transaction_id, stage)
        if f"{stage}_intent" not in kinds:
            self._event(
                request_id,
                f"{stage}_intent",
                {
                    "stage": stage,
                    "requested_operation": operation,
                    "observation_bundle_sha256": observation_bundle_sha256,
                    "live_before_sha256": live["projection_sha256"],
                    "gce_request_id": gce_request_id,
                    "activation_seal_sha256": context[
                        "activation_seal_sha256"
                    ],
                    "firewall_readiness_receipt_sha256": context[
                        "firewall_readiness_receipt_sha256"
                    ],
                },
            )
        return gce_request_id

    def _record_attempt_failure(
        self,
        *,
        request_id: str,
        transaction_id: str,
        stage: str,
        gce_request_id: str,
        attempt_id: str,
        attempt_anchor_event_head_sha256: str,
        live: Mapping[str, Any],
        exc: BaseException,
    ) -> None:
        self._event(
            request_id,
            "attempt_failed",
            {
                "stage": stage,
                "attempt_id": attempt_id,
                "attempt_anchor_event_head_sha256": (
                    attempt_anchor_event_head_sha256
                ),
                "gce_request_id": gce_request_id,
                "failure": type(exc).__name__,
                "live_resource_sha256": live["projection_sha256"],
            },
        )

    def _accept_observation(
        self,
        *,
        request_id: str,
        stage: str,
        observation: Mapping[str, Any],
        context: Mapping[str, Any],
        kinds: set[str],
    ) -> None:
        kind = f"post_{stage}_observation_accepted"
        if kind in kinds:
            return
        self._event(
            request_id,
            kind,
            {
                "after_stage": stage,
                "checkpoint": f"post_{stage}",
                "state": observation["state"],
                "observation_bundle_sha256": context[
                    "observation_bundle_sha256"
                ],
                "observation_bundle": dict(context["observation_bundle"]),
                "observation_nonce_sha256": context[
                    "observation_nonce_sha256"
                ],
            },
        )

    def _progress(
        self,
        *,
        action: Mapping[str, Any],
        after_stage: str,
        required_states: list[str],
        live: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        observation_request = _canonical_observation_request(
            self.database,
            transaction_id=action["transaction_id"],
            release_revision=action["executor_release_sha"],
            now_unix=self.clock(),
        )
        unsigned = {
            "schema": "muncho-passkey-v2-storage-progress.v1",
            "terminal": False,
            "state": "observation_required",
            "after_stage": after_stage,
            "required_states": required_states,
            "live_resource_projection": dict(live),
            "observation_request": observation_request,
        }
        return {**unsigned, "progress_sha256": protocol.sha256_json(unsigned)}

    def _terminal_receipt(
        self,
        *,
        transaction_id: str,
        observation_bundle_sha256: str,
        live: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        current = self.database.read_transaction_state(transaction_id)
        if current["state"] == "terminal":
            stored = self.database.read_terminal_receipt(transaction_id)
            if stored is None:
                raise PasskeyV2ServiceError(
                    "passkey_v2_terminal_receipt_invalid"
                )
            return stored
        events = tuple(current["events"])
        kinds = {event["event_kind"] for event in events}
        if "postflight_complete" not in kinds:
            raise PasskeyV2ServiceError(
                "passkey_v2_terminal_before_postflight_forbidden"
            )
        initial_action = current["authorizations"][0]["action_envelope"]
        latest = current["authorization"]
        unsigned = {
            "schema": "muncho-passkey-v2-storage-terminal-receipt.v1",
            "terminal": True,
            "state": "target_ready",
            "request_id": latest["action_envelope"]["request_id"],
            "transaction_id": transaction_id,
            "release_sha": current["transaction"]["executor_release_sha"],
            "plan_sha256": current["transaction"]["executor_plan_sha256"],
            "authorization_receipt_sha256": latest[
                "authorization_receipt"
            ]["receipt_sha256"],
            "source_observation_sha256": initial_action["action_payload"][
                "source_preflight"
            ]["observation_sha256"],
            "final_observation_bundle_sha256": (
                observation_bundle_sha256
            ),
            "final_live_resource_sha256": live["projection_sha256"],
            "disk_id": storage.DISK_ID,
            "instance_id": storage.VM_INSTANCE_ID,
            "target_size_gb": storage.TARGET_SIZE_GB,
            "conditional_reboot_performed": "stop_complete" in kinds,
            "step_event_heads": [
                event["event_head_sha256"] for event in events
            ],
            # Derived from canonical facts so concurrent reconciliation emits
            # byte-identical terminal bytes.
            "completed_at_unix": max(
                int(event["created_at_unix"]) for event in events
            ),
            "opens_runtime_gate": False,
        }
        return {
            **unsigned,
            "receipt_sha256": protocol.sha256_json(unsigned),
        }

    def _finish_read_only(
        self,
        *,
        transaction_id: str,
        request_id: str,
        observation_bundle_sha256: str,
        live: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        terminal = self._terminal_receipt(
            transaction_id=transaction_id,
            observation_bundle_sha256=observation_bundle_sha256,
            live=live,
        )
        self._event(
            request_id,
            "completed",
            {"terminal_receipt": terminal},
        )
        stored = self.database.read_terminal_receipt(transaction_id)
        if stored != terminal:
            raise PasskeyV2ServiceError(
                "passkey_v2_terminal_receipt_invalid"
            )
        return terminal

    def _read_only_progress(
        self,
        *,
        transaction_id: str,
        remaining_stage: str,
        live: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        state = self.database.read_transaction_state(transaction_id)
        latest = state["authorization"]
        observation_request = _canonical_observation_request(
            self.database,
            transaction_id=transaction_id,
            release_revision=state["transaction"]["executor_release_sha"],
            now_unix=self.clock(),
        )
        unsigned = {
            "schema": "muncho-passkey-v2-read-only-reconciliation.v1",
            "terminal": False,
            "state": "resume_required",
            "remaining_stage": remaining_stage,
            "transaction_id": transaction_id,
            "current_authorization_request_id": latest[
                "action_envelope"
            ]["request_id"],
            "current_execution_window_expires_at_unix": latest[
                "authorization_receipt"
            ]["execution_window_expires_at_unix"],
            "live_resource_projection": dict(live),
            "observation_request": observation_request,
        }
        return {
            **unsigned,
            "reconciliation_sha256": protocol.sha256_json(unsigned),
        }

    def _record_read_only_reconciliation(
        self,
        *,
        request_id: str,
        transaction_id: str,
        stage: str,
        live: Mapping[str, Any],
    ) -> None:
        state = self.database.read_transaction_state(transaction_id)
        attempt_id, anchor = self._attempt_binding(
            transaction_id, stage, state["events"]
        )
        self._event(
            request_id,
            "reconciliation_observed",
            {
                "stage": stage,
                "attempt_id": attempt_id,
                "attempt_anchor_event_head_sha256": anchor,
                "disposition": "already_applied",
                "live_resource_sha256": live["projection_sha256"],
            },
        )
        self._event(
            request_id,
            f"{stage}_complete",
            {
                "stage": stage,
                "disposition": "reconciled",
                "live_after_sha256": live["projection_sha256"],
            },
        )
        required_states = {
            "resize": ["resize_complete_boot_required", "target_ready"],
            "stop": ["terminated_after_growth_intent"],
            "start": ["target_ready"],
        }[stage]
        self._event(
            request_id,
            f"post_{stage}_observation_required",
            {
                "after_stage": stage,
                "checkpoint": f"post_{stage}",
                "required_states": required_states,
                "live_resource_sha256": live["projection_sha256"],
            },
        )

    def reconcile_read_only(
        self,
        *,
        transaction_id: str,
        observation: Mapping[str, Any],
        observation_bundle: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Reconcile exact GET state without readiness, grants, or POSTs."""

        state = self.database.read_transaction_state(transaction_id)
        if state["state"] == "terminal":
            terminal = self.database.read_terminal_receipt(transaction_id)
            if terminal is None:
                raise PasskeyV2ServiceError(
                    "passkey_v2_terminal_receipt_invalid"
                )
            return {
                "schema": "muncho-passkey-v2-read-only-reconciliation.v1",
                "terminal": True,
                "state": "target_ready",
                "terminal_receipt": terminal,
            }
        checked_bundle = growth_evidence.validate_attested_observation_structure(
            observation_bundle
        )
        request_id = state["authorization"]["action_envelope"]["request_id"]
        live = self.client.observe()
        kinds = {event["event_kind"] for event in state["events"]}
        observed_state = str(observation["state"])
        disk_size = live["disk"]["size_gb"]
        instance_status = live["instance"]["status"]

        if "resize_intent" not in kinds:
            raise PasskeyV2ServiceError(
                "passkey_v2_reconciliation_without_intent_forbidden"
            )
        if "resize_complete" not in kinds:
            if (
                observed_state == "source_ready"
                and disk_size == storage.SOURCE_SIZE_GB
                and instance_status == "RUNNING"
            ):
                return self._read_only_progress(
                    transaction_id=transaction_id,
                    remaining_stage="resize",
                    live=live,
                )
            if (
                checked_bundle["checkpoint"] != "post_resize"
                or observed_state not in {
                    "resize_complete_boot_required", "target_ready"
                }
                or disk_size != storage.TARGET_SIZE_GB
                or instance_status != "RUNNING"
            ):
                raise PasskeyV2ServiceError(
                    "passkey_v2_resize_reconciliation_drift"
                )
            self._record_read_only_reconciliation(
                request_id=request_id,
                transaction_id=transaction_id,
                stage="resize",
                live=live,
            )
            state = self.database.read_transaction_state(transaction_id)
            kinds = {event["event_kind"] for event in state["events"]}

        if "post_resize_observation_accepted" not in kinds:
            if (
                checked_bundle["checkpoint"] != "post_resize"
                or observed_state not in {
                    "resize_complete_boot_required", "target_ready"
                }
                or disk_size != storage.TARGET_SIZE_GB
                or instance_status != "RUNNING"
            ):
                raise PasskeyV2ServiceError(
                    "passkey_v2_post_resize_reconciliation_drift"
                )
            self._accept_observation(
                request_id=request_id,
                stage="resize",
                observation=observation,
                context={
                    "observation_bundle": dict(checked_bundle),
                    "observation_bundle_sha256": checked_bundle[
                        "bundle_sha256"
                    ],
                    "observation_nonce_sha256": checked_bundle[
                        "observation_nonce_sha256"
                    ],
                },
                kinds=kinds,
            )
            state = self.database.read_transaction_state(transaction_id)
            kinds = {event["event_kind"] for event in state["events"]}

        if observed_state == "target_ready" and "stop_intent" not in kinds:
            if "postflight_complete" not in kinds:
                self._event(
                    request_id,
                    "postflight_complete",
                    {
                        "state": "target_ready",
                        "observation_bundle_sha256": checked_bundle[
                            "bundle_sha256"
                        ],
                        "live_resource_sha256": live["projection_sha256"],
                    },
                )
            terminal = self._finish_read_only(
                transaction_id=transaction_id,
                request_id=request_id,
                observation_bundle_sha256=checked_bundle["bundle_sha256"],
                live=live,
            )
            return {
                "schema": "muncho-passkey-v2-read-only-reconciliation.v1",
                "terminal": True,
                "state": "target_ready",
                "terminal_receipt": terminal,
            }

        if observed_state == "resize_complete_boot_required":
            if "stop_intent" not in kinds:
                return self._read_only_progress(
                    transaction_id=transaction_id,
                    remaining_stage="stop",
                    live=live,
                )
            if "stop_complete" not in kinds:
                return self._read_only_progress(
                    transaction_id=transaction_id,
                    remaining_stage="stop",
                    live=live,
                )

        if "stop_complete" in kinds and "post_stop_observation_accepted" not in kinds:
            if (
                checked_bundle["checkpoint"] != "post_stop"
                or observed_state != "terminated_after_growth_intent"
                or disk_size != storage.TARGET_SIZE_GB
                or instance_status != "TERMINATED"
            ):
                raise PasskeyV2ServiceError(
                    "passkey_v2_post_stop_reconciliation_drift"
                )
            self._accept_observation(
                request_id=request_id,
                stage="stop",
                observation=observation,
                context={
                    "observation_bundle": dict(checked_bundle),
                    "observation_bundle_sha256": checked_bundle[
                        "bundle_sha256"
                    ],
                    "observation_nonce_sha256": checked_bundle[
                        "observation_nonce_sha256"
                    ],
                },
                kinds=kinds,
            )
            state = self.database.read_transaction_state(transaction_id)
            kinds = {event["event_kind"] for event in state["events"]}

        if "stop_intent" in kinds and "stop_complete" not in kinds:
            if (
                checked_bundle["checkpoint"] == "post_stop"
                and observed_state == "terminated_after_growth_intent"
                and disk_size == storage.TARGET_SIZE_GB
                and instance_status == "TERMINATED"
            ):
                self._record_read_only_reconciliation(
                    request_id=request_id,
                    transaction_id=transaction_id,
                    stage="stop",
                    live=live,
                )
                state = self.database.read_transaction_state(transaction_id)
                kinds = {event["event_kind"] for event in state["events"]}
                self._accept_observation(
                    request_id=request_id,
                    stage="stop",
                    observation=observation,
                    context={
                        "observation_bundle": dict(checked_bundle),
                        "observation_bundle_sha256": checked_bundle[
                            "bundle_sha256"
                        ],
                        "observation_nonce_sha256": checked_bundle[
                            "observation_nonce_sha256"
                        ],
                    },
                    kinds=kinds,
                )
                return self._read_only_progress(
                    transaction_id=transaction_id,
                    remaining_stage="start",
                    live=live,
                )
            return self._read_only_progress(
                transaction_id=transaction_id,
                remaining_stage="stop",
                live=live,
            )

        if "post_stop_observation_accepted" in kinds and "start_intent" not in kinds:
            return self._read_only_progress(
                transaction_id=transaction_id,
                remaining_stage="start",
                live=live,
            )
        if "start_intent" in kinds and "start_complete" not in kinds:
            if (
                checked_bundle["checkpoint"] == "post_start"
                and observed_state == "target_ready"
                and disk_size == storage.TARGET_SIZE_GB
                and instance_status == "RUNNING"
            ):
                self._record_read_only_reconciliation(
                    request_id=request_id,
                    transaction_id=transaction_id,
                    stage="start",
                    live=live,
                )
                state = self.database.read_transaction_state(transaction_id)
                kinds = {event["event_kind"] for event in state["events"]}
            else:
                return self._read_only_progress(
                    transaction_id=transaction_id,
                    remaining_stage="start",
                    live=live,
                )
        if "start_complete" in kinds and "post_start_observation_accepted" not in kinds:
            if (
                checked_bundle["checkpoint"] != "post_start"
                or observed_state != "target_ready"
                or disk_size != storage.TARGET_SIZE_GB
                or instance_status != "RUNNING"
            ):
                raise PasskeyV2ServiceError(
                    "passkey_v2_post_start_reconciliation_drift"
                )
            self._accept_observation(
                request_id=request_id,
                stage="start",
                observation=observation,
                context={
                    "observation_bundle": dict(checked_bundle),
                    "observation_bundle_sha256": checked_bundle[
                        "bundle_sha256"
                    ],
                    "observation_nonce_sha256": checked_bundle[
                        "observation_nonce_sha256"
                    ],
                },
                kinds=kinds,
            )
            state = self.database.read_transaction_state(transaction_id)
            kinds = {event["event_kind"] for event in state["events"]}
        if "post_start_observation_accepted" in kinds:
            if "postflight_complete" not in kinds:
                self._event(
                    request_id,
                    "postflight_complete",
                    {
                        "state": "target_ready",
                        "observation_bundle_sha256": checked_bundle[
                            "bundle_sha256"
                        ],
                        "live_resource_sha256": live["projection_sha256"],
                    },
                )
            terminal = self._finish_read_only(
                transaction_id=transaction_id,
                request_id=request_id,
                observation_bundle_sha256=checked_bundle["bundle_sha256"],
                live=live,
            )
            return {
                "schema": "muncho-passkey-v2-read-only-reconciliation.v1",
                "terminal": True,
                "state": "target_ready",
                "terminal_receipt": terminal,
            }
        raise PasskeyV2ServiceError(
            "passkey_v2_read_only_reconciliation_incomplete"
        )

    def __call__(
        self,
        action: Mapping[str, Any],
        observation: Mapping[str, Any],
        execution_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        checked = storage.validate_storage_growth_envelope(action)
        if set(execution_context) != {
            "authorization_request_id",
            "observation_bundle",
            "observation_bundle_sha256",
            "observation_nonce_sha256",
            "activation_seal_sha256",
            "firewall_readiness_receipt_sha256",
        } or not isinstance(
            execution_context["observation_bundle"], Mapping
        ) or execution_context["observation_bundle"].get(
            "bundle_sha256"
        ) != execution_context["observation_bundle_sha256"] or any(
            _SHA256.fullmatch(str(execution_context[name])) is None
            for name in (
                "observation_bundle_sha256",
                "observation_nonce_sha256",
                "activation_seal_sha256",
                "firewall_readiness_receipt_sha256",
            )
        ) or execution_context["authorization_request_id"] != checked["request_id"]:
            raise PasskeyV2ServiceError(
                "passkey_v2_execution_context_invalid"
            )
        request_id = checked["request_id"]
        state = self.database.read_execution_state(request_id)
        kinds = {event["event_kind"] for event in state["events"]}
        live = self.client.observe()
        transaction_id = checked["transaction_id"]
        observation_bundle_sha = str(
            execution_context["observation_bundle_sha256"]
        )

        if "resize_complete" not in kinds:
            gce_request_id = self._record_intent(
                request_id=request_id,
                transaction_id=transaction_id,
                stage="resize",
                operation="resize_exact_disk_40_to_80",
                observation_bundle_sha256=observation_bundle_sha,
                live=live,
                context=execution_context,
                kinds=kinds,
            )
            state = self.database.read_execution_state(request_id)
            attempt_id, attempt_anchor = self._attempt_binding(
                transaction_id, "resize", state["events"]
            )
            if live["disk"]["size_gb"] == storage.SOURCE_SIZE_GB:
                if observation["state"] != "source_ready":
                    raise PasskeyV2ServiceError(
                        "passkey_v2_resize_source_observation_invalid"
                    )
                try:
                    self.client.resize(gce_request_id)
                    self._event(
                        request_id,
                        "provider_pending",
                        {
                            "stage": "resize",
                            "attempt_id": attempt_id,
                            "attempt_anchor_event_head_sha256": attempt_anchor,
                            "gce_request_id": gce_request_id,
                            "provider_status": "accepted",
                            "live_resource_sha256": live[
                                "projection_sha256"
                            ],
                        },
                    )
                    live = self.client.wait_disk_80()
                except BaseException as exc:
                    self._record_attempt_failure(
                        request_id=request_id,
                        transaction_id=transaction_id,
                        stage="resize",
                        gce_request_id=gce_request_id,
                        attempt_id=attempt_id,
                        attempt_anchor_event_head_sha256=attempt_anchor,
                        live=live,
                        exc=exc,
                    )
                    raise
                disposition = "accepted"
            elif live["disk"]["size_gb"] == storage.TARGET_SIZE_GB:
                disposition = "reconciled"
                self._event(
                    request_id,
                    "reconciliation_observed",
                    {
                        "stage": "resize",
                        "attempt_id": attempt_id,
                        "attempt_anchor_event_head_sha256": attempt_anchor,
                        "disposition": "already_applied",
                        "live_resource_sha256": live["projection_sha256"],
                    },
                )
            else:
                raise PasskeyV2ServiceError("passkey_v2_disk_size_drift")
            self._event(
                request_id,
                "resize_complete",
                {
                    "stage": "resize",
                    "disposition": disposition,
                    "live_after_sha256": live["projection_sha256"],
                },
            )
            self._event(
                request_id,
                "post_resize_observation_required",
                {
                    "after_stage": "resize",
                    "checkpoint": "post_resize",
                    "required_states": [
                        "resize_complete_boot_required", "target_ready"
                    ],
                    "live_resource_sha256": live["projection_sha256"],
                },
            )
            return self._progress(
                action=checked,
                after_stage="resize",
                required_states=["resize_complete_boot_required", "target_ready"],
                live=live,
            )

        if observation["state"] == "target_ready":
            if "post_start_observation_required" in kinds:
                self._accept_observation(
                    request_id=request_id,
                    stage="start",
                    observation=observation,
                    context=execution_context,
                    kinds=kinds,
                )
            elif "post_resize_observation_required" in kinds:
                self._accept_observation(
                    request_id=request_id,
                    stage="resize",
                    observation=observation,
                    context=execution_context,
                    kinds=kinds,
                )
            state = self.database.read_execution_state(request_id)
            kinds = {event["event_kind"] for event in state["events"]}
            if live["disk"]["size_gb"] != storage.TARGET_SIZE_GB or live[
                "instance"
            ]["status"] != "RUNNING":
                raise PasskeyV2ServiceError("passkey_v2_target_live_drift")
            if "postflight_complete" not in kinds:
                self._event(
                    request_id,
                    "postflight_complete",
                    {
                        "state": "target_ready",
                        "observation_bundle_sha256": observation_bundle_sha,
                        "live_resource_sha256": live["projection_sha256"],
                    },
                )
            current = self.database.read_execution_state(request_id)
            step_heads = [event["event_head_sha256"] for event in current["events"]]
            unsigned = {
                "schema": "muncho-passkey-v2-storage-terminal-receipt.v1",
                "terminal": True,
                "state": "target_ready",
                "request_id": request_id,
                "transaction_id": checked["transaction_id"],
                "release_sha": checked["executor_release_sha"],
                "plan_sha256": checked["executor_plan_sha256"],
                "authorization_receipt_sha256": current[
                    "authorization_receipt"
                ]["receipt_sha256"],
                "source_observation_sha256": checked["action_payload"][
                    "source_preflight"
                ]["observation_sha256"],
                "final_observation_bundle_sha256": observation_bundle_sha,
                "final_live_resource_sha256": live["projection_sha256"],
                "disk_id": storage.DISK_ID,
                "instance_id": storage.VM_INSTANCE_ID,
                "target_size_gb": storage.TARGET_SIZE_GB,
                "conditional_reboot_performed": "stop_complete" in kinds,
                "step_event_heads": step_heads,
                "completed_at_unix": self.clock(),
                "opens_runtime_gate": False,
            }
            return {**unsigned, "receipt_sha256": protocol.sha256_json(unsigned)}

        if observation["state"] == "resize_complete_boot_required":
            self._accept_observation(
                request_id=request_id,
                stage="resize",
                observation=observation,
                context=execution_context,
                kinds=kinds,
            )
            state = self.database.read_execution_state(request_id)
            kinds = {event["event_kind"] for event in state["events"]}
            if "stop_complete" not in kinds:
                gce_request_id = self._record_intent(
                    request_id=request_id,
                    transaction_id=transaction_id,
                    stage="stop",
                    operation="conditional_stop_exact_instance",
                    observation_bundle_sha256=observation_bundle_sha,
                    live=live,
                    context=execution_context,
                    kinds=kinds,
                )
                state = self.database.read_execution_state(request_id)
                attempt_id, attempt_anchor = self._attempt_binding(
                    transaction_id, "stop", state["events"]
                )
                status = live["instance"]["status"]
                if status == "RUNNING":
                    try:
                        self.client.stop(gce_request_id)
                        self._event(
                            request_id,
                            "provider_pending",
                            {
                                "stage": "stop",
                                "attempt_id": attempt_id,
                                "attempt_anchor_event_head_sha256": attempt_anchor,
                                "gce_request_id": gce_request_id,
                                "provider_status": "accepted",
                                "live_resource_sha256": live[
                                    "projection_sha256"
                                ],
                            },
                        )
                        live = self.client.wait_terminated()
                    except BaseException as exc:
                        self._record_attempt_failure(
                            request_id=request_id,
                            transaction_id=transaction_id,
                            stage="stop",
                            gce_request_id=gce_request_id,
                            attempt_id=attempt_id,
                            attempt_anchor_event_head_sha256=attempt_anchor,
                            live=live,
                            exc=exc,
                        )
                        raise
                    disposition = "accepted"
                elif status in {"STOPPING", "TERMINATED"}:
                    disposition = "reconciled"
                    live = self.client.wait_terminated()
                    self._event(
                        request_id,
                        "reconciliation_observed",
                        {
                            "stage": "stop",
                            "attempt_id": attempt_id,
                            "attempt_anchor_event_head_sha256": attempt_anchor,
                            "disposition": "already_applied",
                            "live_resource_sha256": live[
                                "projection_sha256"
                            ],
                        },
                    )
                else:
                    raise PasskeyV2ServiceError("passkey_v2_stop_state_invalid")
                self._event(
                    request_id,
                    "stop_complete",
                    {
                        "stage": "stop", "disposition": disposition,
                        "live_after_sha256": live["projection_sha256"],
                    },
                )
                self._event(
                    request_id,
                    "post_stop_observation_required",
                    {
                        "after_stage": "stop",
                        "checkpoint": "post_stop",
                        "required_states": ["terminated_after_growth_intent"],
                        "live_resource_sha256": live["projection_sha256"],
                    },
                )
            return self._progress(
                action=checked,
                after_stage="stop",
                required_states=["terminated_after_growth_intent"],
                live=live,
            )

        if observation["state"] == "terminated_after_growth_intent":
            if "stop_complete" not in kinds:
                raise PasskeyV2ServiceError("passkey_v2_start_without_stop_forbidden")
            self._accept_observation(
                request_id=request_id,
                stage="stop",
                observation=observation,
                context=execution_context,
                kinds=kinds,
            )
            state = self.database.read_execution_state(request_id)
            kinds = {event["event_kind"] for event in state["events"]}
            if "start_complete" not in kinds:
                gce_request_id = self._record_intent(
                    request_id=request_id,
                    transaction_id=transaction_id,
                    stage="start",
                    operation="conditional_start_exact_instance",
                    observation_bundle_sha256=observation_bundle_sha,
                    live=live,
                    context=execution_context,
                    kinds=kinds,
                )
                state = self.database.read_execution_state(request_id)
                attempt_id, attempt_anchor = self._attempt_binding(
                    transaction_id, "start", state["events"]
                )
                status = live["instance"]["status"]
                if status == "TERMINATED":
                    try:
                        self.client.start(gce_request_id)
                        self._event(
                            request_id,
                            "provider_pending",
                            {
                                "stage": "start",
                                "attempt_id": attempt_id,
                                "attempt_anchor_event_head_sha256": attempt_anchor,
                                "gce_request_id": gce_request_id,
                                "provider_status": "accepted",
                                "live_resource_sha256": live[
                                    "projection_sha256"
                                ],
                            },
                        )
                        live = self.client.wait_running()
                    except BaseException as exc:
                        self._record_attempt_failure(
                            request_id=request_id,
                            transaction_id=transaction_id,
                            stage="start",
                            gce_request_id=gce_request_id,
                            attempt_id=attempt_id,
                            attempt_anchor_event_head_sha256=attempt_anchor,
                            live=live,
                            exc=exc,
                        )
                        raise
                    disposition = "accepted"
                elif status in {"STARTING", "PROVISIONING", "STAGING", "RUNNING"}:
                    disposition = "reconciled"
                    live = self.client.wait_running()
                    self._event(
                        request_id,
                        "reconciliation_observed",
                        {
                            "stage": "start",
                            "attempt_id": attempt_id,
                            "attempt_anchor_event_head_sha256": attempt_anchor,
                            "disposition": "already_applied",
                            "live_resource_sha256": live[
                                "projection_sha256"
                            ],
                        },
                    )
                else:
                    raise PasskeyV2ServiceError("passkey_v2_start_state_invalid")
                self._event(
                    request_id,
                    "start_complete",
                    {
                        "stage": "start", "disposition": disposition,
                        "live_after_sha256": live["projection_sha256"],
                    },
                )
                self._event(
                    request_id,
                    "post_start_observation_required",
                    {
                        "after_stage": "start",
                        "checkpoint": "post_start",
                        "required_states": ["target_ready"],
                        "live_resource_sha256": live["projection_sha256"],
                    },
                )
            return self._progress(
                action=checked,
                after_stage="start",
                required_states=["target_ready"],
                live=live,
            )

        raise PasskeyV2ServiceError("passkey_v2_continuation_observation_invalid")


def executor_main(argv: Sequence[str]) -> int:
    operation = tuple(argv[:1])
    command = "preflight" if operation == ("preflight",) else "serve-executor"
    args = _parser(command).parse_args(tuple(argv))
    config = _load_config(
        Path(args.config),
        exact_path=EXECUTOR_CONFIG,
        schema="muncho-owner-gate-executor-config.v1",
        fields=_EXECUTOR_CONFIG_FIELDS,
    )
    if (
        config["api_host"] != "compute.googleapis.com"
        or config["api_private_vip_range"] != "199.36.153.8/30"
        or config["expected_disk_id"] != storage.DISK_ID
        or config["expected_instance_id"] != storage.VM_INSTANCE_ID
        or config["executor_database"] != str(EXECUTOR_DB)
        or config["journal_root"] != str(EXECUTOR_DB.parent)
        or config["mutation_enable_seal"] != str(ACTIVATION_SEAL)
        or config["mutation_enable_seal_uid"] != 0
        or config["mutation_enable_seal_gid"] != EXECUTOR_GID
        or config["mutation_enable_seal_mode"] != "0440"
        or config["receipt_public_key_owner"] != "root:root"
        or config["receipt_public_key_mode"] != "0444"
        or config["metadata_host"] != "169.254.169.254"
        or config["project"] != storage.PROJECT
        or config["target_disk"] != storage.DISK_NAME
        or config["target_instance"] != storage.VM_NAME
        or config["target_boot_device"] != storage.BOOT_DEVICE_NAME
        or config["zone"] != storage.ZONE
        or config["signed_authorization_receipt_required"] is not True
        or config["topology_iam_readiness_seal_required_for_mutation_only"] is not True
    ):
        raise PasskeyV2ServiceError("passkey_v2_executor_config_invalid")
    direct_iam_pins = _direct_iam_pins(config)
    release = _release_revision()
    public_key = _load_executor_public_key(config)
    cloud_observation_key, cloud_observation_key_id = (
        _load_raw_observation_public_key(
            config,
            role="cloud",
            expected_path=CLOUD_OBSERVATION_PUBLIC_KEY,
        )
    )
    host_observation_key, host_observation_key_id = (
        _load_raw_observation_public_key(
            config,
            role="host",
            expected_path=HOST_OBSERVATION_PUBLIC_KEY,
        )
    )
    cloud_signer_runtime_readiness = (
        _require_cloud_signer_runtime_ready(release)
    )
    executor = PasskeyV2ExecutorDatabase(
        EXECUTOR_DB,
        executor_uid=EXECUTOR_UID,
        executor_gid=EXECUTOR_GID,
        pinned_authority_receipt_public_key=public_key,
        pinned_authority_receipt_key_id=protocol.sha256_bytes(
            public_key.public_bytes_raw()
        ),
    )
    if command == "preflight":
        if args.require_firewall_readiness_receipt != config["firewall_readiness_receipt"]:
            raise PasskeyV2ServiceError("passkey_v2_firewall_receipt_path_invalid")
        validate_firewall_readiness(config, now_unix=int(time.time()))
        _require_cloud_signer_runtime_ready(release)
        executor.preflight()
        return 0

    def cloud_attestor_callback(
        frame: Mapping[str, Any], *, now_unix: int
    ) -> Mapping[str, Any]:
        _require_cloud_signer_runtime_ready(release)
        activation = activation_seal_status(
            expected_release_revision=release,
            now_unix=now_unix,
        )
        return trusted_collector.attest_cloud_observation_fixed(
            frame,
            now_unix=now_unix,
            direct_iam_pins=direct_iam_pins,
            expected_mutation_binding_present=bool(
                activation["active"]
            ),
            activation_seal_sha256=activation["seal_sha256"],
        )

    return _serve_activated_socket(
        lambda value, peer: handle_executor_frame(
            value,
            executor=executor,
            peer_uid=peer,
            release_revision=release,
            now_unix=int(time.time()),
            mutation_handler=StorageGrowthComputeExecutor(
                executor, FixedComputeRestClient()
            ),
            readiness_handler=lambda: validate_firewall_readiness(
                config, now_unix=int(time.time())
            ),
            observation_verifier=lambda bundle, expected: _verify_attested_observation_with_direct_iam(
                bundle,
                expected,
                release_revision=release,
                cloud_public_key=cloud_observation_key,
                cloud_public_key_id=cloud_observation_key_id,
                host_public_key=host_observation_key,
                host_public_key_id=host_observation_key_id,
                direct_iam_pins=direct_iam_pins,
                now_unix=int(time.time()),
            ),
            cloud_attestor=cloud_attestor_callback,
            observation_attestors={
                "cloud": {
                    "public_key_id": cloud_observation_key_id,
                    "public_key_b64url": base64.urlsafe_b64encode(
                        cloud_observation_key.public_bytes_raw()
                    ).rstrip(b"=").decode("ascii"),
                },
                "host": {
                    "public_key_id": host_observation_key_id,
                    "public_key_b64url": base64.urlsafe_b64encode(
                        host_observation_key.public_bytes_raw()
                    ).rstrip(b"=").decode("ascii"),
                },
            },
            cloud_signer_runtime_readiness=(
                cloud_signer_runtime_readiness
            ),
        ),
        expected_path=EXECUTOR_SOCKET,
        expected_name="privileged-executor",
    )


def intake_main(argv: Sequence[str]) -> int:
    """Fixed stdin-only IAP entrypoint; no argument or config override exists."""

    if tuple(argv):
        raise PasskeyV2ServiceError("passkey_v2_intake_arguments_forbidden")
    raw = sys.stdin.buffer.read(MAX_FRAME_BYTES + 1)
    if not raw or len(raw) > MAX_FRAME_BYTES or raw.endswith(b"\n"):
        raise PasskeyV2ServiceError("passkey_v2_intake_frame_invalid")
    frame = protocol.decode_canonical_json(raw)
    if not isinstance(frame, Mapping):
        raise PasskeyV2ServiceError("passkey_v2_intake_frame_invalid")
    response = handle_intake_frame(
        frame,
        authority_client=UnixServiceClient(AUTHORITY_SOCKET),
        executor_client=UnixServiceClient(EXECUTOR_SOCKET),
        release_revision=_release_revision(),
        now_unix=int(time.time()),
    )
    sys.stdout.buffer.write(protocol.canonical_json_bytes(response))
    sys.stdout.buffer.flush()
    return 0


def handle_intake_frame(
    value: Any,
    *,
    authority_client: UnixServiceClient,
    executor_client: UnixServiceClient,
    release_revision: str,
    now_unix: int,
    binding_loader: Callable[
        [str], tuple[Mapping[str, Any], str, str]
    ] = _local_action_authority_binding,
    cutover_binding_loader: Callable[
        [str, str],
        tuple[Mapping[str, Any], str, str, Mapping[str, Any]],
    ] = _local_cutover_authority_binding,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "operation", "release_sha", "document", "frame_sha256"
    }:
        raise PasskeyV2ServiceError("passkey_v2_intake_frame_invalid")
    frame = dict(value)
    unsigned = {key: item for key, item in frame.items() if key != "frame_sha256"}
    if (
        frame.get("schema") != storage.REMOTE_FRAME_SCHEMA
        or frame.get("release_sha") != release_revision
        or not isinstance(frame.get("document"), Mapping)
        or frame.get("frame_sha256") != protocol.sha256_json(unsigned)
    ):
        raise PasskeyV2ServiceError("passkey_v2_intake_frame_invalid")
    operation = str(frame["operation"])
    document = dict(frame["document"])
    if operation == "preflight":
        if document:
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        authority = authority_client.call("health", {})
        executor = executor_client.call("preflight", {})
        result = {
            "owner_gate_vm_name": storage.OWNER_GATE_VM_NAME,
            "web_uid": WEB_UID,
            "authority_uid": AUTHORITY_UID,
            "executor_uid": EXECUTOR_UID,
            "authority_socket": str(AUTHORITY_SOCKET),
            "executor_socket": str(EXECUTOR_SOCKET),
            "authority_db": str(AUTHORITY_DB),
            "executor_db": str(EXECUTOR_DB),
            "rp_id": protocol.PRODUCTION_RP_ID,
            "origin": protocol.PRODUCTION_ORIGIN,
            "iap_only": True,
            "local_compute_mutation_available": False,
            "sqlite_synchronous": "FULL",
            "sqlite_begin_immediate": True,
            "totp_dangerous_actions": False,
            "authority_release_sha": release_revision,
            "authority_preflight_sha256": authority["preflight"][
                "preflight_sha256"
            ],
            "executor_preflight_sha256": executor["database"][
                "preflight_sha256"
            ],
            "activation_seal": executor["activation_seal"],
            "observation_attestors": executor[
                "observation_attestors"
            ],
            "cloud_signer_runtime_readiness": executor[
                "cloud_signer_runtime_readiness"
            ],
        }
    elif operation == "observation_request":
        if (
            set(document) != {"plan_sha256", "transaction_id"}
            or document.get("plan_sha256")
            != storage.exact_storage_plan()["plan_sha256"]
        ):
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        result = executor_client.call(
            "observation_request",
            {"transaction_id": document["transaction_id"]},
        )
    elif operation == "attest_cloud_observation":
        if (
            set(document)
            != {"plan_sha256", "transaction_id", "attestation_request"}
            or document.get("plan_sha256")
            != storage.exact_storage_plan()["plan_sha256"]
            or _SHA256.fullmatch(str(document.get("transaction_id"))) is None
            or not isinstance(document.get("attestation_request"), Mapping)
        ):
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        result = executor_client.call(
            "attest_cloud_observation",
            {
                "transaction_id": document["transaction_id"],
                "attestation_request": document["attestation_request"],
            },
        )
    elif operation == "request_production_cutover":
        if set(document) != {"freeze_publication"} or not isinstance(
            document.get("freeze_publication"), Mapping
        ):
            raise PasskeyV2ServiceError(
                "passkey_v2_service_document_invalid"
            )
        publication = document["freeze_publication"]
        plan_sha = publication.get("documents", {}).get(
            "plan", {}
        ).get("plan_sha256") if isinstance(
            publication.get("documents"), Mapping
        ) and isinstance(
            publication.get("documents", {}).get("plan"), Mapping
        ) else None
        if not isinstance(plan_sha, str) or _SHA256.fullmatch(plan_sha) is None:
            raise PasskeyV2ServiceError(
                "passkey_v2_service_document_invalid"
            )
        (
            _runtime_binding,
            manifest_sha,
            host_receipt_sha,
            _trust_bundle,
        ) = cutover_binding_loader(release_revision, plan_sha)
        try:
            action = production_cutover.build_cutover_action_envelope(
                freeze_publication=publication,
                authority_release_sha=release_revision,
                authority_manifest_sha256=manifest_sha,
                authority_host_receipt_sha256=host_receipt_sha,
                issued_at_unix=now_unix,
            )
        except production_cutover.ProductionCutoverPasskeyError:
            raise PasskeyV2ServiceError(
                "passkey_v2_cutover_request_invalid"
            ) from None
        created = authority_client.call(
            "create_request", {"action_envelope": action}
        )
        result = {
            **created,
            "release_sha": release_revision,
            "plan_sha256": action["executor_plan_sha256"],
            "freeze_publication_sha256": action[
                "external_iam_receipt_sha256"
            ],
            "action_payload_sha256": action[
                "action_payload_sha256"
            ],
            "transaction_id": action["transaction_id"],
            "approval_url": (
                f"{protocol.PRODUCTION_ORIGIN}/approve/"
                f"{action['request_id']}"
            ),
            "passkey_only": True,
            "single_use": True,
            "production_mutation_performed": False,
        }
    elif operation == "consume_production_cutover":
        if set(document) != {
            "freeze_publication", "request_id", "consume_attempt_id"
        } or not isinstance(document.get("freeze_publication"), Mapping):
            raise PasskeyV2ServiceError(
                "passkey_v2_service_document_invalid"
            )
        publication = document["freeze_publication"]
        plan_sha = publication.get("documents", {}).get(
            "plan", {}
        ).get("plan_sha256") if isinstance(
            publication.get("documents"), Mapping
        ) and isinstance(
            publication.get("documents", {}).get("plan"), Mapping
        ) else None
        if (
            not isinstance(plan_sha, str)
            or _SHA256.fullmatch(plan_sha) is None
            or not isinstance(document.get("request_id"), str)
            or _SHA256.fullmatch(document["request_id"]) is None
            or not isinstance(document.get("consume_attempt_id"), str)
            or _SHA256.fullmatch(document["consume_attempt_id"]) is None
        ):
            raise PasskeyV2ServiceError(
                "passkey_v2_service_document_invalid"
            )
        runtime_binding, _manifest, _host, trust_bundle = (
            cutover_binding_loader(release_revision, plan_sha)
        )
        preview = authority_client.call(
            "render", {"request_id": document["request_id"]}
        )
        try:
            rendered = preview["exact_action_envelope_canonical_json"]
            if not isinstance(rendered, str):
                raise PasskeyV2ServiceError(
                    "passkey_v2_cutover_request_binding_invalid"
                )
            preview_action = protocol.decode_canonical_json(
                rendered.encode("utf-8", errors="strict")
            )
            production_cutover.validate_cutover_action_envelope(
                preview_action,
                freeze_publication=publication,
            )
        except (
            KeyError,
            UnicodeError,
            protocol.PasskeyV2ProtocolError,
            production_cutover.ProductionCutoverPasskeyError,
        ):
            raise PasskeyV2ServiceError(
                "passkey_v2_cutover_request_binding_invalid"
            ) from None
        consumed = authority_client.call(
            "consume",
            {
                "request_id": document["request_id"],
                "consume_attempt_id": document["consume_attempt_id"],
                "runtime_binding": runtime_binding,
            },
        )
        if consumed.get("action_envelope") != preview_action:
            raise PasskeyV2ServiceError(
                "passkey_v2_cutover_request_binding_invalid"
            )
        try:
            proof = production_cutover.build_passkey_proof(
                freeze_publication=publication,
                action_envelope=consumed["action_envelope"],
                challenge_record=consumed["challenge_record"],
                grant_record=consumed["grant_record"],
                authorization_receipt=consumed["authorization_receipt"],
                trust_bundle=trust_bundle,
            )
        except (
            KeyError,
            production_cutover.ProductionCutoverPasskeyError,
        ):
            raise PasskeyV2ServiceError(
                "passkey_v2_cutover_proof_invalid"
            ) from None
        result = {
            "request_id": document["request_id"],
            "consume_attempt_id": document["consume_attempt_id"],
            "disposition": consumed["disposition"],
            "passkey_proof": proof,
            "release_sha": release_revision,
            "plan_sha256": plan_sha,
            "single_use": True,
            "production_mutation_performed": False,
        }
    elif operation in {"request_initial", "request_resume"}:
        expected_fields = (
            {"plan_sha256", "transaction_id", "source_preflight"}
            if operation == "request_initial"
            else {
                "plan_sha256", "transaction_id",
                "continuation_preflight",
            }
        )
        if set(document) != expected_fields or document.get(
            "plan_sha256"
        ) != storage.exact_storage_plan()["plan_sha256"]:
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        transaction_id = str(document["transaction_id"])
        if _SHA256.fullmatch(transaction_id) is None:
            raise PasskeyV2ServiceError("passkey_v2_transaction_id_invalid")
        runtime_binding, manifest_sha, host_receipt_sha = binding_loader(
            release_revision
        )
        del runtime_binding
        observation_request = executor_client.call(
            "observation_request", {"transaction_id": transaction_id}
        )
        if observation_request["canonical_state"] == "terminal":
            raise PasskeyV2ServiceError(
                "passkey_v2_transaction_already_terminal"
            )
        if operation == "request_initial":
            bundle = document["source_preflight"]
            if storage.transaction_id_for_source(bundle) != transaction_id:
                raise PasskeyV2ServiceError(
                    "passkey_v2_transaction_id_invalid"
                )
            stage = "intent"
            prior_receipt = protocol.GENESIS_JOURNAL_HEAD_SHA256
            prior_event_head = observation_request[
                "prior_event_head_sha256"
            ]
            checkpoint = observation_request["checkpoint"]
            if (
                checkpoint != "source"
                or prior_event_head
                != protocol.GENESIS_JOURNAL_HEAD_SHA256
            ):
                raise PasskeyV2ServiceError(
                    "passkey_v2_initial_observation_request_invalid"
                )
        else:
            bundle = document["continuation_preflight"]
            checkpoint = observation_request["checkpoint"]
            context = executor_client.call(
                "context", {"transaction_id": transaction_id}
            )
            milestones = set(context["milestone_event_kinds"])
            if "resize_complete" not in milestones:
                stage = "resize"
            elif "post_resize_observation_accepted" not in milestones:
                raise PasskeyV2ServiceError(
                    "passkey_v2_resume_reconciliation_required"
                )
            elif "stop_complete" not in milestones:
                stage = "stop"
            elif "post_stop_observation_accepted" not in milestones:
                raise PasskeyV2ServiceError(
                    "passkey_v2_resume_reconciliation_required"
                )
            elif "start_complete" not in milestones:
                stage = "start"
            elif "post_start_observation_accepted" not in milestones:
                raise PasskeyV2ServiceError(
                    "passkey_v2_resume_reconciliation_required"
                )
            else:
                raise PasskeyV2ServiceError(
                    "passkey_v2_resume_mutation_not_required"
                )
            prior_receipt = context[
                "prior_authoritative_receipt_sha256"
            ]
            prior_event_head = observation_request[
                "prior_event_head_sha256"
            ]
        expected_binding = _expected_observation_binding(
            observation_request
        )
        verified = executor_client.call(
            "verify_observation",
            {
                "observation_bundle": bundle,
                "expected_binding": expected_binding,
            },
        )
        observation_state = verified["observation"]["state"]
        if (
            (stage in {"intent", "resize"} and observation_state != "source_ready")
            or (stage == "stop" and observation_state
                != "resize_complete_boot_required")
            or (stage == "start" and observation_state
                != "terminated_after_growth_intent")
        ):
            raise PasskeyV2ServiceError(
                "passkey_v2_resume_observation_state_invalid"
            )
        action = storage.build_storage_growth_envelope(
            source_preflight=bundle,
            transaction_id=transaction_id,
            stage=stage,
            release_sha=release_revision,
            authority_manifest_sha256=manifest_sha,
            authority_host_receipt_sha256=host_receipt_sha,
            prior_authoritative_receipt_sha256=prior_receipt,
            prior_event_head_sha256=prior_event_head,
            issued_at_unix=now_unix,
        )
        result = authority_client.call(
            "create_request",
            {"action_envelope": action},
        )
        result = {
            **result,
            "release_sha": release_revision,
            "plan_sha256": action["executor_plan_sha256"],
            "transaction_id": action["transaction_id"],
            "attested_observation_bundle_sha256": bundle["bundle_sha256"],
            "passkey_only": True,
            "local_mutation_authority": False,
            "opens_runtime_gate": False,
        }
    elif operation == "verify_terminal":
        if (
            set(document) != {"plan_sha256", "transaction_id"}
            or document.get("plan_sha256")
            != storage.exact_storage_plan()["plan_sha256"]
        ):
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        result = executor_client.call(
            "terminal", {"transaction_id": document["transaction_id"]}
        )
        result = {
            **result,
            "release_sha": release_revision,
            "plan_sha256": document["plan_sha256"],
        }
    elif operation == "execute_or_recover":
        if set(document) != {
            "request_id", "consume_attempt_id", "transaction_id",
            "continuation_preflight",
        }:
            raise PasskeyV2ServiceError("passkey_v2_service_document_invalid")
        transaction_id = str(document["transaction_id"])
        terminal = executor_client.call(
            "terminal", {"transaction_id": transaction_id}
        )
        if terminal["terminal"] is True:
            result = {
                "mutation_receipt": terminal["terminal_receipt"],
                "terminal_replay": True,
                "release_sha": release_revision,
                "plan_sha256": storage.exact_storage_plan()["plan_sha256"],
                "transaction_id": transaction_id,
                "request_id": document["request_id"],
                "authoritative_remote_executor": True,
                "local_mutation_performed": False,
                "passkey_method": "passkey",
            }
            unsigned_response = {
                "schema": storage.REMOTE_RESPONSE_SCHEMA,
                "operation": operation,
                "release_sha": release_revision,
                "ok": True,
                "document": result,
            }
            return {
                **unsigned_response,
                "response_sha256": protocol.sha256_json(unsigned_response),
            }
        reconciliation = executor_client.call(
            "reconcile_read_only",
            {
                "transaction_id": transaction_id,
                "continuation_observation": document[
                    "continuation_preflight"
                ],
            },
        )
        if reconciliation["terminal"] is True:
            result = {
                "mutation_receipt": reconciliation["terminal_receipt"],
                "terminal_replay": False,
                "read_only_reconciliation": True,
                "release_sha": release_revision,
                "plan_sha256": storage.exact_storage_plan()["plan_sha256"],
                "transaction_id": transaction_id,
                "request_id": document["request_id"],
                "authoritative_remote_executor": True,
                "local_mutation_performed": False,
                "passkey_method": "passkey",
            }
            unsigned_response = {
                "schema": storage.REMOTE_RESPONSE_SCHEMA,
                "operation": operation,
                "release_sha": release_revision,
                "ok": True,
                "document": result,
            }
            return {
                **unsigned_response,
                "response_sha256": protocol.sha256_json(unsigned_response),
            }
        if reconciliation["state"] == "resume_required":
            observation_request = reconciliation["observation_request"]
            supplied = document["continuation_preflight"]
            observation_is_current = (
                isinstance(supplied, Mapping)
                and supplied.get("prior_event_head_sha256")
                == observation_request["prior_event_head_sha256"]
                and supplied.get("request_binding_sha256")
                == observation_request["request_binding_sha256"]
            )
            current_request = reconciliation[
                "current_authorization_request_id"
            ]
            current_window_live = now_unix < reconciliation[
                "current_execution_window_expires_at_unix"
            ]
            if (
                not observation_is_current
                or (
                    document["request_id"] == current_request
                    and not current_window_live
                )
            ):
                resume = {
                    **reconciliation,
                    "fresh_observation_required": not observation_is_current,
                    "passkey_resume_required": (
                        observation_is_current and not current_window_live
                    ),
                }
                result = {
                    "mutation_receipt": resume,
                    "terminal_replay": False,
                    "read_only_reconciliation": True,
                    "release_sha": release_revision,
                    "plan_sha256": storage.exact_storage_plan()[
                        "plan_sha256"
                    ],
                    "transaction_id": transaction_id,
                    "request_id": document["request_id"],
                    "authoritative_remote_executor": True,
                    "local_mutation_performed": False,
                    "passkey_method": "passkey",
                }
                unsigned_response = {
                    "schema": storage.REMOTE_RESPONSE_SCHEMA,
                    "operation": operation,
                    "release_sha": release_revision,
                    "ok": True,
                    "document": result,
                }
                return {
                    **unsigned_response,
                    "response_sha256": protocol.sha256_json(
                        unsigned_response
                    ),
                }
        runtime_binding, _manifest_sha, _host_receipt_sha = binding_loader(
            release_revision
        )
        consumed = authority_client.call(
            "consume",
            {
                "request_id": document["request_id"],
                "consume_attempt_id": document["consume_attempt_id"],
                "runtime_binding": runtime_binding,
            },
        )
        if consumed["action_envelope"]["transaction_id"] != transaction_id:
            raise PasskeyV2ServiceError(
                "passkey_v2_execution_transaction_invalid"
            )
        executed = executor_client.call(
            "execute",
            {
                "authorization_receipt": consumed["authorization_receipt"],
                "action_envelope": consumed["action_envelope"],
                "challenge_record": consumed["challenge_record"],
                "grant_record": consumed["grant_record"],
                "continuation_observation": document[
                    "continuation_preflight"
                ],
            },
        )
        action = consumed["action_envelope"]
        result = {
            **executed,
            "release_sha": release_revision,
            "plan_sha256": action["executor_plan_sha256"],
            "transaction_id": action["transaction_id"],
            "request_id": action["request_id"],
            "authoritative_remote_executor": True,
            "local_mutation_performed": False,
            "passkey_method": "passkey",
        }
    else:
        raise PasskeyV2ServiceError("passkey_v2_intake_operation_forbidden")
    unsigned_response = {
        "schema": storage.REMOTE_RESPONSE_SCHEMA,
        "operation": operation,
        "release_sha": release_revision,
        "ok": True,
        "document": result,
    }
    return {
        **unsigned_response,
        "response_sha256": protocol.sha256_json(unsigned_response),
    }
