#!/usr/bin/env python3
"""Owner-side transport for production-shaped capability-canary leases.

Secret values are accepted only from a protected local Codex auth file or
inherited stdin, carried in bounded binary frames, and sent to one fixed
packaged module over the existing pinned IAP/SSH transport.  No secret value
or digest is accepted on argv, placed in the environment, or returned in a
receipt.  Codex refresh tokens are deliberately never leased: the canary is
bounded to one still-valid access token so it cannot rotate or invalidate the
owner's desktop OAuth session.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import secrets
import stat
import struct
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from scripts.canary.full_canary_owner_launcher import (
    GcloudOwnerAccessToken,
    IapStoppedReleaseTransport,
    LocalLauncherProvenance,
    OwnerLauncherError,
    PHASE_B_OWNER_PRIVATE_KEY_PATH,
    PHASE_B_OWNER_PUBLIC_KEY_COMMENT,
    PHASE_B_OWNER_PUBLIC_KEY_FINGERPRINT,
    PHASE_B_OWNER_PUBLIC_KEY_PATH,
    PHASE_B_SSH_KEYGEN,
    PinnedGcloudConfiguration,
    _PhaseBOwnerExternalSigner,
    harden_owner_secret_process,
    require_trusted_owner_runtime,
)
from gateway.support_ops_team_registry import (
    SKYVISION_LOCKED_NONPUBLIC_CHANNEL_IDS,
)
RUNTIME_MODULE = "gateway.canonical_capability_canary_runtime"
LIVE_DRIVER_MODULE = "gateway.canonical_capability_canary_live_driver"
PRODUCER_FOUNDATION_MODULE = (
    "gateway.canonical_capability_canary_producer_units"
)
PRODUCER_FOUNDATION_BOOTSTRAP_SCHEMA = (
    "muncho-production-capability-producer-foundation-owner-bootstrap.v1"
)
FOUNDATION_PREPARE_REQUEST_SCHEMA = (
    "muncho-capability-producer-foundation-prepare-request.v1"
)
FOUNDATION_PREPARATION_SCHEMA = (
    "muncho-capability-producer-foundation-preparation.v1"
)
FOUNDATION_INSTALL_REQUEST_SCHEMA = (
    "muncho-capability-producer-foundation-install-request.v1"
)
FOUNDATION_INSTALL_RECEIPT_SCHEMA = (
    "muncho-capability-producer-foundation-installation.v1"
)
PRODUCER_FOUNDATION_SSHSIG_NAMESPACE = (
    "muncho-production-capability-canary-producer-foundation-v1"
)
PRODUCTION_OBSERVATION_SSHSIG_NAMESPACE = (
    "muncho-production-capability-production-observation-v1"
)
PRODUCTION_OBSERVATION_ENVELOPE_SCHEMA = (
    "muncho-production-capability-owner-signed-production-observation.v1"
)
PRODUCTION_OBSERVATION_WAIT_REQUEST_SCHEMA = (
    "muncho-production-capability-production-observation-wait-request.v1"
)
PRODUCTION_OBSERVATION_STAGE_RECEIPT_SCHEMA = (
    "muncho-production-capability-production-observation-stage.v1"
)
FIXTURE_PUBLICATION_RECEIPT_SCHEMA = (
    "muncho-production-capability-canary-fixture-publication.v1"
)
REVIEWED_FIXTURE_PATH = "/etc/muncho/capability-canary/reviewed-live-fixture.json"
FIXTURE_PUBLICATION_ROOT = (
    "/var/lib/muncho-capability-canary-control/fixture-publications"
)
CAPABILITY_PREFLIGHT_SCHEMA = "muncho-production-capability-runtime-preflight.v2"
CAPABILITY_APPROVAL_SCHEMA = "muncho-production-capability-owner-approval.v1"
CAPABILITY_BROWSER_HOST_IDENTITY_SCHEMA = (
    "muncho-production-capability-browser-host-identity.v1"
)
CAPABILITY_EXECUTION_HOST_IDENTITY_SCHEMA = (
    "muncho-production-capability-execution-host-identity.v1"
)
CAPABILITY_LEASE_FRAME_SCHEMA = (
    "muncho-production-capability-secret-lease-frame.v1"
)
CAPABILITY_PLAN_INPUTS_SCHEMA = (
    "muncho-production-capability-plan-publication-inputs.v1"
)
CAPABILITY_PLAN_PUBLICATION_AUTHORITY_SCHEMA = (
    "muncho-production-capability-plan-publication-authority.v1"
)
CAPABILITY_PLAN_PUBLICATION_SCOPE = (
    "production_capability_canary_plan_publication"
)
CAPABILITY_PLAN_AUTHORING_CONTEXT_SCHEMA = (
    "muncho-production-capability-plan-authoring-context.v1"
)
CAPABILITY_PLAN_AUTHORING_RECEIPT_SCHEMA = (
    "muncho-production-capability-plan-authoring-receipt.v1"
)
FULL_CANARY_TERMINAL_RECEIPT_SCHEMA = (
    "muncho-full-canary-session-bound-owner-receipt.v1"
)
FULL_CANARY_STAGED_PLAN_SCHEMA = "muncho-full-canary-runtime-plan.v1"
FULL_CANARY_STAGED_PLAN_PATH = (
    "/etc/muncho/full-canary/staged/runtime-plan.json"
)
CAPABILITY_FIXTURE_AUTHORITY_SCHEMA = (
    "muncho-production-capability-canary-fixture-authority.v1"
)
CAPABILITY_FIXTURE_AUTHORING_RECEIPT_SCHEMA = (
    "muncho-production-capability-canary-fixture-authoring-receipt.v1"
)
CAPABILITY_FIXTURE_SSHSIG_NAMESPACE = (
    "muncho-production-capability-canary-owner-v1"
)
CAPABILITY_PRODUCER_BOOTSTRAP_RECEIPT_SCHEMA = (
    "muncho-production-capability-producer-foundation-owner-bootstrap.v1"
)
PRODUCTION_CANARY_PUBLIC_GUILD_ID = "1282725267068157972"
PRODUCTION_CANARY_PUBLIC_CHANNEL_ID = "1526858760100909066"
PRODUCTION_OWNER_USER_ID = "1279454038731264061"
LOCKED_NONPUBLIC_CHANNEL_IDS = SKYVISION_LOCKED_NONPUBLIC_CHANNEL_IDS
CAPABILITY_BITRIX_FOUNDATION_INPUTS_SCHEMA = (
    "muncho-production-capability-bitrix-foundation-inputs.v1"
)
CAPABILITY_BITRIX_FOUNDATION_AUTHORITY_SCHEMA = (
    "muncho-production-capability-bitrix-foundation-authority.v1"
)
CAPABILITY_BITRIX_FOUNDATION_SCOPE = (
    "production_capability_canary_bitrix_foundation"
)
CODEX_FRAME_MAGIC = b"MCO1"
MAC_OPS_FRAME_MAGIC = b"MCG1"
CONNECTOR_FRAME_MAGIC = b"MCD1"
API_CONTROL_FRAME_MAGIC = b"MCK1"
ROUTEBACK_FRAME_MAGIC = b"MDR1"
BITRIX_FRAME_MAGIC = b"MBX1"
_SECRET_LEASE_MAGIC_BY_KIND = {
    "api_server_control_key": API_CONTROL_FRAME_MAGIC,
    "discord_routeback_token": ROUTEBACK_FRAME_MAGIC,
    "bitrix_operational_edge_webhook": BITRIX_FRAME_MAGIC,
    "discord_connector_token": CONNECTOR_FRAME_MAGIC,
    "mac_ops_gitlab_env": MAC_OPS_FRAME_MAGIC,
    "codex_access_token": CODEX_FRAME_MAGIC,
}
_REVISION_RE = __import__("re").compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LEASE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_MAX_OUTPUT = 2 * 1024 * 1024
_MAX_SECRET_BYTES = 64 * 1024
_MAX_PLAN_INPUT_BYTES = 512 * 1024
_MAX_BITRIX_FOUNDATION_INPUT_BYTES = 128 * 1024
_MAX_FIXTURE_AUTHORITY_BYTES = 512 * 1024
_MAX_FULL_CANARY_TERMINAL_RECEIPT_BYTES = 2 * 1024 * 1024
_MAX_PLAN_AUTHORING_PLAN_BYTES = 1024 * 1024
_MAX_LEASE_SECONDS = 1_200
_BITRIX_FOUNDATION_MAX_SECONDS = 1_200
_FIXTURE_AUTHORITY_MAX_SECONDS = 900
RELEASE_STORAGE_PREFLIGHT_SCHEMA = (
    "muncho-capability-release-storage-preflight.v1"
)
RELEASE_BASE = "/opt/muncho-canary-releases"
MINIMUM_PACKAGING_FREE_BYTES = 8 * 1024 * 1024 * 1024
MAXIMUM_MANAGED_RELEASES = 64


class CapabilityLauncherProvenance(LocalLauncherProvenance):
    """Bind this secret-bearing launcher to its exact committed release byte."""

    _RELATIVE_MODULE = "scripts/canary/capability_canary_owner_launcher.py"


class ProductionCapabilityObserverProvenance(LocalLauncherProvenance):
    """Bind the in-memory production observer to exact committed bytes."""

    _RELATIVE_MODULE = "scripts/canary/production_capability_observer.py"


def require_capability_launcher_provenance(release_sha: str) -> str:
    return CapabilityLauncherProvenance(module_path=__file__)(release_sha)


def _production_observer_source(release_sha: str) -> tuple[str, str]:
    path = Path(__file__).resolve().parent / "production_capability_observer.py"
    digest = ProductionCapabilityObserverProvenance(module_path=path)(release_sha)
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or not 0 < before.st_size <= 512 * 1024
    ):
        raise OwnerLauncherError("production_observer_source_invalid")
    raw = path.read_bytes()
    after = path.lstat()
    if (
        len(raw) != before.st_size
        or hashlib.sha256(raw).hexdigest() != digest
        or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
    ):
        raise OwnerLauncherError("production_observer_source_changed")
    try:
        return raw.decode("utf-8", errors="strict"), digest
    except UnicodeError:
        raise OwnerLauncherError("production_observer_source_invalid") from None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("capability owner JSON contains duplicate keys")
        result[key] = value
    return result


def _decode_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} digest is invalid")
    return value


_PLAN_INPUT_IDENTITY_FIELDS = frozenset(
    {
        "mac_ops_uid",
        "mac_ops_gid",
        "connector_uid",
        "connector_gid",
        "bitrix_operational_edge_uid",
        "bitrix_operational_edge_gid",
        "bitrix_operational_edge_client_gid",
        "browser_uid",
        "browser_gid",
        "worker_uid",
        "worker_gid",
        "worker_client_gid",
    }
)
_PLAN_INPUT_DISCORD_FIELDS = frozenset(
    {
        "connector_bot_user_id",
        "routeback_bot_user_id",
        "allowed_guild_ids",
        "allowed_channel_ids",
        "allowed_user_ids",
    }
)
_PLAN_INPUT_ARTIFACT_FIELDS = frozenset(
    {
        "browser_node_sha256",
        "browser_wrapper_sha256",
        "browser_native_sha256",
        "browser_executable_sha256",
        "agent_browser_config_sha256",
        "worker_bwrap_sha256",
        "worker_shell_sha256",
        "runtime_dependency_manifest_sha256",
        "bitrix_operational_edge_asset_manifest_sha256",
        "bitrix_operational_edge_rendered_unit_sha256",
        "bitrix_operational_edge_rendered_config_sha256",
        "bitrix_operational_edge_rendered_trust_sha256",
        "bitrix_operational_edge_identity_bootstrap_receipt_sha256",
        "bitrix_operational_edge_receipt_public_key_id",
        "bitrix_operational_edge_key_bootstrap_receipt_sha256",
    }
)
_PLAN_AUTHORITY_FIELDS = frozenset(
    {
        "schema",
        "scope",
        "revision",
        "full_canary_plan_sha256",
        "plan_sha256",
        "owner_subject_sha256",
        "authority_kind",
        "cryptographic_owner_proof",
        "inputs",
        "secret_material_recorded",
        "secret_digest_recorded",
        "semantic_content_recorded",
        "authority_sha256",
    }
)


def _snowflake(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.isdigit()
        or value.startswith("0")
        or len(value) > 25
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _snowflake_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 64:
        raise ValueError(f"{label} is invalid")
    result = [_snowflake(item, label) for item in value]
    if result != sorted(set(result)):
        raise ValueError(f"{label} is not canonical")
    return result


def validate_plan_publication_inputs(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "identities",
        "discord",
        "artifacts",
        "inputs_sha256",
    }:
        raise ValueError("capability plan publication inputs are not exact")
    identities = value.get("identities")
    discord = value.get("discord")
    artifacts = value.get("artifacts")
    if (
        value.get("schema") != CAPABILITY_PLAN_INPUTS_SCHEMA
        or not isinstance(identities, Mapping)
        or set(identities) != _PLAN_INPUT_IDENTITY_FIELDS
        or not isinstance(discord, Mapping)
        or set(discord) != _PLAN_INPUT_DISCORD_FIELDS
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != _PLAN_INPUT_ARTIFACT_FIELDS
    ):
        raise ValueError("capability plan publication inputs are invalid")
    if any(
        type(identities[field]) is not int
        or not 0 < identities[field] < (1 << 31)
        for field in _PLAN_INPUT_IDENTITY_FIELDS
    ):
        raise ValueError("capability plan publication identities are invalid")
    connector_bot = _snowflake(
        discord.get("connector_bot_user_id"), "connector bot identity"
    )
    routeback_bot = _snowflake(
        discord.get("routeback_bot_user_id"), "route-back bot identity"
    )
    if connector_bot == routeback_bot:
        raise ValueError("capability canary bot identities are not distinct")
    for field in (
        "allowed_guild_ids",
        "allowed_channel_ids",
        "allowed_user_ids",
    ):
        _snowflake_list(discord.get(field), field)
    if (
        discord["allowed_guild_ids"] != [PRODUCTION_CANARY_PUBLIC_GUILD_ID]
        or discord["allowed_channel_ids"]
        != [PRODUCTION_CANARY_PUBLIC_CHANNEL_ID]
        or discord["allowed_user_ids"] != [PRODUCTION_OWNER_USER_ID]
        or LOCKED_NONPUBLIC_CHANNEL_IDS.intersection(
            discord["allowed_channel_ids"]
        )
    ):
        raise ValueError("capability canary public Discord target is invalid")
    for field in _PLAN_INPUT_ARTIFACT_FIELDS:
        _digest(artifacts.get(field), field)
    unsigned = {
        key: item for key, item in value.items() if key != "inputs_sha256"
    }
    if value.get("inputs_sha256") != hashlib.sha256(
        _canonical_bytes(unsigned)
    ).hexdigest():
        raise ValueError("capability plan publication inputs digest drifted")
    return dict(value)


def read_plan_publication_inputs(path: Path) -> Mapping[str, Any]:
    if (
        not path.is_absolute()
        or path != Path(os.path.normpath(os.fspath(path)))
    ):
        raise OwnerLauncherError("capability_plan_input_path_invalid")
    try:
        raw = _stable_owner_file(path, maximum=_MAX_PLAN_INPUT_BYTES)
        value = _decode_json(raw, label="capability plan publication inputs")
        if raw != _canonical_bytes(value):
            raise ValueError
        return validate_plan_publication_inputs(value)
    except (OSError, ValueError, OwnerLauncherError):
        raise OwnerLauncherError("capability_plan_input_source_invalid") from None


def validate_bitrix_foundation_inputs(value: Any) -> Mapping[str, Any]:
    fields = {
        "schema",
        "service_uid",
        "service_gid",
        "socket_client_gid",
        "business_edge_uid",
        "asset_manifest_sha256",
        "inputs_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Bitrix foundation inputs are not exact")
    if (
        value.get("schema") != CAPABILITY_BITRIX_FOUNDATION_INPUTS_SCHEMA
        or any(
            type(value.get(field)) is not int
            or not 0 < value[field] < (1 << 31)
            for field in (
                "service_uid",
                "service_gid",
                "socket_client_gid",
                "business_edge_uid",
            )
        )
        or len(
            {
                value["service_uid"],
                value["service_gid"],
                value["socket_client_gid"],
                value["business_edge_uid"],
            }
        )
        != 4
    ):
        raise ValueError("Bitrix foundation inputs are invalid")
    _digest(value.get("asset_manifest_sha256"), "Bitrix asset manifest")
    unsigned = {key: item for key, item in value.items() if key != "inputs_sha256"}
    if value.get("inputs_sha256") != hashlib.sha256(
        _canonical_bytes(unsigned)
    ).hexdigest():
        raise ValueError("Bitrix foundation inputs digest drifted")
    return dict(value)


def read_bitrix_foundation_inputs(path: Path) -> Mapping[str, Any]:
    if not path.is_absolute() or path != Path(os.path.normpath(os.fspath(path))):
        raise OwnerLauncherError("capability_bitrix_foundation_input_path_invalid")
    try:
        raw = _stable_owner_file(
            path,
            maximum=_MAX_BITRIX_FOUNDATION_INPUT_BYTES,
        )
        value = _decode_json(raw, label="Bitrix foundation inputs")
        if raw != _canonical_bytes(value):
            raise ValueError
        return validate_bitrix_foundation_inputs(value)
    except (OSError, ValueError, OwnerLauncherError):
        raise OwnerLauncherError(
            "capability_bitrix_foundation_input_source_invalid"
        ) from None


def build_bitrix_foundation_authority(
    *,
    revision: str,
    full_canary_plan_sha256: str,
    release_artifact_sha256: str,
    owner_subject_sha256: str,
    inputs: Mapping[str, Any],
    now_unix: int | None = None,
    ttl_seconds: int = 900,
) -> Mapping[str, Any]:
    validated = validate_bitrix_foundation_inputs(inputs)
    issued = int(time.time()) if now_unix is None else now_unix
    if (
        _REVISION_RE.fullmatch(revision or "") is None
        or type(issued) is not int
        or type(ttl_seconds) is not int
        or not 60 <= ttl_seconds <= _BITRIX_FOUNDATION_MAX_SECONDS
    ):
        raise ValueError("Bitrix foundation authority window is invalid")
    unsigned = {
        "schema": CAPABILITY_BITRIX_FOUNDATION_AUTHORITY_SCHEMA,
        "scope": CAPABILITY_BITRIX_FOUNDATION_SCOPE,
        "revision": revision,
        "full_canary_plan_sha256": _digest(
            full_canary_plan_sha256,
            "full-canary plan",
        ),
        "release_artifact_sha256": _digest(
            release_artifact_sha256,
            "release artifact",
        ),
        "owner_subject_sha256": _digest(owner_subject_sha256, "owner subject"),
        "authority_kind": "trusted_gcloud_owner_explicit_foundation_digest",
        "cryptographic_owner_proof": False,
        "issued_at_unix": issued,
        "expires_at_unix": issued + ttl_seconds,
        "identities": {
            "service_uid": validated["service_uid"],
            "service_gid": validated["service_gid"],
            "socket_client_gid": validated["socket_client_gid"],
            "business_edge_uid": validated["business_edge_uid"],
        },
        "asset_manifest_sha256": validated["asset_manifest_sha256"],
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    if "plan_sha256" in unsigned or "capability_plan_sha256" in unsigned:
        raise AssertionError("Bitrix foundation authority is self-referential")
    return {
        **unsigned,
        "authority_sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
    }


def build_plan_publication_authority(
    *,
    revision: str,
    full_canary_plan_sha256: str,
    plan_sha256: str,
    owner_subject_sha256: str,
    inputs: Mapping[str, Any],
) -> Mapping[str, Any]:
    validated = validate_plan_publication_inputs(inputs)
    if _REVISION_RE.fullmatch(revision or "") is None:
        raise ValueError("capability plan revision is invalid")
    unsigned = {
        "schema": CAPABILITY_PLAN_PUBLICATION_AUTHORITY_SCHEMA,
        "scope": CAPABILITY_PLAN_PUBLICATION_SCOPE,
        "revision": revision,
        "full_canary_plan_sha256": _digest(
            full_canary_plan_sha256, "full-canary plan"
        ),
        "plan_sha256": _digest(plan_sha256, "capability plan"),
        "owner_subject_sha256": _digest(owner_subject_sha256, "owner subject"),
        "authority_kind": "trusted_gcloud_owner_explicit_plan_digest",
        "cryptographic_owner_proof": False,
        "inputs": {
            "identities": validated["identities"],
            "discord": validated["discord"],
            "artifacts": validated["artifacts"],
        },
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    authority = {
        **unsigned,
        "authority_sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
    }
    if set(authority) != _PLAN_AUTHORITY_FIELDS:
        raise AssertionError("capability plan authority fields drifted")
    return authority


def build_plan_publication_inputs(
    *,
    identities: Mapping[str, Any],
    connector_bot_user_id: str,
    routeback_bot_user_id: str,
    artifacts: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build only the exact public mechanical inputs for one capability plan."""

    unsigned = {
        "schema": CAPABILITY_PLAN_INPUTS_SCHEMA,
        "identities": dict(identities),
        "discord": {
            "connector_bot_user_id": connector_bot_user_id,
            "routeback_bot_user_id": routeback_bot_user_id,
            "allowed_guild_ids": [PRODUCTION_CANARY_PUBLIC_GUILD_ID],
            "allowed_channel_ids": [PRODUCTION_CANARY_PUBLIC_CHANNEL_ID],
            "allowed_user_ids": [PRODUCTION_OWNER_USER_ID],
        },
        "artifacts": dict(artifacts),
    }
    return validate_plan_publication_inputs(
        {
            **unsigned,
            "inputs_sha256": hashlib.sha256(
                _canonical_bytes(unsigned)
            ).hexdigest(),
        }
    )


def validate_plan_authoring_context(
    value: Any,
    *,
    revision: str,
    terminal_receipt: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "revision",
        "staged_plan_path",
        "staged_plan_b64",
        "staged_plan_bytes",
        "staged_plan_file_sha256",
        "staged_plan_identity",
        "full_canary_plan_sha256",
        "capability_inputs_sha256",
        "capability_plan_sha256",
        "mutation_performed",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OwnerLauncherError("capability_plan_authoring_context_invalid")
    raw = dict(value)
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    identity = raw.get("staged_plan_identity")
    if (
        raw.get("schema") != CAPABILITY_PLAN_AUTHORING_CONTEXT_SCHEMA
        or raw.get("revision") != revision
        or raw.get("staged_plan_path") != FULL_CANARY_STAGED_PLAN_PATH
        or raw.get("full_canary_plan_sha256")
        != terminal_receipt.get("full_canary_plan_sha256")
        or raw.get("capability_inputs_sha256") != inputs.get("inputs_sha256")
        or raw.get("mutation_performed") is not False
        or not isinstance(raw.get("staged_plan_b64"), str)
        or type(raw.get("staged_plan_bytes")) is not int
        or not 0 < raw["staged_plan_bytes"] <= _MAX_PLAN_AUTHORING_PLAN_BYTES
        or not isinstance(identity, Mapping)
        or set(identity)
        != {"device", "inode", "uid", "gid", "mode", "size", "mtime_ns"}
        or any(
            type(identity.get(field)) is not int or identity[field] < 0
            for field in ("device", "inode", "uid", "gid", "size", "mtime_ns")
        )
        or identity.get("uid") != 0
        or identity.get("gid") != 0
        or identity.get("mode") != "0400"
        or identity.get("size") != raw.get("staged_plan_bytes")
        or raw.get("receipt_sha256")
        != hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    ):
        raise OwnerLauncherError("capability_plan_authoring_context_invalid")
    try:
        plan_payload = base64.b64decode(
            raw["staged_plan_b64"].encode("ascii"),
            validate=True,
        )
        plan_value = _decode_json(plan_payload, label="staged full-canary plan")
    except (ValueError, UnicodeError):
        raise OwnerLauncherError("capability_plan_authoring_context_invalid") from None
    plan_unsigned = {
        key: item
        for key, item in plan_value.items()
        if key != "full_canary_plan_sha256"
    }
    if (
        len(plan_payload) != raw["staged_plan_bytes"]
        or plan_payload != _canonical_bytes(plan_value)
        or hashlib.sha256(plan_payload).hexdigest()
        != raw.get("staged_plan_file_sha256")
        or plan_value.get("schema") != FULL_CANARY_STAGED_PLAN_SCHEMA
        or plan_value.get("revision") != revision
        or plan_value.get("full_canary_plan_sha256")
        != raw.get("full_canary_plan_sha256")
        or hashlib.sha256(_canonical_bytes(plan_unsigned)).hexdigest()
        != raw.get("full_canary_plan_sha256")
    ):
        raise OwnerLauncherError("capability_plan_authoring_context_invalid")
    try:
        _digest(raw.get("staged_plan_file_sha256"), "staged full-canary plan")
        _digest(raw.get("capability_plan_sha256"), "capability plan")
        _digest(raw.get("receipt_sha256"), "plan authoring context")
    except ValueError:
        raise OwnerLauncherError("capability_plan_authoring_context_invalid") from None
    return raw


def author_capability_plan_inputs(
    transport: "CapabilityCanaryTransport",
    *,
    revision: str,
    terminal_receipt_file: Path,
    output_file: Path,
    identities: Mapping[str, Any],
    connector_bot_user_id: str,
    routeback_bot_user_id: str,
    artifacts: Mapping[str, Any],
) -> Mapping[str, Any]:
    terminal = validate_full_canary_terminal_receipt(
        _read_owner_canonical_json(
            terminal_receipt_file,
            maximum=_MAX_FULL_CANARY_TERMINAL_RECEIPT_BYTES,
            label="full-canary terminal receipt",
        ),
        revision=revision,
    )
    inputs = build_plan_publication_inputs(
        identities=identities,
        connector_bot_user_id=connector_bot_user_id,
        routeback_bot_user_id=routeback_bot_user_id,
        artifacts=artifacts,
    )
    context = validate_plan_authoring_context(
        transport.invoke(
            revision,
            "collect-plan-authoring-context",
            frame=_canonical_bytes(inputs),
        ),
        revision=revision,
        terminal_receipt=terminal,
        inputs=inputs,
    )
    payload = _canonical_bytes(inputs)
    file_sha256 = _write_owner_file_no_replace(output_file, payload)
    unsigned = {
        "schema": CAPABILITY_PLAN_AUTHORING_RECEIPT_SCHEMA,
        "revision": revision,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "full_canary_plan_sha256": terminal["full_canary_plan_sha256"],
        "staged_plan_file_sha256": context["staged_plan_file_sha256"],
        "collector_receipt_sha256": context["receipt_sha256"],
        "capability_inputs_sha256": inputs["inputs_sha256"],
        "capability_plan_sha256": context["capability_plan_sha256"],
        "output_file": str(output_file),
        "output_file_sha256": file_sha256,
        "output_file_mode": "0600",
        "mutation_scope": "local_owner_file_create_only",
        "cloud_mutation_performed": False,
        "secret_material_recorded": False,
        "semantic_content_recorded": False,
    }
    return {
        **unsigned,
        "receipt_sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
    }


def _jwt_exp(token: bytes) -> int:
    try:
        pieces = token.decode("ascii", errors="strict").split(".")
        if len(pieces) != 3:
            raise ValueError
        payload = pieces[1] + "=" * (-len(pieces[1]) % 4)
        value = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
        expiry = value.get("exp")
    except Exception as exc:
        raise ValueError("Codex access token is not a bounded JWT") from exc
    if type(expiry) is not int or expiry <= 0:
        raise ValueError("Codex access token expiry is invalid")
    return expiry


def build_secret_lease_frame(
    *,
    kind: str,
    secret: bytes | bytearray,
    plan_sha256: str,
    owner_subject_sha256: str,
    now_unix: int | None = None,
    ttl_seconds: int = 900,
    lease_id: str | None = None,
) -> bytearray:
    """Build the stdlib-only owner half of the fixed secret-frame protocol."""

    if kind not in _SECRET_LEASE_MAGIC_BY_KIND:
        raise ValueError("secret lease kind is invalid")
    payload = bytes(secret)
    if not payload or len(payload) > _MAX_SECRET_BYTES:
        raise ValueError("secret lease payload size is invalid")
    issued = int(time.time()) if now_unix is None else now_unix
    if (
        type(issued) is not int
        or type(ttl_seconds) is not int
        or not 60 <= ttl_seconds <= _MAX_LEASE_SECONDS
    ):
        raise ValueError("secret lease window is invalid")
    lease = uuid.uuid4().hex if lease_id is None else lease_id
    if _LEASE_ID_RE.fullmatch(lease) is None:
        raise ValueError("secret lease id is invalid")
    token_expiry = _jwt_exp(payload) if kind == "codex_access_token" else None
    if token_expiry is not None and token_expiry < issued + ttl_seconds + 120:
        raise ValueError("Codex access token expires inside the canary window")
    metadata = {
        "schema": CAPABILITY_LEASE_FRAME_SCHEMA,
        "kind": kind,
        "plan_sha256": _digest(plan_sha256, "capability plan"),
        "owner_subject_sha256": _digest(owner_subject_sha256, "owner subject"),
        "lease_id": lease,
        "issued_at_unix": issued,
        "expires_at_unix": issued + ttl_seconds,
        "secret_bytes": len(payload),
        "token_expires_at_unix": token_expiry,
    }
    encoded = _canonical_bytes(metadata)
    return bytearray(
        _SECRET_LEASE_MAGIC_BY_KIND[kind]
        + struct.pack(">II", len(encoded), len(payload))
        + encoded
        + payload
    )
_REMOTE_STORAGE_PREFLIGHT = r'''
import hashlib,json,os,re,stat,sys
base="/opt/muncho-canary-releases"
revision=sys.argv[1]
minimum=8589934592
maximum=64
canonical=lambda value:json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(",",":"),allow_nan=False).encode("ascii")
release_re=re.compile(r"[0-9a-f]{40}")
blocker=""
base_record={"path":base,"state":"absent","device":-1,"inode":-1,"uid":-1,"gid":-1,"mode":"0000"}
records=[]
try:
    base_stat=os.lstat(base)
except FileNotFoundError:
    blocker="release_base_absent"
else:
    base_record={"path":base,"state":"present_exact","device":base_stat.st_dev,"inode":base_stat.st_ino,"uid":base_stat.st_uid,"gid":base_stat.st_gid,"mode":format(stat.S_IMODE(base_stat.st_mode),"04o")}
    if not stat.S_ISDIR(base_stat.st_mode) or stat.S_ISLNK(base_stat.st_mode) or base_stat.st_uid != 0 or base_stat.st_gid != 0 or stat.S_IMODE(base_stat.st_mode) != 0o755:
        blocker="release_base_identity_invalid"
    else:
        names=sorted(os.listdir(base))
        if len(names) > maximum:
            blocker="release_inventory_too_large"
        elif any(release_re.fullmatch(name) is None for name in names):
            blocker="release_base_contains_unmanaged_entries"
        else:
            for name in names:
                path=base+"/"+name
                item=os.lstat(path)
                if not stat.S_ISDIR(item.st_mode) or stat.S_ISLNK(item.st_mode) or item.st_uid != 0 or item.st_gid != 0 or stat.S_IMODE(item.st_mode) != 0o755:
                    blocker="release_identity_invalid"
                    records=[]
                    break
                records.append({"revision":name,"path":path,"device":item.st_dev,"inode":item.st_ino,"uid":item.st_uid,"gid":item.st_gid,"mode":"0755","mtime_ns":item.st_mtime_ns,"ctime_ns":item.st_ctime_ns})
            if not blocker:
                second=[]
                for name in sorted(os.listdir(base)):
                    item=os.lstat(base+"/"+name)
                    second.append((name,item.st_dev,item.st_ino,item.st_mode,item.st_uid,item.st_gid,item.st_mtime_ns,item.st_ctime_ns))
                first=[(item["revision"],item["device"],item["inode"],stat.S_IFDIR|int(item["mode"],8),item["uid"],item["gid"],item["mtime_ns"],item["ctime_ns"]) for item in records]
                if first != second:
                    blocker="release_inventory_changed"
capacity_root=base if base_record["state"] == "present_exact" else "/opt"
capacity=os.statvfs(capacity_root)
available=capacity.f_bavail*capacity.f_frsize
protected=[]
by_revision={item["revision"]:item for item in records}
if not blocker:
    if revision in by_revision:
        protected.append(by_revision[revision]["path"])
    rollback=[item for item in records if item["revision"] != revision]
    if rollback:
        newest=max(rollback,key=lambda item:(item["mtime_ns"],item["revision"]))
        protected.append(newest["path"])
protected=sorted(set(protected))
candidates=sorted((item for item in records if item["path"] not in protected),key=lambda item:(item["mtime_ns"],item["revision"])) if not blocker else []
packaging_allowed=not blocker and available >= minimum
if not blocker and not packaging_allowed:
    blocker="minimum_packaging_free_bytes_not_met"
unsigned={"schema":"muncho-capability-release-storage-preflight.v1","ok":packaging_allowed,"revision":revision,"release_base":base_record,"capacity":{"available_bytes":available,"minimum_packaging_free_bytes":minimum,"shortfall_bytes":max(0,minimum-available)},"retention":{"maximum_managed_releases":maximum,"protect_target_release":True,"protect_newest_rollback_release":True},"protected_release_paths":protected,"cleanup_candidates":candidates,"cleanup_mutation_performed":False,"cleanup_requires_fresh_owner_approval":True,"arbitrary_path_cleanup_allowed":False,"blocker":blocker}
result={**unsigned,"receipt_sha256":hashlib.sha256(canonical(unsigned)).hexdigest()}
sys.stdout.buffer.write(canonical(result)+b"\n")
'''.strip()

_REMOTE_PLAN_AUTHORING_CONTEXT = r'''
import base64,hashlib,json,os,stat,sys
from gateway.canonical_full_canary_runtime import FullCanaryPlan
from gateway.canonical_capability_canary_runtime import build_capability_plan
path="/etc/muncho/full-canary/staged/runtime-plan.json"
revision=sys.argv[1]
canonical=lambda value:json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(",",":"),allow_nan=False).encode("ascii")
def pairs(items):
    value={}
    for key,item in items:
        if key in value: raise ValueError("duplicate key")
        value[key]=item
    return value
def decode(raw):
    return json.loads(raw.decode("utf-8","strict"),object_pairs_hook=pairs,parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))
input_raw=sys.stdin.buffer.read(524289)
if not input_raw or len(input_raw)>524288: raise ValueError("plan inputs oversized")
inputs=decode(input_raw)
if input_raw!=canonical(inputs) or set(inputs)!={"schema","identities","discord","artifacts","inputs_sha256"} or inputs.get("schema")!="muncho-production-capability-plan-publication-inputs.v1": raise ValueError("plan inputs invalid")
input_unsigned={key:item for key,item in inputs.items() if key!="inputs_sha256"}
if inputs.get("inputs_sha256")!=hashlib.sha256(canonical(input_unsigned)).hexdigest(): raise ValueError("plan inputs digest invalid")
before=os.lstat(path)
if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_nlink!=1 or before.st_uid!=0 or before.st_gid!=0 or stat.S_IMODE(before.st_mode)!=0o400 or not 0<before.st_size<=1048576: raise ValueError("staged plan identity invalid")
fd=os.open(path,os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0))
try:
    opened=os.fstat(fd); chunks=[]; total=0
    while True:
        chunk=os.read(fd,min(65536,1048577-total))
        if not chunk: break
        chunks.append(chunk); total+=len(chunk)
        if total>1048576: raise ValueError("staged plan oversized")
    after=os.fstat(fd)
finally: os.close(fd)
reachable=os.lstat(path)
identity=lambda item:(item.st_dev,item.st_ino,item.st_mode,item.st_nlink,item.st_uid,item.st_gid,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
if identity(before)!=identity(opened) or identity(before)!=identity(after) or identity(before)!=identity(reachable): raise ValueError("staged plan changed")
plan_raw=b"".join(chunks)
plan_value=decode(plan_raw)
if plan_raw!=canonical(plan_value): raise ValueError("staged plan noncanonical")
full=FullCanaryPlan.from_mapping(plan_value)
if full.revision!=revision: raise ValueError("staged plan revision mismatch")
identities=inputs["identities"]; discord=inputs["discord"]; artifacts=inputs["artifacts"]
capability=build_capability_plan(full_plan=full,**identities,connector_bot_user_id=discord["connector_bot_user_id"],routeback_bot_user_id=discord["routeback_bot_user_id"],connector_allowed_guild_ids=discord["allowed_guild_ids"],connector_allowed_channel_ids=discord["allowed_channel_ids"],connector_allowed_user_ids=discord["allowed_user_ids"],**artifacts)
unsigned={"schema":"muncho-production-capability-plan-authoring-context.v1","revision":revision,"staged_plan_path":path,"staged_plan_b64":base64.b64encode(plan_raw).decode("ascii"),"staged_plan_bytes":len(plan_raw),"staged_plan_file_sha256":hashlib.sha256(plan_raw).hexdigest(),"staged_plan_identity":{"device":before.st_dev,"inode":before.st_ino,"uid":before.st_uid,"gid":before.st_gid,"mode":format(stat.S_IMODE(before.st_mode),"04o"),"size":before.st_size,"mtime_ns":before.st_mtime_ns},"full_canary_plan_sha256":full.sha256,"capability_inputs_sha256":inputs["inputs_sha256"],"capability_plan_sha256":capability.sha256,"mutation_performed":False}
result={**unsigned,"receipt_sha256":hashlib.sha256(canonical(unsigned)).hexdigest()}
sys.stdout.buffer.write(canonical(result)+b"\n")
'''.strip()


def validate_release_storage_preflight(
    value: Mapping[str, Any], *, revision: str
) -> Mapping[str, Any]:
    """Validate the fixed read-only capacity and retention projection."""

    fields = {
        "schema",
        "ok",
        "revision",
        "release_base",
        "capacity",
        "retention",
        "protected_release_paths",
        "cleanup_candidates",
        "cleanup_mutation_performed",
        "cleanup_requires_fresh_owner_approval",
        "arbitrary_path_cleanup_allowed",
        "blocker",
        "receipt_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != RELEASE_STORAGE_PREFLIGHT_SCHEMA
        or value.get("revision") != revision
        or _REVISION_RE.fullmatch(revision) is None
        or value.get("cleanup_mutation_performed") is not False
        or value.get("cleanup_requires_fresh_owner_approval") is not True
        or value.get("arbitrary_path_cleanup_allowed") is not False
        or not isinstance(value.get("blocker"), str)
        or re.fullmatch(r"[a-z0-9_]*", str(value.get("blocker"))) is None
    ):
        raise OwnerLauncherError("capability_release_storage_preflight_invalid")
    base = value.get("release_base")
    if (
        not isinstance(base, Mapping)
        or set(base)
        != {"path", "state", "device", "inode", "uid", "gid", "mode"}
        or base.get("path") != RELEASE_BASE
        or base.get("state") not in {"absent", "present_exact"}
        or any(type(base.get(name)) is not int for name in ("device", "inode", "uid", "gid"))
        or not isinstance(base.get("mode"), str)
        or re.fullmatch(r"[0-7]{4}", base["mode"]) is None
        or (
            base.get("state") == "present_exact"
            and (
                base.get("uid") != 0
                or base.get("gid") != 0
                or base.get("mode") != "0755"
                or base.get("device", -1) < 0
                or base.get("inode", -1) < 0
            )
        )
    ):
        raise OwnerLauncherError("capability_release_storage_preflight_invalid")
    capacity = value.get("capacity")
    retention = value.get("retention")
    if (
        not isinstance(capacity, Mapping)
        or set(capacity)
        != {
            "available_bytes",
            "minimum_packaging_free_bytes",
            "shortfall_bytes",
        }
        or any(
            type(capacity.get(name)) is not int or capacity[name] < 0
            for name in capacity
        )
        or capacity["minimum_packaging_free_bytes"]
        != MINIMUM_PACKAGING_FREE_BYTES
        or capacity["shortfall_bytes"]
        != max(
            0,
            MINIMUM_PACKAGING_FREE_BYTES - capacity["available_bytes"],
        )
        or retention
        != {
            "maximum_managed_releases": MAXIMUM_MANAGED_RELEASES,
            "protect_target_release": True,
            "protect_newest_rollback_release": True,
        }
    ):
        raise OwnerLauncherError("capability_release_storage_preflight_invalid")

    protected = value.get("protected_release_paths")
    candidates = value.get("cleanup_candidates")
    if (
        not isinstance(protected, list)
        or any(not isinstance(path, str) for path in protected)
        or protected != sorted(set(protected))
        or len(protected) > 2
        or any(
            not isinstance(path, str)
            or re.fullmatch(
                rf"{re.escape(RELEASE_BASE)}/[0-9a-f]{{40}}", path
            )
            is None
            for path in protected
        )
        or not isinstance(candidates, list)
        or len(candidates) > MAXIMUM_MANAGED_RELEASES
    ):
        raise OwnerLauncherError("capability_release_storage_preflight_invalid")
    candidate_fields = {
        "revision",
        "path",
        "device",
        "inode",
        "uid",
        "gid",
        "mode",
        "mtime_ns",
        "ctime_ns",
    }
    candidate_paths: list[str] = []
    for item in candidates:
        if (
            not isinstance(item, Mapping)
            or set(item) != candidate_fields
            or not isinstance(item.get("revision"), str)
            or _REVISION_RE.fullmatch(item["revision"]) is None
            or item.get("path")
            != f"{RELEASE_BASE}/{item['revision']}"
            or item["path"] in protected
            or item["revision"] == revision
            or item.get("uid") != 0
            or item.get("gid") != 0
            or item.get("mode") != "0755"
            or any(
                type(item.get(name)) is not int or item[name] < 0
                for name in ("device", "inode", "mtime_ns", "ctime_ns")
            )
        ):
            raise OwnerLauncherError(
                "capability_release_storage_preflight_invalid"
            )
        candidate_paths.append(item["path"])
    if len(candidate_paths) != len(set(candidate_paths)):
        raise OwnerLauncherError("capability_release_storage_preflight_invalid")

    allowed = value.get("ok")
    expected_allowed = (
        value.get("blocker") == ""
        and base.get("state") == "present_exact"
        and capacity["available_bytes"] >= MINIMUM_PACKAGING_FREE_BYTES
    )
    if (
        type(allowed) is not bool
        or allowed is not expected_allowed
        or (
            not allowed
            and not value.get("blocker")
        )
    ):
        raise OwnerLauncherError("capability_release_storage_preflight_invalid")
    unsigned = {
        name: item for name, item in value.items() if name != "receipt_sha256"
    }
    if value.get("receipt_sha256") != hashlib.sha256(
        _canonical_bytes(unsigned)
    ).hexdigest():
        raise OwnerLauncherError("capability_release_storage_preflight_invalid")
    return dict(value)


def _wipe(value: bytearray | None) -> None:
    if value is not None:
        for index in range(len(value)):
            value[index] = 0


def _stable_owner_file(path: Path, *, maximum: int) -> bytes:
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.getuid()  # windows-footgun: ok — macOS/Linux owner boundary
        or stat.S_IMODE(before.st_mode) != 0o600
        or not 0 < before.st_size <= maximum
    ):
        raise OwnerLauncherError("capability_owner_secret_source_invalid")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        raw = os.read(descriptor, maximum + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = os.lstat(path)
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_nlink, item.st_uid,
        item.st_gid, item.st_size, item.st_mtime_ns, item.st_ctime_ns,
    )
    if (
        len(raw) > maximum
        or len(raw) != before.st_size
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
        or identity(before) != identity(reachable)
    ):
        raise OwnerLauncherError("capability_owner_secret_source_changed")
    return raw


def _read_owner_canonical_json(
    path: Path,
    *,
    maximum: int,
    label: str,
) -> Mapping[str, Any]:
    try:
        raw = _stable_owner_file(path, maximum=maximum)
        payload = raw[:-1] if raw.endswith(b"\n") else raw
        if not payload or b"\n" in payload:
            raise ValueError
        value = _decode_json(payload, label=label)
        if payload != _canonical_bytes(value):
            raise ValueError
        return dict(value)
    except (OSError, ValueError, OwnerLauncherError):
        raise OwnerLauncherError("capability_owner_receipt_source_invalid") from None


def _write_owner_file_no_replace(path: Path, payload: bytes) -> str:
    """Publish one owner-only artifact without replacing any existing byte."""

    normalized = Path(os.path.normpath(os.fspath(path)))
    if (
        not path.is_absolute()
        or path != normalized
        or not payload
        or len(payload) > _MAX_PLAN_INPUT_BYTES
    ):
        raise OwnerLauncherError("capability_owner_output_path_invalid")
    parent = path.parent
    try:
        parent_item = os.lstat(parent)
    except OSError:
        raise OwnerLauncherError("capability_owner_output_path_invalid") from None
    if (
        not stat.S_ISDIR(parent_item.st_mode)
        or stat.S_ISLNK(parent_item.st_mode)
        or parent_item.st_uid != os.getuid()  # windows-footgun: ok — macOS/Linux owner boundary
        or stat.S_IMODE(parent_item.st_mode) & 0o022
        or os.path.lexists(path)
    ):
        raise OwnerLauncherError("capability_owner_output_path_invalid")
    temporary = parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("owner artifact write made no progress")
            offset += written
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        item = os.fstat(descriptor)
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_uid != os.getuid()  # windows-footgun: ok — macOS/Linux owner boundary
            or stat.S_IMODE(item.st_mode) != 0o600
            or item.st_size != len(payload)
        ):
            raise OSError("owner artifact temporary identity drifted")
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except (OSError, FileExistsError):
        raise OwnerLauncherError("capability_owner_output_publish_failed") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    if _stable_owner_file(path, maximum=len(payload)) != payload:
        raise OwnerLauncherError("capability_owner_output_readback_drifted")
    return hashlib.sha256(payload).hexdigest()


_FULL_CANARY_TERMINAL_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "ok",
        "state",
        "release_sha",
        "coordinator_input_sha256",
        "full_canary_plan_sha256",
        "owner_approval_sha256",
        "phase_b_readiness_anchor_sha256",
        "api_session_key_sha256",
        "fixture_sha256",
        "discord_token_install_receipt_sha256",
        "coordinator_receipt_sha256",
        "live_driver_receipt_sha256",
        "services_stopped",
        "discord_token_retired",
        "temporary_admin_created",
        "bootstrap_credential_created",
        "completed_at_unix",
        "receipt_sha256",
    }
)


def validate_full_canary_terminal_receipt(
    value: Any,
    *,
    revision: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FULL_CANARY_TERMINAL_RECEIPT_FIELDS:
        raise OwnerLauncherError("capability_full_canary_receipt_invalid")
    raw = dict(value)
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    if (
        raw.get("schema") != FULL_CANARY_TERMINAL_RECEIPT_SCHEMA
        or raw.get("ok") is not True
        or raw.get("state") != "verified_stopped_and_credentials_retired"
        or raw.get("release_sha") != revision
        or _REVISION_RE.fullmatch(revision or "") is None
        or raw.get("services_stopped") is not True
        or raw.get("discord_token_retired") is not True
        or raw.get("temporary_admin_created") is not False
        or raw.get("bootstrap_credential_created") is not False
        or type(raw.get("completed_at_unix")) is not int
        or raw["completed_at_unix"] < 0
        or raw.get("receipt_sha256")
        != hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    ):
        raise OwnerLauncherError("capability_full_canary_receipt_invalid")
    for field in _FULL_CANARY_TERMINAL_RECEIPT_FIELDS - {
        "schema",
        "ok",
        "state",
        "release_sha",
        "services_stopped",
        "discord_token_retired",
        "temporary_admin_created",
        "bootstrap_credential_created",
        "completed_at_unix",
    }:
        try:
            _digest(raw.get(field), field)
        except ValueError:
            raise OwnerLauncherError("capability_full_canary_receipt_invalid") from None
    return raw


def read_codex_access_token(path: Path | None = None) -> bytearray:
    source = Path.home() / ".codex/auth.json" if path is None else path
    raw = _stable_owner_file(source, maximum=256 * 1024)
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
        tokens = value.get("tokens") if isinstance(value, Mapping) else None
        token = tokens.get("access_token") if isinstance(tokens, Mapping) else None
        if not isinstance(token, str) or not token or len(token) > _MAX_SECRET_BYTES:
            raise ValueError
        return bytearray(token.encode("ascii", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError):
        raise OwnerLauncherError("capability_codex_access_token_invalid") from None


def read_mac_ops_env(stream: Any = None) -> bytearray:
    source = sys.stdin.buffer if stream is None else stream
    payload = bytearray(source.read(_MAX_SECRET_BYTES + 1))
    if not payload or len(payload) > _MAX_SECRET_BYTES:
        _wipe(payload)
        raise OwnerLauncherError("capability_mac_ops_secret_input_invalid")
    return payload


class CapabilityCanaryTransport(IapStoppedReleaseTransport):
    """Invoke only the fixed packaged capability-canary commands."""

    _ACTIONS = frozenset(
        {
            "contract",
            "storage-preflight",
            "collect-plan-authoring-context",
            "bootstrap-bitrix-foundation",
            "prepare-producer-foundation",
            "install-producer-foundation",
            "preflight-producer-foundation",
            "publish-plan",
            "preflight-stopped",
            "preflight-live",
            "provision-codex",
            "provision-mac-ops",
            "provision-discord-connector",
            "provision-api-control",
            "provision-discord-routeback",
            "provision-bitrix-operational-edge",
            "install-approval",
            "publish-live-fixture",
            "wait-production-observation-marker",
            "stage-production-observation",
            "start",
            "run-live",
            "stop",
            "retire-secrets",
        }
    )

    def _remote_command(self, revision: str, action: str) -> tuple[str, ...]:
        if _REVISION_RE.fullmatch(revision) is None or action not in self._ACTIONS:
            raise OwnerLauncherError("capability_canary_command_invalid")
        if action == "storage-preflight":
            return (
                *self._fixed_remote_environment(chdir="/"),
                "/usr/bin/python3",
                "-B",
                "-I",
                "-c",
                _REMOTE_STORAGE_PREFLIGHT,
                revision,
            )
        if action == "collect-plan-authoring-context":
            return (
                *self._fixed_remote_environment(chdir="/"),
                f"/opt/muncho-canary-releases/{revision}/venv/bin/python",
                "-B",
                "-I",
                "-c",
                _REMOTE_PLAN_AUTHORING_CONTEXT,
                revision,
            )
        interpreter = f"/opt/muncho-canary-releases/{revision}/venv/bin/python"
        live_action = action in {"publish-live-fixture", "run-live"}
        producer_foundation_action = action in {
            "prepare-producer-foundation",
            "install-producer-foundation",
            "preflight-producer-foundation",
        }
        module = (
            LIVE_DRIVER_MODULE
            if live_action
            else PRODUCER_FOUNDATION_MODULE
            if producer_foundation_action
            else RUNTIME_MODULE
        )
        remote_action = (
            "publish-fixture"
            if action == "publish-live-fixture"
            else "run"
            if action == "run-live"
            else "prepare-foundation"
            if action == "prepare-producer-foundation"
            else "install-foundation"
            if action == "install-producer-foundation"
            else "preflight"
            if action == "preflight-producer-foundation"
            else action
        )
        return (
            *self._fixed_remote_environment(chdir="/"),
            interpreter,
            "-B",
            "-I",
            "-m",
            module,
            remote_action,
        )

    def invoke(
        self,
        revision: str,
        action: str,
        *,
        frame: bytes | bytearray | None = None,
    ) -> Mapping[str, Any]:
        account = self._owner_identity.account_for_read_only_preflight()
        self._owner_identity.require_stable()
        command = self._remote_command(revision, action)
        if frame is None:
            completed = self._run_remote(
                command,
                account=account,
                timeout_seconds=900,
                maximum_output_bytes=_MAX_OUTPUT,
            )
        else:
            completed = self._run_remote_input(
                command,
                account=account,
                input_bytes=bytes(frame),
                timeout_seconds=900,
                maximum_input_bytes=(
                    _MAX_PLAN_INPUT_BYTES
                    if action
                    in {"publish-plan", "collect-plan-authoring-context"}
                    else 2 * 1024 * 1024
                    if action
                    in {
                        "prepare-producer-foundation",
                        "install-producer-foundation",
                        "wait-production-observation-marker",
                        "stage-production-observation",
                    }
                    else _MAX_BITRIX_FOUNDATION_INPUT_BYTES
                    if action == "bootstrap-bitrix-foundation"
                    else _MAX_FIXTURE_AUTHORITY_BYTES
                    if action == "publish-live-fixture"
                    else _MAX_SECRET_BYTES + 128 * 1024
                ),
                maximum_output_bytes=_MAX_OUTPUT,
            )
        self._owner_identity.require_stable()
        if (
            not completed.stdout
            or not completed.stdout.endswith(b"\n")
            or b"\n" in completed.stdout[:-1]
        ):
            raise OwnerLauncherError("capability_canary_output_invalid")
        value = _decode_json(
            completed.stdout[:-1], label="capability-canary owner response"
        )
        if completed.stdout[:-1] != _canonical_bytes(value):
            raise OwnerLauncherError("capability_canary_output_invalid")
        result = dict(value)
        if action == "storage-preflight":
            return validate_release_storage_preflight(
                result,
                revision=revision,
            )
        return result


class CapabilityProducerFoundationOwnerSigner:
    """Sign one non-secret foundation with the fixed owner key via OpenSSH."""

    def __init__(
        self,
        *,
        inspector: Any | None = None,
        command_runner: Any = subprocess.run,
        namespace: str = PRODUCER_FOUNDATION_SSHSIG_NAMESPACE,
    ) -> None:
        if (
            not callable(command_runner)
            or namespace
            not in {
                PRODUCER_FOUNDATION_SSHSIG_NAMESPACE,
                PRODUCTION_OBSERVATION_SSHSIG_NAMESPACE,
                CAPABILITY_FIXTURE_SSHSIG_NAMESPACE,
            }
        ):
            raise OwnerLauncherError("capability_producer_signer_invalid")
        self._inspector = inspector or _PhaseBOwnerExternalSigner(
            private_key_path=PHASE_B_OWNER_PRIVATE_KEY_PATH,
            public_key_path=PHASE_B_OWNER_PUBLIC_KEY_PATH,
            expected_comment=PHASE_B_OWNER_PUBLIC_KEY_COMMENT,
            expected_fingerprint=PHASE_B_OWNER_PUBLIC_KEY_FINGERPRINT,
        )
        if not callable(getattr(self._inspector, "inspect", None)):
            raise OwnerLauncherError("capability_producer_signer_invalid")
        self._command_runner = command_runner
        self._namespace = namespace

    def inspect(self) -> Any:
        return self._inspector.inspect()

    def sign(self, payload: bytes, *, expected_authority: Any) -> str:
        if (
            not isinstance(payload, bytes)
            or not payload
            or len(payload) > 2 * 1024 * 1024
        ):
            raise OwnerLauncherError("capability_producer_signing_request_invalid")
        before = self.inspect()
        if before != expected_authority:
            raise OwnerLauncherError("capability_producer_owner_key_changed")
        try:
            executable = PHASE_B_SSH_KEYGEN.lstat()
        except OSError:
            raise OwnerLauncherError(
                "capability_producer_signer_unavailable"
            ) from None
        if (
            not stat.S_ISREG(executable.st_mode)
            or stat.S_ISLNK(executable.st_mode)
            or executable.st_uid != 0
            or stat.S_IMODE(executable.st_mode) & 0o022
        ):
            raise OwnerLauncherError("capability_producer_signer_untrusted")
        try:
            completed = self._command_runner(
                (
                    str(PHASE_B_SSH_KEYGEN),
                    "-Y",
                    "sign",
                    "-q",
                    "-f",
                    str(PHASE_B_OWNER_PRIVATE_KEY_PATH),
                    "-n",
                    self._namespace,
                ),
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=None,
                check=False,
                shell=False,
                close_fds=True,
                timeout=15,
                env={
                    "PATH": "/usr/bin:/bin",
                    "LANG": "C",
                    "LC_ALL": "C",
                },
            )
        except (OSError, subprocess.SubprocessError, TypeError):
            raise OwnerLauncherError(
                "capability_producer_signer_failed"
            ) from None
        if (
            completed.returncode != 0
            or not isinstance(completed.stdout, bytes)
            or not completed.stdout
            or len(completed.stdout) > 4096
        ):
            raise OwnerLauncherError("capability_producer_signer_failed")
        after = self.inspect()
        if after != before:
            raise OwnerLauncherError("capability_producer_owner_key_changed")
        try:
            return completed.stdout.decode("ascii", errors="strict")
        except UnicodeError:
            raise OwnerLauncherError(
                "capability_producer_signature_invalid"
            ) from None


_PRODUCER_BOOTSTRAP_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "revision",
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "preparation_sha256",
        "foundation_sha256",
        "install_receipt_sha256",
        "unit_bundle_manifest_sha256",
        "preflight_ready",
        "private_key_loaded_by_launcher",
        "secret_material_recorded",
        "receipt_sha256",
    }
)


def validate_producer_bootstrap_receipt(
    value: Any,
    *,
    revision: str,
    terminal_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PRODUCER_BOOTSTRAP_RECEIPT_FIELDS:
        raise OwnerLauncherError("capability_producer_bootstrap_receipt_invalid")
    raw = dict(value)
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    if (
        raw.get("schema") != CAPABILITY_PRODUCER_BOOTSTRAP_RECEIPT_SCHEMA
        or raw.get("revision") != revision
        or raw.get("full_canary_plan_sha256")
        != terminal_receipt.get("full_canary_plan_sha256")
        or raw.get("preflight_ready") is not True
        or raw.get("private_key_loaded_by_launcher") is not False
        or raw.get("secret_material_recorded") is not False
        or raw.get("receipt_sha256")
        != hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    ):
        raise OwnerLauncherError("capability_producer_bootstrap_receipt_invalid")
    for field in _PRODUCER_BOOTSTRAP_RECEIPT_FIELDS - {
        "schema",
        "revision",
        "preflight_ready",
        "private_key_loaded_by_launcher",
        "secret_material_recorded",
    }:
        try:
            _digest(raw.get(field), field)
        except ValueError:
            raise OwnerLauncherError(
                "capability_producer_bootstrap_receipt_invalid"
            ) from None
    return raw


def build_live_fixture_authority(
    *,
    producer_receipt: Mapping[str, Any],
    signer: Any,
    run_id: str,
    now_unix_ms: int | None = None,
    valid_for_seconds: int = 900,
) -> Mapping[str, Any]:
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,239}", run_id) is None
        or type(valid_for_seconds) is not int
        or not 60 <= valid_for_seconds <= _FIXTURE_AUTHORITY_MAX_SECONDS
        or not callable(getattr(signer, "inspect", None))
        or not callable(getattr(signer, "sign", None))
    ):
        raise OwnerLauncherError("capability_fixture_authoring_request_invalid")
    issued = int(time.time() * 1000) if now_unix_ms is None else now_unix_ms
    if type(issued) is not int or issued < 0:
        raise OwnerLauncherError("capability_fixture_authoring_request_invalid")
    authority = signer.inspect()
    if not callable(getattr(authority, "to_mapping", None)):
        raise OwnerLauncherError("capability_fixture_owner_authority_invalid")
    public = authority.to_mapping()
    owner_key_id = public.get("key_id") if isinstance(public, Mapping) else None
    try:
        _digest(owner_key_id, "fixture owner key")
        _digest(producer_receipt.get("foundation_sha256"), "producer foundation")
    except ValueError:
        raise OwnerLauncherError("capability_fixture_owner_authority_invalid") from None
    unsigned = {
        "schema": CAPABILITY_FIXTURE_AUTHORITY_SCHEMA,
        "run_id": run_id,
        "owner_id": PRODUCTION_OWNER_USER_ID,
        "valid_from_unix_ms": issued,
        "valid_until_unix_ms": issued + valid_for_seconds * 1000,
        "public_discord_target": {
            "target_type": "public_channel",
            "guild_id": PRODUCTION_CANARY_PUBLIC_GUILD_ID,
            "channel_id": PRODUCTION_CANARY_PUBLIC_CHANNEL_ID,
        },
        "producer_foundation_sha256": producer_receipt["foundation_sha256"],
        "owner_key_id": owner_key_id,
        "signature_algorithm": "sshsig-ed25519-sha512",
    }
    signature = signer.sign(
        _canonical_bytes(unsigned),
        expected_authority=authority,
    )
    if not isinstance(signature, str) or not signature:
        raise OwnerLauncherError("capability_fixture_signature_invalid")
    return {**unsigned, "owner_signature": signature}


def author_live_fixture_authority(
    *,
    revision: str,
    terminal_receipt_file: Path,
    producer_receipt_file: Path,
    output_file: Path,
    run_id: str,
    valid_for_seconds: int,
    now_unix_ms: int | None = None,
    signer: Any | None = None,
) -> Mapping[str, Any]:
    terminal = validate_full_canary_terminal_receipt(
        _read_owner_canonical_json(
            terminal_receipt_file,
            maximum=_MAX_FULL_CANARY_TERMINAL_RECEIPT_BYTES,
            label="full-canary terminal receipt",
        ),
        revision=revision,
    )
    producer = validate_producer_bootstrap_receipt(
        _read_owner_canonical_json(
            producer_receipt_file,
            maximum=_MAX_FULL_CANARY_TERMINAL_RECEIPT_BYTES,
            label="producer bootstrap receipt",
        ),
        revision=revision,
        terminal_receipt=terminal,
    )
    owner_signer = signer or CapabilityProducerFoundationOwnerSigner(
        namespace=CAPABILITY_FIXTURE_SSHSIG_NAMESPACE
    )
    authority = build_live_fixture_authority(
        producer_receipt=producer,
        signer=owner_signer,
        run_id=run_id,
        now_unix_ms=now_unix_ms,
        valid_for_seconds=valid_for_seconds,
    )
    payload = _canonical_bytes(authority)
    file_sha256 = _write_owner_file_no_replace(output_file, payload)
    unsigned = {
        "schema": CAPABILITY_FIXTURE_AUTHORING_RECEIPT_SCHEMA,
        "revision": revision,
        "capability_plan_sha256": producer["capability_plan_sha256"],
        "full_canary_plan_sha256": producer["full_canary_plan_sha256"],
        "producer_foundation_sha256": producer["foundation_sha256"],
        "producer_bootstrap_receipt_sha256": producer["receipt_sha256"],
        "run_id": run_id,
        "valid_from_unix_ms": authority["valid_from_unix_ms"],
        "valid_until_unix_ms": authority["valid_until_unix_ms"],
        "authority_file": str(output_file),
        "authority_file_sha256": file_sha256,
        "authority_file_mode": "0600",
        "owner_key_id": authority["owner_key_id"],
        "signature_algorithm": authority["signature_algorithm"],
        "mutation_scope": "local_owner_file_create_only",
        "cloud_mutation_performed": False,
        "secret_material_recorded": False,
        "semantic_content_recorded": False,
    }
    return {
        **unsigned,
        "receipt_sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
    }


class CapabilityProductionObservationTransport:
    """Separate pinned IAP transport for the one real production VM."""

    def __init__(
        self,
        transport: Any | None = None,
        *,
        revision: str | None = None,
    ) -> None:
        if transport is None:
            from scripts.canary.production_cutover_owner_launcher import (
                ProductionCutoverTransport,
            )

            if _REVISION_RE.fullmatch(revision or "") is None:
                raise OwnerLauncherError(
                    "production_observation_binding_invalid"
                )
            trusted = require_trusted_owner_runtime(revision)
            configuration = PinnedGcloudConfiguration()
            identity = GcloudOwnerAccessToken(
                gcloud_executable=trusted,
                gcloud_configuration=configuration,
            )
            identity.account_for_read_only_preflight()
            transport = ProductionCutoverTransport(
                identity,
                gcloud_executable=trusted,
                gcloud_configuration=configuration,
            )
        required = (
            "_owner_identity",
            "_authorization_snapshot",
            "_run_remote_input",
            "_fixed_remote_environment",
            "_known_hosts",
        )
        if any(not hasattr(transport, name) for name in required):
            raise OwnerLauncherError("production_observation_transport_invalid")
        self._transport = transport

    def observe(
        self,
        *,
        phase: str,
        revision: str,
        plan_sha256: str,
        full_canary_plan_sha256: str,
        fixture_sha256: str,
        run_id: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if (
            phase not in {"before", "after"}
            or _REVISION_RE.fullmatch(revision or "") is None
            or any(
                re.fullmatch(r"[0-9a-f]{64}", value or "") is None
                for value in (
                    plan_sha256,
                    full_canary_plan_sha256,
                    fixture_sha256,
                )
            )
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id or "")
            is None
        ):
            raise OwnerLauncherError("production_observation_binding_invalid")
        inner = self._transport
        observer_source, observer_source_sha256 = _production_observer_source(
            revision
        )
        observer_source_bytes = observer_source.encode("utf-8", errors="strict")
        account = inner._owner_identity.account_for_read_only_preflight()
        inner._owner_identity.require_stable()
        authorization = inner._authorization_snapshot(account)
        if (
            not isinstance(authorization, tuple)
            or len(authorization) != 3
            or any(re.fullmatch(r"[0-9a-f]{64}", item or "") is None for item in authorization)
        ):
            raise OwnerLauncherError("production_observation_authority_invalid")
        command = (
            *inner._fixed_remote_environment(chdir="/"),
            "/opt/adventico-ai-platform/hermes-agent/.venv/bin/python",
            "-B",
            "-I",
            "-",
            phase,
            "--canary-revision",
            revision,
            "--capability-plan-sha256",
            plan_sha256,
            "--full-canary-plan-sha256",
            full_canary_plan_sha256,
            "--fixture-sha256",
            fixture_sha256,
            "--run-id",
            run_id,
        )
        completed = inner._run_remote_input(
            command,
            account=account,
            input_bytes=observer_source_bytes,
            timeout_seconds=120,
            maximum_input_bytes=512 * 1024,
            maximum_output_bytes=2 * 1024 * 1024,
        )
        inner._owner_identity.require_stable()
        raw = completed.stdout
        if (
            not isinstance(raw, bytes)
            or not raw.endswith(b"\n")
            or b"\n" in raw[:-1]
        ):
            raise OwnerLauncherError("production_observation_output_invalid")
        observation = _decode_json(
            raw[:-1], label="production observation"
        )
        known_hosts_path = Path(inner._known_hosts.absolute_path())
        known_hosts_raw = _stable_owner_file(
            known_hosts_path, maximum=256 * 1024
        )
        authority = {
            "kind": "pinned_owner_gcloud_iap_ssh_read_only",
            "project": "adventico-ai-platform",
            "zone": "europe-west3-a",
            "vm": "ai-platform-runtime-01",
            "instance_id": "1094477181810932795",
            "known_hosts_file_sha256": hashlib.sha256(
                known_hosts_raw
            ).hexdigest(),
            "observer_source_sha256": observer_source_sha256,
            "instance_authorization_sha256": authorization[0],
            "project_authorization_sha256": authorization[1],
            "oslogin_authorization_sha256": authorization[2],
        }
        return observation, authority


def _validate_production_observation(
    value: Any,
    *,
    phase: str,
    revision: str,
    plan_sha256: str,
    full_canary_plan_sha256: str,
    fixture_sha256: str,
    run_id: str,
    now_unix_ms: int,
) -> Mapping[str, Any]:
    try:
        from scripts.canary.production_capability_observer import (
            ProductionObservationError,
            validate_production_observation,
        )
    except ImportError as exc:
        raise OwnerLauncherError("production_observation_invalid") from exc
    try:
        return validate_production_observation(
            value,
            phase=phase,
            canary_revision=revision,
            capability_plan_sha256=plan_sha256,
            full_canary_plan_sha256=full_canary_plan_sha256,
            fixture_sha256=fixture_sha256,
            run_id=run_id,
            now_unix_ms=now_unix_ms,
        )
    except ProductionObservationError as exc:
        raise OwnerLauncherError("production_observation_invalid") from exc


def collect_owner_signed_production_observation(
    transport: Any,
    *,
    phase: str,
    revision: str,
    plan_sha256: str,
    full_canary_plan_sha256: str,
    fixture_sha256: str,
    run_id: str,
    owner_subject_sha256: str,
    owner_signer: Any | None = None,
    now_unix_ms: int | None = None,
) -> Mapping[str, Any]:
    """Observe the real production VM, then sign exact bytes on the owner Mac."""

    _digest(owner_subject_sha256, "production observation owner")
    observation, transport_authority = transport.observe(
        phase=phase,
        revision=revision,
        plan_sha256=plan_sha256,
        full_canary_plan_sha256=full_canary_plan_sha256,
        fixture_sha256=fixture_sha256,
        run_id=run_id,
    )
    signed_at = int(time.time() * 1000) if now_unix_ms is None else now_unix_ms
    observed = _validate_production_observation(
        observation,
        phase=phase,
        revision=revision,
        plan_sha256=plan_sha256,
        full_canary_plan_sha256=full_canary_plan_sha256,
        fixture_sha256=fixture_sha256,
        run_id=run_id,
        now_unix_ms=signed_at,
    )
    transport_fields = {
        "kind",
        "project",
        "zone",
        "vm",
        "instance_id",
        "known_hosts_file_sha256",
        "observer_source_sha256",
        "instance_authorization_sha256",
        "project_authorization_sha256",
        "oslogin_authorization_sha256",
    }
    if (
        not isinstance(transport_authority, Mapping)
        or set(transport_authority) != transport_fields
        or transport_authority.get("kind")
        != "pinned_owner_gcloud_iap_ssh_read_only"
        or transport_authority.get("project") != "adventico-ai-platform"
        or transport_authority.get("zone") != "europe-west3-a"
        or transport_authority.get("vm") != "ai-platform-runtime-01"
        or transport_authority.get("instance_id") != "1094477181810932795"
        or transport_authority.get("observer_source_sha256")
        != _production_observer_source(revision)[1]
        or any(
            re.fullmatch(
                r"[0-9a-f]{64}", str(transport_authority.get(field) or "")
            )
            is None
            for field in (
                "known_hosts_file_sha256",
                "observer_source_sha256",
                "instance_authorization_sha256",
                "project_authorization_sha256",
                "oslogin_authorization_sha256",
            )
        )
    ):
        raise OwnerLauncherError("production_observation_transport_invalid")
    signer = owner_signer or CapabilityProducerFoundationOwnerSigner(
        namespace=PRODUCTION_OBSERVATION_SSHSIG_NAMESPACE
    )
    authority = signer.inspect()
    if not callable(getattr(authority, "to_mapping", None)):
        raise OwnerLauncherError("production_observation_owner_authority_invalid")
    if (
        type(signed_at) is not int
        or signed_at < observed["observed_at_unix_ms"]
        or signed_at - observed["observed_at_unix_ms"] > 300_000
    ):
        raise OwnerLauncherError("production_observation_signing_time_invalid")
    unsigned = {
        "schema": PRODUCTION_OBSERVATION_ENVELOPE_SCHEMA,
        "phase": phase,
        "canary_revision": revision,
        "capability_plan_sha256": plan_sha256,
        "full_canary_plan_sha256": full_canary_plan_sha256,
        "fixture_sha256": fixture_sha256,
        "run_id": run_id,
        "observation": observed,
        "observation_sha256": observed["observation_sha256"],
        "transport_authority": copy.deepcopy(dict(transport_authority)),
        "owner_subject_sha256": owner_subject_sha256,
        "owner_public_authority": authority.to_mapping(),
        "signed_at_unix_ms": signed_at,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    payload = _canonical_bytes(unsigned)
    signature = signer.sign(payload, expected_authority=authority)
    signed = {**unsigned, "owner_signature": signature}
    return {
        **signed,
        "envelope_sha256": hashlib.sha256(_canonical_bytes(signed)).hexdigest(),
    }


def _production_observation_wait_request(
    *,
    phase: str,
    revision: str,
    plan_sha256: str,
    full_canary_plan_sha256: str,
    fixture_sha256: str,
    run_id: str,
    owner_subject_sha256: str,
    timeout_seconds: int,
) -> Mapping[str, Any]:
    if (
        phase not in {"before", "after"}
        or _REVISION_RE.fullmatch(revision or "") is None
        or any(
            re.fullmatch(r"[0-9a-f]{64}", value or "") is None
            for value in (
                plan_sha256,
                full_canary_plan_sha256,
                fixture_sha256,
                owner_subject_sha256,
            )
        )
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id or ""
        )
        is None
        or type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= 300
    ):
        raise OwnerLauncherError("production_observation_binding_invalid")
    return {
        "schema": PRODUCTION_OBSERVATION_WAIT_REQUEST_SCHEMA,
        "phase": phase,
        "canary_revision": revision,
        "capability_plan_sha256": plan_sha256,
        "full_canary_plan_sha256": full_canary_plan_sha256,
        "fixture_sha256": fixture_sha256,
        "run_id": run_id,
        "owner_subject_sha256": owner_subject_sha256,
        "timeout_seconds": timeout_seconds,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


def _validate_production_observation_marker_wait_receipt(
    value: Any,
    *,
    phase: str,
    fixture_sha256: str,
    run_id: str,
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "phase",
        "run_id",
        "fixture_sha256",
        "marker_sha256",
        "observer_live_verified",
        "secret_material_recorded",
        "secret_digest_recorded",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OwnerLauncherError("production_observation_marker_wait_invalid")
    raw = dict(value)
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    if (
        raw["schema"]
        != "muncho-production-capability-production-observation-marker-wait.v1"
        or raw["phase"] != phase
        or raw["fixture_sha256"] != fixture_sha256
        or raw["run_id"] != run_id
        or re.fullmatch(r"[0-9a-f]{64}", str(raw["marker_sha256"] or ""))
        is None
        or raw["observer_live_verified"] is not (phase == "after")
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["receipt_sha256"]
        != hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    ):
        raise OwnerLauncherError("production_observation_marker_wait_invalid")
    return raw


def _validate_production_observation_stage_receipt(
    value: Any,
    *,
    phase: str,
    fixture_sha256: str,
    run_id: str,
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "phase",
        "run_id",
        "fixture_sha256",
        "staged_envelope_sha256",
        "observation_sha256",
        "marker_sha256",
        "production_diff_sha256",
        "secret_material_recorded",
        "secret_digest_recorded",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OwnerLauncherError("production_observation_stage_invalid")
    raw = dict(value)
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    digests = (
        "staged_envelope_sha256",
        "observation_sha256",
        "marker_sha256",
    )
    if (
        raw["schema"] != PRODUCTION_OBSERVATION_STAGE_RECEIPT_SCHEMA
        or raw["phase"] != phase
        or raw["fixture_sha256"] != fixture_sha256
        or raw["run_id"] != run_id
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(raw[field] or "")) is None
            for field in digests
        )
        or (
            phase == "before"
            and raw["production_diff_sha256"] is not None
        )
        or (
            phase == "after"
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(raw["production_diff_sha256"] or ""),
            )
            is None
        )
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["receipt_sha256"]
        != hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    ):
        raise OwnerLauncherError("production_observation_stage_invalid")
    return raw


def run_live_with_owner_production_observations(
    transport: CapabilityCanaryTransport,
    observation_transport: Any,
    *,
    revision: str,
    plan_sha256: str,
    full_canary_plan_sha256: str,
    fixture_sha256: str,
    run_id: str,
    owner_subject_sha256: str,
    owner_signer: Any | None = None,
    timeout_seconds: int = 300,
) -> Mapping[str, Any]:
    """Run the live canary while the owner supplies both pinned observations."""

    # Build once up front so no remote mutation starts from a partial binding.
    requests = {
        phase: _production_observation_wait_request(
            phase=phase,
            revision=revision,
            plan_sha256=plan_sha256,
            full_canary_plan_sha256=full_canary_plan_sha256,
            fixture_sha256=fixture_sha256,
            run_id=run_id,
            owner_subject_sha256=owner_subject_sha256,
            timeout_seconds=timeout_seconds,
        )
        for phase in ("before", "after")
    }
    live_results: list[Mapping[str, Any]] = []
    live_errors: list[BaseException] = []

    def run_live() -> None:
        try:
            result = transport.invoke(revision, "run-live")
            if not isinstance(result, Mapping):
                raise OwnerLauncherError("capability_live_result_invalid")
            live_results.append(dict(result))
        except BaseException as exc:
            live_errors.append(exc)

    thread = threading.Thread(
        target=run_live,
        name="capability-live-observed-run",
        daemon=False,
    )
    thread.start()
    stage_receipts: dict[str, Mapping[str, Any]] = {}
    try:
        for phase in ("before", "after"):
            wait_frame = bytearray(_canonical_bytes(requests[phase]))
            try:
                waited = transport.invoke(
                    revision,
                    "wait-production-observation-marker",
                    frame=wait_frame,
                )
            finally:
                _wipe(wait_frame)
            _validate_production_observation_marker_wait_receipt(
                waited,
                phase=phase,
                fixture_sha256=fixture_sha256,
                run_id=run_id,
            )
            envelope = collect_owner_signed_production_observation(
                observation_transport,
                phase=phase,
                revision=revision,
                plan_sha256=plan_sha256,
                full_canary_plan_sha256=full_canary_plan_sha256,
                fixture_sha256=fixture_sha256,
                run_id=run_id,
                owner_subject_sha256=owner_subject_sha256,
                owner_signer=owner_signer,
            )
            stage_frame = bytearray(_canonical_bytes(envelope))
            try:
                staged = transport.invoke(
                    revision,
                    "stage-production-observation",
                    frame=stage_frame,
                )
            finally:
                _wipe(stage_frame)
            stage_receipts[phase] = (
                _validate_production_observation_stage_receipt(
                    staged,
                    phase=phase,
                    fixture_sha256=fixture_sha256,
                    run_id=run_id,
                )
            )
    finally:
        # The remote live invocation has its own 900-second hard bound and
        # fail-closed cleanup.  Never abandon it while credentials/services
        # could still be active.
        thread.join(930)
    if thread.is_alive():
        raise OwnerLauncherError("capability_live_run_cleanup_timeout")
    if live_errors:
        raise OwnerLauncherError("capability_live_run_failed") from live_errors[0]
    if len(live_results) != 1 or set(stage_receipts) != {"before", "after"}:
        raise OwnerLauncherError("capability_live_result_invalid")
    live = live_results[0]
    if (
        live.get("schema")
        != "muncho-production-capability-canary-live-driver.v1"
        or live.get("ok") is not True
        or live.get("release_sha") != revision
        or live.get("capability_plan_sha256") != plan_sha256
        or live.get("full_canary_plan_sha256") != full_canary_plan_sha256
        or live.get("fixture_sha256") != fixture_sha256
        or live.get("run_id") != run_id
        or live.get("production_before_observation_sha256")
        != stage_receipts["before"]["observation_sha256"]
        or live.get("production_diff_sha256")
        != stage_receipts["after"]["production_diff_sha256"]
    ):
        raise OwnerLauncherError("capability_live_result_invalid")
    unsigned = {
        "schema": "muncho-production-capability-owner-observed-live-run.v1",
        "revision": revision,
        "capability_plan_sha256": plan_sha256,
        "full_canary_plan_sha256": full_canary_plan_sha256,
        "fixture_sha256": fixture_sha256,
        "run_id": run_id,
        "live_evidence_sha256": live.get("evidence_sha256"),
        "before_stage_receipt_sha256": stage_receipts["before"][
            "receipt_sha256"
        ],
        "after_stage_receipt_sha256": stage_receipts["after"][
            "receipt_sha256"
        ],
        "production_diff_sha256": stage_receipts["after"][
            "production_diff_sha256"
        ],
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    for field in ("live_evidence_sha256", "production_diff_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(unsigned[field] or "")) is None:
            raise OwnerLauncherError("capability_live_result_invalid")
    return {**unsigned, "receipt_sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()}


def _validate_producer_install_receipt(
    value: Any,
    *,
    revision: str,
    plan_sha256: str,
    full_canary_plan_sha256: str,
    preparation_sha256: str,
    foundation_sha256: str,
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "revision",
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "preparation_sha256",
        "foundation_sha256",
        "unit_bundle_manifest_sha256",
        "installed_units",
        "installed_configs",
        "installed_auxiliary_files",
        "native_root_contract",
        "config_install_contract",
        "volatile_runtime_contract",
        "authority_key_lifecycle",
        "daemon_reload_completed",
        "volatile_runtime_materialized",
        "services_started",
        "secret_material_recorded",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OwnerLauncherError("capability_producer_install_receipt_invalid")
    raw = dict(value)
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    if (
        raw["schema"] != FOUNDATION_INSTALL_RECEIPT_SCHEMA
        or raw["revision"] != revision
        or raw["capability_plan_sha256"] != plan_sha256
        or raw["full_canary_plan_sha256"] != full_canary_plan_sha256
        or raw["preparation_sha256"] != preparation_sha256
        or raw["foundation_sha256"] != foundation_sha256
        or raw["daemon_reload_completed"] is not True
        or raw["volatile_runtime_materialized"] is not True
        or raw["services_started"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["receipt_sha256"]
        != hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    ):
        raise OwnerLauncherError("capability_producer_install_receipt_invalid")
    _digest(raw["unit_bundle_manifest_sha256"], "producer unit manifest")
    return raw


def bootstrap_producer_foundation(
    transport: CapabilityCanaryTransport,
    *,
    revision: str,
    plan_sha256: str,
    full_canary_plan_sha256: str,
    owner_signer: Any | None = None,
) -> Mapping[str, Any]:
    """Prepare on Cloud, sign exact public bytes locally, install, preflight."""

    from gateway.canonical_capability_canary_producer_units import (
        validate_foundation_preparation,
    )
    from gateway.canonical_capability_canary_producers import (
        producer_foundation_sha256,
        producer_foundation_signature_payload,
        project_pinned_owner_public_key_source,
        seal_producer_foundation,
    )

    if _REVISION_RE.fullmatch(revision or "") is None:
        raise OwnerLauncherError("capability_producer_binding_invalid")
    _digest(plan_sha256, "capability plan")
    _digest(full_canary_plan_sha256, "full-canary plan")
    harden_owner_secret_process()
    signer = owner_signer or CapabilityProducerFoundationOwnerSigner()
    if (
        not callable(getattr(signer, "inspect", None))
        or not callable(getattr(signer, "sign", None))
    ):
        raise OwnerLauncherError("capability_producer_signer_invalid")
    authority = signer.inspect()
    if not callable(getattr(authority, "to_mapping", None)):
        raise OwnerLauncherError("capability_producer_owner_authority_invalid")
    authority_mapping = authority.to_mapping()
    prepare_request = {
        "schema": FOUNDATION_PREPARE_REQUEST_SCHEMA,
        "revision": revision,
        "capability_plan_sha256": plan_sha256,
        "full_canary_plan_sha256": full_canary_plan_sha256,
        "owner_public_authority": authority_mapping,
        "secret_material_recorded": False,
        "semantic_content_recorded": False,
    }
    preparation = validate_foundation_preparation(
        transport.invoke(
            revision,
            "prepare-producer-foundation",
            frame=_canonical_bytes(prepare_request),
        )
    )
    projected_owner = project_pinned_owner_public_key_source(
        authority_mapping,
        expected_comment=PHASE_B_OWNER_PUBLIC_KEY_COMMENT,
    )
    if (
        preparation["schema"] != FOUNDATION_PREPARATION_SCHEMA
        or preparation["revision"] != revision
        or preparation["capability_plan_sha256"] != plan_sha256
        or preparation["full_canary_plan_sha256"]
        != full_canary_plan_sha256
        or preparation["owner_public_key_ed25519_hex"]
        != authority_mapping.get("public_key_ed25519_hex")
        or preparation["owner_public_key_source_sha256"]
        != projected_owner["public_key_source_sha256"]
    ):
        raise OwnerLauncherError("capability_producer_preparation_invalid")
    payload = producer_foundation_signature_payload(
        preparation["unsigned_foundation"]
    )
    signature = signer.sign(payload, expected_authority=authority)
    sealed = seal_producer_foundation(
        preparation["unsigned_foundation"],
        owner_signature=signature,
        pinned_owner_public_key_ed25519_hex=(
            preparation["owner_public_key_ed25519_hex"]
        ),
        pinned_owner_public_key_source_sha256=(
            preparation["owner_public_key_source_sha256"]
        ),
    )
    foundation_sha256 = producer_foundation_sha256(sealed)
    install_request = {
        "schema": FOUNDATION_INSTALL_REQUEST_SCHEMA,
        "preparation_sha256": preparation["preparation_sha256"],
        "owner_signature": signature,
    }
    install = _validate_producer_install_receipt(
        transport.invoke(
            revision,
            "install-producer-foundation",
            frame=_canonical_bytes(install_request),
        ),
        revision=revision,
        plan_sha256=plan_sha256,
        full_canary_plan_sha256=full_canary_plan_sha256,
        preparation_sha256=preparation["preparation_sha256"],
        foundation_sha256=foundation_sha256,
    )
    preflight = transport.invoke(revision, "preflight-producer-foundation")
    if (
        not isinstance(preflight, Mapping)
        or set(preflight)
        != {
            "schema",
            "revision",
            "foundation_sha256",
            "preparation_sha256",
            "unit_bundle_manifest_sha256",
            "ready",
            "mutation_performed",
        }
        or preflight.get("schema")
        != "muncho-capability-producer-installation-preflight.v1"
        or preflight.get("revision") != revision
        or preflight.get("foundation_sha256") != foundation_sha256
        or preflight.get("preparation_sha256")
        != preparation["preparation_sha256"]
        or preflight.get("unit_bundle_manifest_sha256")
        != install["unit_bundle_manifest_sha256"]
        or preflight.get("ready") is not True
        or preflight.get("mutation_performed") is not False
    ):
        raise OwnerLauncherError("capability_producer_preflight_invalid")
    unsigned = {
        "schema": PRODUCER_FOUNDATION_BOOTSTRAP_SCHEMA,
        "revision": revision,
        "capability_plan_sha256": plan_sha256,
        "full_canary_plan_sha256": full_canary_plan_sha256,
        "preparation_sha256": preparation["preparation_sha256"],
        "foundation_sha256": foundation_sha256,
        "install_receipt_sha256": install["receipt_sha256"],
        "unit_bundle_manifest_sha256": install[
            "unit_bundle_manifest_sha256"
        ],
        "preflight_ready": True,
        "private_key_loaded_by_launcher": False,
        "secret_material_recorded": False,
    }
    return {
        **unsigned,
        "receipt_sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
    }


def validate_fixture_publication_receipt(
    value: Any,
    *,
    revision: str,
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "run_id",
        "release_sha",
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "producer_foundation_sha256",
        "authority_sha256",
        "fixture_path",
        "fixture_sha256",
        "fixture_file_identity",
        "receipt_path",
        "published_at_unix_ms",
        "receipt_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != FIXTURE_PUBLICATION_RECEIPT_SCHEMA
        or value.get("release_sha") != revision
        or _REVISION_RE.fullmatch(revision or "") is None
        or not isinstance(value.get("run_id"), str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}", value["run_id"])
        is None
        or value.get("fixture_path") != REVIEWED_FIXTURE_PATH
        or type(value.get("published_at_unix_ms")) is not int
        or value["published_at_unix_ms"] <= 0
    ):
        raise OwnerLauncherError("capability_fixture_publication_receipt_invalid")
    for field in (
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "producer_foundation_sha256",
        "authority_sha256",
        "fixture_sha256",
    ):
        try:
            _digest(value.get(field), field)
        except ValueError:
            raise OwnerLauncherError(
                "capability_fixture_publication_receipt_invalid"
            ) from None
    expected_receipt_path = (
        f"{FIXTURE_PUBLICATION_ROOT}/{value['capability_plan_sha256']}/"
        f"{value['run_id']}/{value['fixture_sha256']}.json"
    )
    identity = value.get("fixture_file_identity")
    if (
        value.get("receipt_path") != expected_receipt_path
        or not isinstance(identity, Mapping)
        or set(identity)
        != {"device", "inode", "uid", "gid", "mode", "size", "mtime_ns"}
        or any(
            type(identity.get(field)) is not int or identity[field] < 0
            for field in ("device", "inode", "uid", "gid", "size", "mtime_ns")
        )
        or identity.get("uid") != 0
        or identity.get("gid") != 0
        or identity.get("mode") != "0400"
        or identity.get("size", 0) <= 1
    ):
        raise OwnerLauncherError("capability_fixture_publication_receipt_invalid")
    unsigned = {
        key: item for key, item in value.items() if key != "receipt_sha256"
    }
    if value.get("receipt_sha256") != hashlib.sha256(
        _canonical_bytes(unsigned)
    ).hexdigest():
        raise OwnerLauncherError("capability_fixture_publication_receipt_invalid")
    return dict(value)


def publish_live_fixture(
    transport: CapabilityCanaryTransport,
    *,
    revision: str,
    stream: Any = None,
) -> Mapping[str, Any]:
    source = sys.stdin.buffer if stream is None else stream
    frame = bytearray(source.read(_MAX_FIXTURE_AUTHORITY_BYTES + 1))
    try:
        if not frame or len(frame) > _MAX_FIXTURE_AUTHORITY_BYTES:
            raise OwnerLauncherError("capability_fixture_authority_input_invalid")
        value = _decode_json(bytes(frame), label="capability fixture authority")
        if bytes(frame) != _canonical_bytes(value):
            raise OwnerLauncherError("capability_fixture_authority_input_invalid")
        receipt = transport.invoke(
            revision,
            "publish-live-fixture",
            frame=frame,
        )
        return validate_fixture_publication_receipt(
            receipt,
            revision=revision,
        )
    finally:
        _wipe(frame)


def publish_capability_plan(
    transport: CapabilityCanaryTransport,
    *,
    revision: str,
    full_canary_plan_sha256: str,
    plan_sha256: str,
    owner_subject_sha256: str,
    plan_file: Path,
) -> Mapping[str, Any]:
    inputs = read_plan_publication_inputs(plan_file)
    authority = build_plan_publication_authority(
        revision=revision,
        full_canary_plan_sha256=full_canary_plan_sha256,
        plan_sha256=plan_sha256,
        owner_subject_sha256=owner_subject_sha256,
        inputs=inputs,
    )
    frame = bytearray(_canonical_bytes(authority))
    try:
        return transport.invoke(revision, "publish-plan", frame=frame)
    finally:
        _wipe(frame)


def bootstrap_bitrix_foundation(
    transport: CapabilityCanaryTransport,
    *,
    revision: str,
    full_canary_plan_sha256: str,
    release_artifact_sha256: str,
    owner_subject_sha256: str,
    foundation_file: Path,
) -> Mapping[str, Any]:
    inputs = read_bitrix_foundation_inputs(foundation_file)
    authority = build_bitrix_foundation_authority(
        revision=revision,
        full_canary_plan_sha256=full_canary_plan_sha256,
        release_artifact_sha256=release_artifact_sha256,
        owner_subject_sha256=owner_subject_sha256,
        inputs=inputs,
    )
    frame = bytearray(_canonical_bytes(authority))
    try:
        return transport.invoke(
            revision,
            "bootstrap-bitrix-foundation",
            frame=frame,
        )
    finally:
        _wipe(frame)


def provision_codex(
    transport: CapabilityCanaryTransport,
    *,
    revision: str,
    plan_sha256: str,
    owner_subject_sha256: str,
    auth_path: Path | None = None,
) -> Mapping[str, Any]:
    token: bytearray | None = None
    frame: bytearray | None = None
    try:
        harden_owner_secret_process()
        token = read_codex_access_token(auth_path)
        frame = build_secret_lease_frame(
            kind="codex_access_token",
            secret=token,
            plan_sha256=plan_sha256,
            owner_subject_sha256=owner_subject_sha256,
        )
        return transport.invoke(revision, "provision-codex", frame=frame)
    finally:
        _wipe(frame)
        _wipe(token)


def provision_mac_ops(
    transport: CapabilityCanaryTransport,
    *,
    revision: str,
    plan_sha256: str,
    owner_subject_sha256: str,
    stream: Any = None,
) -> Mapping[str, Any]:
    secret: bytearray | None = None
    frame: bytearray | None = None
    try:
        harden_owner_secret_process()
        secret = read_mac_ops_env(stream)
        frame = build_secret_lease_frame(
            kind="mac_ops_gitlab_env",
            secret=secret,
            plan_sha256=plan_sha256,
            owner_subject_sha256=owner_subject_sha256,
        )
        return transport.invoke(revision, "provision-mac-ops", frame=frame)
    finally:
        _wipe(frame)
        _wipe(secret)


def read_discord_connector_token(stream: Any = None) -> bytearray:
    """Read one bounded printable ASCII connector credential from inherited stdin."""

    source = sys.stdin.buffer if stream is None else stream
    if bool(getattr(source, "isatty", lambda: False)()):
        raise OwnerLauncherError("capability_discord_connector_stdin_is_tty")
    payload = bytearray(source.read(513))
    if (
        not payload
        or len(payload) > 512
        or any(value <= 0x20 or value == 0x7F for value in payload)
    ):
        _wipe(payload)
        raise OwnerLauncherError(
            "capability_discord_connector_secret_input_invalid"
        )
    try:
        payload.decode("ascii", errors="strict")
    except UnicodeError:
        _wipe(payload)
        raise OwnerLauncherError(
            "capability_discord_connector_secret_input_invalid"
        ) from None
    return payload


def provision_discord_connector(
    transport: CapabilityCanaryTransport,
    *,
    revision: str,
    plan_sha256: str,
    owner_subject_sha256: str,
    stream: Any = None,
) -> Mapping[str, Any]:
    secret: bytearray | None = None
    frame: bytearray | None = None
    try:
        harden_owner_secret_process()
        secret = read_discord_connector_token(stream)
        frame = build_secret_lease_frame(
            kind="discord_connector_token",
            secret=secret,
            plan_sha256=plan_sha256,
            owner_subject_sha256=owner_subject_sha256,
        )
        return transport.invoke(
            revision,
            "provision-discord-connector",
            frame=frame,
        )
    finally:
        _wipe(frame)
        _wipe(secret)


def generate_api_control_key() -> bytearray:
    """Generate one opaque printable control key in trusted launcher memory."""

    raw = bytearray(secrets.token_bytes(48))
    try:
        return bytearray(base64.urlsafe_b64encode(raw).rstrip(b"="))
    finally:
        _wipe(raw)


def provision_api_control(
    transport: CapabilityCanaryTransport,
    *,
    revision: str,
    plan_sha256: str,
    owner_subject_sha256: str,
) -> Mapping[str, Any]:
    secret: bytearray | None = None
    frame: bytearray | None = None
    try:
        harden_owner_secret_process()
        secret = generate_api_control_key()
        frame = build_secret_lease_frame(
            kind="api_server_control_key",
            secret=secret,
            plan_sha256=plan_sha256,
            owner_subject_sha256=owner_subject_sha256,
        )
        return transport.invoke(
            revision,
            "provision-api-control",
            frame=frame,
        )
    finally:
        _wipe(frame)
        _wipe(secret)


def read_discord_routeback_token(stream: Any = None) -> bytearray:
    """Read one bounded route-back token only from inherited non-TTY stdin."""

    source = sys.stdin.buffer if stream is None else stream
    if bool(getattr(source, "isatty", lambda: False)()):
        raise OwnerLauncherError("capability_discord_routeback_stdin_is_tty")
    payload = bytearray(source.read(513))
    if (
        not payload
        or len(payload) > 512
        or any(value <= 0x20 or value == 0x7F for value in payload)
    ):
        _wipe(payload)
        raise OwnerLauncherError(
            "capability_discord_routeback_secret_input_invalid"
        )
    try:
        payload.decode("ascii", errors="strict")
    except UnicodeError:
        _wipe(payload)
        raise OwnerLauncherError(
            "capability_discord_routeback_secret_input_invalid"
        ) from None
    return payload


def provision_discord_routeback(
    transport: CapabilityCanaryTransport,
    *,
    revision: str,
    plan_sha256: str,
    owner_subject_sha256: str,
    stream: Any = None,
) -> Mapping[str, Any]:
    secret: bytearray | None = None
    frame: bytearray | None = None
    try:
        harden_owner_secret_process()
        secret = read_discord_routeback_token(stream)
        frame = build_secret_lease_frame(
            kind="discord_routeback_token",
            secret=secret,
            plan_sha256=plan_sha256,
            owner_subject_sha256=owner_subject_sha256,
        )
        return transport.invoke(
            revision,
            "provision-discord-routeback",
            frame=frame,
        )
    finally:
        _wipe(frame)
        _wipe(secret)


def read_bitrix_operational_edge_webhook(stream: Any = None) -> bytearray:
    """Read the bounded opaque Bitrix webhook only from inherited stdin."""

    source = sys.stdin.buffer if stream is None else stream
    if bool(getattr(source, "isatty", lambda: False)()):
        raise OwnerLauncherError("capability_bitrix_webhook_stdin_is_tty")
    payload = bytearray(source.read(8 * 1024 + 3))
    if payload.endswith(b"\r\n"):
        del payload[-2:]
    elif payload.endswith(b"\n"):
        del payload[-1:]
    if (
        not payload
        or len(payload) > 8 * 1024
        or any(value <= 0x20 or value == 0x7F for value in payload)
    ):
        _wipe(payload)
        raise OwnerLauncherError("capability_bitrix_webhook_input_invalid")
    try:
        payload.decode("ascii", errors="strict")
    except UnicodeError:
        _wipe(payload)
        raise OwnerLauncherError("capability_bitrix_webhook_input_invalid") from None
    return payload


def provision_bitrix_operational_edge(
    transport: CapabilityCanaryTransport,
    *,
    revision: str,
    plan_sha256: str,
    owner_subject_sha256: str,
    stream: Any = None,
) -> Mapping[str, Any]:
    secret: bytearray | None = None
    frame: bytearray | None = None
    try:
        harden_owner_secret_process()
        secret = read_bitrix_operational_edge_webhook(stream)
        frame = build_secret_lease_frame(
            kind="bitrix_operational_edge_webhook",
            secret=secret,
            plan_sha256=plan_sha256,
            owner_subject_sha256=owner_subject_sha256,
        )
        return transport.invoke(
            revision,
            "provision-bitrix-operational-edge",
            frame=frame,
        )
    finally:
        _wipe(frame)
        _wipe(secret)


def install_capability_approval(
    transport: CapabilityCanaryTransport,
    *,
    revision: str,
    owner_subject_sha256: str,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Build and install one fresh approval bound to live stopped preflight."""

    preflight = transport.invoke(revision, "preflight-stopped")
    plan_sha256 = preflight.get("plan_sha256")
    full_plan_sha256 = preflight.get("full_canary_plan_sha256")
    report_sha256 = preflight.get("report_sha256")
    state_sha256 = preflight.get("state_sha256")
    checks = preflight.get("checks")
    evidence = preflight.get("evidence")
    browser_identity = (
        evidence.get("browser.host_identity")
        if isinstance(evidence, Mapping)
        else None
    )
    execution_identity = (
        evidence.get("execution.host_identity")
        if isinstance(evidence, Mapping)
        else None
    )
    if (
        preflight.get("schema") != CAPABILITY_PREFLIGHT_SCHEMA
        or preflight.get("ok") is not True
        or preflight.get("phase") != "stopped"
        or preflight.get("revision") != revision
        or not isinstance(checks, Mapping)
        or any(
            checks.get(name) is not True
            for name in (
                "browser.executable",
                "browser.host_identity",
                "browser.userns_sandbox",
                "browser.principal_smoke",
                "worker.executables",
                "worker.systemd252_tmpfs_contract",
                "execution.host_identity",
            )
        )
        or not isinstance(browser_identity, Mapping)
        or browser_identity.get("schema")
        != CAPABILITY_BROWSER_HOST_IDENTITY_SCHEMA
        or browser_identity.get("plan_sha256") != plan_sha256
        or browser_identity.get("create_only_eligible") is not True
        or not isinstance(browser_identity.get("receipt_sha256"), str)
        or __import__("re").fullmatch(
            r"[0-9a-f]{64}", browser_identity.get("receipt_sha256", "")
        )
        is None
        or not isinstance(execution_identity, Mapping)
        or execution_identity.get("schema")
        != CAPABILITY_EXECUTION_HOST_IDENTITY_SCHEMA
        or execution_identity.get("plan_sha256") != plan_sha256
        or execution_identity.get("create_only_eligible") is not True
        or __import__("re").fullmatch(
            r"[0-9a-f]{64}", execution_identity.get("receipt_sha256", "")
        )
        is None
        or any(
            not isinstance(value, str)
            or __import__("re").fullmatch(r"[0-9a-f]{64}", value) is None
            for value in (
                plan_sha256,
                full_plan_sha256,
                report_sha256,
                state_sha256,
            )
        )
    ):
        raise OwnerLauncherError("capability_stopped_preflight_invalid")
    approved = int(time.time()) if now_unix is None else now_unix
    if type(approved) is not int or approved < 0:
        raise OwnerLauncherError("capability_owner_approval_time_invalid")
    source = {
        "schema": "muncho-production-capability-owner-approval-source.v1",
        "scope": "production_capability_canary_runtime_start",
        "revision": revision,
        "plan_sha256": plan_sha256,
        "full_canary_plan_sha256": full_plan_sha256,
        "owner_subject_sha256": owner_subject_sha256,
        "stopped_preflight_sha256": report_sha256,
        "stopped_preflight_state_sha256": state_sha256,
        "approved_at_unix": approved,
    }
    approval = {
        "schema": CAPABILITY_APPROVAL_SCHEMA,
        "scope": "production_capability_canary_runtime_start",
        "plan_sha256": plan_sha256,
        "full_canary_plan_sha256": full_plan_sha256,
        "authority_kind": "trusted_root_bootstrap_out_of_band_owner",
        "cryptographic_owner_proof": False,
        "owner_subject_sha256": owner_subject_sha256,
        "approval_source_sha256": hashlib.sha256(
            _canonical_bytes(source)
        ).hexdigest(),
        "stopped_preflight_state_sha256": state_sha256,
        "nonce_sha256": hashlib.sha256(secrets.token_bytes(32)).hexdigest(),
        "approved_at_unix": approved,
        "expires_at_unix": approved + 300,
    }
    frame = bytearray(_canonical_bytes(approval))
    try:
        return transport.invoke(revision, "install-approval", frame=frame)
    finally:
        _wipe(frame)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Owner-side production-shaped capability-canary launcher"
    )
    parser.add_argument("revision")
    parser.add_argument(
        "action",
        choices=(
            "contract",
            "storage-preflight",
            "author-plan-inputs",
            "author-live-fixture",
            "bootstrap-bitrix-foundation",
            "bootstrap-producer-foundation",
            "publish-plan",
            "preflight-stopped",
            "preflight-live",
            "provision-codex",
            "provision-mac-ops",
            "provision-discord-connector",
            "provision-api-control",
            "provision-discord-routeback",
            "provision-bitrix-operational-edge",
            "install-approval",
            "publish-live-fixture",
            "start",
            "run-live",
            "run-live-observed",
            "stop",
            "retire-secrets",
        ),
    )
    parser.add_argument("--plan-sha256")
    parser.add_argument("--full-canary-plan-sha256")
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--foundation-file", type=Path)
    parser.add_argument("--release-artifact-sha256")
    parser.add_argument("--codex-auth-path", type=Path)
    parser.add_argument("--fixture-sha256")
    parser.add_argument("--run-id")
    parser.add_argument("--observation-timeout-seconds", type=int)
    parser.add_argument("--full-canary-receipt-file", type=Path)
    parser.add_argument("--producer-receipt-file", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--valid-for-seconds", type=int)
    parser.add_argument("--connector-bot-user-id")
    parser.add_argument("--routeback-bot-user-id")
    for field in sorted(_PLAN_INPUT_IDENTITY_FIELDS):
        parser.add_argument(f"--{field.replace('_', '-')}", type=int)
    for field in sorted(_PLAN_INPUT_ARTIFACT_FIELDS):
        parser.add_argument(f"--{field.replace('_', '-')}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        trusted = require_trusted_owner_runtime(args.revision)
        require_capability_launcher_provenance(args.revision)
        transport: CapabilityCanaryTransport | None = None
        owner_subject: str | None = None
        if args.action != "author-live-fixture":
            configuration = PinnedGcloudConfiguration()
            identity = GcloudOwnerAccessToken(
                gcloud_executable=trusted,
                gcloud_configuration=configuration,
            )
            identity.account_for_read_only_preflight()
            owner_subject = identity.owner_subject_sha256
            if not isinstance(owner_subject, str):
                raise OwnerLauncherError("capability_owner_identity_invalid")
            transport = CapabilityCanaryTransport(
                identity,
                gcloud_executable=trusted,
                gcloud_configuration=configuration,
            )
        if args.action != "run-live-observed" and any(
            value is not None
            for value in (
                args.fixture_sha256,
                args.observation_timeout_seconds,
            )
        ):
            raise OwnerLauncherError("capability_canary_command_invalid")
        if args.action not in {"run-live-observed", "author-live-fixture"} and (
            args.run_id is not None
        ):
            raise OwnerLauncherError("capability_canary_command_invalid")
        plan_identity_values = {
            field: getattr(args, field) for field in _PLAN_INPUT_IDENTITY_FIELDS
        }
        plan_artifact_values = {
            field: getattr(args, field) for field in _PLAN_INPUT_ARTIFACT_FIELDS
        }
        if args.action != "author-plan-inputs" and any(
            value is not None
            for value in (
                args.connector_bot_user_id,
                args.routeback_bot_user_id,
                *plan_identity_values.values(),
                *plan_artifact_values.values(),
            )
        ):
            raise OwnerLauncherError("capability_canary_command_invalid")
        if args.action not in {"author-plan-inputs", "author-live-fixture"} and any(
            value is not None
            for value in (
                args.full_canary_receipt_file,
                args.output_file,
            )
        ):
            raise OwnerLauncherError("capability_canary_command_invalid")
        if args.action != "author-live-fixture" and any(
            value is not None
            for value in (
                args.producer_receipt_file,
                args.valid_for_seconds,
            )
        ):
            raise OwnerLauncherError("capability_canary_command_invalid")
        if args.action != "bootstrap-bitrix-foundation" and (
            args.foundation_file is not None
            or args.release_artifact_sha256 is not None
        ):
            raise OwnerLauncherError("capability_canary_command_invalid")
        if args.action == "author-plan-inputs":
            if (
                args.full_canary_receipt_file is None
                or args.output_file is None
                or not args.connector_bot_user_id
                or not args.routeback_bot_user_id
                or any(value is None for value in plan_identity_values.values())
                or any(value is None for value in plan_artifact_values.values())
                or any(
                    value is not None
                    for value in (
                        args.plan_sha256,
                        args.full_canary_plan_sha256,
                        args.plan_file,
                        args.foundation_file,
                        args.release_artifact_sha256,
                        args.codex_auth_path,
                    )
                )
            ):
                raise OwnerLauncherError("capability_plan_authoring_binding_missing")
            if transport is None:
                raise OwnerLauncherError("capability_canary_transport_invalid")
            result = author_capability_plan_inputs(
                transport,
                revision=args.revision,
                terminal_receipt_file=args.full_canary_receipt_file,
                output_file=args.output_file,
                identities=plan_identity_values,
                connector_bot_user_id=args.connector_bot_user_id,
                routeback_bot_user_id=args.routeback_bot_user_id,
                artifacts=plan_artifact_values,
            )
        elif args.action == "author-live-fixture":
            if (
                args.full_canary_receipt_file is None
                or args.producer_receipt_file is None
                or args.output_file is None
                or not args.run_id
                or args.valid_for_seconds is None
                or any(
                    value is not None
                    for value in (
                        args.plan_sha256,
                        args.full_canary_plan_sha256,
                        args.plan_file,
                        args.foundation_file,
                        args.release_artifact_sha256,
                        args.codex_auth_path,
                    )
                )
            ):
                raise OwnerLauncherError("capability_fixture_authoring_binding_missing")
            result = author_live_fixture_authority(
                revision=args.revision,
                terminal_receipt_file=args.full_canary_receipt_file,
                producer_receipt_file=args.producer_receipt_file,
                output_file=args.output_file,
                run_id=args.run_id,
                valid_for_seconds=args.valid_for_seconds,
            )
        elif args.action == "bootstrap-bitrix-foundation":
            if transport is None or owner_subject is None:
                raise OwnerLauncherError("capability_canary_transport_invalid")
            if (
                not args.full_canary_plan_sha256
                or not args.release_artifact_sha256
                or args.foundation_file is None
                or args.plan_sha256 is not None
                or args.plan_file is not None
                or args.codex_auth_path is not None
            ):
                raise OwnerLauncherError(
                    "capability_bitrix_foundation_binding_missing"
                )
            result = bootstrap_bitrix_foundation(
                transport,
                revision=args.revision,
                full_canary_plan_sha256=args.full_canary_plan_sha256,
                release_artifact_sha256=args.release_artifact_sha256,
                owner_subject_sha256=owner_subject,
                foundation_file=args.foundation_file,
            )
        elif args.action == "publish-plan":
            if transport is None or owner_subject is None:
                raise OwnerLauncherError("capability_canary_transport_invalid")
            if (
                not args.plan_sha256
                or not args.full_canary_plan_sha256
                or args.plan_file is None
                or args.codex_auth_path is not None
            ):
                raise OwnerLauncherError("capability_plan_binding_missing")
            result = publish_capability_plan(
                transport,
                revision=args.revision,
                full_canary_plan_sha256=args.full_canary_plan_sha256,
                plan_sha256=args.plan_sha256,
                owner_subject_sha256=owner_subject,
                plan_file=args.plan_file,
            )
        elif args.action == "bootstrap-producer-foundation":
            if transport is None:
                raise OwnerLauncherError("capability_canary_transport_invalid")
            if (
                not args.plan_sha256
                or not args.full_canary_plan_sha256
                or args.plan_file is not None
                or args.codex_auth_path is not None
            ):
                raise OwnerLauncherError("capability_producer_binding_missing")
            result = bootstrap_producer_foundation(
                transport,
                revision=args.revision,
                plan_sha256=args.plan_sha256,
                full_canary_plan_sha256=args.full_canary_plan_sha256,
            )
        elif args.action == "provision-codex":
            if transport is None or owner_subject is None:
                raise OwnerLauncherError("capability_canary_transport_invalid")
            if (
                args.full_canary_plan_sha256 is not None
                or args.plan_file is not None
            ):
                raise OwnerLauncherError("capability_canary_command_invalid")
            if not args.plan_sha256:
                raise OwnerLauncherError("capability_plan_binding_missing")
            result = provision_codex(
                transport,
                revision=args.revision,
                plan_sha256=args.plan_sha256,
                owner_subject_sha256=owner_subject,
                auth_path=args.codex_auth_path,
            )
        elif args.action == "provision-mac-ops":
            if (
                not args.plan_sha256
                or args.codex_auth_path is not None
                or args.full_canary_plan_sha256 is not None
                or args.plan_file is not None
            ):
                raise OwnerLauncherError("capability_plan_binding_missing")
            result = provision_mac_ops(
                transport,
                revision=args.revision,
                plan_sha256=args.plan_sha256,
                owner_subject_sha256=owner_subject,
            )
        elif args.action == "provision-discord-connector":
            if (
                not args.plan_sha256
                or args.codex_auth_path is not None
                or args.full_canary_plan_sha256 is not None
                or args.plan_file is not None
            ):
                raise OwnerLauncherError("capability_plan_binding_missing")
            result = provision_discord_connector(
                transport,
                revision=args.revision,
                plan_sha256=args.plan_sha256,
                owner_subject_sha256=owner_subject,
            )
        elif args.action == "provision-api-control":
            if (
                not args.plan_sha256
                or args.codex_auth_path is not None
                or args.full_canary_plan_sha256 is not None
                or args.plan_file is not None
            ):
                raise OwnerLauncherError("capability_plan_binding_missing")
            result = provision_api_control(
                transport,
                revision=args.revision,
                plan_sha256=args.plan_sha256,
                owner_subject_sha256=owner_subject,
            )
        elif args.action == "provision-discord-routeback":
            if (
                not args.plan_sha256
                or args.codex_auth_path is not None
                or args.full_canary_plan_sha256 is not None
                or args.plan_file is not None
            ):
                raise OwnerLauncherError("capability_plan_binding_missing")
            result = provision_discord_routeback(
                transport,
                revision=args.revision,
                plan_sha256=args.plan_sha256,
                owner_subject_sha256=owner_subject,
            )
        elif args.action == "provision-bitrix-operational-edge":
            if (
                not args.plan_sha256
                or args.codex_auth_path is not None
                or args.full_canary_plan_sha256 is not None
                or args.plan_file is not None
            ):
                raise OwnerLauncherError("capability_plan_binding_missing")
            result = provision_bitrix_operational_edge(
                transport,
                revision=args.revision,
                plan_sha256=args.plan_sha256,
                owner_subject_sha256=owner_subject,
            )
        elif args.action == "install-approval":
            if any(
                value is not None
                for value in (
                    args.plan_sha256,
                    args.full_canary_plan_sha256,
                    args.plan_file,
                    args.codex_auth_path,
                )
            ):
                raise OwnerLauncherError("capability_canary_command_invalid")
            result = install_capability_approval(
                transport,
                revision=args.revision,
                owner_subject_sha256=owner_subject,
            )
        elif args.action == "publish-live-fixture":
            if any(
                value is not None
                for value in (
                    args.plan_sha256,
                    args.full_canary_plan_sha256,
                    args.plan_file,
                    args.codex_auth_path,
                )
            ):
                raise OwnerLauncherError("capability_canary_command_invalid")
            result = publish_live_fixture(
                transport,
                revision=args.revision,
            )
        elif args.action == "run-live-observed":
            if (
                not args.plan_sha256
                or not args.full_canary_plan_sha256
                or not args.fixture_sha256
                or not args.run_id
                or args.plan_file is not None
                or args.codex_auth_path is not None
                or args.foundation_file is not None
                or args.release_artifact_sha256 is not None
            ):
                raise OwnerLauncherError(
                    "production_observation_binding_invalid"
                )
            result = run_live_with_owner_production_observations(
                transport,
                CapabilityProductionObservationTransport(
                    revision=args.revision
                ),
                revision=args.revision,
                plan_sha256=args.plan_sha256,
                full_canary_plan_sha256=(
                    args.full_canary_plan_sha256
                ),
                fixture_sha256=args.fixture_sha256,
                run_id=args.run_id,
                owner_subject_sha256=owner_subject,
                timeout_seconds=(
                    300
                    if args.observation_timeout_seconds is None
                    else args.observation_timeout_seconds
                ),
            )
        else:
            if any(
                value is not None
                for value in (
                    args.plan_sha256,
                    args.full_canary_plan_sha256,
                    args.plan_file,
                    args.codex_auth_path,
                )
            ):
                raise OwnerLauncherError("capability_canary_command_invalid")
            result = transport.invoke(args.revision, args.action)
        sys.stdout.buffer.write(_canonical_bytes(result) + b"\n")
        return 0
    except Exception as exc:
        failure = {
            "schema": "muncho-production-capability-owner-launcher-failure.v1",
            "ok": False,
            "error_code": getattr(exc, "code", "capability_owner_launcher_failed"),
        }
        sys.stdout.buffer.write(_canonical_bytes(failure) + b"\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CapabilityCanaryTransport",
    "CapabilityProductionObservationTransport",
    "author_capability_plan_inputs",
    "author_live_fixture_authority",
    "build_live_fixture_authority",
    "build_plan_publication_inputs",
    "build_plan_publication_authority",
    "validate_release_storage_preflight",
    "install_capability_approval",
    "publish_live_fixture",
    "publish_capability_plan",
    "provision_codex",
    "provision_discord_connector",
    "provision_mac_ops",
    "read_codex_access_token",
    "read_discord_connector_token",
    "read_mac_ops_env",
    "read_plan_publication_inputs",
    "run_live_with_owner_production_observations",
    "validate_plan_publication_inputs",
    "validate_fixture_publication_receipt",
    "validate_full_canary_terminal_receipt",
    "validate_plan_authoring_context",
    "validate_producer_bootstrap_receipt",
]
