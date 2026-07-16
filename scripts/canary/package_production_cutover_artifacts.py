#!/usr/bin/env python3
"""Build the six exact, self-contained production cutover executables.

The release clone is immutable only after this packaging step.  The builder
embeds the reviewed SQL and privileged connector boundary into each Python
executable, seals a disjoint action allowlist into each artifact, and writes a
canonical manifest containing every byte digest.  Target identifiers and
numeric service identities are accepted only through the separately signed,
revision-bound unit-input authority.  No credential value or mutable Cloud
state is accepted by this build step; later mutation remains bound to the
signed cutover plan.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from gateway.isolated_worker_units import (
    BWRAP_PATH,
    CONFIG_MODE as WORKER_CONFIG_MODE,
    ISOLATED_WORKER_CLIENT_GROUP,
    ISOLATED_WORKER_CONFIG,
    ISOLATED_WORKER_GROUP,
    ISOLATED_WORKER_LEASE_BASE,
    ISOLATED_WORKER_SERVICE_UNIT,
    ISOLATED_WORKER_SOCKET,
    ISOLATED_WORKER_SOCKET_UNIT,
    ISOLATED_WORKER_USER,
    SHELL_PATH,
    render_isolated_worker_units,
)
from gateway.operational_edge_assets import (
    ASSET_MANIFEST_RELATIVE,
    OperationalEdgeAssetError,
    package_operational_assets,
    validate_packaged_operational_asset_verification,
    verify_packaged_operational_assets,
)
from gateway.operational_edge_catalog import CREDENTIALS_BY_DOMAIN
from gateway.operational_edge_units import (
    CLIENT_CONFIG_PATH as OPERATIONAL_EDGE_CLIENT_CONFIG,
    OperationalEdgeUnitError,
    render_operational_edge_units,
    service_config_path as operational_edge_config_path,
    service_identity_name as operational_edge_service_identity_name,
    service_unit as operational_edge_service_unit,
    socket_group_name as operational_edge_socket_group_name,
)
from gateway.production_capability_prerequisites import (
    BROWSER_CONFIG_PATH,
    BROWSER_SOCKET_PATH,
    BROWSER_UNIT,
    MAC_OPS_UNIT,
    PHASE_B_UNIT,
    ROUTEBACK_EDGE_UNIT,
)
from gateway.production_capability_prerequisites import (
    packaged_prerequisite_contract,
)
from gateway.production_capability_units import (
    BROWSER_CONFIG_MODE,
    render_production_capability_units,
)
from gateway.production_cron_continuity_package import (
    PLAN_SCHEMA as PRODUCTION_CRON_CONTINUITY_PLAN_SCHEMA,
)
from gateway import production_owner_runtime


MANIFEST_SCHEMA = "muncho-production-cutover-artifact-manifest.v1"
HOST_ARTIFACT_CONTRACT_SCHEMA = (
    "muncho-production-cutover-host-artifact-contract.v1"
)
RUNTIME_DEPENDENCY_MANIFEST_SCHEMA = "muncho-production-runtime-dependencies.v1"
RUNTIME_DEPENDENCY_MANIFEST = Path(
    "ops/muncho/runtime/dependencies/manifest.json"
)
REVISION = re.compile(r"^[0-9a-f]{40}$")
SENTINELS = {
    "__MUNCHO_ALLOWED_ACTIONS__",
    "__MUNCHO_LEGACY_RECONCILE_SQL__",
    "__MUNCHO_WRITER_MIGRATION_SQL__",
    "__MUNCHO_CONNECTOR_UNIT_TEMPLATE__",
    "__MUNCHO_GATEWAY_CONNECTOR_DROP_IN_BYTES__",
    "__MUNCHO_PRODUCTION_CAPABILITY_PREREQUISITE_CONTRACT__",
    "__MUNCHO_PRODUCTION_CRON_CONTINUITY_PLAN_SCHEMA__",
    "__MUNCHO_SEALED_RUNTIME_ARTIFACT_REQUEST__",
}
UNIT_INPUT_SCHEMA = "muncho-production-cutover-unit-inputs.v2"
SEALED_RUNTIME_ARTIFACT_REQUEST_SCHEMA = (
    "muncho-production-cutover-sealed-runtime-artifacts.v1"
)
CUTOVER_STAGED_ROOT = Path("/var/lib/muncho-production-legacy-cutover/staged")
STAGED_UNIT_INPUT_PLAN_PATH = CUTOVER_STAGED_ROOT / "unit-input-plan.json"
STAGED_UNIT_INPUT_APPROVAL_PATH = (
    CUTOVER_STAGED_ROOT / "unit-input-approval.json"
)
FIXED_UNIT_INPUTS_PATH = CUTOVER_STAGED_ROOT / "production-unit-inputs.json"
FIXED_UNIT_INPUTS_MODE = 0o444
UNIT_INPUT_STAGING_SCHEMA = "muncho-production-cutover-unit-input-staging.v2"
UNIT_INPUT_PAYLOAD_SCHEMA = "muncho-production-cutover-unit-input-payload.v2"
UNIT_INPUT_PLAN_SCHEMA = "muncho-production-cutover-unit-input-plan.v2"
UNIT_INPUT_APPROVAL_SCHEMA = "muncho-production-cutover-unit-input-approval.v2"
ARTIFACTS: Mapping[str, tuple[str, ...]] = {
    "production-observe": (
        "observe_initial",
        "observe_final_tail",
        "observe_before_apply",
    ),
    "production-database-apply": ("database_apply",),
    "production-database-rollback": ("database_rollback",),
    "production-database-postflight": ("database_preflight", "database_terminal"),
    "production-host-activation": (
        "host_apply_stopped",
        "host_start_prerequisites",
        "host_start_writer",
        "host_commit_boot",
    ),
    "production-host-rollback": ("host_rollback",),
}
PLAN_BINDINGS = {
    "observe": "production-observe",
    "database_apply": "production-database-apply",
    "database_rollback": "production-database-rollback",
    "database_postflight": "production-database-postflight",
    "host_activation": "production-host-activation",
    "host_rollback": "production-host-rollback",
}

# Every host file consumed by the sealed host-activation artifact must be
# represented in the release package.  Twenty-seven files have bytes rendered
# and embedded at package time.  The remaining eleven are rendered from
# owner-controlled production inputs (or are root-only verifier artifacts), so
# their final byte digest is bound later by the read-only host-authority receipt
# and the signed FreezePlan.  Keeping the complete set here prevents a release
# from silently adding an unreviewed, uncollected host input.
HOST_ARTIFACT_TARGETS: Mapping[str, tuple[str, str]] = {
    "gateway_unit": (
        "/etc/systemd/system/hermes-cloud-gateway.service",
        "owner_runtime_rendered",
    ),
    "writer_unit": (
        "/etc/systemd/system/muncho-canonical-writer.service",
        "owner_runtime_rendered",
    ),
    "connector_unit": (
        "/etc/systemd/system/muncho-discord-connector.service",
        "owner_runtime_rendered",
    ),
    "phase_b_unit": (
        f"/etc/systemd/system/{PHASE_B_UNIT}",
        "release_sealed_payload",
    ),
    "routeback_unit": (
        f"/etc/systemd/system/{ROUTEBACK_EDGE_UNIT}",
        "release_sealed_payload",
    ),
    "mac_ops_unit": (
        f"/etc/systemd/system/{MAC_OPS_UNIT}",
        "release_sealed_payload",
    ),
    "browser_unit": (
        f"/etc/systemd/system/{BROWSER_UNIT}",
        "release_sealed_payload",
    ),
    "browser_config": (
        str(BROWSER_CONFIG_PATH),
        "release_sealed_payload",
    ),
    "isolated_worker_socket_unit": (
        f"/etc/systemd/system/{ISOLATED_WORKER_SOCKET_UNIT}",
        "release_sealed_payload",
    ),
    "isolated_worker_service_unit": (
        f"/etc/systemd/system/{ISOLATED_WORKER_SERVICE_UNIT}",
        "release_sealed_payload",
    ),
    "isolated_worker_config": (
        str(ISOLATED_WORKER_CONFIG),
        "release_sealed_payload",
    ),
    "gateway_connector_drop_in": (
        "/etc/systemd/system/hermes-cloud-gateway.service.d/"
        "20-discord-connector.conf",
        "release_reviewed_source",
    ),
    "gateway_config": (
        "/opt/adventico-ai-platform/hermes-home/config.yaml",
        "owner_runtime_rendered",
    ),
    "writer_config": (
        "/etc/muncho-canonical-writer/writer.json",
        "owner_runtime_rendered",
    ),
    "connector_config": (
        "/etc/muncho/discord-public-connector.json",
        "owner_runtime_rendered",
    ),
    "routeback_config": (
        "/etc/muncho/discord-edge.json",
        "owner_runtime_rendered",
    ),
    "mac_ops_config": (
        "/etc/muncho/mac-ops-edge/config.json",
        "owner_runtime_rendered",
    ),
    "api_bearer_verifier": (
        "/etc/muncho/keys/api-server-bearer-sha256.json",
        "root_verifier",
    ),
    "api_approval_verifier": (
        "/etc/muncho/keys/api-approval-passkey-scrypt.json",
        "root_verifier",
    ),
    **{
        f"operational_edge_unit_{domain}": (
            f"/etc/systemd/system/{operational_edge_service_unit(domain)}",
            "release_sealed_payload",
        )
        for domain in sorted(CREDENTIALS_BY_DOMAIN)
    },
    **{
        f"operational_edge_config_{domain}": (
            str(operational_edge_config_path(domain)),
            "release_sealed_payload",
        )
        for domain in sorted(CREDENTIALS_BY_DOMAIN)
    },
    "operational_edge_client_config": (
        str(OPERATIONAL_EDGE_CLIENT_CONFIG),
        "release_sealed_payload",
    ),
}


class PackagingError(RuntimeError):
    """Stable packaging failure."""


def _exact_mapping(
    value: Any,
    fields: frozenset[str],
    code: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PackagingError(code)
    return dict(value)


def _identity_input(value: Any, label: str) -> dict[str, Any]:
    raw = _exact_mapping(
        value,
        frozenset({"user", "group", "uid", "gid"}),
        "cutover_packaging_unit_inputs_invalid",
    )
    if (
        not isinstance(raw["user"], str)
        or not isinstance(raw["group"], str)
        or re.fullmatch(r"[a-z_][a-z0-9_-]{0,63}", raw["user"]) is None
        or re.fullmatch(r"[a-z_][a-z0-9_-]{0,63}", raw["group"]) is None
        or type(raw["uid"]) is not int
        or type(raw["gid"]) is not int
        or raw["uid"] <= 0
        or raw["gid"] <= 0
    ):
        raise PackagingError("cutover_packaging_unit_inputs_invalid")
    return raw


def _operational_edge_identity_inputs(
    identities_value: Any,
    socket_groups_value: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    domains = set(CREDENTIALS_BY_DOMAIN)
    identities_raw = _exact_mapping(
        identities_value,
        frozenset(domains),
        "cutover_packaging_unit_inputs_invalid",
    )
    sockets_raw = _exact_mapping(
        socket_groups_value,
        frozenset(domains),
        "cutover_packaging_unit_inputs_invalid",
    )
    identities: dict[str, dict[str, Any]] = {}
    sockets: dict[str, dict[str, Any]] = {}
    for domain in sorted(domains):
        identity = _identity_input(
            identities_raw[domain], f"operational edge {domain}"
        )
        socket = _exact_mapping(
            sockets_raw[domain],
            frozenset({"group", "gid"}),
            "cutover_packaging_unit_inputs_invalid",
        )
        if (
            identity["user"] != operational_edge_service_identity_name(domain)
            or identity["group"] != operational_edge_service_identity_name(domain)
            or socket["group"] != operational_edge_socket_group_name(domain)
            or re.fullmatch(r"[a-z_][a-z0-9_-]{0,63}", str(socket["group"]))
            is None
            or type(socket["gid"]) is not int
            or socket["gid"] <= 0
        ):
            raise PackagingError("cutover_packaging_unit_inputs_invalid")
        identities[domain] = identity
        sockets[domain] = socket
    return identities, sockets


_UNIT_INPUT_PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "database_ip",
        "target",
        "gateway",
        "routeback",
        "mac_ops",
        "browser",
        "worker",
        "worker_client_group",
        "worker_client_gid",
        "operational_edge_identities",
        "operational_edge_socket_groups",
        "writer_capability_public_key_id",
        "operational_edge_key_foundation_sha256",
        "operational_edge_receipt_public_key_ids",
        "release_owner_uid",
        "release_owner_gid",
        "bwrap_sha256",
        "shell_sha256",
        "secret_material_recorded",
        "secret_digest_recorded",
    }
)


_TARGET_INPUT_FIELDS = frozenset({
    "project",
    "zone",
    "vm",
    "database",
    "sql_instance",
    "sql_host",
    "tls_server_name",
    "port",
    "writer_login",
})


def _target_input(value: Any, *, database_ip: str) -> dict[str, Any]:
    raw = _exact_mapping(
        value,
        _TARGET_INPUT_FIELDS,
        "cutover_packaging_unit_inputs_invalid",
    )
    try:
        address = ipaddress.ip_address(str(raw["sql_host"]))
    except ValueError as exc:
        raise PackagingError("cutover_packaging_unit_inputs_invalid") from exc
    if (
        raw["project"] != "adventico-ai-platform"
        or raw["zone"] != "europe-west3-a"
        or raw["vm"] != "ai-platform-runtime-01"
        or raw["database"] != "ai_platform_brain"
        or raw["port"] != 5432
        or str(address) != raw["sql_host"]
        or raw["sql_host"] != database_ip
        or not isinstance(raw["sql_instance"], str)
        or re.fullmatch(r"[a-z][a-z0-9-]{0,62}", raw["sql_instance"]) is None
        or not isinstance(raw["tls_server_name"], str)
        or len(raw["tls_server_name"]) > 253
        or re.fullmatch(r"[A-Za-z0-9.-]+", raw["tls_server_name"]) is None
        or not isinstance(raw["writer_login"], str)
        or re.fullmatch(r"[a-z_][a-z0-9_-]{0,63}", raw["writer_login"]) is None
    ):
        raise PackagingError("cutover_packaging_unit_inputs_invalid")
    return raw


def _unit_input_payload(value: Any) -> dict[str, Any]:
    raw = _exact_mapping(
        value,
        _UNIT_INPUT_PAYLOAD_FIELDS,
        "cutover_packaging_unit_inputs_invalid",
    )
    identities = {
        name: _identity_input(raw[name], name)
        for name in (
            "gateway",
            "routeback",
            "mac_ops",
            "browser",
            "worker",
        )
    }
    operational_identities, operational_socket_groups = (
        _operational_edge_identity_inputs(
            raw["operational_edge_identities"],
            raw["operational_edge_socket_groups"],
        )
    )
    target = _target_input(raw["target"], database_ip=str(raw["database_ip"]))
    receipt_key_ids = raw["operational_edge_receipt_public_key_ids"]
    if (
        raw["schema"] != UNIT_INPUT_PAYLOAD_SCHEMA
        or not isinstance(raw["database_ip"], str)
        or not raw["database_ip"]
        or raw["worker"]["user"] != ISOLATED_WORKER_USER
        or raw["worker"]["group"] != ISOLATED_WORKER_GROUP
        or raw["worker_client_group"] != ISOLATED_WORKER_CLIENT_GROUP
        or type(raw["worker_client_gid"]) is not int
        or raw["worker_client_gid"] <= 0
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(raw["writer_capability_public_key_id"]),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(raw["operational_edge_key_foundation_sha256"]),
        )
        is None
        or not isinstance(receipt_key_ids, Mapping)
        or set(receipt_key_ids) != set(CREDENTIALS_BY_DOMAIN)
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(key_id)) is None
            for key_id in receipt_key_ids.values()
        )
        or len(set(receipt_key_ids.values())) != len(receipt_key_ids)
        or raw["writer_capability_public_key_id"]
        in set(receipt_key_ids.values())
        or type(raw["release_owner_uid"]) is not int
        or type(raw["release_owner_gid"]) is not int
        or raw["release_owner_uid"] != identities["gateway"]["uid"]
        or raw["release_owner_gid"] != identities["gateway"]["gid"]
        or not isinstance(raw["bwrap_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", raw["bwrap_sha256"]) is None
        or not isinstance(raw["shell_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", raw["shell_sha256"]) is None
        or raw["bwrap_sha256"] == raw["shell_sha256"]
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or len(
            {item["uid"] for item in identities.values()}
            | {item["uid"] for item in operational_identities.values()}
        )
        != len(identities) + len(operational_identities)
        or len(
            {item["gid"] for item in identities.values()}
            | {item["gid"] for item in operational_identities.values()}
            | {item["gid"] for item in operational_socket_groups.values()}
            | {raw["worker_client_gid"]}
        )
        != len(identities) + len(operational_identities) * 2 + 1
    ):
        raise PackagingError("cutover_packaging_unit_inputs_invalid")
    return {
        **raw,
        "target": target,
        "operational_edge_identities": operational_identities,
        "operational_edge_socket_groups": operational_socket_groups,
        "operational_edge_receipt_public_key_ids": dict(
            sorted(receipt_key_ids.items())
        ),
        **identities,
    }


def _unit_inputs(
    value: Any,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    raw = _exact_mapping(
        value,
        frozenset(
            {
                "schema",
                "release_revision",
                "authority_plan_sha256",
                "authority_approval_sha256",
                *(_UNIT_INPUT_PAYLOAD_FIELDS - {"schema"}),
            }
        ),
        "cutover_packaging_unit_inputs_invalid",
    )
    if (
        raw["schema"] != UNIT_INPUT_SCHEMA
        or not isinstance(raw["release_revision"], str)
        or REVISION.fullmatch(raw["release_revision"]) is None
        or (revision is not None and raw["release_revision"] != revision)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(raw["authority_plan_sha256"]),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(raw["authority_approval_sha256"]),
        )
        is None
        or raw["authority_plan_sha256"]
        == raw["authority_approval_sha256"]
    ):
        raise PackagingError("cutover_packaging_unit_inputs_invalid")
    payload = _unit_input_payload(
        {
            **{
                key: item
                for key, item in raw.items()
                if key
                not in {
                    "schema",
                    "release_revision",
                    "authority_plan_sha256",
                    "authority_approval_sha256",
                }
            },
            "schema": UNIT_INPUT_PAYLOAD_SCHEMA,
        }
    )
    return {
        "schema": UNIT_INPUT_SCHEMA,
        "release_revision": raw["release_revision"],
        "authority_plan_sha256": raw["authority_plan_sha256"],
        "authority_approval_sha256": raw["authority_approval_sha256"],
        **{key: item for key, item in payload.items() if key != "schema"},
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", errors="strict")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_source(path: Path, *, maximum: int) -> bytes:
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PackagingError("cutover_packaging_source_unavailable") from exc
    if (
        resolved != path
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > maximum
    ):
        raise PackagingError("cutover_packaging_source_invalid")
    payload = path.read_bytes()
    after = path.lstat()
    if (
        len(payload) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        or b"\x00" in payload
    ):
        raise PackagingError("cutover_packaging_source_raced")
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PackagingError("cutover_packaging_source_encoding_invalid") from exc
    return payload


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


def _read_trusted_staged_file(
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int,
    mode: int,
    maximum: int,
) -> bytes:
    """Read one fixed, immutable staging file without following a link."""

    descriptor: int | None = None
    try:
        before = os.lstat(path)
        if (
            not path.is_absolute()
            or ".." in path.parts
            or path.resolve(strict=True) != path
            or stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != expected_uid
            or before.st_gid != expected_gid
            or stat.S_IMODE(before.st_mode) != mode
            or not 0 < before.st_size <= maximum
        ):
            raise PackagingError("cutover_unit_inputs_staging_identity_invalid")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(
                descriptor,
                min(64 * 1024, maximum + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        reachable = os.lstat(path)
    except PackagingError:
        raise
    except OSError as exc:
        raise PackagingError("cutover_unit_inputs_staging_unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        len(payload) != before.st_size
        or len(payload) > maximum
        or _file_identity(before) != _file_identity(opened)
        or _file_identity(before) != _file_identity(after)
        or _file_identity(before) != _file_identity(reachable)
    ):
        raise PackagingError("cutover_unit_inputs_staging_changed")
    return bytes(payload)


def _decode_canonical_json(
    payload: bytes,
    *,
    newline: bool,
    code: str,
) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in items:
            if name in value:
                raise PackagingError(code)
            value[name] = item
        return value

    def constant(_value: str) -> None:
        raise PackagingError(code)

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except PackagingError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PackagingError(code) from exc
    expected = _canonical_bytes(value) + (b"\n" if newline else b"")
    if not isinstance(value, Mapping) or payload != expected:
        raise PackagingError(code)
    return value


def load_fixed_unit_inputs(
    path: Path = FIXED_UNIT_INPUTS_PATH,
    *,
    expected_uid: int = 0,
    expected_gid: int = 0,
) -> Mapping[str, Any]:
    """Load the one root-owned, non-secret input artifact used by build/verify."""

    payload = _read_trusted_staged_file(
        path,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        mode=FIXED_UNIT_INPUTS_MODE,
        maximum=128 * 1024,
    )
    return _unit_inputs(
        _decode_canonical_json(
            payload,
            newline=True,
            code="cutover_unit_inputs_staging_invalid",
        )
    )


_UNIT_INPUT_PLAN_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "unit_inputs",
        "owner_subject_sha256",
        "owner_public_key_ed25519_hex",
        "owner_key_id",
        "owner_runtime_attestation",
        "created_at_unix",
        "secret_material_recorded",
        "secret_digest_recorded",
        "plan_sha256",
    }
)
_UNIT_INPUT_APPROVAL_FIELDS = frozenset(
    {
        "schema",
        "purpose",
        "plan_sha256",
        "release_revision",
        "owner_subject_sha256",
        "owner_public_key_ed25519_hex",
        "owner_key_id",
        "nonce_sha256",
        "issued_at_unix",
        "expires_at_unix",
        "approved",
        "signature_ed25519_hex",
        "approval_sha256",
    }
)


def _self_hashed(
    value: Any,
    *,
    fields: frozenset[str],
    digest_field: str,
    code: str,
) -> dict[str, Any]:
    raw = _exact_mapping(value, fields, code)
    digest = raw[digest_field]
    unsigned = {key: item for key, item in raw.items() if key != digest_field}
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
        or _sha256(_canonical_bytes(unsigned)) != digest
    ):
        raise PackagingError(code)
    return raw


def validate_unit_input_plan(value: Any) -> Mapping[str, Any]:
    raw = _self_hashed(
        value,
        fields=_UNIT_INPUT_PLAN_FIELDS,
        digest_field="plan_sha256",
        code="cutover_unit_input_plan_invalid",
    )
    payload = _unit_input_payload(raw["unit_inputs"])
    public = raw["owner_public_key_ed25519_hex"]
    try:
        runtime_attestation = (
            production_owner_runtime.validate_owner_runtime_attestation(
                raw["owner_runtime_attestation"],
                revision=str(raw["release_revision"]),
            )
        )
    except production_owner_runtime.ProductionOwnerRuntimeError as exc:
        raise PackagingError("cutover_unit_input_plan_invalid") from exc
    if (
        raw["schema"] != UNIT_INPUT_PLAN_SCHEMA
        or not isinstance(raw["release_revision"], str)
        or REVISION.fullmatch(raw["release_revision"]) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(raw["owner_subject_sha256"]))
        is None
        or re.fullmatch(r"[0-9a-f]{64}", str(public)) is None
        or raw["owner_key_id"]
        != _sha256(bytes.fromhex(public))
        or type(raw["created_at_unix"]) is not int
        or raw["created_at_unix"] <= 0
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
    ):
        raise PackagingError("cutover_unit_input_plan_invalid")
    return {
        **raw,
        "unit_inputs": payload,
        "owner_runtime_attestation": runtime_attestation,
    }


def unit_input_approval_signature_payload(value: Mapping[str, Any]) -> bytes:
    if set(value) != _UNIT_INPUT_APPROVAL_FIELDS:
        raise PackagingError("cutover_unit_input_approval_invalid")
    return _canonical_bytes(
        {
            key: item
            for key, item in value.items()
            if key not in {"signature_ed25519_hex", "approval_sha256"}
        }
    )


def validate_unit_input_approval(
    value: Any,
    *,
    plan: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    raw = _self_hashed(
        value,
        fields=_UNIT_INPUT_APPROVAL_FIELDS,
        digest_field="approval_sha256",
        code="cutover_unit_input_approval_invalid",
    )
    signature = raw["signature_ed25519_hex"]
    if (
        raw["schema"] != UNIT_INPUT_APPROVAL_SCHEMA
        or raw["purpose"] != "production_cutover_unit_inputs"
        or raw["plan_sha256"] != plan["plan_sha256"]
        or raw["release_revision"] != plan["release_revision"]
        or raw["owner_subject_sha256"] != plan["owner_subject_sha256"]
        or raw["owner_public_key_ed25519_hex"]
        != plan["owner_public_key_ed25519_hex"]
        or raw["owner_key_id"] != plan["owner_key_id"]
        or re.fullmatch(r"[0-9a-f]{64}", str(raw["nonce_sha256"])) is None
        or type(raw["issued_at_unix"]) is not int
        or type(raw["expires_at_unix"]) is not int
        or not raw["issued_at_unix"] <= now_unix < raw["expires_at_unix"]
        or not 1 <= raw["expires_at_unix"] - raw["issued_at_unix"] <= 3600
        or raw["approved"] is not True
        or re.fullmatch(r"[0-9a-f]{128}", str(signature)) is None
    ):
        raise PackagingError("cutover_unit_input_approval_invalid")
    try:
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(plan["owner_public_key_ed25519_hex"])
        ).verify(
            bytes.fromhex(signature),
            unit_input_approval_signature_payload(raw),
        )
    except (InvalidSignature, ValueError) as exc:
        raise PackagingError("cutover_unit_input_approval_invalid") from exc
    return raw


def build_unit_input_plan(
    *,
    release_revision: str,
    unit_inputs: Mapping[str, Any],
    owner_subject_sha256: str,
    owner_public_key_ed25519_hex: str,
    owner_runtime_attestation: Mapping[str, Any],
    created_at_unix: int,
) -> Mapping[str, Any]:
    public = owner_public_key_ed25519_hex
    unsigned = {
        "schema": UNIT_INPUT_PLAN_SCHEMA,
        "release_revision": release_revision,
        "unit_inputs": dict(unit_inputs),
        "owner_subject_sha256": owner_subject_sha256,
        "owner_public_key_ed25519_hex": public,
        "owner_key_id": _sha256(bytes.fromhex(public)),
        "owner_runtime_attestation": dict(owner_runtime_attestation),
        "created_at_unix": created_at_unix,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return validate_unit_input_plan(
        {**unsigned, "plan_sha256": _sha256(_canonical_bytes(unsigned))}
    )


def _unit_inputs_from_authority(
    plan: Mapping[str, Any],
    approval: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = plan["unit_inputs"]
    return _unit_inputs(
        {
            "schema": UNIT_INPUT_SCHEMA,
            "release_revision": plan["release_revision"],
            "authority_plan_sha256": plan["plan_sha256"],
            "authority_approval_sha256": approval["approval_sha256"],
            **{key: item for key, item in payload.items() if key != "schema"},
        },
        revision=plan["release_revision"],
    )


def _create_or_validate_fixed_unit_inputs(
    path: Path,
    payload: bytes,
    *,
    uid: int,
    gid: int,
) -> bool:
    created = False
    if not os.path.lexists(path):
        temporary = path.with_name(f".{path.name}.bootstrap.{os.getpid()}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            os.fchown(descriptor, uid, gid)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short unit-input staging write")
                view = view[written:]
            os.fchmod(descriptor, FIXED_UNIT_INPUTS_MODE)
            os.fsync(descriptor)
        except BaseException:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path, follow_symlinks=False)
            created = True
        except FileExistsError:
            pass
        finally:
            temporary.unlink()
        parent = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    observed = _read_trusted_staged_file(
        path,
        expected_uid=uid,
        expected_gid=gid,
        mode=FIXED_UNIT_INPUTS_MODE,
        maximum=128 * 1024,
    )
    if observed != payload:
        raise PackagingError("cutover_unit_inputs_staging_conflict")
    return created


def bootstrap_fixed_unit_inputs(
    *,
    authority_plan_path: Path = STAGED_UNIT_INPUT_PLAN_PATH,
    authority_approval_path: Path = STAGED_UNIT_INPUT_APPROVAL_PATH,
    unit_inputs_path: Path = FIXED_UNIT_INPUTS_PATH,
    require_root: bool = True,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Create fixed inputs from a separately signed, pre-package authority."""

    geteuid = getattr(os, "geteuid", None)
    getegid = getattr(os, "getegid", None)
    uname = getattr(os, "uname", None)
    if geteuid is None or getegid is None or uname is None:
        raise PackagingError("cutover_unit_inputs_bootstrap_boundary_invalid")
    effective_uid = geteuid()
    effective_gid = getegid()
    if require_root and (
        effective_uid != 0
        or not uname().sysname.lower().startswith("linux")
        or authority_plan_path != STAGED_UNIT_INPUT_PLAN_PATH
        or authority_approval_path != STAGED_UNIT_INPUT_APPROVAL_PATH
        or unit_inputs_path != FIXED_UNIT_INPUTS_PATH
    ):
        raise PackagingError("cutover_unit_inputs_bootstrap_boundary_invalid")
    uid = 0 if require_root else effective_uid
    gid = 0 if require_root else effective_gid
    try:
        parent = os.lstat(unit_inputs_path.parent)
    except OSError as exc:
        raise PackagingError("cutover_unit_inputs_staging_directory_invalid") from exc
    if (
        stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or unit_inputs_path.parent.resolve(strict=True)
        != unit_inputs_path.parent
        or parent.st_uid != uid
        or parent.st_gid != gid
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise PackagingError("cutover_unit_inputs_staging_directory_invalid")

    plan_value = _decode_canonical_json(
        _read_trusted_staged_file(
            authority_plan_path,
            expected_uid=uid,
            expected_gid=gid,
            mode=0o400,
            maximum=8 * 1024 * 1024,
        ),
        newline=False,
        code="cutover_unit_input_plan_invalid",
    )
    approval_value = _decode_canonical_json(
        _read_trusted_staged_file(
            authority_approval_path,
            expected_uid=uid,
            expected_gid=gid,
            mode=0o400,
            maximum=1024 * 1024,
        ),
        newline=False,
        code="cutover_unit_input_approval_invalid",
    )
    try:
        plan = validate_unit_input_plan(plan_value)
        approval = validate_unit_input_approval(
            approval_value,
            plan=plan,
            now_unix=int(time.time()) if now_unix is None else now_unix,
        )
        unit_inputs = _unit_inputs_from_authority(plan, approval)
    except (PermissionError, TypeError, ValueError) as exc:
        raise PackagingError("cutover_unit_inputs_owner_authority_invalid") from exc
    payload = _canonical_bytes(unit_inputs) + b"\n"
    created = _create_or_validate_fixed_unit_inputs(
        unit_inputs_path,
        payload,
        uid=uid,
        gid=gid,
    )
    unsigned = {
        "schema": UNIT_INPUT_STAGING_SCHEMA,
        "path": str(unit_inputs_path),
        "sha256": _sha256(payload),
        "release_revision": unit_inputs["release_revision"],
        "authority_plan_sha256": unit_inputs["authority_plan_sha256"],
        "authority_approval_sha256": unit_inputs[
            "authority_approval_sha256"
        ],
        "created": created,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {
        **unsigned,
        "receipt_sha256": _sha256(_canonical_bytes(unsigned)),
    }


def _production_reconcile(source: str) -> str:
    canary_header = """-- This artifact is only for a disposable, isolated PostgreSQL 18 copy.  It
-- deliberately refuses the production database name and also requires nine
-- explicit, session-local expectations.  A caller must collect those values
-- from the exact frozen copy before executing this transaction:
"""
    production_header = """-- This rendered artifact is only for the exact owner-approved production
-- final-tail plan and requires nine explicit, session-local expectations.
-- The self-contained executable collects and validates them before execution:
"""
    refusal = """    IF pg_catalog.current_database() = 'ai_platform_brain' THEN
        RAISE EXCEPTION
            'legacy reconciliation refuses the production database name';
    END IF;
"""
    if source.count(refusal) != 1 or source.count(canary_header) != 1:
        raise PackagingError("cutover_reconcile_refusal_contract_changed")
    if source.count("isolated_canary_copy") != 5:
        raise PackagingError("cutover_reconcile_scope_contract_changed")
    if source.count("muncho_canary_brain") != 3:
        raise PackagingError("cutover_reconcile_database_contract_changed")
    rendered = source.replace(canary_header, production_header).replace(refusal, "")
    rendered = rendered.replace("isolated_canary_copy", "owner_approved_cutover")
    rendered = rendered.replace("muncho_canary_brain", "ai_platform_brain")
    if (
        "isolated_canary_copy" in rendered
        or "muncho_canary_brain" in rendered
        or "refuses the production database" in rendered
        or rendered.count("owner_approved_cutover") != 5
        or rendered.count("ai_platform_brain") != 3
    ):
        raise PackagingError("cutover_reconcile_render_invalid")
    banner = (
        "-- OWNER-APPROVED PRODUCTION RENDER. Generated only into an exact release.\n"
        "-- The signed plan supplies the frozen row/storage identities and target.\n"
    )
    return banner + rendered


def render_artifact(
    template: bytes,
    *,
    actions: tuple[str, ...],
    legacy_reconcile_sql: bytes,
    writer_migration_sql: bytes,
    connector_unit_template: bytes,
    gateway_connector_drop_in: bytes,
    prerequisite_contract: Mapping[str, Any],
    sealed_runtime_artifact_request: Mapping[str, Any],
) -> bytes:
    try:
        rendered = template.decode("utf-8", errors="strict")
        legacy = legacy_reconcile_sql.decode("utf-8", errors="strict")
        migration = writer_migration_sql.decode("utf-8", errors="strict")
        connector_unit = connector_unit_template.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PackagingError("cutover_packaging_source_encoding_invalid") from exc
    for sentinel in SENTINELS:
        if rendered.count(sentinel) != 1:
            raise PackagingError("cutover_packaging_template_contract_changed")
    production_sql = _production_reconcile(legacy)
    rendered = rendered.replace("__MUNCHO_ALLOWED_ACTIONS__", repr(tuple(actions)))
    rendered = rendered.replace("__MUNCHO_LEGACY_RECONCILE_SQL__", repr(production_sql))
    rendered = rendered.replace("__MUNCHO_WRITER_MIGRATION_SQL__", repr(migration))
    rendered = rendered.replace(
        "__MUNCHO_CONNECTOR_UNIT_TEMPLATE__", repr(connector_unit)
    )
    rendered = rendered.replace(
        "__MUNCHO_GATEWAY_CONNECTOR_DROP_IN_BYTES__",
        repr(gateway_connector_drop_in),
    )
    rendered = rendered.replace(
        "__MUNCHO_PRODUCTION_CAPABILITY_PREREQUISITE_CONTRACT__",
        repr(dict(prerequisite_contract)),
    )
    rendered = rendered.replace(
        "__MUNCHO_PRODUCTION_CRON_CONTINUITY_PLAN_SCHEMA__",
        repr(PRODUCTION_CRON_CONTINUITY_PLAN_SCHEMA),
    )
    rendered = rendered.replace(
        "__MUNCHO_SEALED_RUNTIME_ARTIFACT_REQUEST__",
        repr(dict(sealed_runtime_artifact_request)),
    )
    if any(sentinel in rendered for sentinel in SENTINELS):
        raise PackagingError("cutover_packaging_template_render_failed")
    payload = rendered.encode("utf-8", errors="strict")
    if not payload.startswith(b"#!/usr/bin/python3\n") or b"\x00" in payload:
        raise PackagingError("cutover_packaging_artifact_invalid")
    return payload


def _atomic_install(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PackagingError("cutover_packaging_write_failed")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _release_address(release: Path, revision: str, value: Path | None) -> Path:
    address = release if value is None else value
    if (
        not address.is_absolute()
        or ".." in address.parts
        or address.name != f"hermes-agent-{revision[:12]}"
    ):
        raise PackagingError("cutover_packaging_release_address_invalid")
    return address


def _runtime_dependency_manifest(release: Path, revision: str) -> tuple[bytes, Mapping[str, Any]]:
    raw = _read_source(release / RUNTIME_DEPENDENCY_MANIFEST, maximum=2 * 1024 * 1024)
    try:
        value = json.loads(raw.decode("ascii", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PackagingError("cutover_runtime_dependency_manifest_invalid") from exc
    if (
        not isinstance(value, Mapping)
        or raw != _canonical_bytes(value) + b"\n"
        or value.get("schema") != RUNTIME_DEPENDENCY_MANIFEST_SCHEMA
        or value.get("release_revision") != revision
        or value.get("secret_material_recorded") is not False
        or not isinstance(value.get("manifest_sha256"), str)
        or value["manifest_sha256"]
        != _sha256(
            _canonical_bytes(
                {key: item for key, item in value.items() if key != "manifest_sha256"}
            )
        )
    ):
        raise PackagingError("cutover_runtime_dependency_manifest_invalid")
    return raw, value


def _runtime_browser_kwargs(
    runtime_dependency: Mapping[str, Any],
) -> dict[str, str]:
    """Extract only the exact release-local identities used by the renderer."""

    try:
        agent_browser = _exact_mapping(
            runtime_dependency["agent_browser"],
            frozenset(
                {
                    "version",
                    "config_path",
                    "config_sha256",
                    "wrapper_path",
                    "wrapper_sha256",
                    "native_path",
                    "native_sha256",
                    "package_tree",
                    "node_path",
                    "node_version",
                    "node_sha256",
                    "npm_path",
                    "npm_version",
                    "npm_target_sha256",
                    "node_tree",
                }
            ),
            "cutover_runtime_dependency_manifest_invalid",
        )
        chrome = _exact_mapping(
            runtime_dependency["chrome"],
            frozenset({"version", "executable_path", "executable_sha256", "tree"}),
            "cutover_runtime_dependency_manifest_invalid",
        )
    except (KeyError, TypeError) as exc:
        raise PackagingError("cutover_runtime_dependency_manifest_invalid") from exc
    fields = {
        "browser_node_path": agent_browser["node_path"],
        "browser_node_sha256": agent_browser["node_sha256"],
        "browser_wrapper_path": agent_browser["wrapper_path"],
        "browser_wrapper_sha256": agent_browser["wrapper_sha256"],
        "browser_native_path": agent_browser["native_path"],
        "browser_native_sha256": agent_browser["native_sha256"],
        "browser_chrome_path": chrome["executable_path"],
        "browser_chrome_sha256": chrome["executable_sha256"],
        "agent_browser_config_path": agent_browser["config_path"],
        "agent_browser_config_sha256": agent_browser["config_sha256"],
    }
    if any(
        not isinstance(value, str) or not value
        for value in fields.values()
    ) or any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for key, value in fields.items()
        if key.endswith("sha256")
    ):
        raise PackagingError("cutover_runtime_dependency_manifest_invalid")
    return fields


def _operational_asset_receipt(
    *,
    release: Path,
    release_address: Path,
    revision: str,
    expected_uid: int,
    expected_gid: int,
    package_if_missing: bool,
) -> Mapping[str, Any]:
    """Verify exact helper bytes and bind their final release address.

    Auto-deploy builds under a temporary sibling and atomically renames that
    directory to ``release_address``. Physical reads therefore use ``release``
    while the sealed receipt records the immutable final address.
    """

    manifest_path = release / ASSET_MANIFEST_RELATIVE
    if package_if_missing and not os.path.lexists(manifest_path):
        try:
            package_operational_assets(
                release_root=release,
                revision=revision,
            )
        except (OperationalEdgeAssetError, OSError) as exc:
            raise PackagingError(
                "cutover_operational_assets_package_failed"
            ) from exc
    try:
        observed = verify_packaged_operational_assets(
            release_root=release,
            revision=revision,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            reported_release_root=release_address,
        )
    except (OperationalEdgeAssetError, OSError) as exc:
        raise PackagingError(
            "cutover_operational_assets_verification_failed"
        ) from exc

    try:
        return validate_packaged_operational_asset_verification(
            observed,
            revision=revision,
            expected_release_root=release_address,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
    except OperationalEdgeAssetError as exc:
        raise PackagingError(
            "cutover_operational_assets_verification_failed"
        ) from exc


def _sealed_runtime_artifact_request(
    *,
    revision: str,
    runtime_dependency: Mapping[str, Any],
    unit_inputs: Mapping[str, Any],
    operational_asset_verification: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Render and seal the complete no-secret operational host boundary."""

    inputs = _unit_inputs(unit_inputs, revision=revision)
    runtime = _runtime_browser_kwargs(runtime_dependency)
    gateway = inputs["gateway"]
    routeback = inputs["routeback"]
    mac_ops = inputs["mac_ops"]
    browser = inputs["browser"]
    worker = inputs["worker"]
    try:
        verified_assets = validate_packaged_operational_asset_verification(
            operational_asset_verification,
            revision=revision,
        )
        capability = render_production_capability_units(
            revision=revision,
            database_ip=inputs["database_ip"],
            gateway_user=gateway["user"],
            gateway_group=gateway["group"],
            gateway_uid=gateway["uid"],
            gateway_gid=gateway["gid"],
            routeback_user=routeback["user"],
            routeback_group=routeback["group"],
            routeback_uid=routeback["uid"],
            routeback_gid=routeback["gid"],
            mac_ops_user=mac_ops["user"],
            mac_ops_group=mac_ops["group"],
            mac_ops_uid=mac_ops["uid"],
            mac_ops_gid=mac_ops["gid"],
            browser_user=browser["user"],
            browser_group=browser["group"],
            browser_uid=browser["uid"],
            browser_gid=browser["gid"],
            socket_client_group=mac_ops["group"],
            **runtime,
        )
        isolated_worker = render_isolated_worker_units(
            revision=revision,
            gateway_uid=gateway["uid"],
            gateway_primary_gid=gateway["gid"],
            socket_root_uid=0,
            socket_client_group=inputs["worker_client_group"],
            socket_client_gid=inputs["worker_client_gid"],
            worker_user=worker["user"],
            worker_group=worker["group"],
            worker_uid=worker["uid"],
            worker_gid=worker["gid"],
            bwrap_sha256=inputs["bwrap_sha256"],
            shell_sha256=inputs["shell_sha256"],
        )
        operational_units = render_operational_edge_units(
            revision=revision,
            service_identities=inputs["operational_edge_identities"],
            socket_groups=inputs["operational_edge_socket_groups"],
            read_peer_uids=(gateway["uid"],),
            mutation_peer_uid=gateway["uid"],
            mutation_peer_gid=gateway["gid"],
            release_owner_uid=verified_assets["expected_uid"],
            release_owner_gid=verified_assets["expected_gid"],
            receipt_public_key_ids=inputs[
                "operational_edge_receipt_public_key_ids"
            ],
            writer_key_id=inputs["writer_capability_public_key_id"],
        )
    except (
        OperationalEdgeAssetError,
        OperationalEdgeUnitError,
        TypeError,
        ValueError,
    ) as exc:
        raise PackagingError("cutover_packaging_unit_render_invalid") from exc

    payloads = {
        "phase_b_unit": capability.phase_b_unit,
        "routeback_unit": capability.routeback_unit,
        "mac_ops_unit": capability.mac_ops_unit,
        "browser_unit": capability.browser_unit,
        "browser_config": capability.browser_config,
        "isolated_worker_socket_unit": isolated_worker.socket_unit,
        "isolated_worker_service_unit": isolated_worker.service_unit,
        "isolated_worker_config": isolated_worker.config,
        **{
            f"operational_edge_unit_{domain}": operational_units.units[
                operational_edge_service_unit(domain)
            ]
            for domain in sorted(CREDENTIALS_BY_DOMAIN)
        },
        **{
            f"operational_edge_config_{domain}": operational_units.configs[
                str(operational_edge_config_path(domain))
            ]
            for domain in sorted(CREDENTIALS_BY_DOMAIN)
        },
        "operational_edge_client_config": operational_units.client_config,
    }
    targets = {
        "phase_b_unit": f"/etc/systemd/system/{PHASE_B_UNIT}",
        "routeback_unit": f"/etc/systemd/system/{ROUTEBACK_EDGE_UNIT}",
        "mac_ops_unit": f"/etc/systemd/system/{MAC_OPS_UNIT}",
        "browser_unit": f"/etc/systemd/system/{BROWSER_UNIT}",
        "browser_config": str(BROWSER_CONFIG_PATH),
        "isolated_worker_socket_unit": (
            f"/etc/systemd/system/{ISOLATED_WORKER_SOCKET_UNIT}"
        ),
        "isolated_worker_service_unit": (
            f"/etc/systemd/system/{ISOLATED_WORKER_SERVICE_UNIT}"
        ),
        "isolated_worker_config": str(ISOLATED_WORKER_CONFIG),
        **{
            f"operational_edge_unit_{domain}": (
                f"/etc/systemd/system/{operational_edge_service_unit(domain)}"
            )
            for domain in sorted(CREDENTIALS_BY_DOMAIN)
        },
        **{
            f"operational_edge_config_{domain}": str(
                operational_edge_config_path(domain)
            )
            for domain in sorted(CREDENTIALS_BY_DOMAIN)
        },
        "operational_edge_client_config": str(
            OPERATIONAL_EDGE_CLIENT_CONFIG
        ),
    }
    gids = {
        "browser_config": browser["gid"],
        "isolated_worker_config": worker["gid"],
    }
    modes = {
        "browser_config": BROWSER_CONFIG_MODE,
        "isolated_worker_config": WORKER_CONFIG_MODE,
        **{
            f"operational_edge_config_{domain}": 0o400
            for domain in sorted(CREDENTIALS_BY_DOMAIN)
        },
        "operational_edge_client_config": 0o444,
    }
    files = {
        name: {
            "target_path": targets[name],
            "sha256": _sha256(payload),
            "uid": 0,
            "gid": gids.get(name, 0),
            "mode": modes.get(name, 0o644),
        }
        for name, payload in payloads.items()
    }
    worker_topology = {
        "socket_unit": ISOLATED_WORKER_SOCKET_UNIT,
        "socket_fragment_sha256": isolated_worker.socket_unit_sha256,
        "service_unit": ISOLATED_WORKER_SERVICE_UNIT,
        "service_fragment_sha256": isolated_worker.service_unit_sha256,
        "config_path": str(ISOLATED_WORKER_CONFIG),
        "config_sha256": isolated_worker.config_sha256,
        "socket_path": str(ISOLATED_WORKER_SOCKET),
        "socket_uid": 0,
        "socket_gid": inputs["worker_client_gid"],
        "server_uid": worker["uid"],
        "server_gid": worker["gid"],
        "gateway_uid": gateway["uid"],
        "gateway_gid": gateway["gid"],
        "bwrap_path": str(BWRAP_PATH),
        "bwrap_sha256": isolated_worker.bwrap_sha256,
        "shell_path": str(SHELL_PATH),
        "shell_sha256": isolated_worker.shell_sha256,
    }
    browser_topology = {
        "unit": BROWSER_UNIT,
        "fragment_sha256": capability.browser_sha256,
        "config_path": str(BROWSER_CONFIG_PATH),
        "config_sha256": capability.browser_config_sha256,
        "socket_path": str(BROWSER_SOCKET_PATH),
        "service_uid": browser["uid"],
        "service_gid": browser["gid"],
        "node_path": runtime["browser_node_path"],
        "node_sha256": runtime["browser_node_sha256"],
        "wrapper_path": runtime["browser_wrapper_path"],
        "wrapper_sha256": runtime["browser_wrapper_sha256"],
        "native_path": runtime["browser_native_path"],
        "native_sha256": runtime["browser_native_sha256"],
        "executable": runtime["browser_chrome_path"],
        "executable_sha256": runtime["browser_chrome_sha256"],
        "agent_browser_config_path": runtime["agent_browser_config_path"],
        "agent_browser_config_sha256": runtime[
            "agent_browser_config_sha256"
        ],
    }
    descriptor_unsigned = {
        "schema": SEALED_RUNTIME_ARTIFACT_REQUEST_SCHEMA,
        "release_revision": revision,
        "target": inputs["target"],
        "files": files,
        "isolated_worker_lease_mountpoint": {
            "target_path": str(ISOLATED_WORKER_LEASE_BASE),
            "uid": 0,
            "gid": 0,
            "mode": 0o700,
        },
        "topology_fragments": {
            "isolated_worker": worker_topology,
            "browser": browser_topology,
            "operational_edge": dict(operational_units.manifest),
        },
        "capability_bundle": dict(capability.manifest()),
        "isolated_worker_bundle": dict(isolated_worker.manifest()),
        "operational_edge_bundle": dict(operational_units.manifest),
        "operational_asset_verification": dict(verified_assets),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    descriptor = {
        **descriptor_unsigned,
        "request_sha256": _sha256(_canonical_bytes(descriptor_unsigned)),
    }
    request = {**descriptor, "payloads": payloads}
    return request, descriptor


def _host_artifact_contract(
    *,
    sealed_descriptor: Mapping[str, Any],
    gateway_connector_drop_in_sha256: str,
) -> Mapping[str, Any]:
    """Bind the complete host-input surface without recording secret bytes.

    Package-rendered payloads carry their final byte digest here.  Dynamic
    production outputs and root-only verifier files deliberately do not: their
    final digest is collected on the target host and becomes part of the
    owner-signed FreezePlan.  Every entry nevertheless has an exact target,
    fixed staging address, binding class, and mandatory readback gate.
    """

    sealed_files = sealed_descriptor.get("files")
    if not isinstance(sealed_files, Mapping):
        raise PackagingError("cutover_host_artifact_contract_invalid")
    staged_root = CUTOVER_STAGED_ROOT / "host"
    files: dict[str, Any] = {}
    for name, (target_path, binding_class) in HOST_ARTIFACT_TARGETS.items():
        package_sha256: str | None = None
        if binding_class == "release_sealed_payload":
            item = sealed_files.get(name)
            if (
                not isinstance(item, Mapping)
                or item.get("target_path") != target_path
                or not isinstance(item.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
            ):
                raise PackagingError("cutover_host_artifact_contract_invalid")
            package_sha256 = str(item["sha256"])
        elif binding_class == "release_reviewed_source":
            if name != "gateway_connector_drop_in" or re.fullmatch(
                r"[0-9a-f]{64}", gateway_connector_drop_in_sha256
            ) is None:
                raise PackagingError("cutover_host_artifact_contract_invalid")
            package_sha256 = gateway_connector_drop_in_sha256
        elif binding_class not in {"owner_runtime_rendered", "root_verifier"}:
            raise PackagingError("cutover_host_artifact_contract_invalid")
        files[name] = {
            "target_path": target_path,
            "staged_path": str(staged_root / Path(target_path).name),
            "binding_class": binding_class,
            "package_sha256": package_sha256,
            "actual_sha256_bound_by": (
                "muncho-production-cutover-host-authority.v1"
            ),
            "required_readback": True,
        }
    if (
        len({item["target_path"] for item in files.values()}) != len(files)
        or len({item["staged_path"] for item in files.values()}) != len(files)
    ):
        raise PackagingError("cutover_host_artifact_contract_invalid")
    unsigned = {
        "schema": HOST_ARTIFACT_CONTRACT_SCHEMA,
        "files": files,
        "required_file_count": len(HOST_ARTIFACT_TARGETS),
        "all_files_require_readback": True,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {
        **unsigned,
        "contract_sha256": _sha256(_canonical_bytes(unsigned)),
    }


def build_release_artifacts(
    release_root: Path,
    revision: str,
    *,
    release_address: Path | None = None,
    unit_inputs: Mapping[str, Any],
) -> Mapping[str, Any]:
    if REVISION.fullmatch(revision) is None:
        raise PackagingError("cutover_packaging_revision_invalid")
    try:
        release = release_root.resolve(strict=True)
    except OSError as exc:
        raise PackagingError("cutover_packaging_release_unavailable") from exc
    if release != release_root or not release.is_dir():
        raise PackagingError("cutover_packaging_release_invalid")
    address = _release_address(release, revision, release_address)
    marker = _read_source(release / ".codex-source-commit", maximum=128)
    if marker != (revision + "\n").encode("ascii"):
        raise PackagingError("cutover_packaging_release_identity_invalid")
    runtime_dependency_raw, runtime_dependency = _runtime_dependency_manifest(
        release, revision
    )

    template_path = release / "ops" / "muncho" / "cutover" / "production_cutover_artifact_runtime.py.in"
    reconcile_path = release / "scripts" / "sql" / "canonical_writer_legacy_reconcile_v1.sql"
    migration_path = release / "scripts" / "sql" / "canonical_writer_v1.sql"
    connector_unit_path = release / "ops/muncho/systemd/muncho-discord-connector.service.in"
    connector_drop_in_path = release / "ops/muncho/systemd/hermes-cloud-gateway.discord-connector.conf"
    connector_config_path = release / "ops/muncho/systemd/discord-public-connector.json.in"
    template = _read_source(template_path, maximum=2 * 1024 * 1024)
    reconcile = _read_source(reconcile_path, maximum=4 * 1024 * 1024)
    migration = _read_source(migration_path, maximum=8 * 1024 * 1024)
    connector_unit = _read_source(connector_unit_path, maximum=1024 * 1024)
    connector_drop_in = _read_source(connector_drop_in_path, maximum=1024 * 1024)
    connector_config = _read_source(connector_config_path, maximum=1024 * 1024)
    output_root = release / "ops" / "muncho" / "cutover" / "artifacts"
    prerequisite_contract = packaged_prerequisite_contract()
    normalized_unit_inputs = _unit_inputs(unit_inputs, revision=revision)
    operational_assets = _operational_asset_receipt(
        release=release,
        release_address=address,
        revision=revision,
        expected_uid=normalized_unit_inputs["release_owner_uid"],
        expected_gid=normalized_unit_inputs["release_owner_gid"],
        package_if_missing=True,
    )
    sealed_request, sealed_descriptor = _sealed_runtime_artifact_request(
        revision=revision,
        runtime_dependency=runtime_dependency,
        unit_inputs=normalized_unit_inputs,
        operational_asset_verification=operational_assets,
    )
    host_artifact_contract = _host_artifact_contract(
        sealed_descriptor=sealed_descriptor,
        gateway_connector_drop_in_sha256=_sha256(connector_drop_in),
    )

    manifest_artifacts: dict[str, Any] = {}
    for name, actions in ARTIFACTS.items():
        payload = render_artifact(
            template,
            actions=actions,
            legacy_reconcile_sql=reconcile,
            writer_migration_sql=migration,
            connector_unit_template=connector_unit,
            gateway_connector_drop_in=connector_drop_in,
            prerequisite_contract=prerequisite_contract,
            sealed_runtime_artifact_request=sealed_request,
        )
        path = output_root / name
        _atomic_install(path, payload, mode=0o500)
        manifest_artifacts[name] = {
            "path": str(address / "ops" / "muncho" / "cutover" / "artifacts" / name),
            "actions": list(actions),
            "sha256": _sha256(payload),
            "size": len(payload),
        }

    unsigned = {
        "schema": MANIFEST_SCHEMA,
        "release_revision": revision,
        "source": {
            "template_sha256": _sha256(template),
            "legacy_reconcile_sha256": _sha256(reconcile),
            "writer_migration_sha256": _sha256(migration),
            "connector_unit_template_sha256": _sha256(connector_unit),
            "gateway_connector_drop_in_sha256": _sha256(connector_drop_in),
            "connector_config_template_sha256": _sha256(connector_config),
            "production_capability_prerequisite_contract_sha256": _sha256(
                _canonical_bytes(prerequisite_contract)
            ),
            "runtime_dependency_manifest_sha256": _sha256(runtime_dependency_raw),
            "runtime_dependency_identity_sha256": runtime_dependency[
                "manifest_sha256"
            ],
            "sealed_runtime_artifact_request_sha256": sealed_descriptor[
                "request_sha256"
            ],
            "operational_asset_manifest_sha256": operational_assets[
                "manifest_sha256"
            ],
            "operational_asset_verification_sha256": operational_assets[
                "verification_sha256"
            ],
        },
        "unit_inputs": normalized_unit_inputs,
        "sealed_runtime_artifact_request": sealed_descriptor,
        "host_artifact_contract": host_artifact_contract,
        "artifacts": manifest_artifacts,
        "plan_bindings": {
            binding: {
                "path": manifest_artifacts[name]["path"],
                "sha256": manifest_artifacts[name]["sha256"],
            }
            for binding, name in PLAN_BINDINGS.items()
        },
        "secret_material_recorded": False,
    }
    manifest = {**unsigned, "manifest_sha256": _sha256(_canonical_bytes(unsigned))}
    _atomic_install(output_root / "manifest.json", _canonical_bytes(manifest) + b"\n", mode=0o444)
    return manifest


def verify_release_artifacts(
    release_root: Path,
    revision: str,
    *,
    release_address: Path | None = None,
    unit_inputs: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        release = release_root.resolve(strict=True)
        supplied = release_root.lstat()
    except OSError as exc:
        raise PackagingError("cutover_packaging_release_unavailable") from exc
    if (
        release != release_root
        or stat.S_ISLNK(supplied.st_mode)
        or not stat.S_ISDIR(supplied.st_mode)
    ):
        raise PackagingError("cutover_packaging_release_invalid")
    address = _release_address(release, revision, release_address)
    marker = _read_source(release / ".codex-source-commit", maximum=128)
    if marker != (revision + "\n").encode("ascii"):
        raise PackagingError("cutover_packaging_release_identity_invalid")
    runtime_dependency_raw, runtime_dependency = _runtime_dependency_manifest(
        release, revision
    )
    template = _read_source(
        release / "ops/muncho/cutover/production_cutover_artifact_runtime.py.in",
        maximum=2 * 1024 * 1024,
    )
    reconcile = _read_source(
        release / "scripts/sql/canonical_writer_legacy_reconcile_v1.sql",
        maximum=4 * 1024 * 1024,
    )
    migration = _read_source(
        release / "scripts/sql/canonical_writer_v1.sql",
        maximum=8 * 1024 * 1024,
    )
    connector_unit = _read_source(
        release / "ops/muncho/systemd/muncho-discord-connector.service.in",
        maximum=1024 * 1024,
    )
    connector_drop_in = _read_source(
        release / "ops/muncho/systemd/hermes-cloud-gateway.discord-connector.conf",
        maximum=1024 * 1024,
    )
    connector_config = _read_source(
        release / "ops/muncho/systemd/discord-public-connector.json.in",
        maximum=1024 * 1024,
    )
    prerequisite_contract = packaged_prerequisite_contract()
    manifest_path = release / "ops" / "muncho" / "cutover" / "artifacts" / "manifest.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PackagingError("cutover_packaging_manifest_invalid") from exc
    if not isinstance(manifest, Mapping) or raw != _canonical_bytes(manifest) + b"\n":
        raise PackagingError("cutover_packaging_manifest_invalid")
    if (
        set(manifest) != {
            "schema", "release_revision", "source", "artifacts",
            "plan_bindings", "unit_inputs", "sealed_runtime_artifact_request",
            "host_artifact_contract",
            "secret_material_recorded", "manifest_sha256",
        }
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("release_revision") != revision
        or manifest.get("secret_material_recorded") is not False
        or set(manifest.get("artifacts", {})) != set(ARTIFACTS)
    ):
        raise PackagingError("cutover_packaging_manifest_invalid")
    unsigned = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _sha256(_canonical_bytes(unsigned)):
        raise PackagingError("cutover_packaging_manifest_invalid")
    normalized_unit_inputs = _unit_inputs(unit_inputs, revision=revision)
    if manifest.get("unit_inputs") != normalized_unit_inputs:
        raise PackagingError("cutover_packaging_manifest_invalid")
    operational_assets = _operational_asset_receipt(
        release=release,
        release_address=address,
        revision=revision,
        expected_uid=normalized_unit_inputs["release_owner_uid"],
        expected_gid=normalized_unit_inputs["release_owner_gid"],
        package_if_missing=False,
    )
    sealed_request, sealed_descriptor = _sealed_runtime_artifact_request(
        revision=revision,
        runtime_dependency=runtime_dependency,
        unit_inputs=normalized_unit_inputs,
        operational_asset_verification=operational_assets,
    )
    if manifest.get("sealed_runtime_artifact_request") != sealed_descriptor:
        raise PackagingError("cutover_packaging_manifest_invalid")
    expected_host_contract = _host_artifact_contract(
        sealed_descriptor=sealed_descriptor,
        gateway_connector_drop_in_sha256=_sha256(connector_drop_in),
    )
    if manifest.get("host_artifact_contract") != expected_host_contract:
        raise PackagingError("cutover_packaging_manifest_invalid")
    if manifest.get("source") != {
        "template_sha256": _sha256(template),
        "legacy_reconcile_sha256": _sha256(reconcile),
        "writer_migration_sha256": _sha256(migration),
        "connector_unit_template_sha256": _sha256(connector_unit),
        "gateway_connector_drop_in_sha256": _sha256(connector_drop_in),
        "connector_config_template_sha256": _sha256(connector_config),
            "production_capability_prerequisite_contract_sha256": _sha256(
                _canonical_bytes(prerequisite_contract)
            ),
            "runtime_dependency_manifest_sha256": _sha256(runtime_dependency_raw),
            "runtime_dependency_identity_sha256": runtime_dependency[
                "manifest_sha256"
            ],
            "sealed_runtime_artifact_request_sha256": sealed_descriptor[
                "request_sha256"
            ],
            "operational_asset_manifest_sha256": operational_assets[
                "manifest_sha256"
            ],
            "operational_asset_verification_sha256": operational_assets[
                "verification_sha256"
            ],
    }:
        raise PackagingError("cutover_packaging_manifest_invalid")
    for name, actions in ARTIFACTS.items():
        item = manifest["artifacts"][name]
        if not isinstance(item, Mapping) or set(item) != {"path", "actions", "sha256", "size"}:
            raise PackagingError("cutover_packaging_manifest_invalid")
        path = release / "ops" / "muncho" / "cutover" / "artifacts" / name
        expected = address / "ops" / "muncho" / "cutover" / "artifacts" / name
        payload = _read_source(path, maximum=16 * 1024 * 1024)
        expected_payload = render_artifact(
            template,
            actions=actions,
            legacy_reconcile_sql=reconcile,
            writer_migration_sql=migration,
            connector_unit_template=connector_unit,
            gateway_connector_drop_in=connector_drop_in,
            prerequisite_contract=prerequisite_contract,
            sealed_runtime_artifact_request=sealed_request,
        )
        if (
            item["path"] != str(expected)
            or item["actions"] != list(actions)
            or item["sha256"] != _sha256(payload)
            or item["size"] != len(payload)
            or stat.S_IMODE(path.stat().st_mode) != 0o500
            or payload != expected_payload
        ):
            raise PackagingError("cutover_packaging_artifact_drifted")
    bindings = manifest.get("plan_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(PLAN_BINDINGS):
        raise PackagingError("cutover_packaging_manifest_invalid")
    for binding, name in PLAN_BINDINGS.items():
        item = bindings[binding]
        artifact = manifest["artifacts"][name]
        if item != {"path": artifact["path"], "sha256": artifact["sha256"]}:
            raise PackagingError("cutover_packaging_manifest_invalid")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package exact production cutover artifacts")
    parser.add_argument(
        "command",
        choices=("bootstrap-unit-inputs", "build", "verify"),
    )
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--release-address", type=Path)
    parser.add_argument("--revision")
    parser.add_argument(
        "--unit-inputs",
        type=Path,
        help=(
            "fixed root-owned non-secret unit identity artifact "
            "(build and verify only)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "bootstrap-unit-inputs":
            if any(
                value is not None
                for value in (
                    args.release_root,
                    args.release_address,
                    args.revision,
                    args.unit_inputs,
                )
            ):
                raise PackagingError("cutover_unit_inputs_bootstrap_argv_invalid")
            result = bootstrap_fixed_unit_inputs()
            print(_canonical_bytes(result).decode("utf-8"))
            return 0
        if (
            args.release_root is None
            or args.revision is None
            or args.unit_inputs != FIXED_UNIT_INPUTS_PATH
        ):
            raise PackagingError("cutover_packaging_unit_inputs_path_invalid")
        unit_inputs = load_fixed_unit_inputs(args.unit_inputs)
        result = (
            build_release_artifacts(
                args.release_root,
                args.revision,
                release_address=args.release_address,
                unit_inputs=unit_inputs,
            )
            if args.command == "build"
            else verify_release_artifacts(
                args.release_root,
                args.revision,
                release_address=args.release_address,
                unit_inputs=unit_inputs,
            )
        )
    except (OSError, PackagingError):
        print('{"error_code":"production_cutover_packaging_failed","ok":false}')
        return 2
    print(_canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
