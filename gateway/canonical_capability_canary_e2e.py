"""Offline verifier for the production-shaped Muncho capability canary.

The live canary is deliberately implemented elsewhere.  This module opens no
socket, invokes no model, reads no database, and starts no service.  It accepts
only an exact reviewed fixture plus cryptographically signed, digest-bound
receipts produced by the live edges.  It then checks mechanical relationships
between those receipts.

There is no task-text inspection, semantic classifier, router, dispatcher, or
effort chooser here.  GPT/Hermes remains responsible for the plan, tool use,
fallback choices, and terminal answer.  This verifier checks only schemas,
identities, signatures, exact configuration, state, idempotency, and receipts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from gateway.canonical_capability_canary_producers import (
    ENDPOINT_ROLES,
    NATIVE_EVIDENCE_SCHEMA,
    PRODUCER_SERVICE_UNITS,
    SLOT_NATIVE_BINDING_KINDS,
    SLOT_ROLE,
)
from gateway.support_ops_team_registry import (
    SKYVISION_LOCKED_NONPUBLIC_CHANNEL_IDS,
)


FIXTURE_SCHEMA = "muncho-production-capability-canary-fixture.v1"
EVIDENCE_SCHEMA = "muncho-production-capability-canary-evidence.v1"
SIGNED_RECEIPT_SCHEMA = "muncho-production-capability-canary-signed-receipt.v1"
VERIFICATION_SCHEMA = "muncho-production-capability-canary-verification.v1"

RUNTIME_RECEIPT_SCHEMA = "muncho-production-capability-runtime-receipt.v1"
TASK_WORKSPACE_GATEWAY_SCHEMA = (
    "muncho-production-capability-canonical-task-workspace-gateway.v3"
)
TASK_WORKSPACE_WRITER_SCHEMA = (
    "muncho-production-capability-canonical-task-workspace-writer.v3"
)
PLAN_APPROVAL_SCHEMA = "muncho-production-capability-plan-approval.v2"
CAPABILITY_DENIAL_SCHEMA = "muncho-production-capability-denials.v1"
DATABASE_RECONCILIATION_SCHEMA = (
    "muncho-production-capability-database-reconciliation.v1"
)
BITRIX_EDGE_SCHEMA = "muncho-production-capability-bitrix-edge.v1"
BITRIX_WRITER_SCHEMA = "muncho-production-capability-bitrix-writer.v1"
DISCORD_EDGE_SCHEMA = "muncho-production-capability-discord-edge.v1"
DISCORD_WRITER_SCHEMA = "muncho-production-capability-discord-writer.v1"
FAILURE_GATEWAY_SCHEMA = "muncho-production-capability-failure-gateway.v1"
FAILURE_WRITER_SCHEMA = "muncho-production-capability-failure-writer.v1"
CLEANUP_RECEIPT_SCHEMA = "muncho-production-capability-cleanup.v1"
CLEANUP_FINALIZATION_SCHEMA = (
    "muncho-production-capability-cleanup-finalization.v1"
)
OBSERVER_STOP_RECEIPT_SCHEMA = (
    "muncho-production-capability-observer-stop-receipt.v1"
)
CREDENTIAL_CONSUMER_STOP_PROOF_SCHEMA = (
    "muncho-production-capability-credential-consumer-stop-proof.v1"
)
CAPABILITY_SERVICE_STOP_PROOF_SCHEMA = (
    "muncho-production-capability-service-stop-proof.v1"
)
PRODUCER_FLEET_RETIREMENT_SCHEMA = (
    "muncho-production-capability-fleet-retirement.v1"
)

MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_RECEIPT_LIST = 64
MAX_COMMANDS = 64
MAX_STEPS = 64
MAX_WINDOW_MS = 24 * 60 * 60 * 1000

AUTHORITY_ROLES = (
    "business_edge",
    "canonical_writer",
    "discord_edge",
    "gateway_observer",
    "owner",
)
AUTHORITY_ALGORITHMS = {
    "business_edge": "ed25519",
    "canonical_writer": "ed25519",
    "discord_edge": "ed25519",
    "gateway_observer": "ed25519",
    "owner": "sshsig-ed25519-sha512",
}
OWNER_SSHSIG_NAMESPACE = "muncho-production-capability-canary-owner-v1"
REQUIRED_TOOLSETS = (
    "browser",
    "canonical_brain",
    "clarify",
    "delegation",
    "file",
    "mac_ops",
    "memory",
    "session_search",
    "skills",
    "terminal",
    "todo",
    "web",
)
DENIAL_KINDS = (
    "unapproved_command",
    "expired_capability",
    "changed_command_bytes",
    "wrong_owner",
    "wrong_session_epoch",
    "stale_plan_revision",
)
FAILURE_COMPONENTS = ("tool", "browser", "database", "writer", "egress")
# Cleanup evidence is an exact projection of the packaged phased stop order.
# The observer producer is deliberately last: its signed cleanup receipt can
# attest every credential consumer stopped while remaining alive and
# credential-blind, then the root finalization proves that signer also stopped.
OBSERVER_PRODUCER_SERVICE_UNIT = PRODUCER_SERVICE_UNITS["gateway_observer"]
NON_OBSERVER_SERVICE_UNITS = (
    "hermes-cloud-gateway.service",
    *(
        PRODUCER_SERVICE_UNITS[role]
        for role in reversed(ENDPOINT_ROLES)
        if role != "gateway_observer"
    ),
    "muncho-operational-edge-bitrix.service",
    "muncho-canonical-writer.service",
    "muncho-capability-browser.service",
    "muncho-isolated-worker.service",
    "muncho-isolated-worker.socket",
    "muncho-mac-ops-edge.service",
    "muncho-discord-connector.service",
    "muncho-discord-egress.service",
    "muncho-canonical-writer-phase-b-readiness.service",
)
SERVICE_UNITS = (*NON_OBSERVER_SERVICE_UNITS, OBSERVER_PRODUCER_SERVICE_UNIT)
PRODUCER_ACTIVATION_PATH = (
    "/run/muncho-capability-canary/producer-activation.json"
)
CREDENTIAL_LEASES = (
    "api_control",
    "bitrix_operational_edge_webhook",
    "discord_canonical_routeback_bot_token",
    "discord_public_session_bot_token",
    "mac_ops_gitlab",
    "openai_codex",
)
CREDENTIAL_LEASE_KINDS = {
    "api_control": "api_server_control_key",
    "bitrix_operational_edge_webhook": "bitrix_operational_edge_webhook",
    "discord_canonical_routeback_bot_token": "discord_routeback_token",
    "discord_public_session_bot_token": "discord_connector_token",
    "mac_ops_gitlab": "mac_ops_gitlab_env",
    "openai_codex": "codex_access_token",
}
SECRET_RETIREMENT_COMPLETION_SCHEMA = (
    "muncho-production-capability-secret-retirement-completion.v2"
)
INTERNAL_KEY_RETIREMENT_SCHEMA = (
    "muncho-production-capability-internal-key-retirement.v1"
)
BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT = "muncho-operational-edge-bitrix.service"
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
BITRIX_OPERATIONAL_EDGE_CONFIG_PATH = "/etc/muncho/operational-edge/bitrix.json"
BITRIX_OPERATIONAL_EDGE_TRUST_PATH = (
    "/etc/muncho/operational-edge/trust/bitrix-receipt-public.pem"
)
BITRIX_WEBHOOK_SOURCE_PATH = (
    "/opt/adventico-ai-platform/hermes-home/secrets/"
    "bitrix_skyvision_crm_webhook.url"
)
BITRIX_WEBHOOK_PROJECTION_PATH = (
    "/run/credentials/muncho-operational-edge-bitrix.service/"
    "bitrix-webhook-url"
)
BITRIX_RECEIPT_PRIVATE_KEY_PATH = (
    "/etc/muncho/keys/operational-edge-bitrix-receipt-private.pem"
)
BITRIX_RECEIPT_PRIVATE_KEY_PROJECTION_PATH = (
    "/run/credentials/muncho-operational-edge-bitrix.service/"
    "receipt-private-key"
)
WRITER_PUBLIC_KEY_PATH = "/etc/muncho/keys/writer-capability-public.pem"
WRITER_PUBLIC_KEY_PROJECTION_PATH = (
    "/run/credentials/muncho-operational-edge-bitrix.service/"
    "writer-public-key"
)
DISCORD_CREDENTIAL_TOPOLOGY = {
    "connector_service_unit": "muncho-discord-connector.service",
    "connector_credential_lease": "discord_public_session_bot_token",
    "connector_operation_class": "ordinary_public_ingress_and_session_replies",
    "connector_discord_transport": "gateway_websocket",
    "routeback_service_unit": "muncho-discord-egress.service",
    "routeback_credential_lease": "discord_canonical_routeback_bot_token",
    "routeback_operation_class": "canonical_signed_route_back",
    "routeback_discord_transport": "rest_only",
    "gateway_service_unit": "hermes-cloud-gateway.service",
    "gateway_discord_credential_lease": None,
    "gateway_discord_token_absent": True,
    "gateway_direct_discord_adapter_enabled": False,
}
CONNECTOR_BOT_ID_PROVENANCE = "discord_gateway_ready_user_id"
ROUTEBACK_BOT_ID_PROVENANCE = "discord_rest_current_user_readback"
PRODUCTION_DISCORD_GUILD_ID = "1282725267068157972"
PRODUCTION_DISCORD_CANARY_CHANNEL_ID = "1526858760100909066"
PRODUCTION_DISCORD_BOT_USER_ID = "1501976597455044801"
PRODUCTION_DISCORD_CONNECTOR_BOT_USER_ID = "1526849374007853086"
PRODUCTION_DISCORD_ROUTEBACK_BOT_USER_ID = "1526850127921283222"
LOCKED_PRIVATE_DISCORD_CHANNEL_IDS = (
    SKYVISION_LOCKED_NONPUBLIC_CHANNEL_IDS
)

INVARIANTS = (
    "reviewed_release_and_runtime_bound",
    "fixed_production_tool_surface_without_kanban",
    "workspace_continued_after_blocker_and_restart",
    "model_authored_high_to_max",
    "one_owner_plan_capability_without_microapprovals",
    "capability_denial_matrix_complete",
    "database_write_reconciled_and_read_back",
    "bitrix_edge_read_and_unapproved_mutation_blocked",
    "public_discord_routeback_sent_only_after_verified_readback",
    "discord_dm_denied_before_dispatch",
    "discord_token_leases_disjoint_and_gateway_credential_free",
    "discord_canary_bot_identities_separate_from_each_other_and_production",
    "failure_matrix_retained_model_control",
    "canonical_task_workspaces_terminal",
    "services_stopped_and_credentials_retired",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")
_PUBLIC_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
_SNOWFLAKE_RE = re.compile(r"^[0-9]{1,25}$")
_SSHSIG_BEGIN = "-----BEGIN SSH SIGNATURE-----"
_SSHSIG_END = "-----END SSH SIGNATURE-----"
_MAX_SSHSIG_BYTES = 4096


class CapabilityCanaryEvidenceError(ValueError):
    """A stable non-secret validation failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise CapabilityCanaryEvidenceError(code)


def _strict(
    value: Any,
    *,
    fields: Sequence[str],
    code: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(code)
    result = dict(value)
    if set(result) != set(fields):
        _fail(code)
    return result


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail("non_canonical_json_value")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(code)
    return value


def _git_sha(value: Any, code: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        _fail(code)
    return value


def _safe_id(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        _fail(code)
    return value


def _snowflake(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SNOWFLAKE_RE.fullmatch(value) is None:
        _fail(code)
    return value


def _positive_int(value: Any, code: str) -> int:
    if type(value) is not int or not 0 < value < 1 << 63:
        _fail(code)
    return value


def _nonnegative_int(value: Any, code: str) -> int:
    if type(value) is not int or not 0 <= value < 1 << 63:
        _fail(code)
    return value


def _bool(value: Any, code: str) -> bool:
    if type(value) is not bool:
        _fail(code)
    return value


def _bounded_digest_list(
    value: Any,
    *,
    code: str,
    minimum: int = 1,
    maximum: int = MAX_RECEIPT_LIST,
    sorted_unique: bool = False,
) -> list[str]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        _fail(code)
    result = [_sha256(item, code) for item in value]
    if len(set(result)) != len(result):
        _fail(code)
    if sorted_unique and result != sorted(result):
        _fail(code)
    return result


def _bounded_id_list(
    value: Any,
    *,
    code: str,
    minimum: int = 1,
    maximum: int = MAX_STEPS,
) -> list[str]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        _fail(code)
    result = [_safe_id(item, code) for item in value]
    if len(set(result)) != len(result):
        _fail(code)
    return result


def _strict_json(raw: bytes, code: str) -> dict[str, Any]:
    def reject_constant(_value: str) -> None:
        raise ValueError("non-json numeric constant")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _fail(code)
    if not isinstance(value, dict):
        _fail(code)
    return value


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


def _read_bound_artifact(path: Path, expected_sha256: str, code: str) -> dict[str, Any]:
    expected = _sha256(expected_sha256, code)
    raw_path = os.fspath(path)
    if not path.is_absolute() or os.path.normpath(raw_path) != raw_path:
        _fail(code)
    try:
        before = path.lstat()
    except OSError:
        _fail(code)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_mode & 0o022
        or not 1 < before.st_size <= MAX_ARTIFACT_BYTES
    ):
        _fail(code)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            remaining = MAX_ARTIFACT_BYTES + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            body = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        reachable = path.lstat()
    except OSError:
        _fail(code)
    if (
        len(body) != before.st_size
        or len(body) > MAX_ARTIFACT_BYTES
        or _identity(before) != _identity(opened)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(reachable)
        or hashlib.sha256(body).hexdigest() != expected
    ):
        _fail(code)
    return _strict_json(body, code)


def _validate_fixture(value: Mapping[str, Any], fixture_sha256: str) -> dict[str, Any]:
    fixture = _strict(
        value,
        fields=(
            "schema",
            "release_sha",
            "release_root",
            "release_artifact_sha256",
            "capability_plan_sha256",
            "full_canary_plan_sha256",
            "installed_wheel_manifest_sha256",
            "effective_config_sha256",
            "tool_inventory_sha256",
            "run_id",
            "owner_id",
            "host_identity_sha256",
            "business_edge_service_identity_sha256",
            "bitrix_operational_edge_contract",
            "valid_from_unix_ms",
            "valid_until_unix_ms",
            "producer_foundation_sha256",
            "model_route",
            "required_toolsets",
            "public_discord_target",
            "discord_bot_identities",
            "authority_keys",
        ),
        code="fixture_invalid",
    )
    if fixture["schema"] != FIXTURE_SCHEMA:
        _fail("fixture_invalid")
    _git_sha(fixture["release_sha"], "fixture_invalid")
    if fixture["release_root"] != (
        f"/opt/muncho-canary-releases/{fixture['release_sha']}"
    ):
        _fail("fixture_invalid")
    for field in (
        "release_artifact_sha256",
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "installed_wheel_manifest_sha256",
        "effective_config_sha256",
        "tool_inventory_sha256",
        "host_identity_sha256",
        "business_edge_service_identity_sha256",
        "producer_foundation_sha256",
    ):
        _sha256(fixture[field], "fixture_invalid")
    bitrix_contract = _strict(
        fixture["bitrix_operational_edge_contract"],
        fields=(
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
        code="fixture_invalid",
    )
    if (
        bitrix_contract["revision"] != fixture["release_sha"]
        or bitrix_contract["service_unit"]
        != BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT
        or bitrix_contract["service_identity_sha256"]
        != fixture["business_edge_service_identity_sha256"]
        or bitrix_contract["asset_names"]
        != list(BITRIX_OPERATIONAL_EDGE_ASSET_NAMES)
        or bitrix_contract["asset_manifest_path"]
        != (
            f"{fixture['release_root']}/ops/muncho/runtime/"
            "operational-assets/manifest.json"
        )
        or bitrix_contract["rendered_unit_path"]
        != BITRIX_OPERATIONAL_EDGE_UNIT_PATH
        or bitrix_contract["rendered_config_path"]
        != BITRIX_OPERATIONAL_EDGE_CONFIG_PATH
        or bitrix_contract["rendered_trust_path"]
        != BITRIX_OPERATIONAL_EDGE_TRUST_PATH
        or bitrix_contract["credential_binding"]
        != "bitrix_operational_edge_webhook"
        or bitrix_contract["staging_protocol"]
        != BITRIX_OPERATIONAL_EDGE_STAGING_PROTOCOL
        or bitrix_contract["secret_material_recorded"] is not False
        or bitrix_contract["secret_digest_recorded"] is not False
    ):
        _fail("fixture_invalid")
    identity = _strict(
        bitrix_contract["identity_bootstrap"],
        fields=(
            "service_user",
            "service_group",
            "service_uid",
            "service_gid",
            "socket_client_group",
            "socket_client_gid",
            "receipt_sha256",
        ),
        code="fixture_invalid",
    )
    if (
        identity["service_user"] != BITRIX_OPERATIONAL_EDGE_SERVICE_USER
        or identity["service_group"] != BITRIX_OPERATIONAL_EDGE_SERVICE_GROUP
        or identity["socket_client_group"]
        != BITRIX_OPERATIONAL_EDGE_SOCKET_GROUP
        or any(
            type(identity[field]) is not int or identity[field] <= 0
            for field in ("service_uid", "service_gid", "socket_client_gid")
        )
        or identity["service_gid"] == identity["socket_client_gid"]
    ):
        _fail("fixture_invalid")
    _sha256(identity["receipt_sha256"], "fixture_invalid")
    projection = _strict(
        bitrix_contract["credential_projection"],
        fields=(
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
        code="fixture_invalid",
    )
    if projection != {
        "name": "bitrix-webhook-url",
        "source_path": BITRIX_WEBHOOK_SOURCE_PATH,
        "projected_path": BITRIX_WEBHOOK_PROJECTION_PATH,
        "bind_target_path": BITRIX_WEBHOOK_SOURCE_PATH,
        "source_owner_uid": 0,
        "source_owner_gid": 0,
        "source_mode": "0400",
        "service_reads_projection": True,
        "original_source_inaccessible": True,
        "value_or_digest_recorded": False,
    }:
        _fail("fixture_invalid")
    receipt_key = _strict(
        bitrix_contract["receipt_key_contract"],
        fields=(
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
        code="fixture_invalid",
    )
    if receipt_key != {
        "private_credential_name": "receipt-private-key",
        "private_source_path": BITRIX_RECEIPT_PRIVATE_KEY_PATH,
        "private_projection_path": (
            BITRIX_RECEIPT_PRIVATE_KEY_PROJECTION_PATH
        ),
        "private_owner_uid": 0,
        "private_owner_gid": 0,
        "private_mode": "0400",
        "public_path": BITRIX_OPERATIONAL_EDGE_TRUST_PATH,
        "public_key_id": receipt_key["public_key_id"],
        "public_trust_sha256": bitrix_contract["rendered_trust_sha256"],
        "writer_public_key_credential_name": "writer-public-key",
        "writer_public_key_source_path": WRITER_PUBLIC_KEY_PATH,
        "writer_public_key_projection_path": WRITER_PUBLIC_KEY_PROJECTION_PATH,
        "key_bootstrap_receipt_sha256": receipt_key[
            "key_bootstrap_receipt_sha256"
        ],
        "create_only": True,
        "retire_private_on_stop": True,
        "retire_public_on_stop": True,
        "private_content_or_digest_recorded": False,
    }:
        _fail("fixture_invalid")
    _sha256(receipt_key["public_key_id"], "fixture_invalid")
    _sha256(receipt_key["key_bootstrap_receipt_sha256"], "fixture_invalid")
    if bitrix_contract["expected_active_service_state"] != {
        "load_state": "loaded",
        "active_state": "active",
        "sub_state": "running",
        "unit_file_state": "disabled",
    } or bitrix_contract["expected_cleanup_service_state"] != {
        "active_state": "inactive",
        "sub_state": "dead",
        "overlay_retired_or_prior_restored": True,
    }:
        _fail("fixture_invalid")
    for field in (
        "service_identity_sha256",
        "asset_manifest_sha256",
        "rendered_unit_sha256",
        "rendered_config_sha256",
        "rendered_trust_sha256",
    ):
        _sha256(bitrix_contract[field], "fixture_invalid")
    _safe_id(fixture["run_id"], "fixture_invalid")
    _snowflake(fixture["owner_id"], "fixture_invalid")
    valid_from = _positive_int(fixture["valid_from_unix_ms"], "fixture_invalid")
    valid_until = _positive_int(fixture["valid_until_unix_ms"], "fixture_invalid")
    if not valid_from < valid_until <= valid_from + MAX_WINDOW_MS:
        _fail("fixture_invalid")

    route = _strict(
        fixture["model_route"],
        fields=(
            "provider",
            "api_mode",
            "model",
            "initial_effort",
            "adaptive_max_effort",
            "max_turns",
        ),
        code="fixture_invalid",
    )
    if route != {
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "model": "gpt-5.6-sol",
        "initial_effort": "high",
        "adaptive_max_effort": "max",
        "max_turns": 90,
    }:
        _fail("fixture_invalid")
    if fixture["required_toolsets"] != list(REQUIRED_TOOLSETS):
        _fail("fixture_invalid")

    target = _strict(
        fixture["public_discord_target"],
        fields=("target_type", "guild_id", "channel_id"),
        code="fixture_invalid",
    )
    if target != {
        "target_type": "public_channel",
        "guild_id": PRODUCTION_DISCORD_GUILD_ID,
        "channel_id": PRODUCTION_DISCORD_CANARY_CHANNEL_ID,
    } or target["channel_id"] in LOCKED_PRIVATE_DISCORD_CHANNEL_IDS:
        _fail("fixture_invalid")
    _snowflake(target["guild_id"], "fixture_invalid")
    _snowflake(target["channel_id"], "fixture_invalid")

    bot_identities = _strict(
        fixture["discord_bot_identities"],
        fields=(
            "production_bot_user_id",
            "connector_bot_user_id",
            "routeback_bot_user_id",
        ),
        code="fixture_invalid",
    )
    bot_user_ids = {
        _snowflake(bot_identities[field], "fixture_invalid")
        for field in (
            "production_bot_user_id",
            "connector_bot_user_id",
            "routeback_bot_user_id",
        )
    }
    if len(bot_user_ids) != 3 or bot_identities != {
        "production_bot_user_id": PRODUCTION_DISCORD_BOT_USER_ID,
        "connector_bot_user_id": PRODUCTION_DISCORD_CONNECTOR_BOT_USER_ID,
        "routeback_bot_user_id": PRODUCTION_DISCORD_ROUTEBACK_BOT_USER_ID,
    }:
        _fail("fixture_invalid")

    authorities = _strict(
        fixture["authority_keys"],
        fields=AUTHORITY_ROLES,
        code="fixture_invalid",
    )
    key_ids: set[str] = set()
    public_keys: set[str] = set()
    for role in AUTHORITY_ROLES:
        authority = _strict(
            authorities[role],
            fields=("key_id", "algorithm", "public_key_ed25519_hex"),
            code="fixture_invalid",
        )
        key_id = _sha256(authority["key_id"], "fixture_invalid")
        public_key = authority["public_key_ed25519_hex"]
        if (
            key_id in key_ids
            or public_key in public_keys
            or authority["algorithm"] != AUTHORITY_ALGORITHMS[role]
            or not isinstance(public_key, str)
            or _PUBLIC_KEY_RE.fullmatch(public_key) is None
            or key_id != hashlib.sha256(bytes.fromhex(public_key)).hexdigest()
        ):
            _fail("fixture_invalid")
        key_ids.add(key_id)
        public_keys.add(public_key)
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key))
        except (TypeError, ValueError):
            _fail("fixture_invalid")
    if _digest(fixture) != fixture_sha256:
        _fail("fixture_digest_noncanonical")
    return fixture


_COMMON_PAYLOAD_FIELDS = (
    "schema",
    "run_id",
    "release_sha",
    "fixture_sha256",
    "observed_at_unix_ms",
)


def _ssh_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def _read_ssh_string(value: bytes, offset: int, code: str) -> tuple[bytes, int]:
    if offset + 4 > len(value):
        _fail(code)
    size = struct.unpack(">I", value[offset : offset + 4])[0]
    start = offset + 4
    end = start + size
    if size > _MAX_SSHSIG_BYTES or end > len(value):
        _fail(code)
    return value[start:end], end


def _verify_owner_sshsig(
    signature: Any,
    *,
    message: bytes,
    public_key_hex: str,
    code: str,
) -> None:
    if (
        not isinstance(signature, str)
        or len(signature.encode("ascii", errors="ignore")) > _MAX_SSHSIG_BYTES
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
            re.fullmatch(r"[A-Za-z0-9+/=]{1,70}", line) is None for line in lines[1:-1]
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
    public_blob, offset = _read_ssh_string(envelope, offset, code)
    namespace, offset = _read_ssh_string(envelope, offset, code)
    reserved, offset = _read_ssh_string(envelope, offset, code)
    hash_algorithm, offset = _read_ssh_string(envelope, offset, code)
    signature_blob, offset = _read_ssh_string(envelope, offset, code)
    if offset != len(envelope):
        _fail(code)
    key_type, key_offset = _read_ssh_string(public_blob, 0, code)
    public_key, key_offset = _read_ssh_string(public_blob, key_offset, code)
    signature_type, signature_offset = _read_ssh_string(signature_blob, 0, code)
    raw_signature, signature_offset = _read_ssh_string(
        signature_blob, signature_offset, code
    )
    try:
        expected_public_key = bytes.fromhex(public_key_hex)
    except ValueError:
        _fail(code)
    if (
        key_offset != len(public_blob)
        or signature_offset != len(signature_blob)
        or key_type != b"ssh-ed25519"
        or signature_type != b"ssh-ed25519"
        or public_key != expected_public_key
        or len(public_key) != 32
        or len(raw_signature) != 64
        or namespace != OWNER_SSHSIG_NAMESPACE.encode("ascii")
        or reserved != b""
        or hash_algorithm != b"sha512"
    ):
        _fail(code)
    signed = (
        b"SSHSIG"
        + _ssh_string(namespace)
        + _ssh_string(reserved)
        + _ssh_string(hash_algorithm)
        + _ssh_string(hashlib.sha512(message).digest())
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(raw_signature, signed)
    except (InvalidSignature, ValueError):
        _fail(code)


def _signed_payload(
    receipt: Any,
    *,
    slot: str,
    role: str,
    payload_schema: str,
    fields: Sequence[str],
    fixture: Mapping[str, Any],
    fixture_sha256: str,
    code: str,
) -> dict[str, Any]:
    if SLOT_ROLE.get(slot) != role:
        _fail(code)
    envelope_fields = (
        "schema",
        "authority_role",
        "key_id",
        "signature_algorithm",
        "payload",
        "signature",
    )
    if role != "owner":
        envelope_fields = (*envelope_fields[:-1], "native_evidence", "signature")
    envelope = _strict(receipt, fields=envelope_fields, code=code)
    authority = fixture["authority_keys"].get(role)
    signature = envelope["signature"]
    if (
        envelope["schema"] != SIGNED_RECEIPT_SCHEMA
        or envelope["authority_role"] != role
        or not isinstance(authority, Mapping)
        or envelope["key_id"] != authority.get("key_id")
        or envelope["signature_algorithm"] != authority.get("algorithm")
    ):
        _fail(code)
    payload = _strict(
        envelope["payload"],
        fields=(*_COMMON_PAYLOAD_FIELDS, *fields),
        code=code,
    )
    observed = _positive_int(payload["observed_at_unix_ms"], code)
    if (
        payload["schema"] != payload_schema
        or payload["run_id"] != fixture["run_id"]
        or payload["release_sha"] != fixture["release_sha"]
        or payload["fixture_sha256"] != fixture_sha256
        or not fixture["valid_from_unix_ms"]
        <= observed
        <= fixture["valid_until_unix_ms"]
    ):
        _fail(code)
    unsigned: dict[str, Any] = {
        "schema": envelope["schema"],
        "authority_role": envelope["authority_role"],
        "key_id": envelope["key_id"],
        "signature_algorithm": envelope["signature_algorithm"],
        "payload": payload,
    }
    if role != "owner":
        native = _strict(
            envelope["native_evidence"],
            fields=("schema", "producer_readiness_sha256", "bindings"),
            code=code,
        )
        bindings = native["bindings"]
        expected_kinds = SLOT_NATIVE_BINDING_KINDS.get(slot)
        if (
            native["schema"] != NATIVE_EVIDENCE_SCHEMA
            or _SHA256_RE.fullmatch(
                str(native["producer_readiness_sha256"])
            )
            is None
            or not isinstance(bindings, list)
            or expected_kinds is None
            or len(bindings) != len(expected_kinds)
        ):
            _fail(code)
        for expected_kind, raw_binding in zip(
            expected_kinds, bindings, strict=True
        ):
            binding = _strict(
                raw_binding,
                fields=(
                    "kind",
                    "source_identity_sha256",
                    "artifact_sha256",
                    "verification_receipt_sha256",
                ),
                code=code,
            )
            if binding["kind"] != expected_kind:
                _fail(code)
            for field in (
                "source_identity_sha256",
                "artifact_sha256",
                "verification_receipt_sha256",
            ):
                _sha256(binding[field], code)
        unsigned["native_evidence"] = native
    message = _canonical_bytes(unsigned)
    if envelope["signature_algorithm"] == "ed25519":
        if not isinstance(signature, str) or _SIGNATURE_RE.fullmatch(signature) is None:
            _fail(code)
        try:
            public_key = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(authority["public_key_ed25519_hex"])
            )
            public_key.verify(bytes.fromhex(signature), message)
        except (InvalidSignature, TypeError, ValueError):
            _fail(code)
    else:
        _verify_owner_sshsig(
            signature,
            message=message,
            public_key_hex=authority["public_key_ed25519_hex"],
            code=code,
        )
    return payload


def _validate_terminal_ctw(
    value: Any,
    *,
    code: str,
    allowed_states: frozenset[str],
) -> dict[str, Any]:
    terminal = _strict(
        value,
        fields=(
            "case_id",
            "plan_id",
            "revision",
            "state",
            "terminal_event_id",
            "terminal_event_sha256",
            "completed_step_ids",
            "verification_event_ids",
            "pending_step_count",
            "blocked_step_count",
            "resumed_after_restart",
            "replayed_mutation_count",
            "blocker_event_id",
            "blocker_receipt_sha256",
        ),
        code=code,
    )
    _safe_id(terminal["case_id"], code)
    _safe_id(terminal["plan_id"], code)
    _positive_int(terminal["revision"], code)
    _safe_id(terminal["terminal_event_id"], code)
    _sha256(terminal["terminal_event_sha256"], code)
    _bounded_id_list(terminal["completed_step_ids"], code=code)
    _bounded_id_list(terminal["verification_event_ids"], code=code)
    pending = _nonnegative_int(terminal["pending_step_count"], code)
    blocked = _nonnegative_int(terminal["blocked_step_count"], code)
    _bool(terminal["resumed_after_restart"], code)
    if _nonnegative_int(terminal["replayed_mutation_count"], code) != 0:
        _fail(code)
    state = terminal["state"]
    if state not in allowed_states:
        _fail(code)
    if state == "completed":
        if (
            pending != 0
            or blocked != 0
            or terminal["blocker_event_id"] is not None
            or terminal["blocker_receipt_sha256"] is not None
        ):
            _fail(code)
    else:
        if (
            blocked < 1
            or not isinstance(terminal["blocker_event_id"], str)
            or not isinstance(terminal["blocker_receipt_sha256"], str)
        ):
            _fail(code)
        _safe_id(terminal["blocker_event_id"], code)
        _sha256(terminal["blocker_receipt_sha256"], code)
    return terminal


def _validate_runtime(
    receipt: Any,
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
) -> None:
    code = "runtime_receipt_invalid"
    payload = _signed_payload(
        receipt,
        slot="runtime",
        role="gateway_observer",
        payload_schema=RUNTIME_RECEIPT_SCHEMA,
        fields=(
            "host_identity_sha256",
            "release_artifact_sha256",
            "installed_wheel_manifest_sha256",
            "effective_config_sha256",
            "tool_inventory_sha256",
            "provider",
            "api_mode",
            "model",
            "initial_effort",
            "adaptive_max_effort",
            "max_turns",
            "toolsets",
            "kanban_auxiliary_planning_enabled",
            "kanban_auto_decompose",
            "kanban_dispatch_in_gateway",
            "prompt_cache_stable",
            "message_alternation_valid",
            "gateway_process_identity_sha256",
            "connector_bot_user_id",
            "connector_bot_user_id_provenance",
            "connector_readiness_receipt_sha256",
        ),
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        code=code,
    )
    route = fixture["model_route"]
    if (
        payload["host_identity_sha256"] != fixture["host_identity_sha256"]
        or payload["release_artifact_sha256"] != fixture["release_artifact_sha256"]
        or payload["installed_wheel_manifest_sha256"]
        != fixture["installed_wheel_manifest_sha256"]
        or payload["effective_config_sha256"] != fixture["effective_config_sha256"]
        or payload["tool_inventory_sha256"] != fixture["tool_inventory_sha256"]
        or any(
            payload[name] != route[name]
            for name in (
                "provider",
                "api_mode",
                "model",
                "initial_effort",
                "adaptive_max_effort",
                "max_turns",
            )
        )
        or payload["toolsets"] != list(REQUIRED_TOOLSETS)
        or payload["kanban_auxiliary_planning_enabled"] is not False
        or payload["kanban_auto_decompose"] is not False
        or payload["kanban_dispatch_in_gateway"] is not False
        or payload["prompt_cache_stable"] is not True
        or payload["message_alternation_valid"] is not True
        or payload["connector_bot_user_id"]
        != fixture["discord_bot_identities"]["connector_bot_user_id"]
        or payload["connector_bot_user_id_provenance"]
        != CONNECTOR_BOT_ID_PROVENANCE
    ):
        _fail(code)
    _sha256(payload["gateway_process_identity_sha256"], code)
    _sha256(payload["connector_readiness_receipt_sha256"], code)


def _validate_task_workspace_bundle(
    bundle: Any,
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
) -> None:
    code = "task_workspace_bundle_invalid"
    value = _strict(
        bundle,
        fields=("gateway_receipt", "writer_receipt", "owner_approval_receipt"),
        code=code,
    )
    gateway = _signed_payload(
        value["gateway_receipt"],
        slot="workspace_gateway",
        role="gateway_observer",
        payload_schema=TASK_WORKSPACE_GATEWAY_SCHEMA,
        fields=(
            "session_id",
            "capability_epoch_sha256",
            "transcript_sha256",
            "task_workspace_evidence_sha256s",
            "first_path_failure_receipt_sha256",
            "alternate_read_receipt_sha256",
            "model_requested_effort",
            "later_request_effort",
            "reasoning_tool_call_id",
            "restart_count",
            "used_command_sha256s",
            "mutation_receipt_sha256s",
            "approval_prompt_count",
            "microapproval_prompt_count",
            "replayed_mutation_count",
            "owner_grant_id",
            "owner_grant_sha256",
            "consumed_command_sha256s",
            "terminal_plan_id",
            "terminal_plan_revision",
        ),
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        code=code,
    )
    writer = _signed_payload(
        value["writer_receipt"],
        slot="workspace_writer",
        role="canonical_writer",
        payload_schema=TASK_WORKSPACE_WRITER_SCHEMA,
        fields=(
            "session_id",
            "capability_epoch_sha256",
            "owner_grant_id",
            "owner_grant_sha256",
            "consumed_command_sha256s",
            "terminal_ctw",
        ),
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        code=code,
    )
    approval = _signed_payload(
        value["owner_approval_receipt"],
        slot="workspace_owner",
        role="owner",
        payload_schema=PLAN_APPROVAL_SCHEMA,
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
        code=code,
    )
    session_id = _safe_id(gateway["session_id"], code)
    epoch = _sha256(gateway["capability_epoch_sha256"], code)
    _sha256(gateway["transcript_sha256"], code)
    task_workspace_evidence = _bounded_digest_list(
        gateway["task_workspace_evidence_sha256s"],
        code=code,
        minimum=1,
        maximum=MAX_STEPS,
        sorted_unique=True,
    )
    if not task_workspace_evidence:
        _fail(code)
    for field in (
        "first_path_failure_receipt_sha256",
        "alternate_read_receipt_sha256",
    ):
        _sha256(gateway[field], code)
    _safe_id(gateway["reasoning_tool_call_id"], code)
    used_commands = _bounded_digest_list(
        gateway["used_command_sha256s"],
        code=code,
        maximum=MAX_COMMANDS,
        sorted_unique=True,
    )
    _bounded_digest_list(gateway["mutation_receipt_sha256s"], code=code)
    terminal = _validate_terminal_ctw(
        writer["terminal_ctw"], code=code, allowed_states=frozenset({"completed"})
    )
    commands = _bounded_digest_list(
        approval["command_sha256s"],
        code=code,
        maximum=MAX_COMMANDS,
        sorted_unique=True,
    )
    gateway_consumed = _bounded_digest_list(
        gateway["consumed_command_sha256s"],
        code=code,
        maximum=MAX_COMMANDS,
        sorted_unique=True,
    )
    writer_consumed = _bounded_digest_list(
        writer["consumed_command_sha256s"],
        code=code,
        maximum=MAX_COMMANDS,
        sorted_unique=True,
    )
    grant_id = _safe_id(approval["approval_id"], code)
    grant_sha256 = _digest(value["owner_approval_receipt"])
    if (
        gateway["model_requested_effort"] != "max"
        or gateway["later_request_effort"] != "max"
        or _positive_int(gateway["restart_count"], code) < 1
        or _nonnegative_int(gateway["approval_prompt_count"], code) != 0
        or _nonnegative_int(gateway["microapproval_prompt_count"], code) != 0
        or _nonnegative_int(gateway["replayed_mutation_count"], code) != 0
        or writer["session_id"] != session_id
        or writer["capability_epoch_sha256"] != epoch
        or gateway["owner_grant_id"] != grant_id
        or writer["owner_grant_id"] != grant_id
        or gateway["owner_grant_sha256"] != grant_sha256
        or writer["owner_grant_sha256"] != grant_sha256
        or gateway_consumed != commands
        or writer_consumed != commands
        or terminal["resumed_after_restart"] is not True
        or gateway["terminal_plan_id"] != terminal["plan_id"]
        or gateway["terminal_plan_revision"] != terminal["revision"]
        or approval["owner_id"] != fixture["owner_id"]
        or approval["session_id"] != session_id
        or approval["capability_epoch_sha256"] != epoch
        or commands != used_commands
        or not 1 <= _positive_int(approval["ttl_seconds"], code) <= 8 * 60 * 60
        or _positive_int(approval["max_uses"], code) < len(used_commands)
        or approval["observed_at_unix_ms"] > gateway["observed_at_unix_ms"]
        or gateway["observed_at_unix_ms"] > writer["observed_at_unix_ms"]
    ):
        _fail(code)
    _safe_id(gateway["terminal_plan_id"], code)
    _positive_int(gateway["terminal_plan_revision"], code)


def _validate_denial_bundle(
    bundle: Any,
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
) -> None:
    code = "capability_denials_invalid"
    payload = _signed_payload(
        bundle,
        slot="capability_denials",
        role="canonical_writer",
        payload_schema=CAPABILITY_DENIAL_SCHEMA,
        fields=("session_id", "capability_epoch_sha256", "denials"),
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        code=code,
    )
    _safe_id(payload["session_id"], code)
    _sha256(payload["capability_epoch_sha256"], code)
    denials = payload["denials"]
    if not isinstance(denials, list) or len(denials) != len(DENIAL_KINDS):
        _fail(code)
    for expected, raw in zip(DENIAL_KINDS, denials, strict=True):
        item = _strict(
            raw,
            fields=("kind", "denied", "dispatch_attempted", "receipt_sha256"),
            code=code,
        )
        if (
            item["kind"] != expected
            or item["denied"] is not True
            or item["dispatch_attempted"] is not False
        ):
            _fail(code)
        _sha256(item["receipt_sha256"], code)


def _validate_database_bundle(
    bundle: Any,
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
) -> None:
    code = "database_bundle_invalid"
    payload = _signed_payload(
        bundle,
        slot="database_reconciliation",
        role="canonical_writer",
        payload_schema=DATABASE_RECONCILIATION_SCHEMA,
        fields=(
            "read_receipt_sha256",
            "transaction_receipt_sha256",
            "idempotency_key_sha256",
            "lost_response_observed",
            "reconciled_before_retry",
            "readback_verified",
            "readback_receipt_sha256",
            "durable_row_count",
            "duplicate_row_count",
            "terminal_ctw",
        ),
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        code=code,
    )
    for field in (
        "read_receipt_sha256",
        "transaction_receipt_sha256",
        "idempotency_key_sha256",
        "readback_receipt_sha256",
    ):
        _sha256(payload[field], code)
    if (
        payload["lost_response_observed"] is not True
        or payload["reconciled_before_retry"] is not True
        or payload["readback_verified"] is not True
        or _positive_int(payload["durable_row_count"], code) != 1
        or _nonnegative_int(payload["duplicate_row_count"], code) != 0
    ):
        _fail(code)
    _validate_terminal_ctw(
        payload["terminal_ctw"], code=code, allowed_states=frozenset({"completed"})
    )


def _validate_bitrix_bundle(
    bundle: Any,
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
) -> None:
    code = "bitrix_bundle_invalid"
    value = _strict(bundle, fields=("edge_receipt", "writer_receipt"), code=code)
    edge = _signed_payload(
        value["edge_receipt"],
        slot="bitrix_edge",
        role="business_edge",
        payload_schema=BITRIX_EDGE_SCHEMA,
        fields=(
            "handoff_id",
            "selected_edge_id",
            "operational_edge_contract_sha256",
            "operational_edge_service_identity_sha256",
            "operational_edge_asset_manifest_sha256",
            "read_probe",
            "request_sha256",
            "operation_id",
            "arguments_sha256",
            "operational_edge_receipt_sha256",
            "authenticated_live_readback_sha256",
            "readback_verified",
        ),
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        code=code,
    )
    writer = _signed_payload(
        value["writer_receipt"],
        slot="bitrix_writer",
        role="canonical_writer",
        payload_schema=BITRIX_WRITER_SCHEMA,
        fields=(
            "handoff_id",
            "selected_edge_id",
            "selection_event_id",
            "selection_event_sha256",
            "mutation_probe",
            "mutation_approval_present",
            "mutation_dispatched",
            "mutation_denial_receipt_sha256",
            "blocked_event_id",
            "blocked_receipt_sha256",
            "terminal_ctw",
        ),
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        code=code,
    )
    read_probe = _strict(
        edge["read_probe"],
        fields=(
            "selected_edge_id",
            "read_operation_id",
            "read_arguments",
            "initial_read_probe_id",
            "readback_probe_id",
            "normalized_equality_excluded_fields",
            "stable_normalized_equality",
        ),
        code=code,
    )
    mutation_probe = _strict(
        writer["mutation_probe"],
        fields=(
            "selected_edge_id",
            "mutation_operation_id",
            "mutation_arguments",
            "mutation_probe_id",
        ),
        code=code,
    )
    for field in (
        "operational_edge_contract_sha256",
        "operational_edge_service_identity_sha256",
        "operational_edge_asset_manifest_sha256",
        "request_sha256",
        "arguments_sha256",
        "operational_edge_receipt_sha256",
        "authenticated_live_readback_sha256",
    ):
        _sha256(edge[field], code)
    for field in (
        "selection_event_sha256",
        "blocked_receipt_sha256",
        "mutation_denial_receipt_sha256",
    ):
        _sha256(writer[field], code)
    if (
        _safe_id(edge["handoff_id"], code) != writer["handoff_id"]
        or _safe_id(edge["selected_edge_id"], code) != writer["selected_edge_id"]
        or edge["selected_edge_id"] != "operational-edge:bitrix"
        or edge["operational_edge_contract_sha256"]
        != _digest(fixture["bitrix_operational_edge_contract"])
        or edge["operational_edge_service_identity_sha256"]
        != fixture["bitrix_operational_edge_contract"][
            "service_identity_sha256"
        ]
        or edge["operational_edge_asset_manifest_sha256"]
        != fixture["bitrix_operational_edge_contract"][
            "asset_manifest_sha256"
        ]
        or read_probe["selected_edge_id"] != "operational-edge:bitrix"
        or read_probe["read_operation_id"] != "bitrix.crm.status_list"
        or read_probe["read_arguments"] != {"entity_id": "STATUS"}
        or read_probe["normalized_equality_excluded_fields"]
        != ["generated_at_utc"]
        or read_probe["stable_normalized_equality"] is not True
        or mutation_probe["selected_edge_id"] != "operational-edge:bitrix"
        or mutation_probe["mutation_operation_id"] != "bitrix.crm.lead_add"
        or mutation_probe["mutation_arguments"]
        != {
            "title": "Muncho capability canary dry-run",
            "requester": "capability-canary",
            "reason": "Verify pre-dispatch denial without mutation",
            "execute": False,
        }
        or len(
            {
                _safe_id(read_probe["initial_read_probe_id"], code),
                _safe_id(read_probe["readback_probe_id"], code),
                _safe_id(mutation_probe["mutation_probe_id"], code),
            }
        )
        != 3
        or edge["operation_id"] != "bitrix.crm.status_list"
        or edge["arguments_sha256"] != _digest({"entity_id": "STATUS"})
        or edge["readback_verified"] is not True
        or writer["mutation_approval_present"] is not False
        or writer["mutation_dispatched"] is not False
        or edge["observed_at_unix_ms"] > writer["observed_at_unix_ms"]
    ):
        _fail(code)
    _safe_id(writer["selection_event_id"], code)
    _safe_id(writer["blocked_event_id"], code)
    _validate_terminal_ctw(
        writer["terminal_ctw"], code=code, allowed_states=frozenset({"completed"})
    )


def _validate_discord_bundle(
    bundle: Any,
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
) -> None:
    code = "discord_bundle_invalid"
    value = _strict(bundle, fields=("edge_receipt", "writer_receipt"), code=code)
    edge = _signed_payload(
        value["edge_receipt"],
        slot="discord_edge",
        role="discord_edge",
        payload_schema=DISCORD_EDGE_SCHEMA,
        fields=(
            "target_type",
            "guild_id",
            "channel_id",
            "idempotency_key_sha256",
            "request_sha256",
            "content_sha256",
            "platform_message_id",
            "adapter_accepted",
            "public_readback_verified",
            "public_receipt_sha256",
            "private_target_kind",
            "private_dispatch_attempted",
            "journal_unchanged_after_private_probe",
            "private_denial_receipt_sha256",
            "routeback_bot_user_id",
            "routeback_bot_user_id_provenance",
        ),
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        code=code,
    )
    writer = _signed_payload(
        value["writer_receipt"],
        slot="discord_writer",
        role="canonical_writer",
        payload_schema=DISCORD_WRITER_SCHEMA,
        fields=(
            "sent_event_id",
            "sent_event_sha256",
            "sent_after_verified_readback",
            "sent_platform_message_id",
            "blocked_event_id",
            "blocked_event_sha256",
            "blocked_before_dispatch",
            "terminal_ctw",
        ),
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        code=code,
    )
    target = fixture["public_discord_target"]
    for field in (
        "idempotency_key_sha256",
        "request_sha256",
        "content_sha256",
        "public_receipt_sha256",
        "private_denial_receipt_sha256",
    ):
        _sha256(edge[field], code)
    for field in ("sent_event_sha256", "blocked_event_sha256"):
        _sha256(writer[field], code)
    message_id = _snowflake(edge["platform_message_id"], code)
    if (
        edge["target_type"] != target["target_type"]
        or edge["guild_id"] != target["guild_id"]
        or edge["channel_id"] != target["channel_id"]
        or edge["adapter_accepted"] is not True
        or edge["public_readback_verified"] is not True
        or edge["private_target_kind"] != "dm"
        or edge["private_dispatch_attempted"] is not False
        or edge["journal_unchanged_after_private_probe"] is not True
        or edge["routeback_bot_user_id"]
        != fixture["discord_bot_identities"]["routeback_bot_user_id"]
        or edge["routeback_bot_user_id_provenance"]
        != ROUTEBACK_BOT_ID_PROVENANCE
        or writer["sent_after_verified_readback"] is not True
        or writer["sent_platform_message_id"] != message_id
        or writer["blocked_before_dispatch"] is not True
        or edge["observed_at_unix_ms"] > writer["observed_at_unix_ms"]
    ):
        _fail(code)
    for field in ("sent_event_id", "blocked_event_id"):
        _safe_id(writer[field], code)
    _validate_terminal_ctw(
        writer["terminal_ctw"], code=code, allowed_states=frozenset({"completed"})
    )


def _validate_failure_bundle(
    bundle: Any,
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
) -> None:
    code = "failure_bundle_invalid"
    value = _strict(bundle, fields=("gateway_receipt", "writer_receipt"), code=code)
    gateway = _signed_payload(
        value["gateway_receipt"],
        slot="failure_gateway",
        role="gateway_observer",
        payload_schema=FAILURE_GATEWAY_SCHEMA,
        fields=("transcript_sha256", "failures", "model_retained_tool_control"),
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        code=code,
    )
    writer = _signed_payload(
        value["writer_receipt"],
        slot="failure_writer",
        role="canonical_writer",
        payload_schema=FAILURE_WRITER_SCHEMA,
        fields=("terminal_ctw",),
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        code=code,
    )
    _sha256(gateway["transcript_sha256"], code)
    if gateway["model_retained_tool_control"] is not True:
        _fail(code)
    failures = gateway["failures"]
    if not isinstance(failures, list) or len(failures) != len(FAILURE_COMPONENTS):
        _fail(code)
    for expected, raw in zip(FAILURE_COMPONENTS, failures, strict=True):
        item = _strict(
            raw,
            fields=(
                "component",
                "failure_observed",
                "failure_receipt_sha256",
                "alternative_available",
                "alternative_attempted",
                "alternative_receipt_sha256",
            ),
            code=code,
        )
        if item["component"] != expected or item["failure_observed"] is not True:
            _fail(code)
        _sha256(item["failure_receipt_sha256"], code)
        available = _bool(item["alternative_available"], code)
        attempted = _bool(item["alternative_attempted"], code)
        if available != attempted:
            _fail(code)
        if available:
            _sha256(item["alternative_receipt_sha256"], code)
        elif item["alternative_receipt_sha256"] is not None:
            _fail(code)
        if expected in {"tool", "browser"} and not available:
            _fail(code)
    _validate_terminal_ctw(
        writer["terminal_ctw"],
        code=code,
        allowed_states=frozenset({"completed", "blocked"}),
    )
    if gateway["observed_at_unix_ms"] > writer["observed_at_unix_ms"]:
        _fail(code)


def _all_evidence_receipts(
    runtime_receipt: Any,
    bundles: Mapping[str, Any],
    cleanup_receipt: Any,
) -> list[Any]:
    """Return only the fixed receipt positions; never discover by prose/content."""

    try:
        return [
            runtime_receipt,
            bundles["workspace_continuation"]["gateway_receipt"],
            bundles["workspace_continuation"]["writer_receipt"],
            bundles["workspace_continuation"]["owner_approval_receipt"],
            bundles["capability_denials"],
            bundles["database_reconciliation"],
            bundles["bitrix_boundary"]["edge_receipt"],
            bundles["bitrix_boundary"]["writer_receipt"],
            bundles["discord_routeback"]["edge_receipt"],
            bundles["discord_routeback"]["writer_receipt"],
            bundles["failure_recovery"]["gateway_receipt"],
            bundles["failure_recovery"]["writer_receipt"],
            cleanup_receipt,
        ]
    except (KeyError, TypeError):
        _fail("evidence_invalid")


def _validate_cleanup(
    receipt: Any,
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
) -> None:
    code = "cleanup_receipt_invalid"
    payload = _signed_payload(
        receipt,
        slot="cleanup",
        role="gateway_observer",
        payload_schema=CLEANUP_RECEIPT_SCHEMA,
        fields=(
            "non_observer_service_units",
            "non_observer_services_stopped",
            "non_observer_services_state_sha256",
            "gateway_observer_signer_identity",
            "credential_consumer_stop_proof",
            "credential_leases",
            "credential_leases_retired",
            "retirements",
            "retirement_receipt_sha256s",
            "credential_absence",
            "credentials_absent",
            "bitrix_receipt_key_retirement",
            "bitrix_receipt_key_absence",
            "discord_credential_topology",
            "browser_session_retired",
            "isolated_worker_lease_cleanup_verified",
            "production_diff_sha256",
        ),
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        code=code,
    )
    if (
        payload["non_observer_service_units"]
        != list(NON_OBSERVER_SERVICE_UNITS)
        or payload["non_observer_services_stopped"] is not True
        or payload["credential_leases"] != list(CREDENTIAL_LEASES)
        or payload["credential_leases_retired"] is not True
        or payload["credentials_absent"] is not True
        or payload["browser_session_retired"] is not True
        or payload["isolated_worker_lease_cleanup_verified"] is not True
    ):
        _fail(code)
    observer = _strict(
        payload["gateway_observer_signer_identity"],
        fields=(
            "role",
            "service_unit",
            "live",
            "signing_only",
            "credential_read_access",
            "service_state_sha256",
            "producer_foundation_sha256",
            "unit_bundle_manifest_sha256",
            "credential_inaccessibility_contract_sha256",
        ),
        code=code,
    )
    if (
        observer["role"] != "gateway_observer"
        or observer["service_unit"] != OBSERVER_PRODUCER_SERVICE_UNIT
        or observer["live"] is not True
        or observer["signing_only"] is not True
        or observer["credential_read_access"] is not False
        or observer["producer_foundation_sha256"]
        != fixture["producer_foundation_sha256"]
    ):
        _fail(code)
    for field in (
        "service_state_sha256",
        "producer_foundation_sha256",
        "unit_bundle_manifest_sha256",
        "credential_inaccessibility_contract_sha256",
    ):
        _sha256(observer[field], code)
    stop_proof = _strict(
        payload["credential_consumer_stop_proof"],
        fields=(
            "schema",
            "plan_sha256",
            "non_observer_stop_order",
            "non_observer_services_state_sha256",
            "all_credential_consumers_stopped",
            "observer_service_unit",
            "observer_state_sha256",
            "observer_live_signing_only",
            "observer_credential_read_access",
            "producer_foundation_sha256",
            "unit_bundle_manifest_sha256",
            "credential_inaccessibility_contract_sha256",
            "observed_at_unix",
            "secret_material_recorded",
            "secret_digest_recorded",
            "stop_proof_sha256",
        ),
        code=code,
    )
    stop_unsigned = {
        key: item for key, item in stop_proof.items() if key != "stop_proof_sha256"
    }
    if (
        stop_proof["schema"] != CREDENTIAL_CONSUMER_STOP_PROOF_SCHEMA
        or stop_proof["plan_sha256"] != fixture["capability_plan_sha256"]
        or stop_proof["non_observer_stop_order"]
        != list(NON_OBSERVER_SERVICE_UNITS)
        or stop_proof["all_credential_consumers_stopped"] is not True
        or stop_proof["observer_service_unit"]
        != OBSERVER_PRODUCER_SERVICE_UNIT
        or stop_proof["observer_live_signing_only"] is not True
        or stop_proof["observer_credential_read_access"] is not False
        or stop_proof["secret_material_recorded"] is not False
        or stop_proof["secret_digest_recorded"] is not False
        or stop_proof["producer_foundation_sha256"]
        != fixture["producer_foundation_sha256"]
        or stop_proof["observer_state_sha256"]
        != observer["service_state_sha256"]
        or stop_proof["unit_bundle_manifest_sha256"]
        != observer["unit_bundle_manifest_sha256"]
        or stop_proof["credential_inaccessibility_contract_sha256"]
        != observer["credential_inaccessibility_contract_sha256"]
        or payload["non_observer_services_state_sha256"]
        != stop_proof["non_observer_services_state_sha256"]
        or stop_proof["stop_proof_sha256"] != _digest(stop_unsigned)
        or not fixture["valid_from_unix_ms"] // 1000
        <= _nonnegative_int(stop_proof["observed_at_unix"], code)
        <= payload["observed_at_unix_ms"] // 1000
    ):
        _fail(code)
    for field in (
        "plan_sha256",
        "non_observer_services_state_sha256",
        "observer_state_sha256",
        "producer_foundation_sha256",
        "unit_bundle_manifest_sha256",
        "credential_inaccessibility_contract_sha256",
        "stop_proof_sha256",
    ):
        _sha256(stop_proof[field], code)
    topology = _strict(
        payload["discord_credential_topology"],
        fields=tuple(DISCORD_CREDENTIAL_TOPOLOGY),
        code=code,
    )
    if (
        topology != DISCORD_CREDENTIAL_TOPOLOGY
        or topology["connector_service_unit"]
        == topology["routeback_service_unit"]
        or topology["connector_credential_lease"]
        == topology["routeback_credential_lease"]
    ):
        _fail(code)
    credential_stop_proof_sha256 = stop_proof["stop_proof_sha256"]
    retirements = _strict(
        payload["retirements"], fields=CREDENTIAL_LEASES, code=code
    )
    retirement_digests = _strict(
        payload["retirement_receipt_sha256s"],
        fields=CREDENTIAL_LEASES,
        code=code,
    )
    absence = _strict(
        payload["credential_absence"], fields=CREDENTIAL_LEASES, code=code
    )
    target_paths: set[str] = set()
    for binding in CREDENTIAL_LEASES:
        completion = _strict(
            retirements[binding],
            fields=(
                "operation",
                "state",
                "kind",
                "credential_binding",
                "revision",
                "plan_sha256",
                "full_canary_plan_sha256",
                "lease_id",
                "target_path",
                "target_device",
                "target_inode",
                "target_uid",
                "target_gid",
                "target_mode",
                "target_size",
                "target_mtime_ns",
                "target_ctime_ns",
                "install_receipt_path",
                "install_receipt_sha256",
                "retirement_intent_path",
                "retirement_intent_sha256",
                "service_stop_proof_sha256",
                "service_stop_observed_at_unix",
                "removed",
                "absent",
                "absent_after_stop",
                "retired_at_unix",
                "schema",
                "receipt_path",
                "secret_material_recorded",
                "secret_digest_recorded",
                "receipt_sha256",
            ),
            code=code,
        )
        unsigned_completion = {
            key: item
            for key, item in completion.items()
            if key != "receipt_sha256"
        }
        paths = tuple(
            completion[field]
            for field in (
                "target_path",
                "install_receipt_path",
                "retirement_intent_path",
                "receipt_path",
            )
        )
        if any(
            not isinstance(item, str)
            or not Path(item).is_absolute()
            or ".." in Path(item).parts
            for item in paths
        ):
            _fail(code)
        lease_id = _safe_id(completion["lease_id"], code)
        receipt_parent = Path(completion["receipt_path"]).parent
        if (
            completion["schema"] != SECRET_RETIREMENT_COMPLETION_SCHEMA
            or completion["operation"] != "retirement_completion"
            or completion["state"] != "retired"
            or completion["kind"] != CREDENTIAL_LEASE_KINDS[binding]
            or completion["credential_binding"] != binding
            or completion["revision"] != fixture["release_sha"]
            or completion["plan_sha256"]
            != fixture["capability_plan_sha256"]
            or completion["full_canary_plan_sha256"]
            != fixture["full_canary_plan_sha256"]
            or receipt_parent.name != lease_id
            or Path(completion["receipt_path"]).name
            != "retirement-completion.json"
            or Path(completion["install_receipt_path"]).parent
            != receipt_parent
            or Path(completion["install_receipt_path"]).name
            != "install-receipt.json"
            or Path(completion["retirement_intent_path"]).parent
            != receipt_parent
            or Path(completion["retirement_intent_path"]).name
            != "retirement-intent.json"
            or completion["removed"] is not True
            or completion["absent"] is not True
            or completion["absent_after_stop"] is not True
            or completion["secret_material_recorded"] is not False
            or completion["secret_digest_recorded"] is not False
            or completion["service_stop_proof_sha256"]
            != credential_stop_proof_sha256
            or completion["service_stop_observed_at_unix"]
            != stop_proof["observed_at_unix"]
            or _positive_int(completion["retired_at_unix"], code)
            < stop_proof["observed_at_unix"]
            or not isinstance(completion["target_mode"], str)
            or re.fullmatch(r"0[0-7]{3}", completion["target_mode"]) is None
            or any(
                type(completion[field]) is not int
                or completion[field] < 0
                for field in (
                    "target_device",
                    "target_inode",
                    "target_uid",
                    "target_gid",
                    "target_size",
                    "target_mtime_ns",
                    "target_ctime_ns",
                )
            )
            or _sha256(completion["install_receipt_sha256"], code)
            == _sha256(completion["retirement_intent_sha256"], code)
            or completion["receipt_sha256"]
            != _digest(unsigned_completion)
            or retirement_digests[binding] != completion["receipt_sha256"]
        ):
            _fail(code)
        absence_item = _strict(
            absence[binding], fields=("path", "absent"), code=code
        )
        if absence_item != {
            "path": completion["target_path"],
            "absent": True,
        }:
            _fail(code)
        target_paths.add(completion["target_path"])
    if len(target_paths) != len(CREDENTIAL_LEASES):
        _fail(code)
    key_retirement = _strict(
        payload["bitrix_receipt_key_retirement"],
        fields=(
            "schema",
            "operation",
            "reason",
            "revision",
            "full_canary_plan_sha256",
            "key_bootstrap_receipt_path",
            "key_bootstrap_receipt_sha256",
            "retirement_intent_path",
            "retirement_intent_sha256",
            "public_key_id",
            "private_path",
            "public_path",
            "private_absent",
            "public_absent",
            "both_pair_members_absent",
            "service_stop_proof_sha256",
            "retired_at_unix",
            "private_content_or_digest_recorded",
            "receipt_path",
            "receipt_sha256",
        ),
        code=code,
    )
    key_unsigned = {
        key: item
        for key, item in key_retirement.items()
        if key != "receipt_sha256"
    }
    key_contract = fixture["bitrix_operational_edge_contract"][
        "receipt_key_contract"
    ]
    key_paths = (
        key_retirement["key_bootstrap_receipt_path"],
        key_retirement["retirement_intent_path"],
        key_retirement["private_path"],
        key_retirement["public_path"],
        key_retirement["receipt_path"],
    )
    if any(
        not isinstance(item, str)
        or not Path(item).is_absolute()
        or ".." in Path(item).parts
        for item in key_paths
    ) or (
        key_retirement["schema"] != INTERNAL_KEY_RETIREMENT_SCHEMA
        or key_retirement["operation"]
        != "retire_bitrix_receipt_key_pair"
        or key_retirement["reason"] != "service_stop"
        or key_retirement["revision"] != fixture["release_sha"]
        or key_retirement["full_canary_plan_sha256"]
        != fixture["full_canary_plan_sha256"]
        or key_retirement["public_key_id"] != key_contract["public_key_id"]
        or key_retirement["private_path"]
        != key_contract["private_source_path"]
        or key_retirement["public_path"] != key_contract["public_path"]
        or key_retirement["key_bootstrap_receipt_sha256"]
        != key_contract["key_bootstrap_receipt_sha256"]
        or Path(key_retirement["key_bootstrap_receipt_path"]).name
        != "bootstrap.json"
        or Path(key_retirement["key_bootstrap_receipt_path"]).parent.name
        != key_contract["public_key_id"]
        or Path(key_retirement["retirement_intent_path"]).name
        != "service_stop-intent.json"
        or Path(key_retirement["receipt_path"]).name
        != "service_stop-completion.json"
        or Path(key_retirement["retirement_intent_path"]).parent
        != Path(key_retirement["receipt_path"]).parent
        or key_retirement["private_absent"] is not True
        or key_retirement["public_absent"] is not True
        or key_retirement["both_pair_members_absent"] is not True
        or key_retirement["private_content_or_digest_recorded"] is not False
        or _positive_int(key_retirement["retired_at_unix"], code)
        < stop_proof["observed_at_unix"]
        or key_retirement["receipt_sha256"] != _digest(key_unsigned)
    ):
        _fail(code)
    _sha256(key_retirement["retirement_intent_sha256"], code)
    stop_proof_sha256 = _sha256(
        key_retirement["service_stop_proof_sha256"], code
    )
    if stop_proof_sha256 != credential_stop_proof_sha256:
        _fail(code)
    key_absence = _strict(
        payload["bitrix_receipt_key_absence"],
        fields=(
            "private_path",
            "private_absent",
            "public_path",
            "public_absent",
            "both_pair_members_absent",
        ),
        code=code,
    )
    if key_absence != {
        "private_path": key_contract["private_source_path"],
        "private_absent": True,
        "public_path": key_contract["public_path"],
        "public_absent": True,
        "both_pair_members_absent": True,
    }:
        _fail(code)
    production_diff_sha256 = _sha256(payload["production_diff_sha256"], code)
    native_bindings = receipt.get("native_evidence", {}).get("bindings", [])
    production_diff_bindings = [
        item
        for item in native_bindings
        if isinstance(item, Mapping)
        and item.get("kind") == "production_diff_observation"
    ]
    if (
        len(production_diff_bindings) != 1
        or production_diff_bindings[0].get("artifact_sha256")
        != production_diff_sha256
    ):
        _fail(code)


def _validate_cleanup_finalization(
    value: Any,
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
    producer_readiness_sha256: str,
    cleanup_receipt: Mapping[str, Any],
) -> int:
    """Validate the root proof written after the observer signer stops."""

    code = "cleanup_finalization_invalid"
    finalization = _strict(
        value,
        fields=(
            "schema",
            "release_sha",
            "capability_plan_sha256",
            "full_canary_plan_sha256",
            "fixture_sha256",
            "run_id",
            "cleanup_receipt_sha256",
            "observer_stop_receipt",
            "service_stop_proof",
            "producer_fleet_retirement",
            "producer_activation_absent",
            "credentials_absent",
            "bitrix_receipt_key_pair_absent",
            "full_canary_stopped_preflight_sha256",
            "finalized_at_unix_ms",
            "finalization_sha256",
        ),
        code=code,
    )
    finalization_unsigned = {
        key: item
        for key, item in finalization.items()
        if key != "finalization_sha256"
    }
    finalized_at = _positive_int(finalization["finalized_at_unix_ms"], code)
    cleanup_observed_at = _positive_int(
        cleanup_receipt.get("payload", {}).get("observed_at_unix_ms"), code
    )
    if (
        finalization["schema"] != CLEANUP_FINALIZATION_SCHEMA
        or finalization["release_sha"] != fixture["release_sha"]
        or finalization["capability_plan_sha256"]
        != fixture["capability_plan_sha256"]
        or finalization["full_canary_plan_sha256"]
        != fixture["full_canary_plan_sha256"]
        or finalization["fixture_sha256"] != fixture_sha256
        or finalization["run_id"] != fixture["run_id"]
        or finalization["cleanup_receipt_sha256"]
        != _digest(cleanup_receipt)
        or finalization["producer_activation_absent"] is not True
        or finalization["credentials_absent"] is not True
        or finalization["bitrix_receipt_key_pair_absent"] is not True
        or not cleanup_observed_at <= finalized_at
        or finalization["finalization_sha256"]
        != _digest(finalization_unsigned)
    ):
        _fail(code)

    observer_stop = _strict(
        finalization["observer_stop_receipt"],
        fields=(
            "schema",
            "plan_sha256",
            "service_unit",
            "service_state_sha256",
            "stopped",
            "stopped_at_unix_ms",
            "secret_material_recorded",
            "receipt_sha256",
        ),
        code=code,
    )
    observer_stop_unsigned = {
        key: item
        for key, item in observer_stop.items()
        if key != "receipt_sha256"
    }
    observer_stopped_at = _positive_int(
        observer_stop["stopped_at_unix_ms"], code
    )
    if (
        observer_stop["schema"] != OBSERVER_STOP_RECEIPT_SCHEMA
        or observer_stop["plan_sha256"]
        != fixture["capability_plan_sha256"]
        or observer_stop["service_unit"]
        != OBSERVER_PRODUCER_SERVICE_UNIT
        or observer_stop["stopped"] is not True
        or observer_stop["secret_material_recorded"] is not False
        or observer_stop["receipt_sha256"]
        != _digest(observer_stop_unsigned)
        or not cleanup_observed_at <= observer_stopped_at <= finalized_at
    ):
        _fail(code)
    for field in ("plan_sha256", "service_state_sha256", "receipt_sha256"):
        _sha256(observer_stop[field], code)

    stop_proof = _strict(
        finalization["service_stop_proof"],
        fields=(
            "schema",
            "plan_sha256",
            "stop_order",
            "services_state_sha256",
            "all_services_stopped",
            "observed_at_unix",
            "secret_material_recorded",
            "secret_digest_recorded",
            "stop_proof_sha256",
        ),
        code=code,
    )
    stop_proof_unsigned = {
        key: item
        for key, item in stop_proof.items()
        if key != "stop_proof_sha256"
    }
    stop_observed_at = _nonnegative_int(stop_proof["observed_at_unix"], code)
    if (
        stop_proof["schema"] != CAPABILITY_SERVICE_STOP_PROOF_SCHEMA
        or stop_proof["plan_sha256"] != fixture["capability_plan_sha256"]
        or stop_proof["stop_order"] != list(SERVICE_UNITS)
        or stop_proof["all_services_stopped"] is not True
        or stop_proof["secret_material_recorded"] is not False
        or stop_proof["secret_digest_recorded"] is not False
        or stop_proof["stop_proof_sha256"]
        != _digest(stop_proof_unsigned)
        or stop_observed_at < observer_stopped_at // 1000
        or stop_observed_at > finalized_at // 1000
    ):
        _fail(code)
    for field in (
        "plan_sha256",
        "services_state_sha256",
        "stop_proof_sha256",
    ):
        _sha256(stop_proof[field], code)

    retirement = _strict(
        finalization["producer_fleet_retirement"],
        fields=(
            "schema",
            "readiness_sha256",
            "foundation_sha256",
            "release_sha",
            "capability_plan_sha256",
            "full_canary_plan_sha256",
            "fixture_sha256",
            "run_id",
            "path",
            "retired",
            "absence_verified",
            "retired_at_unix_ms",
            "receipt_sha256",
        ),
        code=code,
    )
    retirement_unsigned = {
        key: item
        for key, item in retirement.items()
        if key != "receipt_sha256"
    }
    retired_at = _positive_int(retirement["retired_at_unix_ms"], code)
    if (
        retirement["schema"] != PRODUCER_FLEET_RETIREMENT_SCHEMA
        or retirement["readiness_sha256"] != producer_readiness_sha256
        or retirement["foundation_sha256"]
        != fixture["producer_foundation_sha256"]
        or retirement["release_sha"] != fixture["release_sha"]
        or retirement["capability_plan_sha256"]
        != fixture["capability_plan_sha256"]
        or retirement["full_canary_plan_sha256"]
        != fixture["full_canary_plan_sha256"]
        or retirement["fixture_sha256"] != fixture_sha256
        or retirement["run_id"] != fixture["run_id"]
        or retirement["path"] != PRODUCER_ACTIVATION_PATH
        or retirement["retired"] is not True
        or retirement["absence_verified"] is not True
        or retirement["receipt_sha256"] != _digest(retirement_unsigned)
        or not observer_stopped_at <= retired_at <= finalized_at
    ):
        _fail(code)
    for field in (
        "readiness_sha256",
        "foundation_sha256",
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "fixture_sha256",
        "receipt_sha256",
    ):
        _sha256(retirement[field], code)
    _sha256(finalization["cleanup_receipt_sha256"], code)
    _sha256(finalization["full_canary_stopped_preflight_sha256"], code)
    _sha256(finalization["finalization_sha256"], code)
    return finalized_at


def verify_capability_canary(
    fixture: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    fixture_sha256: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    """Verify one externally collected production-shaped evidence bundle."""

    fixture_sha256 = _sha256(fixture_sha256, "fixture_invalid")
    evidence_sha256 = _sha256(evidence_sha256, "evidence_invalid")
    fixture_value = _validate_fixture(fixture, fixture_sha256)
    evidence_value = _strict(
        evidence,
        fields=(
            "schema",
            "execution_mode",
            "synthetic",
            "fixture_sha256",
            "release_sha",
            "release_artifact_sha256",
            "installed_wheel_manifest_sha256",
            "producer_readiness_sha256",
            "run_id",
            "started_at_unix_ms",
            "api_started_at_unix_ms",
            "api_completed_at_unix_ms",
            "completed_at_unix_ms",
            "runtime_receipt",
            "bundles",
            "cleanup_receipt",
            "cleanup_finalization",
        ),
        code="evidence_invalid",
    )
    if _digest(evidence_value) != evidence_sha256:
        _fail("evidence_digest_noncanonical")
    started = _positive_int(evidence_value["started_at_unix_ms"], "evidence_invalid")
    api_started = _positive_int(
        evidence_value["api_started_at_unix_ms"], "evidence_invalid"
    )
    api_completed = _positive_int(
        evidence_value["api_completed_at_unix_ms"], "evidence_invalid"
    )
    completed = _positive_int(
        evidence_value["completed_at_unix_ms"], "evidence_invalid"
    )
    if (
        evidence_value["schema"] != EVIDENCE_SCHEMA
        or evidence_value["execution_mode"] != "live_production_shaped_canary"
        or evidence_value["synthetic"] is not False
        or evidence_value["fixture_sha256"] != fixture_sha256
        or evidence_value["release_sha"] != fixture_value["release_sha"]
        or evidence_value["release_artifact_sha256"]
        != fixture_value["release_artifact_sha256"]
        or evidence_value["installed_wheel_manifest_sha256"]
        != fixture_value["installed_wheel_manifest_sha256"]
        or _SHA256_RE.fullmatch(
            str(evidence_value["producer_readiness_sha256"])
        )
        is None
        or evidence_value["run_id"] != fixture_value["run_id"]
        or not fixture_value["valid_from_unix_ms"]
        <= started
        <= api_started
        <= api_completed
        <= completed
        <= fixture_value["valid_until_unix_ms"]
    ):
        _fail("evidence_invalid")

    _validate_runtime(
        evidence_value["runtime_receipt"],
        fixture=fixture_value,
        fixture_sha256=fixture_sha256,
    )
    bundles = _strict(
        evidence_value["bundles"],
        fields=(
            "workspace_continuation",
            "capability_denials",
            "database_reconciliation",
            "bitrix_boundary",
            "discord_routeback",
            "failure_recovery",
        ),
        code="evidence_invalid",
    )
    receipt_times: list[int] = []
    owner_receipt = bundles["workspace_continuation"]["owner_approval_receipt"]
    all_receipts = _all_evidence_receipts(
        evidence_value["runtime_receipt"],
        bundles,
        evidence_value["cleanup_receipt"],
    )
    for raw_receipt in all_receipts:
        if not isinstance(raw_receipt, Mapping):
            _fail("evidence_invalid")
        raw_payload = raw_receipt.get("payload")
        if not isinstance(raw_payload, Mapping):
            _fail("evidence_invalid")
        observed = _positive_int(
            raw_payload.get("observed_at_unix_ms"), "evidence_invalid"
        )
        if raw_receipt is owner_receipt:
            if not fixture_value["valid_from_unix_ms"] <= observed <= api_started:
                _fail("evidence_invalid")
        elif not started <= observed <= completed:
            _fail("evidence_invalid")
        receipt_times.append(observed)
        if raw_receipt is not owner_receipt:
            native = raw_receipt.get("native_evidence")
            if (
                not isinstance(native, Mapping)
                or native.get("producer_readiness_sha256")
                != evidence_value["producer_readiness_sha256"]
            ):
                _fail("evidence_invalid")
    if evidence_value["cleanup_receipt"]["payload"]["observed_at_unix_ms"] != max(
        receipt_times
    ):
        _fail("evidence_invalid")
    _validate_task_workspace_bundle(
        bundles["workspace_continuation"],
        fixture=fixture_value,
        fixture_sha256=fixture_sha256,
    )
    _validate_denial_bundle(
        bundles["capability_denials"],
        fixture=fixture_value,
        fixture_sha256=fixture_sha256,
    )
    _validate_database_bundle(
        bundles["database_reconciliation"],
        fixture=fixture_value,
        fixture_sha256=fixture_sha256,
    )
    _validate_bitrix_bundle(
        bundles["bitrix_boundary"],
        fixture=fixture_value,
        fixture_sha256=fixture_sha256,
    )
    _validate_discord_bundle(
        bundles["discord_routeback"],
        fixture=fixture_value,
        fixture_sha256=fixture_sha256,
    )
    _validate_failure_bundle(
        bundles["failure_recovery"],
        fixture=fixture_value,
        fixture_sha256=fixture_sha256,
    )
    _validate_cleanup(
        evidence_value["cleanup_receipt"],
        fixture=fixture_value,
        fixture_sha256=fixture_sha256,
    )
    finalized_at = _validate_cleanup_finalization(
        evidence_value["cleanup_finalization"],
        fixture=fixture_value,
        fixture_sha256=fixture_sha256,
        producer_readiness_sha256=evidence_value[
            "producer_readiness_sha256"
        ],
        cleanup_receipt=evidence_value["cleanup_receipt"],
    )
    if completed < finalized_at:
        _fail("evidence_invalid")

    unsigned = {
        "schema": VERIFICATION_SCHEMA,
        "ok": True,
        "fixture_sha256": fixture_sha256,
        "evidence_sha256": evidence_sha256,
        "release_sha": fixture_value["release_sha"],
        "run_id": fixture_value["run_id"],
        "invariants": list(INVARIANTS),
    }
    return {**unsigned, "verification_receipt_sha256": _digest(unsigned)}


def verify_files(
    *,
    fixture_path: Path,
    fixture_sha256: str,
    evidence_path: Path,
    evidence_sha256: str,
) -> dict[str, Any]:
    fixture = _read_bound_artifact(
        fixture_path, fixture_sha256, "fixture_artifact_invalid"
    )
    evidence = _read_bound_artifact(
        evidence_path, evidence_sha256, "evidence_artifact_invalid"
    )
    return verify_capability_canary(
        fixture,
        evidence,
        fixture_sha256=fixture_sha256,
        evidence_sha256=evidence_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify signed production-shaped Muncho canary evidence"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--fixture", type=Path, required=True)
    verify.add_argument("--fixture-sha256", required=True)
    verify.add_argument("--evidence", type=Path, required=True)
    verify.add_argument("--evidence-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_files(
            fixture_path=args.fixture,
            fixture_sha256=args.fixture_sha256,
            evidence_path=args.evidence,
            evidence_sha256=args.evidence_sha256,
        )
    except CapabilityCanaryEvidenceError as exc:
        print(
            json.dumps(
                {"schema": VERIFICATION_SCHEMA, "ok": False, "failure_code": exc.code},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
