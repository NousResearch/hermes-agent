#!/usr/bin/env python3
"""Fixed, release-bound Caddy bridge and cutover for ``auth.lomliev.com``.

The production entrypoint has exactly five allow-listed phases and accepts no
paths, hosts, upstreams, commands, or configuration bytes from its caller.
The two bootstrap phases read one canonical, digest-self-bound bridge document
from standard input.  ``prepare-bridge`` snapshots the exact Caddyfile and
creates or adopts one legacy-v1 passkey request without changing ingress.
``activate-bridge`` temporarily stops the legacy verifier, atomically consumes
that one grant, installs only the exact v2 approval paths, restores the
verifier, and proves the legacy default route remains available.  Before the
database cutover intent, every bridge failure restores the exact Caddy bytes.

The fixed ``prepare`` and ``commit`` phases accept no input.  They re-read the
staged FreezePlan and CutoverPlan and the same durable v2 production-cutover
passkey claim.  ``prepare`` validates and journals the exact private-v2 and
maintenance candidates without changing live ingress.  ``commit`` is
unavailable until the legacy cutover journal contains both the irreversible
activation intent and its terminal receipt.  It atomically installs the
private-v2 candidate, reloads Caddy, and verifies the public readiness
endpoint.  Any failure after that irreversible boundary converges to a fixed
503 maintenance route; it never restores the v1 route.

The fixed ``converge`` phase is the crash-resumable production coordinator.
It makes the verified public 503 maintenance route durable before invoking
the irreversible writer cutover, permanently retires the exact legacy-v1
verifier only after that intent exists, and then commits private-v2 ingress.
"""

from __future__ import annotations

import argparse
import base64
import copy
import ctypes
import fcntl
import grp
import hashlib
import http.client
import json
import os
import pwd
import re
import ssl
import stat
import subprocess
import sys
import time
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from gateway import canonical_writer_activation as activation
from gateway import canonical_writer_production_cutover as cutover
from scripts.canary import owner_gate_production_ingress_observation as ingress


PREPARE_RECEIPT_SCHEMA = "muncho-owner-gate-caddy-cutover-prepare.v1"
BRIDGE_INPUT_SCHEMA = (
    "muncho-production-cutover-bridge-bootstrap-input.v2"
)
BRIDGE_REQUEST_SCHEMA = (
    "muncho-owner-gate-caddy-approval-bridge-request.v2"
)
BRIDGE_RECEIPT_SCHEMA = "muncho-owner-gate-caddy-approval-bridge.v2"
TERMINAL_RECEIPT_SCHEMA = "muncho-owner-gate-caddy-cutover-terminal.v1"
MAINTENANCE_OBSERVATION_SCHEMA = (
    "muncho-owner-gate-caddy-cutover-maintenance-observation.v1"
)
CADDY_PREPARED_DEPENDENCY_SCHEMA = (
    "muncho-production-caddy-prepared-dependency.v1"
)
CADDY_COMMIT_STARTED_SCHEMA = "muncho-owner-gate-caddy-commit-started.v1"
CADDY_POST_INTENT_FLOOR_SCHEMA = (
    "muncho-owner-gate-caddy-post-intent-maintenance-floor.v1"
)
LEGACY_RETIREMENT_SCHEMA = (
    "muncho-owner-gate-legacy-v1-retirement.v1"
)
LEGACY_RETIREMENT_INTENT_SCHEMA = (
    "muncho-owner-gate-legacy-v1-retirement-intent.v1"
)
CONVERGENCE_RECEIPT_SCHEMA = "muncho-owner-gate-production-convergence.v1"
JOURNAL_ENTRY_SCHEMA = "muncho-owner-gate-caddy-cutover-journal-entry.v2"
AUTHORITY_SCHEMA = "muncho-owner-gate-caddy-cutover-authority.v1"

CADDYFILE_PATH = Path("/etc/caddy/Caddyfile")
CADDY_EXECUTABLE = "/usr/bin/caddy"
SYSTEMCTL = "/usr/bin/systemctl"
CADDY_UNIT = "caddy.service"
PUBLIC_HOST = "auth.lomliev.com"
PUBLIC_PATH = "/readyz"
PRIVATE_V2_UPSTREAM = "10.80.3.2:8080"
MAINTENANCE_RESPONSE = b'respond "Service temporarily unavailable" 503'
PUBLIC_READY_CONTENT_TYPE = "application/json"
PUBLIC_READY_SCHEMA = "muncho-passkey-v2-readiness.v1"
PUBLIC_READY_BODY = (
    b'{"schema":"muncho-passkey-v2-readiness.v1","ok":true,'
    b'"service":"muncho-passkey-v2-web",'
    b'"authority_ready":true}'
)
PUBLIC_MAINTENANCE_CONTENT_TYPE = "text/plain; charset=utf-8"
PUBLIC_MAINTENANCE_BODY = b"Service temporarily unavailable"
BRIDGE_GET_MATCHER = "muncho_cutover_passkey_get"
BRIDGE_POST_MATCHER = "muncho_cutover_passkey_post"
BRIDGE_REQUEST_ID_TEMPLATE = "MUNCHO_V2_APPROVAL_REQUEST_ID"
_LEGACY_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{32}$")
_V2_REQUEST_ID = re.compile(r"^[0-9a-f]{64}$")
MINIMUM_V2_APPROVAL_MARGIN_SECONDS = 30
CADDY_JOURNAL_ROOT = cutover.EVIDENCE_ROOT / "caddy-cutover"
CADDY_LOCK = Path("/run/muncho-owner-gate-caddy-cutover.lock")
LEGACY_STEP_UP_ROOT = Path("/opt/adventico-ai-platform/hermes-home/state")
LEGACY_STEP_UP_REQUESTS = LEGACY_STEP_UP_ROOT / "step_up_requests"
LEGACY_STEP_UP_GRANTS = LEGACY_STEP_UP_ROOT / "step_up_verifications"
LEGACY_STEP_UP_LOCK = (
    LEGACY_STEP_UP_GRANTS / ".muncho-caddy-v2-bootstrap.lock"
)
LEGACY_STEP_UP_UID = 999
LEGACY_STEP_UP_GID = 994
LEGACY_STEP_UP_HELPER = Path(
    "/opt/adventico-ai-platform/hermes-home/bin/muncho_step_up_verify"
)
LEGACY_STEP_UP_HELPER_SHA256 = (
    "375982bce838bccfe112cdfb226668c01b347334ec9661a58f647185be03e31b"
)
LEGACY_STEP_UP_HELPER_SIZE = 26997
LEGACY_STEP_UP_UNIT = "muncho-passkey-stepup.service"
LEGACY_STEP_UP_FRAGMENT = Path(
    "/etc/systemd/system/muncho-passkey-stepup.service"
)


def _running_as_root() -> bool:
    """Treat a missing POSIX effective-uid API as non-root."""

    getter = getattr(os, "geteuid", None)
    return bool(callable(getter) and int(getter()) == 0)
LEGACY_STEP_UP_FRAGMENT_SHA256 = (
    "ab395d191e17c4b94cb19153338fd37a866d109dfec6b55373e3a1e7fb6dabc4"
)
LEGACY_STEP_UP_USER = "ai-platform-brain"
LEGACY_STEP_UP_GROUP = "ai-platform-brain"
RUNUSER = Path("/usr/sbin/runuser")
OWNER_DISCORD_USER_ID = "1279454038731264061"
LEGACY_CREDENTIAL_ID_SHA256 = (
    "63bbfca0778101d21dddf2b53cc774460565042391b918eb2d1c87b9d6d19860"
)
LEGACY_REQUEST_SCHEMA = "muncho.dangerous_action.request.v1"
LEGACY_GRANT_SCHEMA = "muncho.dangerous_action.grant.v1"
BRIDGE_APPROVAL_SCOPE = "runtime_config_mutation"
BRIDGE_CASE_ID = "case:muncho-caddy-v2-bootstrap-bridge"
BRIDGE_TARGET_SYSTEM = (
    "gce:adventico-ai-platform/europe-west3-a/"
    "ai-platform-runtime-01/caddy/auth.lomliev.com"
)
BRIDGE_ACTION_SUMMARY = (
    "Install the exact reversible path-only Muncho passkey-v2 approval bridge "
    "while preserving local-v1 default routing."
)
BRIDGE_ACTION_RISK = (
    "Mutates production Caddy routing only for one exact v2 approval request "
    "and its fixed asset; a malformed route could affect owner authentication."
)
BRIDGE_ACTION_ROLLBACK = (
    "Validate and reload the byte-exact previous Caddy configuration; before "
    "durable cutover intent automatically restore it on any failure."
)
MAX_CADDYFILE_BYTES = 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_CADDY_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_PUBLIC_BODY_BYTES = 16 * 1024
MAX_SYSTEMD_OUTPUT_BYTES = 1024 * 1024
LEGACY_SERVICE_READY_TIMEOUT_SECONDS = 15.0
LEGACY_SERVICE_READY_POLL_SECONDS = 0.1
V2_APPROVAL_JS_SHA256 = (
    "918397ec05b7492794b5ffa9fa2c499fd4e52f2a4113074491cc830a19c772ce"
)
V2_APPROVAL_HTML_SHA256 = (
    "c90ca5d440afc1ba2363d65f1ac5b617eaf44bacd4cd899c71e39b1dd2024e71"
)
V2_UI_SECURITY_HEADERS = {
    "cache-control": "no-store, max-age=0",
    "content-security-policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; img-src 'self'; form-action 'self'; "
        "frame-ancestors 'none'; base-uri 'none'"
    ),
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-resource-policy": "same-origin",
    "pragma": "no-cache",
    "referrer-policy": "no-referrer",
    "strict-transport-security": "max-age=31536000; includeSubDomains",
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_FIXED_ARTIFACT_NAMES = frozenset(
    {
        "original.Caddyfile",
        "approval-bridge.Caddyfile",
        "private-v2.Caddyfile",
        "maintenance.Caddyfile",
        "legacy-stepup.service",
    }
)
_LEGACY_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "request_id",
        "requester_discord_user_id",
        "approver_discord_user_id",
        "approval_scope",
        "case_id",
        "target_system",
        "action_summary",
        "risk",
        "rollback",
        "action_hash",
        "action_payload",
        "created_at",
        "expires_at",
        "expires_at_ts",
        "approved_methods",
        "approver_label",
    }
)
_LEGACY_GRANT_FIELDS = frozenset(
    {
        "schema",
        "grant_id",
        "request_id",
        "approved_by_discord_user_id",
        "approval_scope",
        "case_id",
        "action_hash",
        "granted_at",
        "expires_at",
        "expires_at_ts",
        "method",
        "single_use",
        "used_at",
        "used_at_ts",
        "approver_label",
        "credential_id_hash",
        "credential_sign_count",
        "credential_backed_up",
    }
)
_PREPARE_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "freeze_plan_sha256",
        "cutover_plan_sha256",
        "freeze_approval_sha256",
        "authority_sha256",
        "passkey_claim_entry_sha256",
        "passkey_claim_recorded_at_unix",
        "passkey_authorization_receipt_sha256",
        "passkey_action_envelope_sha256",
        "passkey_request_id",
        "passkey_consume_attempt_id",
        "bridge_request_receipt_sha256",
        "bridge_receipt_sha256",
        "approval_bridge_caddy_sha256",
        "source_route",
        "target_route",
        "candidate_validated",
        "maintenance_validated",
        "live_config_mutated",
        "rollback_mode",
        "caller_selected_input_accepted",
        "secret_material_recorded",
        "secret_digest_recorded",
        "prepared_at_unix",
        "receipt_sha256",
    }
)
_BRIDGE_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "freeze_plan_sha256",
        "freeze_approval_sha256",
        "freeze_publication_sha256",
        "v2_request_id",
        "v2_expires_at_unix",
        "v2_transaction_id",
        "v2_approval_url_sha256",
        "v2_action_payload_sha256",
        "bootstrap_input_sha256",
        "bridge_request_receipt_sha256",
        "legacy_passkey_request_id",
        "legacy_passkey_request_sha256",
        "legacy_passkey_grant_id",
        "legacy_passkey_grant_sha256",
        "legacy_passkey_consumed_grant_sha256",
        "legacy_passkey_consume_entry_sha256",
        "legacy_service_active_before_sha256",
        "legacy_service_inactive_sha256",
        "legacy_service_active_after_sha256",
        "legacy_service_local_health_sha256",
        "bridge_action_sha256",
        "route_contract_sha256",
        "original_caddy_sha256",
        "approval_bridge_caddy_sha256",
        "active_route_projection_sha256",
        "default_local_v1_route_preserved",
        "exact_v2_approval_routes_only",
        "caddy_validated",
        "caddy_reloaded",
        "caddy_readback_verified",
        "rollback_mode",
        "control_plane_mutation_performed",
        "source_data_mutation_performed",
        "production_host_mutation_performed",
        "caller_selected_input_accepted",
        "secret_material_recorded",
        "secret_digest_recorded",
        "activated_at_unix",
        "receipt_sha256",
    }
)
_BRIDGE_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "freeze_plan_sha256",
        "freeze_approval_sha256",
        "freeze_publication_sha256",
        "v2_request_id",
        "v2_expires_at_unix",
        "v2_transaction_id",
        "v2_approval_url_sha256",
        "v2_action_payload_sha256",
        "bootstrap_input_sha256",
        "legacy_passkey_request_id",
        "legacy_passkey_request_sha256",
        "legacy_approval_url",
        "bridge_action_sha256",
        "route_contract_sha256",
        "original_caddy_sha256",
        "approval_bridge_template_sha256",
        "approval_bridge_caddy_sha256",
        "default_local_v1_route_preserved",
        "control_plane_mutation_performed",
        "source_data_mutation_performed",
        "production_host_mutation_performed",
        "caller_selected_input_accepted",
        "secret_material_recorded",
        "secret_digest_recorded",
        "requested_at_unix",
        "receipt_sha256",
    }
)
_BRIDGE_INPUT_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "freeze_plan_sha256",
        "freeze_approval_sha256",
        "freeze_publication_sha256",
        "v2_request_id",
        "v2_expires_at_unix",
        "v2_transaction_id",
        "v2_approval_url_sha256",
        "v2_action_payload_sha256",
        "document_sha256",
    }
)
_TERMINAL_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "freeze_plan_sha256",
        "cutover_plan_sha256",
        "freeze_approval_sha256",
        "authority_sha256",
        "prepare_receipt_sha256",
        "passkey_claim_entry_sha256",
        "passkey_authorization_receipt_sha256",
        "passkey_request_id",
        "passkey_consume_attempt_id",
        "legacy_activation_commit_intent_receipt_sha256",
        "legacy_terminal_receipt_sha256",
        "outcome",
        "public_status",
        "active_route_projection_sha256",
        "caddy_validated",
        "caddy_reloaded",
        "public_verified",
        "v1_route_restored",
        "rollback_mode",
        "caller_selected_input_accepted",
        "secret_material_recorded",
        "secret_digest_recorded",
        "completed_at_unix",
        "receipt_sha256",
    }
)
_MAINTENANCE_OBSERVATION_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "freeze_plan_sha256",
        "cutover_plan_sha256",
        "freeze_approval_sha256",
        "authority_sha256",
        "prepare_receipt_sha256",
        "passkey_claim_entry_sha256",
        "passkey_authorization_receipt_sha256",
        "passkey_request_id",
        "passkey_consume_attempt_id",
        "legacy_activation_commit_intent_receipt_sha256",
        "legacy_terminal_receipt_sha256",
        "outcome",
        "public_status",
        "active_route_projection_sha256",
        "caddy_validated",
        "caddy_reloaded",
        "public_verified",
        "v1_route_restored",
        "forward_recovery_required",
        "rollback_mode",
        "caller_selected_input_accepted",
        "secret_material_recorded",
        "secret_digest_recorded",
        "observed_at_unix",
        "receipt_sha256",
    }
)
_CADDY_PREPARED_DEPENDENCY_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "freeze_plan_sha256",
        "cutover_plan_sha256",
        "freeze_approval_sha256",
        "authority_sha256",
        "caddy_prepare_receipt_sha256",
        "original_caddy_sha256",
        "approval_bridge_caddy_sha256",
        "private_v2_caddy_sha256",
        "maintenance_caddy_sha256",
        "maintenance_caddy_b64",
        "production_mutation_performed",
        "caller_selected_input_accepted",
        "secret_material_recorded",
        "secret_digest_recorded",
        "prepared_at_unix",
        "receipt_sha256",
    }
)
_MAINTENANCE_ARM_FIELDS = frozenset({
    "schema",
    "release_revision",
    "freeze_plan_sha256",
    "cutover_plan_sha256",
    "freeze_approval_sha256",
    "authority_sha256",
    "caddy_prepare_receipt_sha256",
    "legacy_service_active_sha256",
    "maintenance_caddy_sha256",
    "active_route_projection_sha256",
    "public_status",
    "caddy_validated",
    "caddy_reloaded",
    "public_verified",
    "v1_public_route_closed",
    "rollback_mode",
    "production_mutation_performed",
    "caller_selected_input_accepted",
    "secret_material_recorded",
    "secret_digest_recorded",
    "armed_at_unix",
    "receipt_sha256",
})
_PRE_INTENT_RESTORE_INTENT_FIELDS = frozenset({
    "schema",
    "release_revision",
    "freeze_plan_sha256",
    "cutover_plan_sha256",
    "freeze_approval_sha256",
    "authority_sha256",
    "caddy_prepare_receipt_sha256",
    "maintenance_arm_receipt_sha256",
    "original_caddy_sha256",
    "maintenance_caddy_sha256",
    "active_route_projection_sha256",
    "public_status",
    "public_verified",
    "v1_public_route_closed",
    "exact_original_artifact_available",
    "forward_apply_invalidated",
    "recovery_basis",
    "rollback_terminal_receipt_sha256",
    "rollback_mode",
    "production_mutation_performed",
    "caller_selected_input_accepted",
    "secret_material_recorded",
    "secret_digest_recorded",
    "restore_started_at_unix",
    "receipt_sha256",
})
_PRE_INTENT_RESTORE_FIELDS = frozenset({
    "schema",
    "release_revision",
    "freeze_plan_sha256",
    "cutover_plan_sha256",
    "freeze_approval_sha256",
    "authority_sha256",
    "caddy_prepare_receipt_sha256",
    "maintenance_arm_receipt_sha256",
    "restore_intent_receipt_sha256",
    "original_caddy_sha256",
    "active_route_projection_sha256",
    "exact_original_caddy_restored",
    "caddy_validated",
    "caddy_reloaded",
    "live_readback_verified",
    "v1_public_route_restored",
    "gateway_terminal_event",
    "gateway_terminal_receipt_sha256",
    "recovery_basis",
    "legacy_service_active_sha256",
    "legacy_service_health_sha256",
    "rollback_mode",
    "production_mutation_performed",
    "caller_selected_input_accepted",
    "secret_material_recorded",
    "secret_digest_recorded",
    "restored_at_unix",
    "receipt_sha256",
})
_CADDY_COMMIT_STARTED_FIELDS = frozenset(
    {
        "schema",
        "cutover_plan_sha256",
        "authority_sha256",
        "prepare_receipt_sha256",
        "legacy_activation_commit_intent_receipt_sha256",
        "legacy_terminal_receipt_sha256",
        "rollback_mode",
        "started_at_unix",
        "receipt_sha256",
    }
)
_CADDY_POST_INTENT_FLOOR_FIELDS = frozenset(
    {
        "schema",
        "cutover_plan_sha256",
        "authority_sha256",
        "prepare_receipt_sha256",
        "legacy_activation_commit_intent_receipt_sha256",
        "rollback_mode",
        "observed_at_unix",
        "receipt_sha256",
    }
)
_LEGACY_RETIREMENT_FIELDS = frozenset({
    "schema",
    "release_revision",
    "freeze_plan_sha256",
    "cutover_plan_sha256",
    "authority_sha256",
    "maintenance_arm_receipt_sha256",
    "legacy_terminal_receipt_sha256",
    "legacy_service_active_before_sha256",
    "legacy_service_fragment_backup_sha256",
    "legacy_service_retired_sha256",
    "unit",
    "fragment_path",
    "fragment_masked",
    "service_inactive",
    "permanent",
    "v1_public_route_closed",
    "rollback_mode",
    "production_mutation_performed",
    "caller_selected_input_accepted",
    "secret_material_recorded",
    "secret_digest_recorded",
    "retired_at_unix",
    "receipt_sha256",
})
_LEGACY_RETIREMENT_INTENT_FIELDS = frozenset({
    "schema",
    "release_revision",
    "freeze_plan_sha256",
    "cutover_plan_sha256",
    "authority_sha256",
    "maintenance_arm_receipt_sha256",
    "legacy_terminal_receipt_sha256",
    "legacy_service_fragment_backup_sha256",
    "rollback_mode",
    "recorded_at_unix",
    "receipt_sha256",
})
_CONVERGENCE_RECEIPT_FIELDS = frozenset({
    "schema",
    "release_revision",
    "freeze_plan_sha256",
    "cutover_plan_sha256",
    "preflight_receipt_sha256",
    "caddy_prepare_receipt_sha256",
    "maintenance_arm_receipt_sha256",
    "cutover_terminal_receipt_sha256",
    "caddy_terminal_receipt_sha256",
    "caddy_outcome",
    "legacy_service_retirement_receipt_sha256",
    "control_plane_mutation_performed",
    "source_data_mutation_performed",
    "production_host_mutation_performed",
    "secret_material_recorded",
    "secret_digest_recorded",
    "receipt_sha256",
})
_BRIDGE_INTENT_FIELDS = frozenset(
    {
        "bootstrap_input_sha256",
        "bridge_request_receipt_sha256",
        "bridge_action_sha256",
        "legacy_passkey_request_id",
        "legacy_service_active_before_sha256",
        "legacy_service_active_before",
        "legacy_grants_root_path",
        "legacy_grants_root_device",
        "legacy_grants_root_inode",
        "legacy_grants_root_uid",
        "legacy_grants_root_gid",
        "temporary_service_stop_required",
        "exact_preimage_restore_required_before_cutover_intent",
    }
)
_LEGACY_TERMINAL_FIELDS = frozenset(
    {
        "schema",
        "plan_sha256",
        "freeze_plan_sha256",
        "freeze_approval_sha256",
        "approval_sha256",
        "final_tail_receipt_sha256",
        "capability_prerequisite_receipt_sha256",
        "capability_prerequisite_file_sha256",
        "isolated_canary_goal_continuation_terminal_sha256",
        "isolated_canary_workspace_gateway_receipt_sha256",
        "isolation_equivalence_projection_sha256",
        "zero_canonical_database_mutation_observed",
        "pre_db_zero_write_observation_sha256",
        "capability_topology_identity_sha256",
        "database_apply_receipt_sha256",
        "host_apply_receipt_sha256",
        "host_boot_commit_receipt_sha256",
        "activation_commit_intent_receipt_sha256",
        "database_postflight_receipt_sha256",
        "gateway_observation_sha256",
        "writer_observation_sha256",
        "connector_observation_sha256",
        "direct_discord_disabled",
        "discord_dm_allowed",
        "rollback_used",
        "secret_material_recorded",
        "completed_at_unix",
        "receipt_sha256",
    }
)


class OwnerGateCaddyCutoverError(RuntimeError):
    """Stable, secret-free fixed Caddy cutover failure."""


def _canonical(value: Any) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_json_invalid"
        ) from exc
    if not payload or len(payload) > MAX_JSON_BYTES:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_json_invalid")
    return payload


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _bridge_get_paths(request_id: str) -> tuple[str, ...]:
    return (
        f"/approve/{request_id}",
        f"/approve/{request_id}/view",
        f"/approve/{request_id}/options",
        "/static/approve.js",
    )


def _bridge_post_paths(request_id: str) -> tuple[str, ...]:
    return (f"/approve/{request_id}/verify",)


def _bridge_route_contract() -> Mapping[str, Any]:
    return {
        "schema": "muncho-caddy-v2-bootstrap-route-contract.v1",
        "public_host": PUBLIC_HOST,
        "tls_terminated_by_caddy": True,
        "private_upstream": PRIVATE_V2_UPSTREAM,
        "default_route": "unchanged_verified_local_v1",
        "request_id_substitution": {
            "template": BRIDGE_REQUEST_ID_TEMPLATE,
            "grammar": "[0-9a-f]{64}",
            "get_paths": list(_bridge_get_paths(BRIDGE_REQUEST_ID_TEMPLATE)),
            "post_paths": list(_bridge_post_paths(BRIDGE_REQUEST_ID_TEMPLATE)),
        },
        "forward_host_origin_cookie_csrf_content_type_unchanged": True,
        "forward_upstream_headers_unchanged": True,
        "health_and_readiness_paths_bridged": False,
        "redirect_or_rewrite_present": False,
        "caller_selected_input_accepted": False,
    }


def _bridge_action(
    *,
    foundation: "_BridgeFoundation",
    original_sha256: str,
    bridge_template_sha256: str,
    bridge_sha256: str,
) -> Mapping[str, Any]:
    route_contract = _bridge_route_contract()
    return {
        "schema": "muncho-caddy-v2-bootstrap-bridge-action.v1",
        "operation": "activate_exact_v2_passkey_approval_bridge",
        "release_revision": foundation.release_revision,
        "freeze_plan_sha256": foundation.freeze_plan_sha256,
        "freeze_approval_sha256": foundation.freeze_approval_sha256,
        "freeze_publication_sha256": foundation.freeze_publication_sha256,
        "v2_request_id": foundation.v2_request_id,
        "v2_expires_at_unix": foundation.v2_expires_at_unix,
        "v2_transaction_id": foundation.v2_transaction_id,
        "v2_approval_url_sha256": foundation.v2_approval_url_sha256,
        "v2_action_payload_sha256": foundation.v2_action_payload_sha256,
        "bootstrap_input_sha256": foundation.document_sha256,
        "original_caddy_sha256": original_sha256,
        "approval_bridge_template_sha256": bridge_template_sha256,
        "approval_bridge_caddy_sha256": bridge_sha256,
        "route_contract": route_contract,
        "route_contract_sha256": _sha256(_canonical(route_contract)),
        "private_upstream": PRIVATE_V2_UPSTREAM,
        "pre_migration_reversible": True,
        "default_local_v1_route_preserved": True,
        "caller_selected_input_accepted": False,
    }


def _hashed(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != fields:
        raise OwnerGateCaddyCutoverError(f"owner_gate_caddy_{label}_invalid")
    raw = copy.deepcopy(dict(value))
    digest = raw.get("receipt_sha256")
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    if not isinstance(digest, str) or digest != _sha256(_canonical(unsigned)):
        raise OwnerGateCaddyCutoverError(f"owner_gate_caddy_{label}_invalid")
    return raw


@dataclass(frozen=True)
class _Authority:
    freeze: cutover.FreezePlan
    plan: cutover.CutoverPlan
    approval_sha256: str
    claim_entry_sha256: str
    claim_recorded_at_unix: int
    claim: Mapping[str, Any]
    value: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        value = self.value.get("authority_sha256")
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_authority_invalid"
            )
        return value


@dataclass(frozen=True)
class _BridgeFoundation:
    release_revision: str
    freeze_plan_sha256: str
    freeze_approval_sha256: str
    freeze_publication_sha256: str
    v2_request_id: str
    v2_expires_at_unix: int
    v2_transaction_id: str
    v2_approval_url_sha256: str
    v2_action_payload_sha256: str
    document_sha256: str


@dataclass(frozen=True)
class _LegacyBridgeAuthorization:
    request: Mapping[str, Any]
    grant_before: Mapping[str, Any]
    request_sha256: str
    grant_before_sha256: str
    grant_after_sha256: str
    consume_entry_sha256: str
    bridge_action_sha256: str


@dataclass(frozen=True)
class _StableOwnedFile:
    path: Path
    raw: bytes
    identity: tuple[int, ...]


@dataclass(frozen=True)
class _PinnedLegacyHelper:
    raw: bytes
    identity: tuple[int, ...]


@dataclass(frozen=True)
class _StableConfig:
    raw: bytes
    identity: tuple[int, ...]


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int
    line: int
    depth: int


@dataclass(frozen=True)
class _DerivedConfigs:
    original: bytes
    approval_bridge: bytes
    private_v2: bytes
    maintenance: bytes


def validate_bridge_bootstrap_input(value: Any) -> _BridgeFoundation:
    """Validate the owner-workflow projection available before first write."""

    if not isinstance(value, Mapping) or frozenset(value) != _BRIDGE_INPUT_FIELDS:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_input_invalid"
        )
    raw = copy.deepcopy(dict(value))
    unsigned = {
        key: item for key, item in raw.items() if key != "document_sha256"
    }
    request_id = raw.get("v2_request_id")
    expected_url = f"https://{PUBLIC_HOST}/approve/{request_id}"
    if (
        raw.get("schema") != BRIDGE_INPUT_SCHEMA
        or _REVISION.fullmatch(str(raw.get("release_revision", ""))) is None
        or any(
            _SHA256.fullmatch(str(raw.get(field, ""))) is None
            for field in (
                "freeze_plan_sha256",
                "freeze_approval_sha256",
                "freeze_publication_sha256",
                "v2_transaction_id",
                "v2_approval_url_sha256",
                "v2_action_payload_sha256",
                "document_sha256",
            )
        )
        or not isinstance(request_id, str)
        or _V2_REQUEST_ID.fullmatch(request_id) is None
        or type(raw.get("v2_expires_at_unix")) is not int
        or raw["v2_expires_at_unix"] <= 0
        or raw["v2_approval_url_sha256"]
        != _sha256(expected_url.encode("ascii", errors="strict"))
        or raw["document_sha256"] != _sha256(_canonical(unsigned))
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_input_invalid"
        )
    return _BridgeFoundation(
        release_revision=raw["release_revision"],
        freeze_plan_sha256=raw["freeze_plan_sha256"],
        freeze_approval_sha256=raw["freeze_approval_sha256"],
        freeze_publication_sha256=raw["freeze_publication_sha256"],
        v2_request_id=request_id,
        v2_expires_at_unix=raw["v2_expires_at_unix"],
        v2_transaction_id=raw["v2_transaction_id"],
        v2_approval_url_sha256=raw["v2_approval_url_sha256"],
        v2_action_payload_sha256=raw["v2_action_payload_sha256"],
        document_sha256=raw["document_sha256"],
    )


def _require_v2_fresh(
    foundation: _BridgeFoundation,
    *,
    now_unix: int,
) -> None:
    if (
        type(now_unix) is not int
        or now_unix <= 0
        or now_unix + MINIMUM_V2_APPROVAL_MARGIN_SECONDS
        >= foundation.v2_expires_at_unix
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_v2_approval_window_stale"
        )


@dataclass(frozen=True)
class _StrictServerRouteProjection:
    listeners: tuple[tuple[str, str], ...]
    exact_host_route_count: int
    reverse_proxy_handler_count: int
    dials: tuple[str, ...]
    terminal_handler_count: int
    maintenance_handler_count: int
    forbidden_terminal_handler_count: int
    alternate_error_route_present: bool
    proxy_shapes: tuple[str, ...]


class CaddyBoundary(Protocol):
    def stable_read(self) -> _StableConfig: ...

    def validate_payload(self, payload: bytes, *, mode: str) -> Mapping[str, Any]: ...

    def replace(self, payload: bytes, *, expected: _StableConfig) -> None: ...

    def reload(self) -> None: ...

    def observe(self, *, mode: str) -> Mapping[str, Any]: ...

    def verify_public(self, *, expected_status: int) -> Mapping[str, Any]: ...

    def verify_bridge(self, *, request_id: str) -> Mapping[str, Any]: ...


class LegacyRequestBoundary(Protocol):
    def create_bridge_request(
        self,
        *,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class LegacyServiceBoundary(Protocol):
    def observe_active(self) -> Mapping[str, Any]: ...

    def stop_exact(self, expected: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def start_exact(self, expected: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def verify_local_v1(self) -> Mapping[str, Any]: ...


class LegacyRetirementBoundary(Protocol):
    def observe_active(self) -> Mapping[str, Any]: ...

    def verify_local_v1(self) -> Mapping[str, Any]: ...

    def snapshot_exact_fragment(
        self, expected: Mapping[str, Any]
    ) -> bytes: ...

    def retire_exact(self) -> Mapping[str, Any]: ...

    def observe_retired(self) -> Mapping[str, Any]: ...


def _lex_caddyfile(raw: bytes) -> tuple[_Token, ...]:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_CADDYFILE_BYTES:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_config_invalid")
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_config_invalid"
        ) from exc
    tokens: list[_Token] = []
    index = 0
    line = 1
    depth = 0
    while index < len(raw):
        byte = raw[index]
        if byte in b" \t\r":
            index += 1
            continue
        if byte == 0x0A:
            line += 1
            index += 1
            continue
        if byte == ord("#"):
            newline = raw.find(b"\n", index)
            index = len(raw) if newline < 0 else newline
            continue
        start = index
        token_line = line
        if byte in (ord("{"), ord("}")):
            text = chr(byte)
            token_depth = depth
            if text == "}":
                depth -= 1
                token_depth = depth
                if depth < 0:
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_config_invalid"
                    )
            else:
                depth += 1
            index += 1
            tokens.append(_Token(text, start, index, token_line, token_depth))
            continue
        if byte == ord('"'):
            index += 1
            escaped = False
            while index < len(raw):
                current = raw[index]
                if current == 0x0A:
                    line += 1
                if current == ord('"') and not escaped:
                    index += 1
                    break
                if current == ord("\\") and not escaped:
                    escaped = True
                else:
                    escaped = False
                index += 1
            else:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_config_invalid"
                )
        elif byte == ord("`"):
            closing = raw.find(b"`", index + 1)
            if closing < 0:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_config_invalid"
                )
            line += raw[index : closing + 1].count(b"\n")
            index = closing + 1
        else:
            while index < len(raw) and raw[index] not in b" \t\r\n{}#":
                index += 1
        try:
            text = raw[start:index].decode("utf-8", errors="strict")
        except UnicodeError as exc:  # pragma: no cover - whole-file check above.
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_config_invalid"
            ) from exc
        tokens.append(_Token(text, start, index, token_line, depth))
    if depth != 0:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_config_invalid")
    return tuple(tokens)


def _derive_configs(
    original: bytes,
    bridge_request_id: str = BRIDGE_REQUEST_ID_TEMPLATE,
) -> _DerivedConfigs:
    if (
        bridge_request_id != BRIDGE_REQUEST_ID_TEMPLATE
        and _V2_REQUEST_ID.fullmatch(bridge_request_id) is None
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_request_invalid"
        )
    tokens = _lex_caddyfile(original)
    site_candidates: list[tuple[int, int]] = []
    for index, token in enumerate(tokens[:-1]):
        if (
            token.depth == 0
            and token.text == PUBLIC_HOST
            and tokens[index + 1].text == "{"
            and tokens[index + 1].depth == 0
        ):
            site_candidates.append((index, index + 1))
    if len(site_candidates) != 1:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_route_invalid")
    _site_label, site_open_index = site_candidates[0]
    site_close_index = next(
        (
            index
            for index in range(site_open_index + 1, len(tokens))
            if tokens[index].text == "}" and tokens[index].depth == 0
        ),
        -1,
    )
    if site_close_index < 0:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_route_invalid")
    proxy_indexes = [
        index
        for index in range(site_open_index + 1, site_close_index)
        if tokens[index].text == "reverse_proxy"
    ]
    if len(proxy_indexes) != 1:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_route_invalid")
    proxy_index = proxy_indexes[0]
    proxy = tokens[proxy_index]
    if proxy.depth != 1 or proxy_index + 1 >= site_close_index:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_route_invalid")
    source = tokens[proxy_index + 1]
    if source.line != proxy.line or not ingress._dial_is_on_current_host(source.text):
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_source_route_invalid")
    following = tokens[proxy_index + 2] if proxy_index + 2 < site_close_index else None
    if following is not None and following.line == proxy.line and following.text != "{":
        # More than one upstream or an inline option would make a one-token
        # mutation ambiguous.
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_route_invalid")

    private = original[: source.start] + PRIVATE_V2_UPSTREAM.encode("ascii") + original[source.end :]

    line_start = original.rfind(b"\n", 0, proxy.start) + 1
    indent_end = line_start
    while indent_end < proxy.start and original[indent_end] in b" \t":
        indent_end += 1
    if indent_end != proxy.start:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_route_invalid")
    indent = original[line_start:proxy.start]
    bridge_lines = (
        indent + b"@" + BRIDGE_GET_MATCHER.encode("ascii") + b" {",
        indent + b"    method GET",
        indent
        + b"    path "
        + b" ".join(
            path.encode("ascii") for path in _bridge_get_paths(bridge_request_id)
        ),
        indent + b"}",
        indent
        + b"reverse_proxy @"
        + BRIDGE_GET_MATCHER.encode("ascii")
        + b" "
        + PRIVATE_V2_UPSTREAM.encode("ascii"),
        indent + b"@" + BRIDGE_POST_MATCHER.encode("ascii") + b" {",
        indent + b"    method POST",
        indent
        + b"    path "
        + b" ".join(
            path.encode("ascii") for path in _bridge_post_paths(bridge_request_id)
        ),
        indent + b"}",
        indent
        + b"reverse_proxy @"
        + BRIDGE_POST_MATCHER.encode("ascii")
        + b" "
        + PRIVATE_V2_UPSTREAM.encode("ascii"),
    )
    approval_bridge = (
        original[:line_start]
        + b"\n".join(bridge_lines)
        + b"\n"
        + original[line_start:]
    )
    directive_end: int
    if following is not None and following.line == proxy.line and following.text == "{":
        block_depth = following.depth
        closing_index = next(
            (
                index
                for index in range(proxy_index + 3, site_close_index)
                if tokens[index].text == "}" and tokens[index].depth == block_depth
            ),
            -1,
        )
        if closing_index < 0:
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_route_invalid")
        directive_end = tokens[closing_index].end
    else:
        newline = original.find(b"\n", source.end)
        directive_end = len(original) if newline < 0 else newline
    newline = original.find(b"\n", directive_end)
    if newline >= 0:
        directive_end = newline
    maintenance_directive = indent + MAINTENANCE_RESPONSE
    maintenance = (
        original[:line_start]
        + maintenance_directive
        + original[directive_end:]
    )
    if (
        approval_bridge == original
        or private == original
        or maintenance == original
    ):
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_route_invalid")
    return _DerivedConfigs(original, approval_bridge, private, maintenance)


def _proxy_match_shape(route: Mapping[str, Any]) -> str:
    matchers = route.get("match")
    if matchers in (None, []):
        return "default"
    if not isinstance(matchers, list) or len(matchers) != 1:
        return "other"
    matcher = matchers[0]
    if not isinstance(matcher, Mapping) or set(matcher) != {
        "method",
        "path",
    }:
        return "other"
    method = matcher.get("method")
    paths = matcher.get("path")
    if not isinstance(paths, list) or any(not isinstance(item, str) for item in paths):
        return "other"
    request_ids: set[str] = set()
    for path in paths:
        match = re.fullmatch(
            r"/approve/([0-9a-f]{64})(?:/(?:view|options|verify))?",
            path,
        )
        if match is not None:
            request_ids.add(match.group(1))
    if len(request_ids) != 1:
        return "other"
    request_id = next(iter(request_ids))
    if method == ["GET"] and paths == list(_bridge_get_paths(request_id)):
        return f"bridge_get={request_id}"
    if method == ["POST"] and paths == list(_bridge_post_paths(request_id)):
        return f"bridge_post={request_id}"
    return "other"


def _strict_analyze_routes(
    routes: Any,
    *,
    inherited_public_host_capability: bool = True,
    depth: int = 0,
) -> tuple[int, int, tuple[str, ...], int, int, int, tuple[str, ...]]:
    """Inventory every route that can handle the fixed public host.

    This deliberately does not stop after the first consuming route.  A
    second terminal handler may look unreachable in today's ordering but is
    still an alternate public behavior after a future matcher/order change.
    The cutover contract therefore proves one terminal for the whole exact
    host graph, not merely one terminal on the currently winning path.
    """

    if not isinstance(routes, list) or depth > 64:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_adapted_route_invalid"
        )
    exact_host_route_count = 0
    reverse_proxy_handler_count = 0
    dials: list[str] = []
    terminal_handler_count = 0
    maintenance_handler_count = 0
    forbidden_terminal_handler_count = 0
    proxy_shapes: list[str] = []
    for route in routes:
        if not isinstance(route, Mapping):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_adapted_route_invalid"
            )
        route_capability, exact_host, _covers_all = ingress._route_host_capability(
            route
        )
        capable = inherited_public_host_capability and route_capability
        group = route.get("group")
        terminal = route.get("terminal", False)
        if group is not None and (
            not isinstance(group, str)
            or not group
            or len(group) > 512
            or "\x00" in group
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_adapted_route_invalid"
            )
        if type(terminal) is not bool:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_adapted_route_invalid"
            )
        if capable and group is not None:
            # Group exclusivity is another alternate route-selection surface.
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_adapted_route_invalid"
            )
        if capable and exact_host:
            exact_host_route_count += 1
        handlers = route.get("handle", [])
        if not isinstance(handlers, list):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_adapted_route_invalid"
            )
        handler_chain_consumes = False
        for index, handler in enumerate(handlers):
            if not isinstance(handler, Mapping):
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_adapted_route_invalid"
                )
            handler_name = handler.get("handler")
            if not isinstance(handler_name, str) or not handler_name:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_adapted_route_invalid"
                )
            if not capable:
                continue
            if handler_name == "subroute":
                if index != len(handlers) - 1 or set(handler) != {
                    "handler",
                    "routes",
                }:
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_adapted_route_invalid"
                    )
                (
                    nested_exact,
                    nested_proxies,
                    nested_dials,
                    nested_terminals,
                    nested_maintenance,
                    nested_forbidden,
                    nested_proxy_shapes,
                ) = _strict_analyze_routes(
                    handler.get("routes"),
                    inherited_public_host_capability=True,
                    depth=depth + 1,
                )
                exact_host_route_count += nested_exact
                reverse_proxy_handler_count += nested_proxies
                dials.extend(nested_dials)
                terminal_handler_count += nested_terminals
                maintenance_handler_count += nested_maintenance
                forbidden_terminal_handler_count += nested_forbidden
                proxy_shapes.extend(nested_proxy_shapes)
                handler_chain_consumes = nested_terminals > 0
                continue
            if "routes" in handler:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_adapted_route_invalid"
                )
            if handler_name == "reverse_proxy":
                if index != len(handlers) - 1:
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_adapted_route_invalid"
                    )
                handler_dials = ingress._reverse_proxy_dials(handler)
                reverse_proxy_handler_count += 1
                dials.extend(handler_dials)
                target_shape = "other"
                if len(handler_dials) == 1:
                    if ingress._dial_is_on_current_host(handler_dials[0]):
                        target_shape = "local_v1"
                    elif ingress._dial_targets_private_v2(handler_dials[0]):
                        target_shape = "private_v2"
                proxy_shapes.append(
                    f"{_proxy_match_shape(route)}:{target_shape}"
                )
                terminal_handler_count += 1
                handler_chain_consumes = True
                continue
            if handler_name == "static_response":
                if index != len(handlers) - 1:
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_adapted_route_invalid"
                    )
                terminal_handler_count += 1
                if (
                    set(handler) == {"handler", "body", "status_code"}
                    and handler.get("body")
                    == MAINTENANCE_RESPONSE.split(b'"')[1].decode("ascii")
                    and handler.get("status_code") == 503
                ):
                    maintenance_handler_count += 1
                else:
                    forbidden_terminal_handler_count += 1
                handler_chain_consumes = True
                continue
            if handler_name == "file_server":
                if index != len(handlers) - 1:
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_adapted_route_invalid"
                    )
                terminal_handler_count += 1
                forbidden_terminal_handler_count += 1
                handler_chain_consumes = True
                continue
            if handler_name not in ingress._LOCAL_MIDDLEWARE_HANDLERS:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_adapted_route_invalid"
                )
        if capable and terminal and not handler_chain_consumes:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_adapted_route_invalid"
            )
    return (
        exact_host_route_count,
        reverse_proxy_handler_count,
        tuple(dials),
        terminal_handler_count,
        maintenance_handler_count,
        forbidden_terminal_handler_count,
        tuple(proxy_shapes),
    )


def _effective_route_projection(raw: bytes, *, mode: str) -> Mapping[str, Any]:
    try:
        value = ingress._decode_json(raw, canonical=False)
        apps = value.get("apps")
        if (
            not isinstance(apps, Mapping)
            or "http" not in apps
            or any(name not in {"http", "pki", "tls"} for name in apps)
        ):
            raise ValueError
        http = apps.get("http") if isinstance(apps, Mapping) else None
        servers = http.get("servers") if isinstance(http, Mapping) else None
        if not isinstance(servers, Mapping) or not servers:
            raise ValueError
        projections: list[_StrictServerRouteProjection] = []
        for server in servers.values():
            if not isinstance(server, Mapping) or server.get("listener_wrappers") not in (None, []):
                raise ValueError
            (
                exact,
                proxies,
                dials,
                terminals,
                maintenance,
                forbidden,
                proxy_shapes,
            ) = (
                _strict_analyze_routes(
                server.get("routes", [])
            )
            )
            errors = server.get("errors")
            projections.append(
                _StrictServerRouteProjection(
                    listeners=ingress._listener_projection(server.get("listen")),
                    exact_host_route_count=exact,
                    reverse_proxy_handler_count=proxies,
                    dials=dials,
                    terminal_handler_count=terminals,
                    maintenance_handler_count=maintenance,
                    forbidden_terminal_handler_count=forbidden,
                    alternate_error_route_present=errors not in (None, {}),
                    proxy_shapes=proxy_shapes,
                )
            )
    except (AttributeError, TypeError, ValueError, ingress.ProductionIngressObservationError) as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_adapted_route_invalid"
        ) from exc
    if sum(item.exact_host_route_count for item in projections) != 1:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_adapted_route_invalid")
    anchor = next(item for item in projections if item.exact_host_route_count)
    effective = [
        item
        for item in projections
        if ingress._listeners_may_overlap(anchor.listeners, item.listeners)
    ]
    proxy_count = sum(item.reverse_proxy_handler_count for item in effective)
    dials = tuple(dial for item in effective for dial in item.dials)
    terminal_count = sum(item.terminal_handler_count for item in effective)
    maintenance_count = sum(item.maintenance_handler_count for item in effective)
    forbidden_terminal_count = sum(
        item.forbidden_terminal_handler_count for item in effective
    )
    alternate_error_route_present = any(
        item.alternate_error_route_present for item in effective
    )
    proxy_shapes = tuple(
        shape for item in effective for shape in item.proxy_shapes
    )
    if mode == "legacy":
        valid = (
            proxy_count == 1
            and len(dials) == 1
            and terminal_count == 1
            and maintenance_count == 0
            and forbidden_terminal_count == 0
            and not alternate_error_route_present
            and proxy_shapes == ("default:local_v1",)
            and ingress._dial_is_on_current_host(dials[0])
        )
        still_local, private_active, maintenance_active = True, False, False
    elif mode == "private_v2":
        valid = (
            proxy_count == 1
            and len(dials) == 1
            and terminal_count == 1
            and maintenance_count == 0
            and forbidden_terminal_count == 0
            and not alternate_error_route_present
            and proxy_shapes == ("default:private_v2",)
            and ingress._dial_targets_private_v2(dials[0])
        )
        still_local, private_active, maintenance_active = False, True, False
    elif mode == "approval_bridge":
        bridge_get = next(
            (
                shape
                for shape in proxy_shapes[:2]
                if shape.startswith("bridge_get=")
            ),
            "",
        )
        bridge_post = next(
            (
                shape
                for shape in proxy_shapes[:2]
                if shape.startswith("bridge_post=")
            ),
            "",
        )
        bridge_get_id = (
            bridge_get.removeprefix("bridge_get=").removesuffix(":private_v2")
        )
        bridge_post_id = (
            bridge_post.removeprefix("bridge_post=").removesuffix(":private_v2")
        )
        valid = (
            proxy_count == 3
            and len(dials) == 3
            and terminal_count == 3
            and maintenance_count == 0
            and forbidden_terminal_count == 0
            and not alternate_error_route_present
            and len(proxy_shapes) == 3
            and bridge_get.endswith(":private_v2")
            and bridge_post.endswith(":private_v2")
            and _V2_REQUEST_ID.fullmatch(bridge_get_id) is not None
            and bridge_post_id == bridge_get_id
            and proxy_shapes[2] == "default:local_v1"
        )
        still_local, private_active, maintenance_active = True, False, False
    elif mode == "maintenance":
        valid = (
            proxy_count == 0
            and not dials
            and terminal_count == 1
            and maintenance_count == 1
            and forbidden_terminal_count == 0
            and not alternate_error_route_present
        )
        still_local, private_active, maintenance_active = False, False, True
    else:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_mode_invalid")
    if not valid:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_adapted_route_invalid")
    projection = {
        "auth_host_route_count": 1,
        "reverse_proxy_handler_count": proxy_count,
        "reverse_proxy_upstream_count": len(dials),
        "still_on_current_host": still_local,
        "private_v2_upstream_active": private_active,
        "maintenance_active": maintenance_active,
    }
    if mode == "approval_bridge":
        projection["bridge_request_id"] = bridge_get_id
    return {**projection, "projection_sha256": _sha256(_canonical(projection))}


def _authority_value(
    *,
    freeze: cutover.FreezePlan,
    plan: cutover.CutoverPlan,
    claim_entry: cutover.JournalEntry,
    claim: Mapping[str, Any],
) -> Mapping[str, Any]:
    fields = {
        "freeze_plan_sha256",
        "freeze_approval_sha256",
        "freeze_publication_sha256",
        "passkey_proof_sha256",
        "authorization_receipt_sha256",
        "action_envelope_sha256",
        "action_payload_sha256",
        "request_id",
        "consume_attempt_id",
        "authority_release_sha",
        "execution_window_expires_at_unix",
        "schema",
    }
    if (
        not isinstance(claim, Mapping)
        or set(claim) != fields
        or claim.get("schema")
        != getattr(cutover, "PASSKEY_CLAIM_SCHEMA", "muncho-production-cutover-passkey-claim.v1")
        or claim.get("freeze_plan_sha256") != freeze.sha256
        or claim.get("freeze_approval_sha256") != plan.value["freeze_approval_sha256"]
        or claim.get("authority_release_sha") != plan.value["release_revision"]
        or any(
            _SHA256.fullmatch(str(claim.get(name))) is None
            for name in (
                "freeze_publication_sha256",
                "passkey_proof_sha256",
                "authorization_receipt_sha256",
                "action_envelope_sha256",
                "action_payload_sha256",
            )
        )
        or not isinstance(claim.get("request_id"), str)
        or _V2_REQUEST_ID.fullmatch(claim["request_id"]) is None
        or _SHA256.fullmatch(str(claim.get("consume_attempt_id"))) is None
        or type(claim.get("execution_window_expires_at_unix")) is not int
        or claim["execution_window_expires_at_unix"] < claim_entry.value["recorded_at_unix"]
    ):
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_passkey_claim_invalid")
    unsigned = {
        "schema": AUTHORITY_SCHEMA,
        "release_revision": plan.value["release_revision"],
        "freeze_plan_sha256": freeze.sha256,
        "cutover_plan_sha256": plan.sha256,
        "freeze_approval_sha256": plan.value["freeze_approval_sha256"],
        "owner_subject_sha256": plan.value["owner_subject_sha256"],
        "owner_key_id": plan.value["owner_key_id"],
        "passkey_claim_entry_sha256": claim_entry.sha256,
        "passkey_claim_recorded_at_unix": claim_entry.value["recorded_at_unix"],
        "passkey_authorization_receipt_sha256": claim["authorization_receipt_sha256"],
        "passkey_action_envelope_sha256": claim["action_envelope_sha256"],
        "passkey_request_id": claim["request_id"],
        "passkey_consume_attempt_id": claim["consume_attempt_id"],
        "claim_before_any_caddy_write": True,
        "caller_selected_input_accepted": False,
    }
    return {**unsigned, "authority_sha256": _sha256(_canonical(unsigned))}


def _load_authority(
    *,
    now_unix: int,
    journal: cutover.CutoverJournal,
    staged_loader: Callable[[Path], Mapping[str, Any]],
) -> _Authority:
    freeze = cutover.FreezePlan.from_mapping(
        staged_loader(cutover.STAGED_FREEZE_PLAN_PATH)
    )
    plan = cutover.CutoverPlan.from_mapping(
        staged_loader(cutover.STAGED_CUTOVER_PLAN_PATH)
    )
    if (
        plan.value["freeze_plan"] != freeze.to_mapping()
        or plan.value["freeze_plan_sha256"] != freeze.sha256
        or plan.value["release_revision"] != freeze.value["release_revision"]
    ):
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_plan_binding_invalid")
    require_claim = getattr(cutover, "require_recorded_passkey_claim", None)
    if not callable(require_claim):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_passkey_contract_unavailable"
        )
    try:
        claim_entry, claim = require_claim(
            journal,
            plan_sha256=freeze.sha256,
            approval_sha256=plan.value["freeze_approval_sha256"],
            release_revision=plan.value["release_revision"],
        )
    except BaseException as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_passkey_claim_invalid"
        ) from exc
    if not isinstance(claim_entry, cutover.JournalEntry):
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_passkey_claim_invalid")
    try:
        approval = cutover.CutoverApproval.from_mapping(
            staged_loader(cutover.STAGED_FREEZE_APPROVAL_PATH),
            plan=freeze,
            now_unix=claim_entry.value["recorded_at_unix"],
        )
    except BaseException as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_freeze_approval_invalid"
        ) from exc
    if approval.sha256 != plan.value["freeze_approval_sha256"]:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_plan_binding_invalid")
    value = _authority_value(
        freeze=freeze,
        plan=plan,
        claim_entry=claim_entry,
        claim=claim,
    )
    if type(now_unix) is not int or now_unix <= 0:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_time_invalid")
    return _Authority(
        freeze=freeze,
        plan=plan,
        approval_sha256=approval.sha256,
        claim_entry_sha256=claim_entry.sha256,
        claim_recorded_at_unix=claim_entry.value["recorded_at_unix"],
        claim=copy.deepcopy(dict(claim)),
        value=value,
    )


class CaddyJournalEntry:
    def __init__(self, value: Mapping[str, Any]) -> None:
        fields = {
            "schema",
            "authority_plan_sha256",
            "sequence",
            "event",
            "previous_entry_sha256",
            "evidence",
            "recorded_at_unix",
            "entry_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_journal_invalid")
        unsigned = {key: item for key, item in value.items() if key != "entry_sha256"}
        if (
            value.get("schema") != JOURNAL_ENTRY_SCHEMA
            or _SHA256.fullmatch(str(value.get("authority_plan_sha256"))) is None
            or type(value.get("sequence")) is not int
            or value["sequence"] < 0
            or not isinstance(value.get("event"), str)
            or not isinstance(value.get("evidence"), Mapping)
            or type(value.get("recorded_at_unix")) is not int
            or value.get("entry_sha256") != _sha256(_canonical(unsigned))
        ):
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_journal_invalid")
        self.value = copy.deepcopy(dict(value))

    @property
    def sha256(self) -> str:
        return str(self.value["entry_sha256"])


class CaddyTransactionStore:
    """No-clobber private artifacts and an fsynced hash-chained journal."""

    def __init__(
        self,
        root: Path = CADDY_JOURNAL_ROOT,
        *,
        expected_uid: int = 0,
        expected_gid: int = 0,
    ) -> None:
        self.root = root
        self.expected_uid = expected_uid
        self.expected_gid = expected_gid

    def _plan_root(self, plan_sha256: str) -> Path:
        if _SHA256.fullmatch(plan_sha256 or "") is None:
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_journal_invalid")
        return self.root / "plans" / plan_sha256

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _ensure_directory(self, path: Path, *, mode: int = 0o700) -> None:
        if not path.is_absolute() or ".." in path.parts:
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_journal_invalid")
        ancestor = path
        while True:
            if os.path.lexists(ancestor):
                observed_ancestor = os.lstat(ancestor)
                if stat.S_ISLNK(observed_ancestor.st_mode) or not stat.S_ISDIR(
                    observed_ancestor.st_mode
                ):
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_journal_invalid"
                    )
            if ancestor == ancestor.parent:
                break
            ancestor = ancestor.parent
        missing: list[Path] = []
        current = path
        while not os.path.lexists(current):
            missing.append(current)
            current = current.parent
        for item in reversed(missing):
            os.mkdir(item, mode)
            os.chown(item, self.expected_uid, self.expected_gid)
            os.chmod(item, mode)
            self._fsync_directory(item.parent)
        observed = os.lstat(path)
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != self.expected_uid
            or observed.st_gid != self.expected_gid
            or stat.S_IMODE(observed.st_mode) != mode
        ):
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_journal_invalid")

    def _read_exact(self, path: Path, *, maximum: int) -> bytes:
        ancestor = path.parent
        while True:
            reached_ancestor = os.lstat(ancestor)
            if stat.S_ISLNK(reached_ancestor.st_mode) or not stat.S_ISDIR(
                reached_ancestor.st_mode
            ):
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_journal_invalid"
                )
            if ancestor == ancestor.parent:
                break
            ancestor = ancestor.parent
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            reached = os.lstat(path)
            if (
                stat.S_ISLNK(reached.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_uid != self.expected_uid
                or opened.st_gid != self.expected_gid
                or stat.S_IMODE(opened.st_mode) != 0o400
                or (opened.st_dev, opened.st_ino) != (reached.st_dev, reached.st_ino)
                or not 0 < opened.st_size <= maximum
            ):
                raise OwnerGateCaddyCutoverError("owner_gate_caddy_journal_invalid")
            raw = bytearray()
            while len(raw) <= maximum:
                chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(raw)))
                if not chunk:
                    break
                raw.extend(chunk)
            after = os.fstat(descriptor)
            if len(raw) != opened.st_size or (
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
                raise OwnerGateCaddyCutoverError("owner_gate_caddy_journal_invalid")
            return bytes(raw)
        finally:
            os.close(descriptor)

    def _install_no_clobber(self, path: Path, payload: bytes) -> None:
        if os.path.lexists(path):
            if self._read_exact(path, maximum=max(len(payload), 1)) != payload:
                raise OwnerGateCaddyCutoverError("owner_gate_caddy_journal_conflict")
            return
        self._ensure_directory(path.parent)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o400)
        try:
            os.fchown(descriptor, self.expected_uid, self.expected_gid)
            os.fchmod(descriptor, 0o400)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._fsync_directory(path.parent)

    def install_artifact(self, plan_sha256: str, name: str, payload: bytes) -> None:
        if name not in _FIXED_ARTIFACT_NAMES or not 0 < len(payload) <= MAX_CADDYFILE_BYTES:
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_artifact_invalid")
        self._install_no_clobber(self._plan_root(plan_sha256) / "private" / name, payload)

    def read_artifact(self, plan_sha256: str, name: str) -> bytes:
        if name not in _FIXED_ARTIFACT_NAMES:
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_artifact_invalid")
        return self._read_exact(
            self._plan_root(plan_sha256) / "private" / name,
            maximum=MAX_CADDYFILE_BYTES,
        )

    def load(self, plan_sha256: str) -> list[CaddyJournalEntry]:
        entries_root = self._plan_root(plan_sha256) / "entries"
        if not os.path.lexists(entries_root):
            return []
        self._ensure_directory(entries_root)
        result: list[CaddyJournalEntry] = []
        for expected, path in enumerate(sorted(entries_root.iterdir())):
            if path.name != f"{expected:06d}.json":
                raise OwnerGateCaddyCutoverError("owner_gate_caddy_journal_invalid")
            raw = self._read_exact(path, maximum=MAX_JSON_BYTES)
            try:
                value = json.loads(raw.decode("ascii", errors="strict"))
            except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_journal_invalid"
                ) from exc
            if not isinstance(value, Mapping) or raw != _canonical(value):
                raise OwnerGateCaddyCutoverError("owner_gate_caddy_journal_invalid")
            entry = CaddyJournalEntry(value)
            previous = None if not result else result[-1].sha256
            if (
                entry.value["sequence"] != expected
                or entry.value["authority_plan_sha256"] != plan_sha256
                or entry.value["previous_entry_sha256"] != previous
            ):
                raise OwnerGateCaddyCutoverError("owner_gate_caddy_journal_invalid")
            result.append(entry)
        return result

    def append(
        self,
        plan_sha256: str,
        event: str,
        evidence: Mapping[str, Any],
        now_unix: int,
    ) -> CaddyJournalEntry:
        if not isinstance(event, str) or not event or type(now_unix) is not int:
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_journal_invalid")
        entries = self.load(plan_sha256)
        unsigned = {
            "schema": JOURNAL_ENTRY_SCHEMA,
            "authority_plan_sha256": plan_sha256,
            "sequence": len(entries),
            "event": event,
            "previous_entry_sha256": None if not entries else entries[-1].sha256,
            "evidence": copy.deepcopy(dict(evidence)),
            "recorded_at_unix": now_unix,
        }
        value = {**unsigned, "entry_sha256": _sha256(_canonical(unsigned))}
        entry = CaddyJournalEntry(value)
        self._install_no_clobber(
            self._plan_root(plan_sha256) / "entries" / f"{len(entries):06d}.json",
            _canonical(entry.value),
        )
        return entry


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> Mapping[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate key")
        value[key] = item
    return value


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON value")


def _require_legacy_directory(
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int,
) -> tuple[int, ...]:
    try:
        item = os.lstat(path)
    except OSError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_store_invalid"
        ) from exc
    identity = (
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_dev,
        item.st_ino,
        item.st_nlink,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISDIR(item.st_mode)
        or item.st_uid != expected_uid
        or item.st_gid != expected_gid
        or stat.S_IMODE(item.st_mode) != 0o700
        or item.st_nlink < 2
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_store_invalid"
        )
    return identity


def _restore_fenced_legacy_grants_root(
    intent: Mapping[str, Any],
    *,
    grants_root: Path,
    expected_uid: int,
    expected_gid: int,
) -> tuple[int, ...]:
    """Recover only the exact directory fenced by this durable intent."""

    if (
        not grants_root.is_absolute()
        or ".." in grants_root.parts
        or intent.get("legacy_grants_root_path") != str(grants_root)
        or type(intent.get("legacy_grants_root_device")) is not int
        or type(intent.get("legacy_grants_root_inode")) is not int
        or type(intent.get("legacy_grants_root_uid")) is not int
        or type(intent.get("legacy_grants_root_gid")) is not int
        or type(expected_uid) is not int
        or type(expected_gid) is not int
        or intent.get("legacy_grants_root_inode", 0) <= 0
        or intent.get("legacy_grants_root_uid") != expected_uid
        or intent.get("legacy_grants_root_gid") != expected_gid
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_grant_fence_replay_invalid"
        )
    try:
        reached = os.lstat(grants_root)
    except OSError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_grant_fence_replay_invalid"
        ) from exc
    expected_identity = (
        intent["legacy_grants_root_device"],
        intent["legacy_grants_root_inode"],
        expected_uid,
        expected_gid,
    )
    if (
        stat.S_ISLNK(reached.st_mode)
        or not stat.S_ISDIR(reached.st_mode)
        or reached.st_nlink < 2
        or (
            reached.st_dev,
            reached.st_ino,
            reached.st_uid,
            reached.st_gid,
        )
        != expected_identity
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_grant_fence_replay_invalid"
        )
    mode = stat.S_IMODE(reached.st_mode)
    if mode == 0o700:
        return _require_legacy_directory(
            grants_root,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
    if mode != 0:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_grant_fence_replay_invalid"
        )

    descriptor: int | None = None
    try:
        if _running_as_root():
            descriptor = os.open(
                grants_root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_uid,
                opened.st_gid,
            ) != expected_identity or stat.S_IMODE(opened.st_mode) != 0:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_legacy_grant_fence_replay_invalid"
                )
            os.fchmod(descriptor, 0o700)
        else:
            # Production runs as root and restores through the already-open
            # directory descriptor.  A non-root test process cannot open a
            # mode-000 directory, so exercise the same exact-identity checks
            # around a no-follow path chmod.
            os.chmod(grants_root, 0o700, follow_symlinks=False)
            descriptor = os.open(
                grants_root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        os.fsync(descriptor)
        restored = os.fstat(descriptor)
        final = os.lstat(grants_root)
        if (
            (
                restored.st_dev,
                restored.st_ino,
                restored.st_uid,
                restored.st_gid,
            )
            != expected_identity
            or (
                final.st_dev,
                final.st_ino,
                final.st_uid,
                final.st_gid,
            )
            != expected_identity
            or stat.S_IMODE(restored.st_mode) != 0o700
            or stat.S_IMODE(final.st_mode) != 0o700
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_grant_fence_replay_invalid"
            )
    except OwnerGateCaddyCutoverError:
        raise
    except OSError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_grant_fence_replay_invalid"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return _require_legacy_directory(
        grants_root,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )


def _stable_legacy_file(
    path: Path,
    *,
    parent: Path,
    expected_uid: int,
    expected_gid: int,
) -> _StableOwnedFile:
    if path.parent != parent or not path.is_absolute() or ".." in path.parts:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_file_invalid"
        )
    parent_before = _require_legacy_directory(
        parent,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )
    descriptor: int | None = None
    try:
        reached = os.lstat(path)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(reached.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (reached.st_dev, reached.st_ino)
            != (opened.st_dev, opened.st_ino)
            or opened.st_uid != expected_uid
            or opened.st_gid != expected_gid
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_nlink != 1
            or not 0 < opened.st_size <= MAX_JSON_BYTES
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_passkey_file_invalid"
            )
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_legacy_passkey_file_changed"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        final = os.lstat(path)
        identity = (
            opened.st_mode,
            opened.st_uid,
            opened.st_gid,
            opened.st_dev,
            opened.st_ino,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if (
            identity
            != (
                after.st_mode,
                after.st_uid,
                after.st_gid,
                after.st_dev,
                after.st_ino,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (final.st_dev, final.st_ino)
            != (opened.st_dev, opened.st_ino)
            or _require_legacy_directory(
                parent,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
            )
            != parent_before
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_passkey_file_changed"
            )
        return _StableOwnedFile(path, b"".join(chunks), identity)
    except OwnerGateCaddyCutoverError:
        raise
    except OSError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_file_invalid"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _decode_legacy_json(snapshot: _StableOwnedFile) -> Mapping[str, Any]:
    try:
        value = json.loads(
            snapshot.raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_json_invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_json_invalid"
        )
    return copy.deepcopy(dict(value))


def _utc_timestamp(value: Any, *, label: str) -> int:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value)
        is None
    ):
        raise OwnerGateCaddyCutoverError(
            f"owner_gate_caddy_legacy_{label}_invalid"
        )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=UTC
        )
    except ValueError as exc:
        raise OwnerGateCaddyCutoverError(
            f"owner_gate_caddy_legacy_{label}_invalid"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise OwnerGateCaddyCutoverError(
            f"owner_gate_caddy_legacy_{label}_invalid"
        )
    timestamp = int(parsed.timestamp())
    if time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)) != value:
        raise OwnerGateCaddyCutoverError(
            f"owner_gate_caddy_legacy_{label}_invalid"
        )
    return timestamp


@contextmanager
def _legacy_passkey_lock(
    *,
    lock_path: Path = LEGACY_STEP_UP_LOCK,
    grants_root: Path = LEGACY_STEP_UP_GRANTS,
    expected_uid: int = LEGACY_STEP_UP_UID,
    expected_gid: int = LEGACY_STEP_UP_GID,
) -> Iterator[None]:
    if lock_path.parent != grants_root or lock_path.name != (
        ".muncho-caddy-v2-bootstrap.lock"
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_lock_invalid"
        )
    _require_legacy_directory(
        grants_root,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        opened = os.fstat(descriptor)
        reached = os.lstat(lock_path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(reached.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (reached.st_dev, reached.st_ino)
            or opened.st_nlink != 1
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_passkey_lock_invalid"
            )
        if (opened.st_uid, opened.st_gid) != (expected_uid, expected_gid):
            os.fchown(descriptor, expected_uid, expected_gid)
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    except OwnerGateCaddyCutoverError:
        raise
    except OSError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_lock_invalid"
        ) from exc
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _assert_no_open_legacy_grant_consumers(
    snapshot: _StableOwnedFile,
    *,
    proc_root: Path = Path("/proc"),
) -> None:
    """Prove no process retained the unused grant before it is claimed."""

    if not proc_root.is_dir():
        if sys.platform.startswith("linux"):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_grant_consumer_proof_unavailable"
            )
        return
    expected = (snapshot.identity[3], snapshot.identity[4])
    try:
        processes = sorted(
            item for item in proc_root.iterdir() if item.name.isdigit()
        )
    except OSError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_grant_consumer_proof_failed"
        ) from exc
    for process in processes:
        descriptors = process / "fd"
        try:
            names = list(descriptors.iterdir())
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_grant_consumer_proof_failed"
            ) from exc
        except OSError:
            continue
        for descriptor in names:
            try:
                observed = os.stat(descriptor)
            except (FileNotFoundError, PermissionError):
                continue
            except OSError:
                continue
            if (observed.st_dev, observed.st_ino) == expected:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_legacy_grant_consumer_present"
                )


@contextmanager
def _legacy_grant_consumer_fence(
    snapshot: _StableOwnedFile,
    *,
    grants_root: Path,
    expected_uid: int,
    expected_gid: int,
) -> Iterator[None]:
    """Fence uid-owned noncooperative consumers with directory permissions."""

    before = _require_legacy_directory(
        grants_root,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )
    descriptor: int | None = None
    locked = False
    try:
        descriptor = os.open(
            grants_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (before[3], before[4])
            or opened.st_uid != expected_uid
            or opened.st_gid != expected_gid
            or stat.S_IMODE(opened.st_mode) != 0o700
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_grant_fence_invalid"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_grant_fence_busy"
            ) from exc
        locked = True
        # Production execution is root.  Non-root unit tests still exercise
        # the same kernel-serialized claim protocol, while the deployed fixed
        # runtime additionally fences noncooperative uid-owned readers by
        # removing directory traversal permission.
        if not _running_as_root():
            yield
            return
        os.fchmod(descriptor, 0)
        os.fsync(descriptor)
        fenced = os.fstat(descriptor)
        reached = os.lstat(grants_root)
        if (
            (fenced.st_dev, fenced.st_ino) != (before[3], before[4])
            or (reached.st_dev, reached.st_ino) != (before[3], before[4])
            or fenced.st_uid != expected_uid
            or fenced.st_gid != expected_gid
            or stat.S_IMODE(fenced.st_mode) != 0
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_grant_fence_invalid"
            )
        _assert_no_open_legacy_grant_consumers(snapshot)
        yield
    except OwnerGateCaddyCutoverError:
        raise
    except OSError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_grant_fence_invalid"
        ) from exc
    finally:
        if descriptor is not None:
            try:
                if _running_as_root():
                    os.fchown(descriptor, expected_uid, expected_gid)
                    os.fchmod(descriptor, 0o700)
                    os.fsync(descriptor)
                    restored = os.fstat(descriptor)
                    reached = os.lstat(grants_root)
                    if (
                        (restored.st_dev, restored.st_ino)
                        != (before[3], before[4])
                        or (reached.st_dev, reached.st_ino)
                        != (before[3], before[4])
                        or restored.st_uid != expected_uid
                        or restored.st_gid != expected_gid
                        or stat.S_IMODE(restored.st_mode) != 0o700
                    ):
                        raise OwnerGateCaddyCutoverError(
                            "owner_gate_caddy_legacy_grant_fence_restore_failed"
                        )
            finally:
                if locked:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


def _read_claimed_legacy_grant(
    path: Path,
    *,
    grants_root: Path,
    expected_uid: int,
    expected_gid: int,
) -> _StableOwnedFile:
    canonical_name = path.name.removesuffix(".json")
    claim_match = re.fullmatch(
        r"\.([A-Za-z0-9_-]{32})\.muncho-caddy-claim", path.name
    )
    consume_match = re.fullmatch(
        r"\.([A-Za-z0-9_-]{32})\.muncho-caddy-consume", path.name
    )
    if (
        path.parent != grants_root
        or not (
            _LEGACY_REQUEST_ID.fullmatch(canonical_name) is not None
            or claim_match is not None
            or consume_match is not None
        )
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_grant_claim_invalid"
        )
    descriptor: int | None = None
    try:
        reached = os.lstat(path)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(reached.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (reached.st_dev, reached.st_ino)
            != (opened.st_dev, opened.st_ino)
            or opened.st_uid != expected_uid
            or opened.st_gid != expected_gid
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_nlink != 1
            or not 0 < opened.st_size <= MAX_JSON_BYTES
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_grant_claim_invalid"
            )
        raw = bytearray()
        while len(raw) < opened.st_size:
            chunk = os.read(descriptor, opened.st_size - len(raw))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        identity = (
            opened.st_mode,
            opened.st_uid,
            opened.st_gid,
            opened.st_dev,
            opened.st_ino,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if len(raw) != opened.st_size or identity != (
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_dev,
            after.st_ino,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_grant_claim_invalid"
            )
        return _StableOwnedFile(path, bytes(raw), identity)
    except OwnerGateCaddyCutoverError:
        raise
    except OSError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_grant_claim_invalid"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_consume_legacy_grant(
    snapshot: _StableOwnedFile,
    value: Mapping[str, Any],
    *,
    consumed_at_unix: int,
    grants_root: Path,
    expected_uid: int,
    expected_gid: int,
) -> tuple[Mapping[str, Any], str]:
    canonical_match = re.fullmatch(r"([A-Za-z0-9_-]{32})\.json", snapshot.path.name)
    claim_match = re.fullmatch(
        r"\.([A-Za-z0-9_-]{32})\.muncho-caddy-claim",
        snapshot.path.name,
    )
    request_id = (
        canonical_match.group(1)
        if canonical_match is not None
        else claim_match.group(1)
        if claim_match is not None
        else None
    )
    if request_id is None:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_grant_claim_invalid"
        )
    canonical = grants_root / f"{request_id}.json"
    temporary = grants_root / f".{request_id}.muncho-caddy-consume"
    claim = grants_root / f".{request_id}.muncho-caddy-claim"
    consumed = copy.deepcopy(dict(value))
    consumed["used_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(consumed_at_unix)
    )
    consumed["used_at_ts"] = consumed_at_unix
    payload = _canonical(consumed)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    parent_descriptor: int | None = None
    try:
        with _legacy_grant_consumer_fence(
            snapshot,
            grants_root=grants_root,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        ):
            already_claimed = snapshot.path == claim
            if os.path.lexists(claim) and not already_claimed:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_legacy_grant_claim_conflict"
                )
            fenced_snapshot = _read_claimed_legacy_grant(
                snapshot.path,
                grants_root=grants_root,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
            )
            if fenced_snapshot != snapshot:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_legacy_passkey_cas_failed"
                )
            if not already_claimed:
                os.rename(snapshot.path, claim)
            claimed = _read_claimed_legacy_grant(
                claim,
                grants_root=grants_root,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
            )
            if (
                claimed.raw != snapshot.raw
                or (claimed.identity[3], claimed.identity[4])
                != (snapshot.identity[3], snapshot.identity[4])
            ):
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_legacy_grant_claim_invalid"
                )
            # Only the process holding the kernel directory lock may create,
            # adopt, replace, or clean the deterministic staged consume file.
            # This avoids a losing concurrent consumer deleting the winner's
            # staged bytes, while keeping every crash state replayable.
            if os.path.lexists(temporary):
                staged = _read_claimed_legacy_grant(
                    temporary,
                    grants_root=grants_root,
                    expected_uid=expected_uid,
                    expected_gid=expected_gid,
                )
                staged_value = _decode_legacy_json(staged)
                staged_used_at = _utc_timestamp(
                    staged_value.get("used_at"),
                    label="passkey_grant_consume_time",
                )
                expected_unconsumed = copy.deepcopy(dict(staged_value))
                expected_unconsumed["used_at"] = None
                expected_unconsumed["used_at_ts"] = None
                if (
                    set(staged_value) != _LEGACY_GRANT_FIELDS
                    or staged_value.get("used_at_ts") != staged_used_at
                    or staged_used_at > consumed_at_unix
                    or expected_unconsumed != dict(value)
                    or staged.raw != _canonical(staged_value)
                ):
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_legacy_grant_staged_invalid"
                    )
                consumed = copy.deepcopy(dict(staged_value))
                payload = staged.raw
            else:
                descriptor = os.open(temporary, flags, 0o600)
                os.fchown(descriptor, expected_uid, expected_gid)
                os.fchmod(descriptor, 0o600)
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written <= 0:
                        raise OSError("write made no progress")
                    offset += written
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = None
            os.replace(temporary, canonical)
            parent_descriptor = os.open(
                grants_root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            os.fsync(parent_descriptor)
            installed = _read_claimed_legacy_grant(
                canonical,
                grants_root=grants_root,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
            )
            if installed.raw != payload:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_legacy_passkey_consume_unconfirmed"
                )
            # Complete the deterministic claim cleanup while the consumer
            # directory is still fenced.  A crash before this unlink leaves a
            # recoverable consumed canonical + claim pair; a crash after it
            # leaves the durable consumed canonical as the sole state.
            os.unlink(claim)
            os.fsync(parent_descriptor)
        confirmed = _stable_legacy_file(
            canonical,
            parent=grants_root,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
        if confirmed.raw != payload:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_passkey_consume_unconfirmed"
            )
        return consumed, _sha256(payload)
    except OwnerGateCaddyCutoverError:
        raise
    except OSError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_consume_failed"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        if os.path.lexists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _legacy_request_for_action(
    *,
    action: Mapping[str, Any],
    now_unix: int,
    requests_root: Path,
    expected_uid: int,
    expected_gid: int,
    allow_expired: bool = False,
) -> tuple[_StableOwnedFile, Mapping[str, Any]]:
    if type(now_unix) is not int or now_unix <= 0:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_time_invalid")
    _require_legacy_directory(
        requests_root,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )
    action_hash = _sha256(_canonical(action))
    candidates: list[tuple[_StableOwnedFile, Mapping[str, Any]]] = []
    try:
        names = sorted(os.listdir(requests_root))
    except OSError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_store_invalid"
        ) from exc
    for name in names:
        if not name.endswith(".json"):
            continue
        request_id = name.removesuffix(".json")
        if _LEGACY_REQUEST_ID.fullmatch(request_id) is None:
            continue
        snapshot = _stable_legacy_file(
            requests_root / name,
            parent=requests_root,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
        value = _decode_legacy_json(snapshot)
        if (
            value.get("action_hash") == action_hash
            and value.get("action_payload") == action
        ):
            candidates.append((snapshot, value))
    if not candidates:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_request_missing"
        )
    if len(candidates) != 1:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_request_ambiguous"
        )
    request_snapshot, request = candidates[0]
    request_id = request_snapshot.path.stem
    if set(request) != _LEGACY_REQUEST_FIELDS:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_request_invalid"
        )
    created_at = _utc_timestamp(
        request.get("created_at"), label="passkey_request_time"
    )
    expires_at = _utc_timestamp(
        request.get("expires_at"), label="passkey_request_time"
    )
    approved_methods = request.get("approved_methods")
    if (
        request.get("schema") != LEGACY_REQUEST_SCHEMA
        or request.get("request_id") != request_id
        or request.get("requester_discord_user_id") != OWNER_DISCORD_USER_ID
        or request.get("approver_discord_user_id") != OWNER_DISCORD_USER_ID
        or request.get("approval_scope") != BRIDGE_APPROVAL_SCOPE
        or request.get("case_id") != BRIDGE_CASE_ID
        or request.get("target_system") != BRIDGE_TARGET_SYSTEM
        or request.get("action_summary") != BRIDGE_ACTION_SUMMARY
        or request.get("risk") != BRIDGE_ACTION_RISK
        or request.get("rollback") != BRIDGE_ACTION_ROLLBACK
        or request.get("action_hash") != action_hash
        or request.get("action_payload") != action
        or type(request.get("expires_at_ts")) is not int
        or request["expires_at_ts"] != expires_at
        or not created_at < expires_at <= created_at + 900
        or (not allow_expired and not created_at <= now_unix < expires_at)
        or not isinstance(approved_methods, list)
        or not approved_methods
        or "passkey" not in approved_methods
        or any(
            not isinstance(method, str) or not method
            for method in approved_methods
        )
        or len(set(approved_methods)) != len(approved_methods)
        or not isinstance(request.get("approver_label"), str)
        or not request["approver_label"]
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_request_invalid"
        )
    return request_snapshot, request


def _legacy_request_and_grant(
    *,
    action: Mapping[str, Any],
    now_unix: int,
    requests_root: Path,
    grants_root: Path,
    expected_uid: int,
    expected_gid: int,
    allow_consumed: bool = False,
) -> tuple[
    _StableOwnedFile,
    Mapping[str, Any],
    _StableOwnedFile,
    Mapping[str, Any],
]:
    request_snapshot, request = _legacy_request_for_action(
        action=action,
        now_unix=now_unix,
        requests_root=requests_root,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        allow_expired=allow_consumed,
    )
    _require_legacy_directory(
        grants_root,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )
    action_hash = _sha256(_canonical(action))
    request_id = request_snapshot.path.stem
    created_at = _utc_timestamp(
        request.get("created_at"), label="passkey_request_time"
    )
    expires_at = _utc_timestamp(
        request.get("expires_at"), label="passkey_request_time"
    )
    canonical_grant_path = grants_root / f"{request_id}.json"
    claimed_grant_path = grants_root / (
        f".{request_id}.muncho-caddy-claim"
    )
    canonical_exists = os.path.lexists(canonical_grant_path)
    claim_exists = os.path.lexists(claimed_grant_path)
    if not canonical_exists and not claim_exists:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_grant_missing"
        )
    grant_snapshot = (
        _stable_legacy_file(
            canonical_grant_path,
            parent=grants_root,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
        if canonical_exists
        else _read_claimed_legacy_grant(
            claimed_grant_path,
            grants_root=grants_root,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
    )
    grant = _decode_legacy_json(grant_snapshot)
    if set(grant) != _LEGACY_GRANT_FIELDS:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_grant_invalid"
        )
    granted_at = _utc_timestamp(
        grant.get("granted_at"), label="passkey_grant_time"
    )
    grant_expires_at = _utc_timestamp(
        grant.get("expires_at"), label="passkey_grant_time"
    )
    unused = grant.get("used_at") is None and grant.get("used_at_ts") is None
    consumed = (
        allow_consumed
        and isinstance(grant.get("used_at"), str)
        and type(grant.get("used_at_ts")) is int
        and _utc_timestamp(
            grant.get("used_at"), label="passkey_grant_consume_time"
        )
        == grant.get("used_at_ts")
        and granted_at <= grant["used_at_ts"] <= now_unix
        and grant["used_at_ts"] < grant_expires_at
    )
    if (
        grant.get("schema") != LEGACY_GRANT_SCHEMA
        or not isinstance(grant.get("grant_id"), str)
        or re.fullmatch(r"[A-Za-z0-9_-]{16,128}", grant["grant_id"]) is None
        or grant.get("request_id") != request_id
        or grant.get("approved_by_discord_user_id") != OWNER_DISCORD_USER_ID
        or grant.get("approval_scope") != BRIDGE_APPROVAL_SCOPE
        or grant.get("case_id") != BRIDGE_CASE_ID
        or grant.get("action_hash") != action_hash
        or not created_at <= granted_at < grant_expires_at
        or granted_at > now_unix
        or grant_expires_at != expires_at
        or type(grant.get("expires_at_ts")) is not int
        or grant["expires_at_ts"] != grant_expires_at
        or (unused and not now_unix < grant_expires_at)
        or grant.get("method") != "passkey"
        or grant.get("single_use") is not True
        or not (unused or consumed)
        or grant.get("approver_label") != request["approver_label"]
        or grant.get("credential_id_hash") != LEGACY_CREDENTIAL_ID_SHA256
        or type(grant.get("credential_sign_count")) is not int
        or grant["credential_sign_count"] < 0
        or grant.get("credential_backed_up") is not True
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_passkey_grant_invalid"
        )
    if claim_exists and canonical_exists:
        claimed = _read_claimed_legacy_grant(
            claimed_grant_path,
            grants_root=grants_root,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
        if (
            not consumed
            or _sha256(claimed.raw) != _unconsumed_grant_sha256(grant)
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_grant_claim_conflict"
            )
    return request_snapshot, request, grant_snapshot, grant


@contextmanager
def _immutable_legacy_helper_copy(
    pinned: _PinnedLegacyHelper,
    *,
    root: Path = Path("/run"),
) -> Iterator[Path]:
    """Stage verified helper bytes in a root-owned, non-writable inode."""

    if (
        not isinstance(pinned, _PinnedLegacyHelper)
        or len(pinned.raw) != LEGACY_STEP_UP_HELPER_SIZE
        or _sha256(pinned.raw) != LEGACY_STEP_UP_HELPER_SHA256
        or not root.is_absolute()
        or ".." in root.parts
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_helper_identity_invalid"
        )
    path = root / f".muncho-caddy-legacy-helper.{os.getpid()}"
    descriptor: int | None = None
    directory_descriptor: int | None = None
    try:
        root_stat = os.lstat(root)
        if (
            stat.S_ISLNK(root_stat.st_mode)
            or not stat.S_ISDIR(root_stat.st_mode)
            or root_stat.st_uid != 0
            or root_stat.st_gid != 0
            or root_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_helper_copy_invalid"
            )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags, 0o500)
        os.fchown(descriptor, 0, 0)
        os.fchmod(descriptor, 0o555)
        offset = 0
        while offset < len(pinned.raw):
            written = os.write(descriptor, pinned.raw[offset:])
            if written <= 0:
                raise OSError("write made no progress")
            offset += written
        os.fsync(descriptor)
        installed = os.fstat(descriptor)
        reached = os.lstat(path)
        if (
            not stat.S_ISREG(installed.st_mode)
            or stat.S_ISLNK(reached.st_mode)
            or (installed.st_dev, installed.st_ino)
            != (reached.st_dev, reached.st_ino)
            or installed.st_uid != 0
            or installed.st_gid != 0
            or stat.S_IMODE(installed.st_mode) != 0o555
            or installed.st_nlink != 1
            or installed.st_size != len(pinned.raw)
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_helper_copy_invalid"
            )
        os.close(descriptor)
        descriptor = None
        directory_descriptor = os.open(
            root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        os.fsync(directory_descriptor)
        copied = _stable_legacy_helper_copy(path)
        if copied != pinned.raw:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_helper_copy_invalid"
            )
        yield path
        if _stable_legacy_helper_copy(path) != pinned.raw:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_helper_copy_changed"
            )
    except OwnerGateCaddyCutoverError:
        raise
    except OSError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_helper_copy_invalid"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        if os.path.lexists(path):
            try:
                os.unlink(path)
                root_fd = os.open(
                    root,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    os.fsync(root_fd)
                finally:
                    os.close(root_fd)
            except OSError:
                pass


def _stable_legacy_helper_copy(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        reached = os.lstat(path)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(reached.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (reached.st_dev, reached.st_ino)
            != (opened.st_dev, opened.st_ino)
            or opened.st_uid != 0
            or opened.st_gid != 0
            or stat.S_IMODE(opened.st_mode) != 0o555
            or opened.st_nlink != 1
            or opened.st_size != LEGACY_STEP_UP_HELPER_SIZE
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_helper_copy_invalid"
            )
        raw = bytearray()
        while len(raw) < opened.st_size:
            chunk = os.read(descriptor, opened.st_size - len(raw))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        if (
            len(raw) != opened.st_size
            or _sha256(bytes(raw)) != LEGACY_STEP_UP_HELPER_SHA256
            or (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_helper_copy_changed"
            )
        return bytes(raw)
    except OwnerGateCaddyCutoverError:
        raise
    except OSError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_helper_copy_invalid"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


class ProductionLegacyRequestBoundary:
    """Create only the fixed bridge request through the deployed v1 helper."""

    _OUTPUT_FIELDS = frozenset(
        {
            "ok",
            "status",
            "request_id",
            "case_id",
            "scope",
            "target_system",
            "action_hash",
            "action_hash_prefix",
            "approval_url",
            "approver_discord_user_id",
            "approver_label",
            "passkey_status",
            "totp_fallback_allowed",
            "instructions_bg",
        }
    )

    @staticmethod
    def _require_identity() -> _PinnedLegacyHelper:
        descriptor: int | None = None
        try:
            user = pwd.getpwnam(LEGACY_STEP_UP_USER)
            group = grp.getgrnam(LEGACY_STEP_UP_GROUP)
            runuser = os.lstat(RUNUSER)
            helper = os.lstat(LEGACY_STEP_UP_HELPER)
            descriptor = os.open(
                LEGACY_STEP_UP_HELPER,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
        except (KeyError, OSError) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_helper_identity_invalid"
            ) from exc
        if (
            user.pw_uid != LEGACY_STEP_UP_UID
            or user.pw_gid != LEGACY_STEP_UP_GID
            or group.gr_gid != LEGACY_STEP_UP_GID
            or stat.S_ISLNK(runuser.st_mode)
            or not stat.S_ISREG(runuser.st_mode)
            or runuser.st_uid != 0
            or runuser.st_gid != 0
            or runuser.st_nlink != 1
            or not runuser.st_mode & stat.S_IXUSR
            or runuser.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            or stat.S_ISLNK(helper.st_mode)
            or not stat.S_ISREG(helper.st_mode)
            or helper.st_uid != LEGACY_STEP_UP_UID
            or helper.st_gid != LEGACY_STEP_UP_GID
            or helper.st_nlink != 1
            or stat.S_IMODE(helper.st_mode) != 0o700
            or (helper.st_dev, helper.st_ino)
            != (opened.st_dev, opened.st_ino)
            or opened.st_size != LEGACY_STEP_UP_HELPER_SIZE
        ):
            os.close(descriptor)
            descriptor = None
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_helper_identity_invalid"
            )
        raw = bytearray()
        try:
            while len(raw) < opened.st_size:
                chunk = os.read(descriptor, opened.st_size - len(raw))
                if not chunk:
                    break
                raw.extend(chunk)
            after = os.fstat(descriptor)
        finally:
            if descriptor is not None:
                os.close(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if (
            len(raw) != opened.st_size
            or _sha256(bytes(raw)) != LEGACY_STEP_UP_HELPER_SHA256
            or identity
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_helper_identity_invalid"
            )
        return _PinnedLegacyHelper(bytes(raw), identity)

    def create_bridge_request(
        self,
        *,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        pinned_helper = self._require_identity()
        action_raw = _canonical(action)
        action_hash = _sha256(action_raw)
        try:
            with _immutable_legacy_helper_copy(pinned_helper) as helper_copy:
                argv = (
                    str(RUNUSER),
                    "--user",
                    LEGACY_STEP_UP_USER,
                    "--",
                    str(helper_copy),
                    "create-action-request",
                    "--requester-discord-user-id",
                    OWNER_DISCORD_USER_ID,
                    "--approver-discord-user-id",
                    OWNER_DISCORD_USER_ID,
                    "--scope",
                    BRIDGE_APPROVAL_SCOPE,
                    "--case-id",
                    BRIDGE_CASE_ID,
                    "--target-system",
                    BRIDGE_TARGET_SYSTEM,
                    "--action-summary",
                    BRIDGE_ACTION_SUMMARY,
                    "--risk",
                    BRIDGE_ACTION_RISK,
                    "--rollback",
                    BRIDGE_ACTION_ROLLBACK,
                    "--action-json",
                    action_raw.decode("ascii", errors="strict"),
                    "--action-hash",
                    action_hash,
                    "--ttl-seconds",
                    "900",
                    "--approval-base-url",
                    f"https://{PUBLIC_HOST}",
                )
                completed = subprocess.run(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={
                        "LC_ALL": "C.UTF-8",
                        "PATH": "/usr/bin:/bin",
                        "PYTHONNOUSERSITE": "1",
                    },
                    shell=False,
                    timeout=30,
                    check=False,
                )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_request_failed"
            ) from exc
        stdout = completed.stdout if isinstance(completed.stdout, bytes) else b""
        stderr = completed.stderr if isinstance(completed.stderr, bytes) else b""
        if (
            completed.returncode != 2
            or not stdout.endswith(b"\n")
            or b"\n" in stdout[:-1]
            or len(stdout) > MAX_JSON_BYTES
            or len(stderr) > MAX_JSON_BYTES
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_request_failed"
            )
        try:
            value = json.loads(
                stdout[:-1].decode("utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_request_output_invalid"
            ) from exc
        request_id = value.get("request_id") if isinstance(value, Mapping) else None
        prefix = value.get("action_hash_prefix") if isinstance(value, Mapping) else None
        if (
            not isinstance(value, Mapping)
            or frozenset(value) != self._OUTPUT_FIELDS
            or value.get("ok") is not False
            or value.get("status") != "DANGEROUS_ACTION_STEP_UP_REQUIRED"
            or not isinstance(request_id, str)
            or _LEGACY_REQUEST_ID.fullmatch(request_id) is None
            or value.get("case_id") != BRIDGE_CASE_ID
            or value.get("scope") != BRIDGE_APPROVAL_SCOPE
            or value.get("target_system") != BRIDGE_TARGET_SYSTEM
            or value.get("action_hash") != action_hash
            or not isinstance(prefix, str)
            or not prefix
            or len(prefix) > len(action_hash)
            or not action_hash.startswith(prefix)
            or value.get("approval_url")
            != f"https://{PUBLIC_HOST}/approve/{request_id}"
            or value.get("approver_discord_user_id") != OWNER_DISCORD_USER_ID
            or not isinstance(value.get("approver_label"), str)
            or not value["approver_label"]
            or value.get("passkey_status")
            != "PENDING_HTTPS_WEBAUTHN_SERVICE"
            or value.get("totp_fallback_allowed") is not True
            or not isinstance(value.get("instructions_bg"), str)
            or not value["instructions_bg"]
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_request_output_invalid"
            )
        return copy.deepcopy(dict(value))


class ProductionLegacyServiceBoundary:
    """Quiesce the exact v1 WebAuthn writer during grant consumption."""

    _UVICORN = (
        "/opt/adventico-ai-platform/hermes-home/services/passkey-stepup/"
        "venv/bin/uvicorn"
    )
    _PROCESS_PYTHON = (
        "/opt/adventico-ai-platform/hermes-home/services/passkey-stepup/"
        "venv/bin/python"
    )
    _CMDLINE = (
        _UVICORN,
        "muncho_passkey_service:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8787",
    )
    _PROCESS_CMDLINE = (_PROCESS_PYTHON, *_CMDLINE)
    _PROPERTIES = (
        "Id,LoadState,ActiveState,SubState,MainPID,ExecMainPID,"
        "UnitFileState,FragmentPath,DropInPaths,NeedDaemonReload,"
        "User,Group,ExecStart"
    )

    @staticmethod
    def _run(argv: tuple[str, ...], *, accepted: frozenset[int] = frozenset({0})) -> bytes:
        allowed = {
            (
                SYSTEMCTL,
                "show",
                LEGACY_STEP_UP_UNIT,
                f"--property={ProductionLegacyServiceBoundary._PROPERTIES}",
                "--no-pager",
            ),
            (SYSTEMCTL, "stop", "--", LEGACY_STEP_UP_UNIT),
            (SYSTEMCTL, "start", "--", LEGACY_STEP_UP_UNIT),
            (SYSTEMCTL, "daemon-reload"),
            (SYSTEMCTL, "is-enabled", "--", LEGACY_STEP_UP_UNIT),
            (SYSTEMCTL, "is-active", "--", LEGACY_STEP_UP_UNIT),
        }
        if argv not in allowed:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_command_invalid"
            )
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                    "PYTHONNOUSERSITE": "1",
                },
                shell=False,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_command_failed"
            ) from exc
        stdout = completed.stdout if isinstance(completed.stdout, bytes) else b""
        stderr = completed.stderr if isinstance(completed.stderr, bytes) else b""
        if (
            completed.returncode not in accepted
            or len(stdout) > MAX_SYSTEMD_OUTPUT_BYTES
            or len(stderr) > MAX_SYSTEMD_OUTPUT_BYTES
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_command_failed"
            )
        return stdout

    @staticmethod
    def _fragment_bytes() -> bytes:
        descriptor: int | None = None
        try:
            reached = os.lstat(LEGACY_STEP_UP_FRAGMENT)
            descriptor = os.open(
                LEGACY_STEP_UP_FRAGMENT,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if (
                stat.S_ISLNK(reached.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or (reached.st_dev, reached.st_ino)
                != (opened.st_dev, opened.st_ino)
                or opened.st_uid != 0
                or opened.st_gid != 0
                or opened.st_nlink != 1
                or stat.S_IMODE(opened.st_mode) != 0o644
                or not 0 < opened.st_size <= MAX_JSON_BYTES
            ):
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_legacy_service_identity_invalid"
                )
            raw = bytearray()
            while len(raw) < opened.st_size:
                chunk = os.read(descriptor, opened.st_size - len(raw))
                if not chunk:
                    break
                raw.extend(chunk)
            after = os.fstat(descriptor)
            if (
                len(raw) != opened.st_size
                or (opened.st_dev, opened.st_ino, opened.st_mtime_ns, opened.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns)
                or _sha256(bytes(raw)) != LEGACY_STEP_UP_FRAGMENT_SHA256
            ):
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_legacy_service_identity_invalid"
                )
            return bytes(raw)
        except OwnerGateCaddyCutoverError:
            raise
        except OSError as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_identity_invalid"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @classmethod
    def _fragment_sha256(cls) -> str:
        raw = cls._fragment_bytes()
        if _sha256(raw) != LEGACY_STEP_UP_FRAGMENT_SHA256:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_identity_invalid"
            )
        return LEGACY_STEP_UP_FRAGMENT_SHA256

    @classmethod
    def _show(cls) -> Mapping[str, str]:
        raw = cls._run(
            (
                SYSTEMCTL,
                "show",
                LEGACY_STEP_UP_UNIT,
                f"--property={cls._PROPERTIES}",
                "--no-pager",
            )
        )
        try:
            lines = raw.decode("utf-8", errors="strict").splitlines()
        except UnicodeError as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_identity_invalid"
            ) from exc
        values: dict[str, str] = {}
        for line in lines:
            key, separator, value = line.partition("=")
            if separator != "=" or key in values:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_legacy_service_identity_invalid"
                )
            values[key] = value
        expected = frozenset(cls._PROPERTIES.split(","))
        if frozenset(values) != expected:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_identity_invalid"
            )
        return values

    @classmethod
    def _normalized(cls, *, active: bool) -> Mapping[str, Any]:
        values = cls._show()
        try:
            main_pid = int(values["MainPID"])
            exec_main_pid = int(values["ExecMainPID"])
        except ValueError as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_identity_invalid"
            ) from exc
        exec_start = values["ExecStart"]
        expected_argv = " ".join(cls._CMDLINE)
        common = (
            values["Id"] == LEGACY_STEP_UP_UNIT
            and values["LoadState"] == "loaded"
            and values["UnitFileState"] == "enabled"
            and values["FragmentPath"] == str(LEGACY_STEP_UP_FRAGMENT)
            and values["DropInPaths"] == ""
            and values["NeedDaemonReload"] == "no"
            and values["User"] == LEGACY_STEP_UP_USER
            and values["Group"] == LEGACY_STEP_UP_GROUP
            and f"path={cls._UVICORN}" in exec_start
            and f"argv[]={expected_argv}" in exec_start
        )
        if not common or (
            active
            and not (
                values["ActiveState"] == "active"
                and values["SubState"] == "running"
                and main_pid > 1
                and exec_main_pid == main_pid
            )
        ) or (
            not active
            and not (
                values["ActiveState"] == "inactive"
                and values["SubState"] == "dead"
                and main_pid == 0
                and exec_main_pid == 0
            )
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_identity_invalid"
            )
        fragment_sha256 = cls._fragment_sha256()
        process: Mapping[str, Any] | None = None
        if active:
            process = cls._process(main_pid)
        unsigned = {
            "schema": "muncho-caddy-v1-service-observation.v1",
            "unit": LEGACY_STEP_UP_UNIT,
            "fragment_path": str(LEGACY_STEP_UP_FRAGMENT),
            "fragment_sha256": fragment_sha256,
            "unit_file_state": "enabled",
            "active_state": values["ActiveState"],
            "sub_state": values["SubState"],
            "main_pid": main_pid,
            "exec_start_path": cls._UVICORN,
            "exec_start_argv": list(cls._CMDLINE),
            "user": LEGACY_STEP_UP_USER,
            "group": LEGACY_STEP_UP_GROUP,
            "drop_in_paths": [],
            "need_daemon_reload": False,
            "process": process,
        }
        return {
            **unsigned,
            "projection_sha256": _sha256(_canonical(unsigned)),
        }

    @classmethod
    def _process(cls, pid: int) -> Mapping[str, Any]:
        proc = Path("/proc") / str(pid)
        try:
            status = (proc / "status").read_bytes()
            cmdline_raw = (proc / "cmdline").read_bytes()
            stat_raw = (proc / "stat").read_text(
                encoding="utf-8", errors="strict"
            )
            cgroup_raw = (proc / "cgroup").read_bytes()
        except (OSError, UnicodeError) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_process_invalid"
            ) from exc
        cmdline = tuple(
            item.decode("utf-8", errors="strict")
            for item in cmdline_raw.rstrip(b"\x00").split(b"\x00")
        )
        uid_match = re.search(rb"(?m)^Uid:\s+([0-9]+)\s", status)
        gid_match = re.search(rb"(?m)^Gid:\s+([0-9]+)\s", status)
        close = stat_raw.rfind(")")
        fields = stat_raw[close + 2 :].split() if close >= 0 else []
        if (
            cmdline != cls._PROCESS_CMDLINE
            or uid_match is None
            or int(uid_match.group(1)) != LEGACY_STEP_UP_UID
            or gid_match is None
            or int(gid_match.group(1)) != LEGACY_STEP_UP_GID
            or len(fields) < 20
            or not fields[19].isdigit()
            or f"/{LEGACY_STEP_UP_UNIT}".encode() not in cgroup_raw
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_process_invalid"
            )
        return {
            "pid": pid,
            "uid": LEGACY_STEP_UP_UID,
            "gid": LEGACY_STEP_UP_GID,
            "start_time_ticks": int(fields[19]),
            "cmdline_sha256": _sha256(cmdline_raw),
            "cgroup_sha256": _sha256(cgroup_raw),
        }

    def observe_active(self) -> Mapping[str, Any]:
        return self._normalized(active=True)

    def stop_exact(self, expected: Mapping[str, Any]) -> Mapping[str, Any]:
        before = self._normalized(active=True)
        if before != expected:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_changed"
            )
        self._run((SYSTEMCTL, "stop", "--", LEGACY_STEP_UP_UNIT))
        inactive = self._normalized(active=False)
        if (
            inactive["fragment_sha256"] != before["fragment_sha256"]
            or inactive["exec_start_argv"] != before["exec_start_argv"]
            or os.path.lexists(Path("/proc") / str(before["main_pid"]))
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_stop_unconfirmed"
            )
        return inactive

    def start_exact(self, expected: Mapping[str, Any]) -> Mapping[str, Any]:
        inactive = self._normalized(active=False)
        if (
            inactive["fragment_sha256"] != expected.get("fragment_sha256")
            or inactive["exec_start_argv"] != expected.get("exec_start_argv")
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_changed"
            )
        self._run((SYSTEMCTL, "start", "--", LEGACY_STEP_UP_UNIT))
        deadline = time.monotonic() + LEGACY_SERVICE_READY_TIMEOUT_SECONDS
        while True:
            try:
                active = self._normalized(active=True)
                break
            except OwnerGateCaddyCutoverError:
                if time.monotonic() >= deadline:
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_legacy_service_start_unconfirmed"
                    ) from None
                time.sleep(LEGACY_SERVICE_READY_POLL_SECONDS)
        if (
            active["fragment_sha256"] != expected.get("fragment_sha256")
            or active["exec_start_argv"] != expected.get("exec_start_argv")
            or active["main_pid"] == expected.get("main_pid")
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_start_unconfirmed"
            )
        return active

    def snapshot_exact_fragment(
        self, expected: Mapping[str, Any]
    ) -> bytes:
        if self.observe_active() != expected:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_changed"
            )
        raw = self._fragment_bytes()
        if _sha256(raw) != LEGACY_STEP_UP_FRAGMENT_SHA256:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_identity_invalid"
            )
        return raw

    @staticmethod
    def _masked_fragment() -> None:
        try:
            reached = os.lstat(LEGACY_STEP_UP_FRAGMENT)
            target = os.readlink(LEGACY_STEP_UP_FRAGMENT)
        except OSError as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_retirement_invalid"
            ) from exc
        if (
            not stat.S_ISLNK(reached.st_mode)
            or reached.st_uid != 0
            or reached.st_gid != 0
            or target != "/dev/null"
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_retirement_invalid"
            )

    @classmethod
    def _retired_observation(cls) -> Mapping[str, Any]:
        cls._masked_fragment()
        enabled = cls._run(
            (SYSTEMCTL, "is-enabled", "--", LEGACY_STEP_UP_UNIT),
            accepted=frozenset({1}),
        )
        active = cls._run(
            (SYSTEMCTL, "is-active", "--", LEGACY_STEP_UP_UNIT),
            accepted=frozenset({3}),
        )
        if enabled != b"masked\n" or active != b"inactive\n":
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_retirement_invalid"
            )
        unsigned = {
            "schema": "muncho-caddy-v1-service-retired.v1",
            "unit": LEGACY_STEP_UP_UNIT,
            "fragment_path": str(LEGACY_STEP_UP_FRAGMENT),
            "mask_target": "/dev/null",
            "unit_file_state": "masked",
            "active_state": "inactive",
            "permanent": True,
        }
        return {
            **unsigned,
            "projection_sha256": _sha256(_canonical(unsigned)),
        }

    def observe_retired(self) -> Mapping[str, Any]:
        return self._retired_observation()

    def retire_exact(self) -> Mapping[str, Any]:
        try:
            self._masked_fragment()
        except OwnerGateCaddyCutoverError:
            pass
        else:
            # The on-disk mask may already be visible to ``is-enabled`` while
            # PID 1 still has the old fragment cached.  Always reload before a
            # masked fast-path can mint durable retirement evidence.
            self._run((SYSTEMCTL, "daemon-reload"))
            return self._retired_observation()
        try:
            active = self.observe_active()
        except OwnerGateCaddyCutoverError:
            # A process may have died after the durable retirement intent but
            # before the unit path was masked.  Continue only if the exact
            # pinned unit remains loaded, enabled, and inactive.
            self._normalized(active=False)
        else:
            self.stop_exact(active)
        temporary = LEGACY_STEP_UP_FRAGMENT.with_name(
            f".{LEGACY_STEP_UP_FRAGMENT.name}.muncho-mask-{os.getpid()}"
        )
        if os.path.lexists(temporary):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_retirement_invalid"
            )
        descriptor: int | None = None
        try:
            os.symlink("/dev/null", temporary)
            reached = os.lstat(temporary)
            if (
                not stat.S_ISLNK(reached.st_mode)
                or reached.st_uid != 0
                or reached.st_gid != 0
                or os.readlink(temporary) != "/dev/null"
            ):
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_legacy_service_retirement_invalid"
                )
            os.replace(temporary, LEGACY_STEP_UP_FRAGMENT)
            descriptor = os.open(
                LEGACY_STEP_UP_FRAGMENT.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            os.fsync(descriptor)
            self._run((SYSTEMCTL, "daemon-reload"))
            return self._retired_observation()
        except OwnerGateCaddyCutoverError:
            raise
        except OSError as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_retirement_invalid"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if os.path.lexists(temporary):
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _verify_local_v1_once(self) -> Mapping[str, Any]:
        connection = http.client.HTTPConnection("127.0.0.1", 8787, timeout=5)
        try:
            connection.request(
                "GET",
                "/healthz",
                headers={
                    "Accept": "application/json",
                    "Connection": "close",
                    "Host": "auth.lomliev.com",
                },
            )
            response = connection.getresponse()
            body = response.read(MAX_PUBLIC_BODY_BYTES + 1)
            content_type = response.getheader("Content-Type", "")
            if (
                response.status != 200
                or not content_type.lower().startswith("application/json")
                or len(body) > MAX_PUBLIC_BODY_BYTES
            ):
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_legacy_service_health_invalid"
                )
            try:
                value = json.loads(
                    body.decode("utf-8", errors="strict"),
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_json_constant,
                )
            except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_legacy_service_health_invalid"
                ) from exc
            expected = {
                "ok": True,
                "service": "muncho-passkey-stepup",
                "rp_id": "lomliev.com",
                "origin": f"https://{PUBLIC_HOST}",
                "credentials_enabled": 1,
            }
            if value != expected or type(value.get("credentials_enabled")) is not int:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_legacy_service_health_invalid"
                )
            unsigned = {
                "schema": "muncho-caddy-v1-service-health.v1",
                "endpoint": "http://127.0.0.1:8787/healthz",
                "status": 200,
                "body": expected,
                "loopback_only": True,
            }
            return {
                **unsigned,
                "projection_sha256": _sha256(_canonical(unsigned)),
            }
        except OwnerGateCaddyCutoverError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_health_invalid"
            ) from exc
        finally:
            connection.close()

    def verify_local_v1(self) -> Mapping[str, Any]:
        deadline = time.monotonic() + LEGACY_SERVICE_READY_TIMEOUT_SECONDS
        while True:
            try:
                return self._verify_local_v1_once()
            except OwnerGateCaddyCutoverError as exc:
                if not isinstance(
                    exc.__cause__,
                    (OSError, http.client.HTTPException),
                ) or time.monotonic() >= deadline:
                    raise
                time.sleep(LEGACY_SERVICE_READY_POLL_SECONDS)


def _last(entries: Sequence[CaddyJournalEntry], event: str) -> CaddyJournalEntry | None:
    return next((item for item in reversed(entries) if item.value["event"] == event), None)


def validate_bridge_request_receipt(
    value: Any,
    *,
    foundation: _BridgeFoundation,
) -> Mapping[str, Any]:
    raw = _hashed(value, _BRIDGE_REQUEST_FIELDS, "bridge_request_receipt")
    expected_approval_url = (
        f"https://{PUBLIC_HOST}/approve/{raw.get('legacy_passkey_request_id')}"
    )
    if (
        raw["schema"] != BRIDGE_REQUEST_SCHEMA
        or raw["release_revision"] != foundation.release_revision
        or raw["freeze_plan_sha256"] != foundation.freeze_plan_sha256
        or raw["freeze_approval_sha256"] != foundation.freeze_approval_sha256
        or raw["freeze_publication_sha256"]
        != foundation.freeze_publication_sha256
        or raw["v2_request_id"] != foundation.v2_request_id
        or raw["v2_expires_at_unix"] != foundation.v2_expires_at_unix
        or raw["v2_transaction_id"] != foundation.v2_transaction_id
        or raw["v2_approval_url_sha256"]
        != foundation.v2_approval_url_sha256
        or raw["v2_action_payload_sha256"]
        != foundation.v2_action_payload_sha256
        or raw["bootstrap_input_sha256"] != foundation.document_sha256
        or _LEGACY_REQUEST_ID.fullmatch(
            str(raw["legacy_passkey_request_id"])
        )
        is None
        or _SHA256.fullmatch(str(raw["legacy_passkey_request_sha256"])) is None
        or raw["legacy_approval_url"] != expected_approval_url
        or any(
            _SHA256.fullmatch(str(raw[name])) is None
            for name in (
                "bridge_action_sha256",
                "route_contract_sha256",
                "original_caddy_sha256",
                "approval_bridge_template_sha256",
                "approval_bridge_caddy_sha256",
            )
        )
        or raw["default_local_v1_route_preserved"] is not True
        or raw["control_plane_mutation_performed"] is not True
        or raw["source_data_mutation_performed"] is not False
        or raw["production_host_mutation_performed"] is not True
        or raw["caller_selected_input_accepted"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or type(raw["requested_at_unix"]) is not int
        or raw["requested_at_unix"] <= 0
        or raw["requested_at_unix"] + MINIMUM_V2_APPROVAL_MARGIN_SECONDS
        >= raw["v2_expires_at_unix"]
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_request_receipt_invalid"
        )
    return raw


def validate_bridge_receipt(
    value: Any,
    *,
    foundation: _BridgeFoundation,
    request_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    requested = validate_bridge_request_receipt(
        request_receipt,
        foundation=foundation,
    )
    raw = _hashed(value, _BRIDGE_RECEIPT_FIELDS, "bridge_receipt")
    if (
        raw["schema"] != BRIDGE_RECEIPT_SCHEMA
        or raw["release_revision"] != foundation.release_revision
        or raw["freeze_plan_sha256"] != foundation.freeze_plan_sha256
        or raw["freeze_approval_sha256"] != foundation.freeze_approval_sha256
        or raw["freeze_publication_sha256"]
        != foundation.freeze_publication_sha256
        or raw["v2_request_id"] != foundation.v2_request_id
        or raw["v2_expires_at_unix"] != foundation.v2_expires_at_unix
        or raw["v2_transaction_id"] != foundation.v2_transaction_id
        or raw["v2_approval_url_sha256"]
        != foundation.v2_approval_url_sha256
        or raw["v2_action_payload_sha256"]
        != foundation.v2_action_payload_sha256
        or raw["bootstrap_input_sha256"] != foundation.document_sha256
        or raw["bridge_request_receipt_sha256"] != requested["receipt_sha256"]
        or raw["legacy_passkey_request_id"]
        != requested["legacy_passkey_request_id"]
        or raw["legacy_passkey_request_sha256"]
        != requested["legacy_passkey_request_sha256"]
        or raw["bridge_action_sha256"] != requested["bridge_action_sha256"]
        or raw["route_contract_sha256"] != requested["route_contract_sha256"]
        or raw["original_caddy_sha256"] != requested["original_caddy_sha256"]
        or raw["approval_bridge_caddy_sha256"]
        != requested["approval_bridge_caddy_sha256"]
        or not isinstance(raw["legacy_passkey_grant_id"], str)
        or re.fullmatch(
            r"[A-Za-z0-9_-]{16,128}", raw["legacy_passkey_grant_id"]
        )
        is None
        or any(
            _SHA256.fullmatch(str(raw[name])) is None
            for name in (
                "legacy_passkey_grant_sha256",
                "legacy_passkey_consumed_grant_sha256",
                "legacy_passkey_consume_entry_sha256",
                "legacy_service_active_before_sha256",
                "legacy_service_inactive_sha256",
                "legacy_service_active_after_sha256",
                "legacy_service_local_health_sha256",
                "active_route_projection_sha256",
            )
        )
        or raw["default_local_v1_route_preserved"] is not True
        or raw["exact_v2_approval_routes_only"] is not True
        or raw["caddy_validated"] is not True
        or raw["caddy_reloaded"] is not True
        or raw["caddy_readback_verified"] is not True
        or raw["rollback_mode"] != "pre_migration_exact_bytes"
        or raw["control_plane_mutation_performed"] is not True
        or raw["source_data_mutation_performed"] is not False
        or raw["production_host_mutation_performed"] is not True
        or raw["caller_selected_input_accepted"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or type(raw["activated_at_unix"]) is not int
        or raw["activated_at_unix"] <= 0
        or not requested["requested_at_unix"] <= raw["activated_at_unix"]
        or raw["activated_at_unix"] + MINIMUM_V2_APPROVAL_MARGIN_SECONDS
        >= raw["v2_expires_at_unix"]
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_receipt_invalid"
        )
    return raw


def validate_bridge_bootstrap_request(
    value: Any,
    *,
    document: Mapping[str, Any],
) -> Mapping[str, Any]:
    return validate_bridge_request_receipt(
        value,
        foundation=validate_bridge_bootstrap_input(document),
    )


def validate_bridge_bootstrap_receipt(
    value: Any,
    *,
    document: Mapping[str, Any],
    request_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    return validate_bridge_receipt(
        value,
        foundation=validate_bridge_bootstrap_input(document),
        request_receipt=request_receipt,
    )


def validate_prepare_receipt(
    value: Any,
    *,
    plan: cutover.CutoverPlan,
) -> Mapping[str, Any]:
    raw = _hashed(value, _PREPARE_FIELDS, "prepare_receipt")
    if (
        raw["schema"] != PREPARE_RECEIPT_SCHEMA
        or raw["release_revision"] != plan.value["release_revision"]
        or raw["freeze_plan_sha256"] != plan.value["freeze_plan_sha256"]
        or raw["cutover_plan_sha256"] != plan.sha256
        or raw["freeze_approval_sha256"] != plan.value["freeze_approval_sha256"]
        or any(
            _SHA256.fullmatch(str(raw[name])) is None
            for name in (
                "authority_sha256",
                "passkey_claim_entry_sha256",
                "passkey_authorization_receipt_sha256",
                "passkey_action_envelope_sha256",
            )
        )
        or type(raw["passkey_claim_recorded_at_unix"]) is not int
        or not isinstance(raw["passkey_request_id"], str)
        or not raw["passkey_request_id"]
        or not isinstance(raw["passkey_consume_attempt_id"], str)
        or not raw["passkey_consume_attempt_id"]
        or any(
            _SHA256.fullmatch(str(raw[name])) is None
            for name in (
                "bridge_request_receipt_sha256",
                "bridge_receipt_sha256",
                "approval_bridge_caddy_sha256",
            )
        )
        or raw["source_route"] != "exact_v2_approval_bridge"
        or raw["target_route"] != "fixed_private_v2"
        or raw["candidate_validated"] is not True
        or raw["maintenance_validated"] is not True
        or raw["live_config_mutated"] is not False
        or raw["rollback_mode"] != "pre_migration_exact_bytes"
        or raw["caller_selected_input_accepted"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or type(raw["prepared_at_unix"]) is not int
        or raw["prepared_at_unix"] <= 0
    ):
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_prepare_receipt_invalid")
    return raw


def validate_terminal_receipt(
    value: Any,
    *,
    plan: cutover.CutoverPlan,
    prepare_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    prepared = validate_prepare_receipt(prepare_receipt, plan=plan)
    raw = _hashed(value, _TERMINAL_FIELDS, "terminal_receipt")
    outcome = raw.get("outcome")
    expected_status = 200 if outcome == "private_v2_active" else 503
    if (
        raw["schema"] != TERMINAL_RECEIPT_SCHEMA
        or raw["release_revision"] != plan.value["release_revision"]
        or raw["freeze_plan_sha256"] != plan.value["freeze_plan_sha256"]
        or raw["cutover_plan_sha256"] != plan.sha256
        or raw["freeze_approval_sha256"] != plan.value["freeze_approval_sha256"]
        or raw["authority_sha256"] != prepared["authority_sha256"]
        or raw["prepare_receipt_sha256"] != prepared["receipt_sha256"]
        or raw["passkey_claim_entry_sha256"] != prepared["passkey_claim_entry_sha256"]
        or raw["passkey_authorization_receipt_sha256"]
        != prepared["passkey_authorization_receipt_sha256"]
        or raw["passkey_request_id"] != prepared["passkey_request_id"]
        or raw["passkey_consume_attempt_id"] != prepared["passkey_consume_attempt_id"]
        or any(
            _SHA256.fullmatch(str(raw[name])) is None
            for name in (
                "legacy_activation_commit_intent_receipt_sha256",
                "legacy_terminal_receipt_sha256",
                "active_route_projection_sha256",
            )
        )
        or outcome not in {"private_v2_active", "maintenance_active"}
        or raw["public_status"] != expected_status
        or raw["caddy_validated"] is not True
        or raw["caddy_reloaded"] is not True
        or raw["public_verified"] is not True
        or raw["v1_route_restored"] is not False
        or raw["rollback_mode"] != "post_migration_maintenance_only"
        or raw["caller_selected_input_accepted"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or type(raw["completed_at_unix"]) is not int
        or raw["completed_at_unix"] <= 0
    ):
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_terminal_receipt_invalid")
    return raw


def validate_maintenance_observation(
    value: Any,
    *,
    plan: cutover.CutoverPlan,
    prepare_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    prepared = validate_prepare_receipt(prepare_receipt, plan=plan)
    raw = _hashed(
        value,
        _MAINTENANCE_OBSERVATION_FIELDS,
        "maintenance_observation",
    )
    if (
        raw["schema"] != MAINTENANCE_OBSERVATION_SCHEMA
        or raw["release_revision"] != plan.value["release_revision"]
        or raw["freeze_plan_sha256"] != plan.value["freeze_plan_sha256"]
        or raw["cutover_plan_sha256"] != plan.sha256
        or raw["freeze_approval_sha256"] != plan.value["freeze_approval_sha256"]
        or raw["authority_sha256"] != prepared["authority_sha256"]
        or raw["prepare_receipt_sha256"] != prepared["receipt_sha256"]
        or raw["passkey_claim_entry_sha256"] != prepared["passkey_claim_entry_sha256"]
        or raw["passkey_authorization_receipt_sha256"]
        != prepared["passkey_authorization_receipt_sha256"]
        or raw["passkey_request_id"] != prepared["passkey_request_id"]
        or raw["passkey_consume_attempt_id"] != prepared["passkey_consume_attempt_id"]
        or _SHA256.fullmatch(
            str(raw["legacy_activation_commit_intent_receipt_sha256"])
        )
        is None
        or raw["legacy_terminal_receipt_sha256"] is not None
        or raw["outcome"] != "maintenance_active_forward_recovery_required"
        or raw["public_status"] != 503
        or _SHA256.fullmatch(str(raw["active_route_projection_sha256"])) is None
        or raw["caddy_validated"] is not True
        or raw["caddy_reloaded"] is not True
        or raw["public_verified"] is not True
        or raw["v1_route_restored"] is not False
        or raw["forward_recovery_required"] is not True
        or raw["rollback_mode"] != "post_migration_maintenance_only"
        or raw["caller_selected_input_accepted"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or type(raw["observed_at_unix"]) is not int
        or raw["observed_at_unix"] <= 0
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_maintenance_observation_invalid"
        )
    return raw


def _stable_config_path(path: Path) -> _StableConfig:
    """Read one root-owned Caddy config without following or changing it."""

    if path.parent != CADDYFILE_PATH.parent or not path.is_absolute():
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_config_invalid")
    descriptor: int | None = None
    try:
        reached = os.lstat(path)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(reached.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (reached.st_dev, reached.st_ino)
            != (opened.st_dev, opened.st_ino)
            or opened.st_uid != 0
            or opened.st_gid != 0
            or stat.S_IMODE(opened.st_mode) != ingress.CADDYFILE_MODE
            or opened.st_nlink != 1
            or not 0 < opened.st_size <= MAX_CADDYFILE_BYTES
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_config_invalid"
            )
        raw = bytearray()
        while len(raw) < opened.st_size:
            chunk = os.read(descriptor, opened.st_size - len(raw))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        final = os.lstat(path)
        identity = (
            opened.st_mode,
            opened.st_uid,
            opened.st_gid,
            opened.st_dev,
            opened.st_ino,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if (
            len(raw) != opened.st_size
            or identity
            != (
                after.st_mode,
                after.st_uid,
                after.st_gid,
                after.st_dev,
                after.st_ino,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (final.st_dev, final.st_ino)
            != (opened.st_dev, opened.st_ino)
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_config_changed"
            )
        return _StableConfig(bytes(raw), identity)
    except OwnerGateCaddyCutoverError:
        raise
    except OSError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_config_invalid"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_exchange_paths(left: Path, right: Path) -> None:
    """Atomically exchange two same-directory names using the host kernel."""

    if (
        left.parent != CADDYFILE_PATH.parent
        or right != CADDYFILE_PATH
        or left == right
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_atomic_exchange_invalid"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    left_raw = os.fsencode(left)
    right_raw = os.fsencode(right)
    result: int
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_atomic_exchange_unavailable"
            )
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, left_raw, -100, right_raw, 2)
    elif sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_atomic_exchange_unavailable"
            )
        renamex_np.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renamex_np.restype = ctypes.c_int
        result = renamex_np(left_raw, right_raw, 2)
    else:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_atomic_exchange_unavailable"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_atomic_exchange_failed"
        ) from OSError(error_number, os.strerror(error_number))


def _exchange_preimage_matches(
    observed: _StableConfig,
    expected: _StableConfig,
) -> bool:
    # rename(2) may advance ctime.  All other inode metadata, the inode itself,
    # and every byte must match the transaction's stable snapshot.
    return (
        observed.raw == expected.raw
        and observed.identity[:-1] == expected.identity[:-1]
    )


class ProductionCaddyBoundary:
    """Concrete fixed-path Caddy boundary; constructor accepts no coordinates."""

    @staticmethod
    def _run(argv: tuple[str, ...]) -> bytes:
        allowed = {
            (
                CADDY_EXECUTABLE,
                "adapt",
                "--config",
                str(CADDYFILE_PATH),
                "--adapter",
                "caddyfile",
            ),
            (SYSTEMCTL, "reload", "--", CADDY_UNIT),
        }
        if argv not in allowed:
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_command_invalid")
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                    "PYTHONNOUSERSITE": "1",
                },
                shell=False,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_command_failed"
            ) from exc
        if (
            completed.returncode != 0
            or not isinstance(completed.stdout, bytes)
            or len(completed.stdout) > MAX_CADDY_OUTPUT_BYTES
            or not isinstance(completed.stderr, bytes)
            or len(completed.stderr) > MAX_CADDY_OUTPUT_BYTES
        ):
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_command_failed")
        return completed.stdout

    @staticmethod
    def _write_temporary(payload: bytes, *, purpose: str) -> Path:
        if purpose not in {"validate", "replace"}:
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_temporary_invalid")
        path = CADDYFILE_PATH.parent / (
            f".{CADDYFILE_PATH.name}.muncho-{purpose}.{os.getpid()}"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
            try:
                os.fchown(descriptor, 0, 0)
                os.fchmod(descriptor, ingress.CADDYFILE_MODE)
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written <= 0:
                        raise OSError("write made no progress")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return path
        except BaseException:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise

    @staticmethod
    def _run_temporary(path: Path, *, operation: str) -> bytes:
        if path.parent != CADDYFILE_PATH.parent or path.name not in {
            f".{CADDYFILE_PATH.name}.muncho-validate.{os.getpid()}",
            f".{CADDYFILE_PATH.name}.muncho-replace.{os.getpid()}",
        }:
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_temporary_invalid")
        if operation not in {"validate", "adapt"}:
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_command_invalid")
        argv = (
            CADDY_EXECUTABLE,
            operation,
            "--config",
            str(path),
            "--adapter",
            "caddyfile",
        )
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                    "PYTHONNOUSERSITE": "1",
                },
                shell=False,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_command_failed"
            ) from exc
        if (
            completed.returncode != 0
            or not isinstance(completed.stdout, bytes)
            or len(completed.stdout) > MAX_CADDY_OUTPUT_BYTES
            or not isinstance(completed.stderr, bytes)
            or len(completed.stderr) > MAX_CADDY_OUTPUT_BYTES
        ):
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_command_failed")
        return completed.stdout

    def stable_read(self) -> _StableConfig:
        stable = ingress._stable_caddyfile()
        return _StableConfig(raw=stable.raw, identity=stable.identity)

    def validate_payload(self, payload: bytes, *, mode: str) -> Mapping[str, Any]:
        path = self._write_temporary(payload, purpose="validate")
        try:
            self._run_temporary(path, operation="validate")
            adapted = self._run_temporary(path, operation="adapt")
            return _effective_route_projection(adapted, mode=mode)
        finally:
            try:
                os.unlink(path)
            except OSError as exc:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_temporary_cleanup_failed"
                ) from exc

    def replace(self, payload: bytes, *, expected: _StableConfig) -> None:
        if (
            not isinstance(payload, bytes)
            or not 0 < len(payload) <= MAX_CADDYFILE_BYTES
            or not isinstance(expected, _StableConfig)
            or not isinstance(expected.raw, bytes)
            or not expected.identity
            or any(type(item) is not int for item in expected.identity)
        ):
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_config_invalid")
        path = self._write_temporary(payload, purpose="replace")
        descriptor: int | None = None
        preserve_captured_preimage = False
        try:
            candidate = _stable_config_path(path)
            if candidate.raw != payload:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_temporary_invalid"
                )
            # renameat2(RENAME_EXCHANGE) makes acquiring the live preimage and
            # installing the candidate one indivisible kernel operation.  The
            # previous live inode moves to ``path`` and is validated there.
            _atomic_exchange_paths(path, CADDYFILE_PATH)
            preserve_captured_preimage = True
            captured = _stable_config_path(path)
            if not _exchange_preimage_matches(captured, expected):
                live = self.stable_read()
                candidate_inode = (candidate.identity[3], candidate.identity[4])
                live_inode = (live.identity[3], live.identity[4])
                if live.raw == payload and live_inode == candidate_inode:
                    _atomic_exchange_paths(path, CADDYFILE_PATH)
                    restored = self.stable_read()
                    if (
                        restored.raw != captured.raw
                        or (restored.identity[3], restored.identity[4])
                        != (captured.identity[3], captured.identity[4])
                    ):
                        preserve_captured_preimage = True
                        raise OwnerGateCaddyCutoverError(
                            "owner_gate_caddy_cas_restore_unconfirmed"
                        )
                    preserve_captured_preimage = False
                else:
                    # A second writer replaced our candidate after the atomic
                    # exchange.  Never overwrite it and never delete the first
                    # captured third-party inode.
                    preserve_captured_preimage = True
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_compare_and_swap_failed"
                )
            descriptor = os.open(
                CADDYFILE_PATH.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            os.fsync(descriptor)
            installed = self.stable_read()
            if (
                installed.raw != payload
                or (installed.identity[3], installed.identity[4])
                != (candidate.identity[3], candidate.identity[4])
            ):
                preserve_captured_preimage = True
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_replace_unconfirmed"
                )
            preserve_captured_preimage = False
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if os.path.lexists(path) and not preserve_captured_preimage:
                try:
                    os.unlink(path)
                    parent = os.open(
                        CADDYFILE_PATH.parent,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                    )
                    try:
                        os.fsync(parent)
                    finally:
                        os.close(parent)
                except OSError:
                    pass

    def reload(self) -> None:
        self._run((SYSTEMCTL, "reload", "--", CADDY_UNIT))

    def observe(self, *, mode: str) -> Mapping[str, Any]:
        first_file = self.stable_read()
        first_service = ingress._caddy_service_projection()
        first_process = ingress._caddy_process_snapshot(first_service)
        adapted = self._run(
            (
                CADDY_EXECUTABLE,
                "adapt",
                "--config",
                str(CADDYFILE_PATH),
                "--adapter",
                "caddyfile",
            )
        )
        adapted_value = ingress._decode_json(adapted, canonical=False)
        adapted_projection = _effective_route_projection(adapted, mode=mode)
        live = ingress._read_live_caddy_config(first_process)
        live_value = ingress._decode_json(live, canonical=False)
        live_projection = _effective_route_projection(live, mode=mode)
        final_file = self.stable_read()
        final_service = ingress._caddy_service_projection()
        final_process = ingress._caddy_process_snapshot(final_service)
        if (
            first_file.identity != final_file.identity
            or first_file.raw != final_file.raw
            or ingress._canonical(adapted_value) != ingress._canonical(live_value)
            or adapted_projection != live_projection
            or first_service != final_service
            or first_process.identity != final_process.identity
            or not ingress._admin_config_matches_process(live_value, first_process)
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_live_observation_invalid"
            )
        return copy.deepcopy(dict(live_projection))

    def verify_public(self, *, expected_status: int) -> Mapping[str, Any]:
        if expected_status not in {200, 503}:
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_public_verify_invalid")
        connection = http.client.HTTPSConnection(
            PUBLIC_HOST,
            443,
            timeout=15,
            context=ssl.create_default_context(),
        )
        try:
            connection.request(
                "GET",
                PUBLIC_PATH,
                headers={
                    "Accept": "application/json",
                    "Connection": "close",
                    "Host": PUBLIC_HOST,
                },
            )
            response = connection.getresponse()
            encoding = response.getheader("Content-Encoding")
            content_type = response.getheader("Content-Type")
            content_length = response.getheader("Content-Length")
            body = response.read(MAX_PUBLIC_BODY_BYTES + 1)
            expected_body = (
                PUBLIC_READY_BODY
                if expected_status == 200
                else PUBLIC_MAINTENANCE_BODY
            )
            expected_content_type = (
                PUBLIC_READY_CONTENT_TYPE
                if expected_status == 200
                else PUBLIC_MAINTENANCE_CONTENT_TYPE
            )
            if (
                response.status != expected_status
                or encoding not in {None, "", "identity"}
                or len(body) > MAX_PUBLIC_BODY_BYTES
                or content_type != expected_content_type
                or content_length != str(len(expected_body))
                or body != expected_body
            ):
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_public_verify_failed"
                )
            authority_ready = False
            service = "muncho-owner-gate-maintenance"
            schema = "muncho-owner-gate-maintenance.v1"
            if expected_status == 200:
                try:
                    value = json.loads(
                        body.decode("utf-8", errors="strict"),
                        object_pairs_hook=_reject_duplicate_keys,
                        parse_constant=_reject_json_constant,
                    )
                except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_public_verify_failed"
                    ) from exc
                expected_value = {
                    "schema": PUBLIC_READY_SCHEMA,
                    "ok": True,
                    "service": "muncho-passkey-v2-web",
                    "authority_ready": True,
                }
                if value != expected_value:
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_public_verify_failed"
                    )
                authority_ready = True
                service = "muncho-passkey-v2-web"
                schema = PUBLIC_READY_SCHEMA
            return {
                "status": response.status,
                "body_size": len(body),
                "body_sha256": _sha256(body),
                "content_type": content_type,
                "schema": schema,
                "service": service,
                "authority_ready": authority_ready,
                "tls_verified": True,
            }
        except OwnerGateCaddyCutoverError:
            raise
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_public_verify_failed"
            ) from exc
        finally:
            connection.close()

    @staticmethod
    def _verify_v2_security_headers(response: http.client.HTTPResponse) -> None:
        observed = {
            key: response.getheader(key)
            for key in V2_UI_SECURITY_HEADERS
        }
        if observed != V2_UI_SECURITY_HEADERS:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_bridge_security_headers_invalid"
            )

    def verify_bridge(self, *, request_id: str) -> Mapping[str, Any]:
        if not isinstance(request_id, str) or _V2_REQUEST_ID.fullmatch(request_id) is None:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_bridge_verify_invalid"
            )

        def request(path: str) -> tuple[http.client.HTTPResponse, bytes]:
            connection = http.client.HTTPSConnection(
                PUBLIC_HOST,
                443,
                timeout=15,
                context=ssl.create_default_context(),
            )
            try:
                connection.request(
                    "GET",
                    path,
                    headers={
                        "Accept": "text/html,application/javascript",
                        "Connection": "close",
                        "Host": PUBLIC_HOST,
                    },
                )
                response = connection.getresponse()
                body = response.read(MAX_PUBLIC_BODY_BYTES + 1)
                if (
                    response.status != 200
                    or len(body) > MAX_PUBLIC_BODY_BYTES
                    or response.getheader("Content-Encoding")
                    not in {None, "", "identity"}
                ):
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_bridge_verify_failed"
                    )
                self._verify_v2_security_headers(response)
                return response, body
            except OwnerGateCaddyCutoverError:
                raise
            except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_bridge_verify_failed"
                ) from exc
            finally:
                connection.close()

        render, html = request(f"/approve/{request_id}")
        cookie = render.getheader("Set-Cookie", "")
        cookie_parts = [item.strip() for item in cookie.split(";") if item.strip()]
        content_type = render.getheader("Content-Type", "").lower()
        if (
            _sha256(html) != V2_APPROVAL_HTML_SHA256
            or not content_type.startswith("text/html")
            or len(cookie_parts) != 5
            or re.fullmatch(r"muncho_csrf=[A-Za-z0-9_-]{43}", cookie_parts[0])
            is None
            or frozenset(cookie_parts[1:])
            != frozenset(
                {
                    f"Path=/approve/{request_id}",
                    "Max-Age=300",
                    "SameSite=strict",
                    "Secure",
                }
            )
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_bridge_verify_failed"
            )
        javascript, body = request("/static/approve.js")
        if (
            _sha256(body) != V2_APPROVAL_JS_SHA256
            or not javascript.getheader("Content-Type", "").lower().startswith(
                "application/javascript"
            )
            or javascript.getheader("Set-Cookie") is not None
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_bridge_verify_failed"
            )
        unsigned = {
            "schema": "muncho-caddy-v2-approval-bridge-verification.v1",
            "public_host": PUBLIC_HOST,
            "request_id": request_id,
            "render_path": f"/approve/{request_id}",
            "javascript_path": "/static/approve.js",
            "render_sha256": V2_APPROVAL_HTML_SHA256,
            "javascript_sha256": V2_APPROVAL_JS_SHA256,
            "strict_security_headers_preserved": True,
            "tls_verified": True,
        }
        return {
            **unsigned,
            "projection_sha256": _sha256(_canonical(unsigned)),
        }


def _bridge_configs_and_action(
    foundation: _BridgeFoundation,
    original: bytes,
) -> tuple[_DerivedConfigs, str, Mapping[str, Any]]:
    template = _derive_configs(original)
    configs = _derive_configs(original, bridge_request_id=foundation.v2_request_id)
    action = _bridge_action(
        foundation=foundation,
        original_sha256=_sha256(original),
        bridge_template_sha256=_sha256(template.approval_bridge),
        bridge_sha256=_sha256(configs.approval_bridge),
    )
    return configs, _sha256(template.approval_bridge), action


def _bridge_request_unsigned(
    foundation: _BridgeFoundation,
    *,
    request_snapshot: _StableOwnedFile,
    request: Mapping[str, Any],
    action: Mapping[str, Any],
    configs: _DerivedConfigs,
    bridge_template_sha256: str,
    now_unix: int,
) -> Mapping[str, Any]:
    request_id = request_snapshot.path.stem
    route_contract = _bridge_route_contract()
    return {
        "schema": BRIDGE_REQUEST_SCHEMA,
        "release_revision": foundation.release_revision,
        "freeze_plan_sha256": foundation.freeze_plan_sha256,
        "freeze_approval_sha256": foundation.freeze_approval_sha256,
        "freeze_publication_sha256": foundation.freeze_publication_sha256,
        "v2_request_id": foundation.v2_request_id,
        "v2_expires_at_unix": foundation.v2_expires_at_unix,
        "v2_transaction_id": foundation.v2_transaction_id,
        "v2_approval_url_sha256": foundation.v2_approval_url_sha256,
        "v2_action_payload_sha256": foundation.v2_action_payload_sha256,
        "bootstrap_input_sha256": foundation.document_sha256,
        "legacy_passkey_request_id": request_id,
        "legacy_passkey_request_sha256": _sha256(request_snapshot.raw),
        "legacy_approval_url": f"https://{PUBLIC_HOST}/approve/{request_id}",
        "bridge_action_sha256": _sha256(_canonical(action)),
        "route_contract_sha256": _sha256(_canonical(route_contract)),
        "original_caddy_sha256": _sha256(configs.original),
        "approval_bridge_template_sha256": bridge_template_sha256,
        "approval_bridge_caddy_sha256": _sha256(configs.approval_bridge),
        "default_local_v1_route_preserved": True,
        "control_plane_mutation_performed": True,
        "source_data_mutation_performed": False,
        "production_host_mutation_performed": True,
        "caller_selected_input_accepted": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "requested_at_unix": now_unix,
    }


def prepare_bridge_bootstrap(
    document: Mapping[str, Any],
    *,
    boundary: CaddyBoundary,
    store: CaddyTransactionStore,
    request_boundary: LegacyRequestBoundary,
    now_unix: int,
    requests_root: Path = LEGACY_STEP_UP_REQUESTS,
    expected_uid: int = LEGACY_STEP_UP_UID,
    expected_gid: int = LEGACY_STEP_UP_GID,
    freshness_clock: Callable[[], float] | None = None,
) -> Mapping[str, Any]:
    """Create the old-v1 authorization request without changing Caddy."""

    foundation = validate_bridge_bootstrap_input(document)
    _require_v2_fresh(
        foundation,
        now_unix=int(freshness_clock()) if freshness_clock else now_unix,
    )
    entries = store.load(foundation.freeze_plan_sha256)
    prior = _last(entries, "bridge_authorization_requested")
    if prior is not None:
        receipt = validate_bridge_request_receipt(
            prior.value["evidence"], foundation=foundation
        )
        original = store.read_artifact(
            foundation.freeze_plan_sha256, "original.Caddyfile"
        )
        configs, template_sha256, action = _bridge_configs_and_action(
            foundation, original
        )
        if (
            store.read_artifact(
                foundation.freeze_plan_sha256, "approval-bridge.Caddyfile"
            )
            != configs.approval_bridge
            or receipt["approval_bridge_template_sha256"] != template_sha256
            or receipt["bridge_action_sha256"] != _sha256(_canonical(action))
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_bridge_artifact_invalid"
            )
        request_snapshot, _request = _legacy_request_for_action(
            action=action,
            now_unix=now_unix,
            requests_root=requests_root,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
        if (
            request_snapshot.path.stem != receipt["legacy_passkey_request_id"]
            or _sha256(request_snapshot.raw)
            != receipt["legacy_passkey_request_sha256"]
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_bridge_request_replay_invalid"
            )
        return receipt

    original = boundary.stable_read()
    configs, template_sha256, action = _bridge_configs_and_action(
        foundation, original.raw
    )
    boundary.validate_payload(configs.original, mode="legacy")
    boundary.validate_payload(configs.approval_bridge, mode="approval_bridge")
    boundary.validate_payload(configs.private_v2, mode="private_v2")
    boundary.validate_payload(configs.maintenance, mode="maintenance")
    if boundary.stable_read() != original:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_config_changed")
    action_sha256 = _sha256(_canonical(action))
    if _last(entries, "bridge_request_intent") is None:
        store.append(
            foundation.freeze_plan_sha256,
            "bridge_request_intent",
            {
                "bootstrap_input_sha256": foundation.document_sha256,
                "bridge_action": copy.deepcopy(dict(action)),
                "bridge_action_sha256": action_sha256,
                "original_caddy_sha256": _sha256(configs.original),
                "approval_bridge_template_sha256": template_sha256,
                "approval_bridge_caddy_sha256": _sha256(
                    configs.approval_bridge
                ),
                "production_caddy_mutated": False,
            },
            now_unix,
        )
    store.install_artifact(
        foundation.freeze_plan_sha256, "original.Caddyfile", configs.original
    )
    store.install_artifact(
        foundation.freeze_plan_sha256,
        "approval-bridge.Caddyfile",
        configs.approval_bridge,
    )
    store.install_artifact(
        foundation.freeze_plan_sha256,
        "private-v2.Caddyfile",
        configs.private_v2,
    )
    store.install_artifact(
        foundation.freeze_plan_sha256,
        "maintenance.Caddyfile",
        configs.maintenance,
    )
    try:
        request_snapshot, request = _legacy_request_for_action(
            action=action,
            now_unix=now_unix,
            requests_root=requests_root,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
    except OwnerGateCaddyCutoverError as exc:
        if str(exc) != "owner_gate_caddy_legacy_passkey_request_missing":
            raise
        created = request_boundary.create_bridge_request(action=action)
        request_snapshot, request = _legacy_request_for_action(
            action=action,
            now_unix=now_unix,
            requests_root=requests_root,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
        if (
            created.get("request_id") != request_snapshot.path.stem
            or created.get("action_hash") != action_sha256
            or created.get("approval_url")
            != f"https://{PUBLIC_HOST}/approve/{request_snapshot.path.stem}"
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_request_output_invalid"
            )
    unsigned = _bridge_request_unsigned(
        foundation,
        request_snapshot=request_snapshot,
        request=request,
        action=action,
        configs=configs,
        bridge_template_sha256=template_sha256,
        now_unix=now_unix,
    )
    receipt = validate_bridge_request_receipt(
        {**unsigned, "receipt_sha256": _sha256(_canonical(unsigned))},
        foundation=foundation,
    )
    store.append(
        foundation.freeze_plan_sha256,
        "bridge_authorization_requested",
        receipt,
        now_unix,
    )
    return receipt


def _unconsumed_grant_sha256(grant: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(grant))
    value["used_at"] = None
    value["used_at_ts"] = None
    return _sha256(_canonical(value))


def _finalize_consumed_legacy_claim(
    *,
    request_id: str,
    consumed: Mapping[str, Any],
    grants_root: Path,
    expected_uid: int,
    expected_gid: int,
) -> None:
    claim = grants_root / f".{request_id}.muncho-caddy-claim"
    temporary = grants_root / f".{request_id}.muncho-caddy-consume"
    if not os.path.lexists(claim) and not os.path.lexists(temporary):
        return
    if not os.path.lexists(claim):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_grant_claim_conflict"
        )
    claimed = _read_claimed_legacy_grant(
        claim,
        grants_root=grants_root,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )
    if _sha256(claimed.raw) != _unconsumed_grant_sha256(consumed):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_grant_claim_conflict"
        )
    if os.path.lexists(temporary):
        staged = _read_claimed_legacy_grant(
            temporary,
            grants_root=grants_root,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
        if staged.raw != _canonical(consumed):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_grant_staged_invalid"
            )
    with _legacy_grant_consumer_fence(
        claimed,
        grants_root=grants_root,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    ):
        confirmed = _read_claimed_legacy_grant(
            claim,
            grants_root=grants_root,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
        if confirmed != claimed:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_grant_claim_conflict"
            )
        os.unlink(claim)
        if os.path.lexists(temporary):
            os.unlink(temporary)
        descriptor = os.open(
            grants_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def activate_bridge_bootstrap(
    document: Mapping[str, Any],
    *,
    boundary: CaddyBoundary,
    store: CaddyTransactionStore,
    service: LegacyServiceBoundary,
    now_unix: int,
    requests_root: Path = LEGACY_STEP_UP_REQUESTS,
    grants_root: Path = LEGACY_STEP_UP_GRANTS,
    expected_uid: int = LEGACY_STEP_UP_UID,
    expected_gid: int = LEGACY_STEP_UP_GID,
    legacy_lock: Callable[[], AbstractContextManager[Any]] | None = None,
    freshness_clock: Callable[[], float] | None = None,
) -> Mapping[str, Any]:
    """Consume one old-v1 grant and install the exact reversible bridge."""

    foundation = validate_bridge_bootstrap_input(document)
    fresh_now = freshness_clock or (lambda: float(now_unix))
    if not grants_root.is_absolute() or ".." in grants_root.parts:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_grant_fence_invalid"
        )
    entries = store.load(foundation.freeze_plan_sha256)
    requested_entry = _last(entries, "bridge_authorization_requested")
    if requested_entry is None:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_request_missing"
        )
    requested = validate_bridge_request_receipt(
        requested_entry.value["evidence"], foundation=foundation
    )
    terminal = _last(entries, "bridge_activated")
    original = store.read_artifact(
        foundation.freeze_plan_sha256, "original.Caddyfile"
    )
    configs, template_sha256, action = _bridge_configs_and_action(
        foundation, original
    )
    if (
        template_sha256 != requested["approval_bridge_template_sha256"]
        or _sha256(_canonical(action)) != requested["bridge_action_sha256"]
        or store.read_artifact(
            foundation.freeze_plan_sha256, "approval-bridge.Caddyfile"
        )
        != configs.approval_bridge
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_artifact_invalid"
        )
    if terminal is not None:
        receipt = validate_bridge_receipt(
            terminal.value["evidence"],
            foundation=foundation,
            request_receipt=requested,
        )
        if terminal.value["recorded_at_unix"] != receipt["activated_at_unix"]:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_bridge_replay_invalid"
            )
        projection = boundary.observe(mode="approval_bridge")
        active = service.observe_active()
        health = service.verify_local_v1()
        if (
            boundary.stable_read().raw != configs.approval_bridge
            or projection.get("projection_sha256")
            != receipt["active_route_projection_sha256"]
            or projection.get("bridge_request_id") != foundation.v2_request_id
            or active.get("projection_sha256")
            != receipt["legacy_service_active_after_sha256"]
            or health.get("projection_sha256")
            != receipt["legacy_service_local_health_sha256"]
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_bridge_replay_drifted"
            )
        boundary.verify_bridge(request_id=foundation.v2_request_id)
        return receipt

    existing_intent = _last(entries, "bridge_intent")
    if existing_intent is None:
        _require_v2_fresh(
            foundation,
            now_unix=int(fresh_now()),
        )
        active_before = service.observe_active()
        grants_identity = _require_legacy_directory(
            grants_root,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
    else:
        intent = existing_intent.value["evidence"]
        stored_active = intent.get("legacy_service_active_before")
        if (
            frozenset(intent) != _BRIDGE_INTENT_FIELDS
            or intent.get("bootstrap_input_sha256")
            != foundation.document_sha256
            or intent.get("bridge_request_receipt_sha256")
            != requested["receipt_sha256"]
            or intent.get("bridge_action_sha256")
            != requested["bridge_action_sha256"]
            or intent.get("legacy_passkey_request_id")
            != requested["legacy_passkey_request_id"]
            or not isinstance(stored_active, Mapping)
            or intent.get("legacy_service_active_before_sha256")
            != stored_active.get("projection_sha256")
            or intent.get("temporary_service_stop_required") is not True
            or intent.get("exact_preimage_restore_required_before_cutover_intent")
            is not True
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_bridge_intent_invalid"
            )
        try:
            _restore_fenced_legacy_grants_root(
                intent,
                grants_root=grants_root,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
            )
            try:
                active_before = service.observe_active()
            except BaseException:
                active_before = service.start_exact(stored_active)
                service.verify_local_v1()
            _require_v2_fresh(
                foundation,
                now_unix=int(fresh_now()),
            )
        except BaseException as primary:
            recovery_errors: list[BaseException] = []
            try:
                current = boundary.stable_read()
                if current.raw != configs.original:
                    boundary.validate_payload(configs.original, mode="legacy")
                    _replace_transaction_owned(
                        boundary,
                        configs.original,
                        allowed_current_payloads=(configs.approval_bridge,),
                    )
                    boundary.reload()
                boundary.observe(mode="legacy")
            except BaseException as exc:
                recovery_errors.append(exc)
            try:
                try:
                    active = service.observe_active()
                except BaseException:
                    active = service.start_exact(stored_active)
                health_after_failure = service.verify_local_v1()
                store.append(
                    foundation.freeze_plan_sha256,
                    "bridge_exact_restore",
                    {
                        "bridge_request_receipt_sha256": requested[
                            "receipt_sha256"
                        ],
                        "original_caddy_sha256": _sha256(configs.original),
                        "legacy_service_active_sha256": active[
                            "projection_sha256"
                        ],
                        "legacy_service_health_sha256": health_after_failure[
                            "projection_sha256"
                        ],
                        "rollback_mode": "pre_migration_exact_bytes",
                    },
                    now_unix,
                )
            except BaseException as exc:
                recovery_errors.append(exc)
            if recovery_errors:
                raise BaseExceptionGroup(
                    "Caddy bridge replay failed and exact recovery was incomplete",
                    [primary, *recovery_errors],
                ) from None
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_bridge_activation_rolled_back"
            ) from primary

    # Check that the exact approval exists while the service is live.  A
    # replay may see the one exact grant already consumed by this intent.
    _legacy_request_and_grant(
        action=action,
        now_unix=now_unix,
        requests_root=requests_root,
        grants_root=grants_root,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        allow_consumed=existing_intent is not None,
    )
    if existing_intent is None:
        confirmed_grants_identity = _require_legacy_directory(
            grants_root,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
        if confirmed_grants_identity[1:5] != grants_identity[1:5]:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_grant_fence_invalid"
            )
        store.append(
            foundation.freeze_plan_sha256,
            "bridge_intent",
            {
                "bootstrap_input_sha256": foundation.document_sha256,
                "bridge_request_receipt_sha256": requested["receipt_sha256"],
                "bridge_action_sha256": requested["bridge_action_sha256"],
                "legacy_passkey_request_id": requested[
                    "legacy_passkey_request_id"
                ],
                "legacy_service_active_before_sha256": active_before[
                    "projection_sha256"
                ],
                "legacy_service_active_before": copy.deepcopy(
                    dict(active_before)
                ),
                "legacy_grants_root_path": str(grants_root),
                "legacy_grants_root_device": grants_identity[3],
                "legacy_grants_root_inode": grants_identity[4],
                "legacy_grants_root_uid": grants_identity[1],
                "legacy_grants_root_gid": grants_identity[2],
                "temporary_service_stop_required": True,
                "exact_preimage_restore_required_before_cutover_intent": True,
            },
            now_unix,
        )

    service_inactive: Mapping[str, Any] | None = None
    service_active_after: Mapping[str, Any] | None = None
    health: Mapping[str, Any] | None = None
    try:
        service_inactive = service.stop_exact(active_before)
        store.append(
            foundation.freeze_plan_sha256,
            "legacy_service_stopped",
            {
                "bridge_request_receipt_sha256": requested["receipt_sha256"],
                "active_before_sha256": active_before["projection_sha256"],
                "inactive_sha256": service_inactive["projection_sha256"],
            },
            now_unix,
        )
        lock_factory = legacy_lock or (
            lambda: _legacy_passkey_lock(
                grants_root=grants_root,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
            )
        )
        with lock_factory():
            (
                request_snapshot,
                _request,
                grant_snapshot,
                grant,
            ) = _legacy_request_and_grant(
                action=action,
                now_unix=now_unix,
                requests_root=requests_root,
                grants_root=grants_root,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
                allow_consumed=True,
            )
            if (
                request_snapshot.path.stem
                != requested["legacy_passkey_request_id"]
                or _sha256(request_snapshot.raw)
                != requested["legacy_passkey_request_sha256"]
            ):
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_bridge_request_changed"
                )
            grant_before_sha256 = _unconsumed_grant_sha256(grant)
            if grant.get("used_at") is None:
                consumed, consumed_sha256 = _atomic_consume_legacy_grant(
                    grant_snapshot,
                    grant,
                    consumed_at_unix=now_unix,
                    grants_root=grants_root,
                    expected_uid=expected_uid,
                    expected_gid=expected_gid,
                )
            else:
                if (
                    existing_intent is None
                    or type(grant.get("used_at_ts")) is not int
                    or
                    grant.get("used_at_ts")
                    < existing_intent.value["recorded_at_unix"]
                ):
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_legacy_consume_replay_invalid"
                    )
                consumed = grant
                consumed_sha256 = _sha256(grant_snapshot.raw)
                _finalize_consumed_legacy_claim(
                    request_id=request_snapshot.path.stem,
                    consumed=consumed,
                    grants_root=grants_root,
                    expected_uid=expected_uid,
                    expected_gid=expected_gid,
                )
            prior_consume = _last(
                store.load(foundation.freeze_plan_sha256),
                "legacy_grant_consumed",
            )
            consume_evidence = {
                "bridge_request_receipt_sha256": requested["receipt_sha256"],
                "legacy_passkey_request_id": request_snapshot.path.stem,
                "legacy_passkey_request_sha256": _sha256(
                    request_snapshot.raw
                ),
                "legacy_passkey_grant_id": consumed["grant_id"],
                "legacy_passkey_grant_sha256": grant_before_sha256,
                "legacy_passkey_consumed_grant_sha256": consumed_sha256,
                "bridge_action_sha256": requested["bridge_action_sha256"],
                "consumed_at_unix": consumed["used_at_ts"],
                "service_inactive_sha256": service_inactive[
                    "projection_sha256"
                ],
            }
            if prior_consume is None:
                consume_entry = store.append(
                    foundation.freeze_plan_sha256,
                    "legacy_grant_consumed",
                    consume_evidence,
                    now_unix,
                )
            else:
                if prior_consume.value["evidence"] != consume_evidence:
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_legacy_consume_replay_invalid"
                    )
                consume_entry = prior_consume
        boundary.validate_payload(
            configs.approval_bridge, mode="approval_bridge"
        )
        _require_v2_fresh(foundation, now_unix=int(fresh_now()))
        _replace_transaction_owned(
            boundary,
            configs.approval_bridge,
            allowed_current_payloads=(
                configs.original,
                configs.approval_bridge,
            ),
        )
        _require_v2_fresh(foundation, now_unix=int(fresh_now()))
        boundary.reload()
        projection = boundary.observe(mode="approval_bridge")
        if projection.get("bridge_request_id") != foundation.v2_request_id:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_bridge_readback_invalid"
            )
        service_active_after = service.start_exact(active_before)
        health = service.verify_local_v1()
        boundary.verify_bridge(request_id=foundation.v2_request_id)
        completed_at_unix = int(fresh_now())
        _require_v2_fresh(foundation, now_unix=completed_at_unix)
        unsigned = {
            "schema": BRIDGE_RECEIPT_SCHEMA,
            "release_revision": foundation.release_revision,
            "freeze_plan_sha256": foundation.freeze_plan_sha256,
            "freeze_approval_sha256": foundation.freeze_approval_sha256,
            "freeze_publication_sha256": foundation.freeze_publication_sha256,
            "v2_request_id": foundation.v2_request_id,
            "v2_expires_at_unix": foundation.v2_expires_at_unix,
            "v2_transaction_id": foundation.v2_transaction_id,
            "v2_approval_url_sha256": foundation.v2_approval_url_sha256,
            "v2_action_payload_sha256": foundation.v2_action_payload_sha256,
            "bootstrap_input_sha256": foundation.document_sha256,
            "bridge_request_receipt_sha256": requested["receipt_sha256"],
            "legacy_passkey_request_id": request_snapshot.path.stem,
            "legacy_passkey_request_sha256": _sha256(request_snapshot.raw),
            "legacy_passkey_grant_id": consumed["grant_id"],
            "legacy_passkey_grant_sha256": grant_before_sha256,
            "legacy_passkey_consumed_grant_sha256": consumed_sha256,
            "legacy_passkey_consume_entry_sha256": consume_entry.sha256,
            "legacy_service_active_before_sha256": active_before[
                "projection_sha256"
            ],
            "legacy_service_inactive_sha256": service_inactive[
                "projection_sha256"
            ],
            "legacy_service_active_after_sha256": service_active_after[
                "projection_sha256"
            ],
            "legacy_service_local_health_sha256": health[
                "projection_sha256"
            ],
            "bridge_action_sha256": requested["bridge_action_sha256"],
            "route_contract_sha256": requested["route_contract_sha256"],
            "original_caddy_sha256": requested["original_caddy_sha256"],
            "approval_bridge_caddy_sha256": requested[
                "approval_bridge_caddy_sha256"
            ],
            "active_route_projection_sha256": projection[
                "projection_sha256"
            ],
            "default_local_v1_route_preserved": True,
            "exact_v2_approval_routes_only": True,
            "caddy_validated": True,
            "caddy_reloaded": True,
            "caddy_readback_verified": True,
            "rollback_mode": "pre_migration_exact_bytes",
            "control_plane_mutation_performed": True,
            "source_data_mutation_performed": False,
            "production_host_mutation_performed": True,
            "caller_selected_input_accepted": False,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
            "activated_at_unix": completed_at_unix,
        }
        receipt = validate_bridge_receipt(
            {**unsigned, "receipt_sha256": _sha256(_canonical(unsigned))},
            foundation=foundation,
            request_receipt=requested,
        )
        store.append(
            foundation.freeze_plan_sha256,
            "bridge_activated",
            receipt,
            completed_at_unix,
        )
        return receipt
    except BaseException as primary:
        recovery_errors: list[BaseException] = []
        try:
            current = boundary.stable_read()
            if current.raw != configs.original:
                boundary.validate_payload(configs.original, mode="legacy")
                _replace_transaction_owned(
                    boundary,
                    configs.original,
                    allowed_current_payloads=(configs.approval_bridge,),
                )
                boundary.reload()
            boundary.observe(mode="legacy")
        except BaseException as exc:
            recovery_errors.append(exc)
        try:
            try:
                active = service.observe_active()
            except BaseException:
                active = service.start_exact(active_before)
            health_after_failure = service.verify_local_v1()
            store.append(
                foundation.freeze_plan_sha256,
                "bridge_exact_restore",
                {
                    "bridge_request_receipt_sha256": requested[
                        "receipt_sha256"
                    ],
                    "original_caddy_sha256": _sha256(configs.original),
                    "legacy_service_active_sha256": active[
                        "projection_sha256"
                    ],
                    "legacy_service_health_sha256": health_after_failure[
                        "projection_sha256"
                    ],
                    "rollback_mode": "pre_migration_exact_bytes",
                },
                now_unix,
            )
        except BaseException as exc:
            recovery_errors.append(exc)
        if recovery_errors:
            raise BaseExceptionGroup(
                "Caddy bridge activation failed and exact recovery was incomplete",
                [primary, *recovery_errors],
            ) from None
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_activation_rolled_back"
        ) from primary


def _foundation_from_bridge_request(
    value: Mapping[str, Any],
) -> _BridgeFoundation:
    unsigned = {
        "schema": BRIDGE_INPUT_SCHEMA,
        "release_revision": value.get("release_revision"),
        "freeze_plan_sha256": value.get("freeze_plan_sha256"),
        "freeze_approval_sha256": value.get("freeze_approval_sha256"),
        "freeze_publication_sha256": value.get("freeze_publication_sha256"),
        "v2_request_id": value.get("v2_request_id"),
        "v2_expires_at_unix": value.get("v2_expires_at_unix"),
        "v2_transaction_id": value.get("v2_transaction_id"),
        "v2_approval_url_sha256": value.get("v2_approval_url_sha256"),
        "v2_action_payload_sha256": value.get("v2_action_payload_sha256"),
    }
    document = {
        **unsigned,
        "document_sha256": value.get("bootstrap_input_sha256"),
    }
    return validate_bridge_bootstrap_input(document)


def _load_required_bridge(
    authority: _Authority,
    *,
    store: CaddyTransactionStore,
) -> tuple[_BridgeFoundation, Mapping[str, Any], Mapping[str, Any]]:
    entries = store.load(authority.freeze.sha256)
    requested_entry = _last(entries, "bridge_authorization_requested")
    terminal_entry = _last(entries, "bridge_activated")
    if requested_entry is None or terminal_entry is None:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_bridge_missing")
    foundation = _foundation_from_bridge_request(
        requested_entry.value["evidence"]
    )
    requested = validate_bridge_request_receipt(
        requested_entry.value["evidence"], foundation=foundation
    )
    bridge = validate_bridge_receipt(
        terminal_entry.value["evidence"],
        foundation=foundation,
        request_receipt=requested,
    )
    claim = authority.claim
    if (
        foundation.release_revision != authority.plan.value["release_revision"]
        or foundation.freeze_plan_sha256 != authority.freeze.sha256
        or foundation.freeze_approval_sha256 != authority.approval_sha256
        or foundation.freeze_publication_sha256
        != claim.get("freeze_publication_sha256")
        or foundation.v2_request_id != claim.get("request_id")
        or foundation.v2_action_payload_sha256
        != claim.get("action_payload_sha256")
        or terminal_entry.value["recorded_at_unix"]
        != bridge["activated_at_unix"]
        or terminal_entry.value["recorded_at_unix"]
        > authority.claim_recorded_at_unix
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_claim_binding_invalid"
        )
    return foundation, requested, bridge


def _prepare_unsigned(
    authority: _Authority,
    *,
    bridge_request: Mapping[str, Any],
    bridge_receipt: Mapping[str, Any],
    now_unix: int,
) -> dict[str, Any]:
    claim = authority.claim
    return {
        "schema": PREPARE_RECEIPT_SCHEMA,
        "release_revision": authority.plan.value["release_revision"],
        "freeze_plan_sha256": authority.freeze.sha256,
        "cutover_plan_sha256": authority.plan.sha256,
        "freeze_approval_sha256": authority.approval_sha256,
        "authority_sha256": authority.sha256,
        "passkey_claim_entry_sha256": authority.claim_entry_sha256,
        "passkey_claim_recorded_at_unix": authority.claim_recorded_at_unix,
        "passkey_authorization_receipt_sha256": claim[
            "authorization_receipt_sha256"
        ],
        "passkey_action_envelope_sha256": claim["action_envelope_sha256"],
        "passkey_request_id": claim["request_id"],
        "passkey_consume_attempt_id": claim["consume_attempt_id"],
        "bridge_request_receipt_sha256": bridge_request["receipt_sha256"],
        "bridge_receipt_sha256": bridge_receipt["receipt_sha256"],
        "approval_bridge_caddy_sha256": bridge_receipt[
            "approval_bridge_caddy_sha256"
        ],
        "source_route": "exact_v2_approval_bridge",
        "target_route": "fixed_private_v2",
        "candidate_validated": True,
        "maintenance_validated": True,
        "live_config_mutated": False,
        "rollback_mode": "pre_migration_exact_bytes",
        "caller_selected_input_accepted": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "prepared_at_unix": now_unix,
    }


def _caddy_prepared_dependency(
    authority: _Authority,
    *,
    prepare_receipt: Mapping[str, Any],
    configs: _DerivedConfigs,
) -> Mapping[str, Any]:
    unsigned = {
        "schema": CADDY_PREPARED_DEPENDENCY_SCHEMA,
        "release_revision": authority.plan.value["release_revision"],
        "freeze_plan_sha256": authority.freeze.sha256,
        "cutover_plan_sha256": authority.plan.sha256,
        "freeze_approval_sha256": authority.approval_sha256,
        "authority_sha256": authority.sha256,
        "caddy_prepare_receipt_sha256": prepare_receipt["receipt_sha256"],
        "original_caddy_sha256": _sha256(configs.original),
        "approval_bridge_caddy_sha256": _sha256(configs.approval_bridge),
        "private_v2_caddy_sha256": _sha256(configs.private_v2),
        "maintenance_caddy_sha256": _sha256(configs.maintenance),
        "maintenance_caddy_b64": base64.b64encode(configs.maintenance).decode(
            "ascii"
        ),
        "production_mutation_performed": False,
        "caller_selected_input_accepted": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "prepared_at_unix": prepare_receipt["prepared_at_unix"],
    }
    return {**unsigned, "receipt_sha256": _sha256(_canonical(unsigned))}


def validate_caddy_prepared_dependency(
    value: Any,
    *,
    authority: _Authority,
    prepare_receipt: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], bytes]:
    raw = _hashed(
        value,
        _CADDY_PREPARED_DEPENDENCY_FIELDS,
        "prepared_dependency",
    )
    try:
        maintenance = base64.b64decode(
            raw["maintenance_caddy_b64"].encode("ascii", errors="strict"),
            validate=True,
        )
    except (UnicodeError, ValueError) as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_prepared_dependency_invalid"
        ) from exc
    if (
        raw["schema"] != CADDY_PREPARED_DEPENDENCY_SCHEMA
        or raw["release_revision"] != authority.plan.value["release_revision"]
        or raw["freeze_plan_sha256"] != authority.freeze.sha256
        or raw["cutover_plan_sha256"] != authority.plan.sha256
        or raw["freeze_approval_sha256"] != authority.approval_sha256
        or raw["authority_sha256"] != authority.sha256
        or any(
            _SHA256.fullmatch(str(raw[field])) is None
            for field in (
                "caddy_prepare_receipt_sha256",
                "original_caddy_sha256",
                "approval_bridge_caddy_sha256",
                "private_v2_caddy_sha256",
                "maintenance_caddy_sha256",
            )
        )
        or not 0 < len(maintenance) <= MAX_CADDYFILE_BYTES
        or base64.b64encode(maintenance).decode("ascii")
        != raw["maintenance_caddy_b64"]
        or _sha256(maintenance) != raw["maintenance_caddy_sha256"]
        or raw["production_mutation_performed"] is not False
        or raw["caller_selected_input_accepted"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or type(raw["prepared_at_unix"]) is not int
        or raw["prepared_at_unix"] <= 0
        or (
            prepare_receipt is not None
            and (
                raw["caddy_prepare_receipt_sha256"]
                != prepare_receipt.get("receipt_sha256")
                or raw["prepared_at_unix"]
                != prepare_receipt.get("prepared_at_unix")
            )
        )
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_prepared_dependency_invalid"
        )
    return raw, maintenance


def _publish_caddy_prepared_dependency(
    authority: _Authority,
    *,
    prepare_receipt: Mapping[str, Any],
    configs: _DerivedConfigs,
    journal: cutover.CutoverJournal,
    now_unix: int,
) -> Mapping[str, Any]:
    expected = _caddy_prepared_dependency(
        authority,
        prepare_receipt=prepare_receipt,
        configs=configs,
    )
    entries = journal.load(authority.plan.sha256)
    matches = [
        entry
        for entry in entries
        if entry.value["event"] == "caddy_prepared"
    ]
    if len(matches) > 1:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_prepared_dependency_invalid"
        )
    if matches:
        observed, maintenance = validate_caddy_prepared_dependency(
            matches[0].value["evidence"],
            authority=authority,
            prepare_receipt=prepare_receipt,
        )
        if observed != expected or maintenance != configs.maintenance:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_prepared_dependency_invalid"
            )
        return observed
    journal.append(
        authority.plan.sha256,
        "caddy_prepared",
        expected,
        now_unix,
    )
    entries = journal.load(authority.plan.sha256)
    match = _last(entries, "caddy_prepared")
    if match is None:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_prepared_dependency_invalid"
        )
    observed, maintenance = validate_caddy_prepared_dependency(
        match.value["evidence"],
        authority=authority,
        prepare_receipt=prepare_receipt,
    )
    if observed != expected or maintenance != configs.maintenance:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_prepared_dependency_invalid"
        )
    return observed


def validate_maintenance_arm_receipt(
    value: Any,
    *,
    authority: _Authority,
    prepare_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    prepared = validate_prepare_receipt(
        prepare_receipt, plan=authority.plan
    )
    raw = _hashed(value, _MAINTENANCE_ARM_FIELDS, "maintenance_arm")
    if (
        raw["schema"] != cutover.CADDY_MAINTENANCE_ARM_SCHEMA
        or raw["release_revision"]
        != authority.plan.value["release_revision"]
        or raw["freeze_plan_sha256"] != authority.freeze.sha256
        or raw["cutover_plan_sha256"] != authority.plan.sha256
        or raw["freeze_approval_sha256"] != authority.approval_sha256
        or raw["authority_sha256"] != authority.sha256
        or raw["caddy_prepare_receipt_sha256"]
        != prepared["receipt_sha256"]
        or any(
            _SHA256.fullmatch(str(raw[field])) is None
            for field in (
                "legacy_service_active_sha256",
                "maintenance_caddy_sha256",
                "active_route_projection_sha256",
            )
        )
        or raw["public_status"] != 503
        or raw["caddy_validated"] is not True
        or raw["caddy_reloaded"] is not True
        or raw["public_verified"] is not True
        or raw["v1_public_route_closed"] is not True
        or raw["rollback_mode"]
        != "pre_intent_exact_restore_available"
        or raw["production_mutation_performed"] is not True
        or raw["caller_selected_input_accepted"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or type(raw["armed_at_unix"]) is not int
        or raw["armed_at_unix"] < prepared["prepared_at_unix"]
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_maintenance_arm_invalid"
        )
    return raw


def _publish_maintenance_arm_dependency(
    authority: _Authority,
    *,
    receipt: Mapping[str, Any],
    journal: cutover.CutoverJournal,
    now_unix: int,
) -> Mapping[str, Any]:
    entries = journal.load(authority.plan.sha256)
    matches = [
        entry
        for entry in entries
        if entry.value["event"] == "caddy_maintenance_armed"
    ]
    if len(matches) > 1:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_maintenance_arm_invalid"
        )
    if matches:
        if matches[0].value["evidence"] != receipt:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_maintenance_arm_invalid"
            )
    else:
        journal.append(
            authority.plan.sha256,
            "caddy_maintenance_armed",
            receipt,
            now_unix,
        )
        entries = journal.load(authority.plan.sha256)
    try:
        observed = cutover.require_caddy_maintenance_arm_dependency(
            entries, authority.plan
        )
    except (TypeError, ValueError, cutover.ProductionCutoverError) as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_maintenance_arm_invalid"
        ) from exc
    if observed != receipt:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_maintenance_arm_invalid"
        )
    return observed


def arm_pre_intent_maintenance(
    authority: _Authority,
    *,
    boundary: CaddyBoundary,
    store: CaddyTransactionStore,
    legacy_journal: cutover.CutoverJournal,
    service: LegacyServiceBoundary,
    now_unix: int,
) -> Mapping[str, Any]:
    """Durably close public v1 before any irreversible activation intent."""

    legacy_entries = legacy_journal.load(authority.plan.sha256)
    if _has_legacy_activation_intent(legacy_entries):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_maintenance_arm_after_intent"
        )
    local_entries = store.load(authority.plan.sha256)
    prepared_entry = _last(local_entries, "prepared")
    if prepared_entry is None:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_prepare_missing")
    prepared = validate_prepare_receipt(
        prepared_entry.value["evidence"], plan=authority.plan
    )
    configs = _DerivedConfigs(
        store.read_artifact(authority.plan.sha256, "original.Caddyfile"),
        store.read_artifact(
            authority.plan.sha256, "approval-bridge.Caddyfile"
        ),
        store.read_artifact(authority.plan.sha256, "private-v2.Caddyfile"),
        store.read_artifact(authority.plan.sha256, "maintenance.Caddyfile"),
    )
    if configs != _derive_configs(
        configs.original, bridge_request_id=prepared["passkey_request_id"]
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_artifact_invalid"
        )
    local_entries = store.load(authority.plan.sha256)
    prior = _last(local_entries, "maintenance_armed")
    legacy_active = service.observe_active()
    if (
        not isinstance(legacy_active, Mapping)
        or _SHA256.fullmatch(
            str(legacy_active.get("projection_sha256"))
        )
        is None
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_service_identity_invalid"
        )
    boundary.validate_payload(configs.maintenance, mode="maintenance")
    _replace_transaction_owned(
        boundary,
        configs.maintenance,
        allowed_current_payloads=(
            configs.original,
            configs.approval_bridge,
            configs.maintenance,
        ),
    )
    boundary.reload()
    projection = boundary.observe(mode="maintenance")
    public = boundary.verify_public(expected_status=503)
    _require_public_verification(public, expected_status=503)
    if _SHA256.fullmatch(str(projection.get("projection_sha256"))) is None:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_maintenance_arm_invalid"
        )
    unsigned = {
        "schema": cutover.CADDY_MAINTENANCE_ARM_SCHEMA,
        "release_revision": authority.plan.value["release_revision"],
        "freeze_plan_sha256": authority.freeze.sha256,
        "cutover_plan_sha256": authority.plan.sha256,
        "freeze_approval_sha256": authority.approval_sha256,
        "authority_sha256": authority.sha256,
        "caddy_prepare_receipt_sha256": prepared["receipt_sha256"],
        "legacy_service_active_sha256": legacy_active[
            "projection_sha256"
        ],
        "maintenance_caddy_sha256": _sha256(configs.maintenance),
        "active_route_projection_sha256": projection[
            "projection_sha256"
        ],
        "public_status": 503,
        "caddy_validated": True,
        "caddy_reloaded": True,
        "public_verified": True,
        "v1_public_route_closed": True,
        "rollback_mode": "pre_intent_exact_restore_available",
        "production_mutation_performed": True,
        "caller_selected_input_accepted": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "armed_at_unix": (
            now_unix
            if prior is None
            else prior.value["evidence"]["armed_at_unix"]
        ),
    }
    receipt = validate_maintenance_arm_receipt(
        {**unsigned, "receipt_sha256": _sha256(_canonical(unsigned))},
        authority=authority,
        prepare_receipt=prepared,
    )
    if prior is None:
        store.append(
            authority.plan.sha256,
            "maintenance_armed",
            receipt,
            receipt["armed_at_unix"],
        )
    else:
        observed = validate_maintenance_arm_receipt(
            prior.value["evidence"],
            authority=authority,
            prepare_receipt=prepared,
        )
        if observed != receipt:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_maintenance_arm_invalid"
            )
    return _publish_maintenance_arm_dependency(
        authority,
        receipt=receipt,
        journal=legacy_journal,
        now_unix=receipt["armed_at_unix"],
    )


def _replace_transaction_owned(
    boundary: CaddyBoundary,
    payload: bytes,
    *,
    allowed_current_payloads: Sequence[bytes],
) -> bool:
    """CAS-replace only a byte-exact state owned by this transaction."""

    current = boundary.stable_read()
    if current.raw == payload:
        return False
    if not any(current.raw == allowed for allowed in allowed_current_payloads):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_unowned_drift_detected"
        )
    boundary.replace(payload, expected=current)
    return True


def _prepare_cutover_inner(
    authority: _Authority,
    *,
    boundary: CaddyBoundary,
    store: CaddyTransactionStore,
    legacy_journal: cutover.CutoverJournal | None,
    post_intent: bool,
    now_unix: int,
) -> Mapping[str, Any]:
    """Prepare candidates without changing the live Caddyfile."""

    foundation, bridge_request, bridge_receipt = _load_required_bridge(
        authority, store=store
    )
    bridge_original = store.read_artifact(
        authority.freeze.sha256, "original.Caddyfile"
    )
    configs, _template_sha256, _action = _bridge_configs_and_action(
        foundation, bridge_original
    )
    if (
        _sha256(configs.approval_bridge)
        != bridge_receipt["approval_bridge_caddy_sha256"]
        or store.read_artifact(
            authority.freeze.sha256, "approval-bridge.Caddyfile"
        )
        != configs.approval_bridge
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_artifact_invalid"
        )
    entries = store.load(authority.plan.sha256)
    terminal = _last(entries, "prepared")
    if terminal is not None:
        receipt = validate_prepare_receipt(
            terminal.value["evidence"], plan=authority.plan
        )
        if receipt["authority_sha256"] != authority.sha256:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_authority_replay_invalid"
            )
        derived = _DerivedConfigs(
            store.read_artifact(authority.plan.sha256, "original.Caddyfile"),
            store.read_artifact(
                authority.plan.sha256, "approval-bridge.Caddyfile"
            ),
            store.read_artifact(authority.plan.sha256, "private-v2.Caddyfile"),
            store.read_artifact(authority.plan.sha256, "maintenance.Caddyfile"),
        )
        expected = _derive_configs(
            derived.original, bridge_request_id=foundation.v2_request_id
        )
        if derived != expected:
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_artifact_invalid")
        if (
            receipt["bridge_receipt_sha256"]
            != bridge_receipt["receipt_sha256"]
            or receipt["bridge_request_receipt_sha256"]
            != bridge_request["receipt_sha256"]
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_bridge_replay_invalid"
            )
        if legacy_journal is not None:
            _publish_caddy_prepared_dependency(
                authority,
                prepare_receipt=receipt,
                configs=derived,
                journal=legacy_journal,
                now_unix=now_unix,
            )
        return receipt

    original = boundary.stable_read()
    if original.raw != configs.approval_bridge:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_not_active"
        )
    boundary.validate_payload(configs.approval_bridge, mode="approval_bridge")
    projection = boundary.observe(mode="approval_bridge")
    if (
        projection.get("projection_sha256")
        != bridge_receipt["active_route_projection_sha256"]
        or projection.get("bridge_request_id") != foundation.v2_request_id
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_readback_invalid"
        )
    if boundary.stable_read() != original:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_config_changed")
    store.append(
        authority.plan.sha256,
        "prepare_intent",
        {
            "authority": copy.deepcopy(dict(authority.value)),
            "bridge_receipt_sha256": bridge_receipt["receipt_sha256"],
            "live_config_mutated": False,
        },
        now_unix,
    )
    try:
        store.install_artifact(
            authority.plan.sha256, "original.Caddyfile", configs.original
        )
        store.install_artifact(
            authority.plan.sha256,
            "approval-bridge.Caddyfile",
            configs.approval_bridge,
        )
        store.install_artifact(
            authority.plan.sha256, "private-v2.Caddyfile", configs.private_v2
        )
        store.install_artifact(
            authority.plan.sha256, "maintenance.Caddyfile", configs.maintenance
        )
        boundary.validate_payload(configs.private_v2, mode="private_v2")
        boundary.validate_payload(configs.maintenance, mode="maintenance")
        if boundary.stable_read() != original:
            raise OwnerGateCaddyCutoverError("owner_gate_caddy_config_changed")
        unsigned = _prepare_unsigned(
            authority,
            bridge_request=bridge_request,
            bridge_receipt=bridge_receipt,
            now_unix=now_unix,
        )
        receipt = validate_prepare_receipt(
            {**unsigned, "receipt_sha256": _sha256(_canonical(unsigned))},
            plan=authority.plan,
        )
        store.append(authority.plan.sha256, "prepared", receipt, now_unix)
        if legacy_journal is not None:
            _publish_caddy_prepared_dependency(
                authority,
                prepare_receipt=receipt,
                configs=configs,
                journal=legacy_journal,
                now_unix=now_unix,
            )
        return receipt
    except BaseException as primary:
        if post_intent:
            raise
        recovery_errors: list[BaseException] = []
        try:
            if boundary.stable_read().raw != configs.original:
                boundary.validate_payload(configs.original, mode="legacy")
                _replace_transaction_owned(
                    boundary,
                    configs.original,
                    allowed_current_payloads=(
                        configs.approval_bridge,
                        configs.private_v2,
                        configs.maintenance,
                    ),
                )
                boundary.reload()
                boundary.observe(mode="legacy")
        except BaseException as exc:
            recovery_errors.append(exc)
        try:
            store.append(
                authority.plan.sha256,
                "pre_migration_exact_restore",
                {
                    "authority_sha256": authority.sha256,
                    "v1_route_restored": True,
                    "rollback_mode": "pre_migration_exact_bytes",
                },
                now_unix,
            )
        except BaseException as exc:
            recovery_errors.append(exc)
        if recovery_errors:
            raise BaseExceptionGroup(
                "Caddy prepare failed and exact pre-migration recovery was incomplete",
                [primary, *recovery_errors],
            ) from None
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_prepare_rolled_back") from primary


def _has_legacy_activation_intent(
    entries: Sequence[cutover.JournalEntry],
) -> bool:
    return any(
        entry.value.get("event") == "activation_commit_intent"
        for entry in entries
    )


def _legacy_activation_intent_receipt_sha256(
    authority: _Authority,
    entries: Sequence[cutover.JournalEntry],
) -> tuple[str, int]:
    matches = [
        entry
        for entry in entries
        if entry.value.get("event") == "activation_commit_intent"
    ]
    if len(matches) != 1:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_lineage_invalid"
        )
    evidence = matches[0].value.get("evidence")
    try:
        accepted = cutover._accepted_activation_commit_intent(
            list(entries), authority.plan
        )
    except BaseException as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_lineage_invalid"
        ) from exc
    if (
        not isinstance(evidence, Mapping)
        or accepted is None
        or dict(accepted) != dict(evidence)
        or _SHA256.fullmatch(str(accepted.get("receipt_sha256"))) is None
        or type(matches[0].value.get("recorded_at_unix")) is not int
        or matches[0].value["recorded_at_unix"] <= 0
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_lineage_invalid"
        )
    return (
        str(accepted["receipt_sha256"]),
        matches[0].value["recorded_at_unix"],
    )


def _validate_caddy_post_intent_floor(
    authority: _Authority,
    entry: CaddyJournalEntry,
    *,
    prepare_receipt: Mapping[str, Any] | None = None,
    legacy_intent_receipt_sha256: str | None = None,
) -> Mapping[str, Any]:
    raw = _hashed(
        entry.value["evidence"],
        _CADDY_POST_INTENT_FLOOR_FIELDS,
        "post_intent_floor",
    )
    if (
        entry.value["event"] != "post_intent_maintenance_floor"
        or raw["schema"] != CADDY_POST_INTENT_FLOOR_SCHEMA
        or raw["cutover_plan_sha256"] != authority.plan.sha256
        or raw["authority_sha256"] != authority.sha256
        or _SHA256.fullmatch(str(raw["prepare_receipt_sha256"])) is None
        or _SHA256.fullmatch(
            str(raw["legacy_activation_commit_intent_receipt_sha256"])
        )
        is None
        or raw["rollback_mode"] != "post_migration_maintenance_only"
        or type(raw["observed_at_unix"]) is not int
        or raw["observed_at_unix"] <= 0
        or raw["observed_at_unix"] != entry.value["recorded_at_unix"]
        or (
            prepare_receipt is not None
            and raw["prepare_receipt_sha256"]
            != prepare_receipt.get("receipt_sha256")
        )
        or (
            legacy_intent_receipt_sha256 is not None
            and raw["legacy_activation_commit_intent_receipt_sha256"]
            != legacy_intent_receipt_sha256
        )
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_post_intent_floor_invalid"
        )
    return raw


def _ensure_caddy_post_intent_floor(
    authority: _Authority,
    *,
    store: CaddyTransactionStore,
    caddy_entries: Sequence[CaddyJournalEntry],
    prepare_receipt: Mapping[str, Any],
    legacy_intent_receipt_sha256: str,
    now_unix: int,
    legacy_intent_recorded_at_unix: int | None = None,
) -> Mapping[str, Any]:
    prepared = validate_prepare_receipt(
        prepare_receipt,
        plan=authority.plan,
    )
    if prepared["authority_sha256"] != authority.sha256:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_authority_replay_invalid"
        )
    prepared_entries = [
        entry for entry in caddy_entries if entry.value["event"] == "prepared"
    ]
    markers = [
        entry
        for entry in caddy_entries
        if entry.value["event"] == "post_intent_maintenance_floor"
    ]
    if (
        len(prepared_entries) != 1
        or prepared_entries[0].value["evidence"] != prepared
        or prepared_entries[0].value["recorded_at_unix"]
        != prepared["prepared_at_unix"]
        or len(markers) > 1
        or type(now_unix) is not int
        or now_unix <= 0
        or now_unix < prepared_entries[0].value["recorded_at_unix"]
        or now_unix < caddy_entries[-1].value["recorded_at_unix"]
        or (
            legacy_intent_recorded_at_unix is not None
            and (
                type(legacy_intent_recorded_at_unix) is not int
                or legacy_intent_recorded_at_unix <= 0
                or now_unix < legacy_intent_recorded_at_unix
            )
        )
        or (
            markers
            and (
                markers[0].value["sequence"]
                <= prepared_entries[0].value["sequence"]
                or markers[0].value["recorded_at_unix"]
                < prepared_entries[0].value["recorded_at_unix"]
            )
        )
        or _SHA256.fullmatch(str(legacy_intent_receipt_sha256)) is None
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_post_intent_floor_invalid"
        )
    if markers:
        return _validate_caddy_post_intent_floor(
            authority,
            markers[0],
            prepare_receipt=prepared,
            legacy_intent_receipt_sha256=legacy_intent_receipt_sha256,
        )
    unsigned = {
        "schema": CADDY_POST_INTENT_FLOOR_SCHEMA,
        "cutover_plan_sha256": authority.plan.sha256,
        "authority_sha256": authority.sha256,
        "prepare_receipt_sha256": prepared["receipt_sha256"],
        "legacy_activation_commit_intent_receipt_sha256": (
            legacy_intent_receipt_sha256
        ),
        "rollback_mode": "post_migration_maintenance_only",
        "observed_at_unix": now_unix,
    }
    expected = {
        **unsigned,
        "receipt_sha256": _sha256(_canonical(unsigned)),
    }
    store.append(
        authority.plan.sha256,
        "post_intent_maintenance_floor",
        expected,
        now_unix,
    )
    reloaded = store.load(authority.plan.sha256)
    markers = [
        entry
        for entry in reloaded
        if entry.value["event"] == "post_intent_maintenance_floor"
    ]
    if len(markers) != 1:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_post_intent_floor_invalid"
        )
    observed = _validate_caddy_post_intent_floor(
        authority,
        markers[0],
        prepare_receipt=prepared,
        legacy_intent_receipt_sha256=legacy_intent_receipt_sha256,
    )
    if observed != expected:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_post_intent_floor_invalid"
        )
    return observed


def _prepared_dependency_from_entries(
    authority: _Authority,
    entries: Sequence[cutover.JournalEntry],
) -> tuple[Mapping[str, Any], bytes]:
    matches = [
        entry
        for entry in entries
        if entry.value.get("event") == "caddy_prepared"
    ]
    if len(matches) != 1:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_prepared_dependency_invalid"
        )
    return validate_caddy_prepared_dependency(
        matches[0].value["evidence"], authority=authority
    )


def _force_post_intent_maintenance(
    authority: _Authority,
    *,
    boundary: CaddyBoundary,
    legacy_entries: Sequence[cutover.JournalEntry],
) -> Mapping[str, Any]:
    dependency, maintenance = _prepared_dependency_from_entries(
        authority, legacy_entries
    )
    allowed_hashes = {
        dependency["original_caddy_sha256"],
        dependency["approval_bridge_caddy_sha256"],
        dependency["private_v2_caddy_sha256"],
        dependency["maintenance_caddy_sha256"],
    }
    return _force_exact_maintenance_payload(
        boundary,
        maintenance=maintenance,
        allowed_hashes=allowed_hashes,
    )


def _force_exact_maintenance_payload(
    boundary: CaddyBoundary,
    *,
    maintenance: bytes,
    allowed_hashes: set[str],
) -> Mapping[str, Any]:
    boundary.validate_payload(maintenance, mode="maintenance")
    for _attempt in range(3):
        current = boundary.stable_read()
        if _sha256(current.raw) not in allowed_hashes:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_unowned_drift_detected"
            )
        if current.raw == maintenance:
            break
        try:
            boundary.replace(maintenance, expected=current)
            break
        except OwnerGateCaddyCutoverError as exc:
            if "compare_and_swap_failed" not in str(exc):
                raise
    else:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_maintenance_cas_exhausted"
        )
    boundary.reload()
    projection = boundary.observe(mode="maintenance")
    public = boundary.verify_public(expected_status=503)
    _require_public_verification(public, expected_status=503)
    if boundary.stable_read().raw != maintenance:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_maintenance_readback_invalid"
        )
    return {
        "maintenance_caddy_sha256": _sha256(maintenance),
        "active_route_projection_sha256": projection["projection_sha256"],
        "public_status": 503,
        "public_verified": True,
        "v1_route_restored": False,
    }


def _force_caddy_local_post_intent_maintenance(
    authority: _Authority,
    *,
    boundary: CaddyBoundary,
    store: CaddyTransactionStore,
    caddy_entries: Sequence[CaddyJournalEntry],
) -> Mapping[str, Any]:
    prepared_entry = _last(caddy_entries, "prepared")
    if prepared_entry is None:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_prepare_missing"
        )
    prepared = validate_prepare_receipt(
        prepared_entry.value["evidence"], plan=authority.plan
    )
    if prepared["authority_sha256"] != authority.sha256:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_authority_replay_invalid"
        )
    configs = _DerivedConfigs(
        store.read_artifact(authority.plan.sha256, "original.Caddyfile"),
        store.read_artifact(
            authority.plan.sha256, "approval-bridge.Caddyfile"
        ),
        store.read_artifact(authority.plan.sha256, "private-v2.Caddyfile"),
        store.read_artifact(authority.plan.sha256, "maintenance.Caddyfile"),
    )
    if configs != _derive_configs(
        configs.original, bridge_request_id=prepared["passkey_request_id"]
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_artifact_invalid"
        )
    return _force_exact_maintenance_payload(
        boundary,
        maintenance=configs.maintenance,
        allowed_hashes={
            _sha256(configs.original),
            _sha256(configs.approval_bridge),
            _sha256(configs.private_v2),
            _sha256(configs.maintenance),
        },
    )


def _recover_post_intent_maintenance(
    authority: _Authority,
    *,
    boundary: CaddyBoundary,
    store: CaddyTransactionStore,
    legacy_entries: Sequence[cutover.JournalEntry],
    caddy_entries: Sequence[CaddyJournalEntry],
) -> Mapping[str, Any]:
    """Recover from either independently durable prepared lineage.

    The cutover-plan dependency and the Caddy-local prepared artifacts are
    deliberately redundant.  Once either journal proves that activation has
    crossed its irreversible boundary, a malformed peer journal must not make
    recovery depend on parsing that peer again.
    """

    errors: list[BaseException] = []
    if legacy_entries:
        try:
            return _force_post_intent_maintenance(
                authority,
                boundary=boundary,
                legacy_entries=legacy_entries,
            )
        except BaseException as exc:
            errors.append(exc)
    if caddy_entries:
        try:
            return _force_caddy_local_post_intent_maintenance(
                authority,
                boundary=boundary,
                store=store,
                caddy_entries=caddy_entries,
            )
        except BaseException as exc:
            errors.append(exc)
    if not errors:
        errors.append(
            OwnerGateCaddyCutoverError(
                "owner_gate_caddy_maintenance_lineage_unavailable"
            )
        )
    raise BaseExceptionGroup(
        "No durable Caddy maintenance lineage could be recovered", errors
    )


def prepare_cutover(
    authority: _Authority,
    *,
    boundary: CaddyBoundary,
    store: CaddyTransactionStore,
    legacy_journal: cutover.CutoverJournal | None = None,
    now_unix: int,
) -> Mapping[str, Any]:
    caddy_entries: Sequence[CaddyJournalEntry] = ()
    caddy_marker = False
    caddy_error: BaseException | None = None
    try:
        caddy_entries = store.load(authority.plan.sha256)
        caddy_marker = _accepted_caddy_post_intent_marker(
            authority, caddy_entries
        )
    except BaseException as exc:
        caddy_error = exc

    legacy_entries: Sequence[cutover.JournalEntry] = ()
    post_intent = False
    legacy_error: BaseException | None = None
    if legacy_journal is not None:
        try:
            legacy_entries = legacy_journal.load(authority.plan.sha256)
            post_intent = _has_legacy_activation_intent(legacy_entries)
        except BaseException as exc:
            legacy_error = exc

    if legacy_error is not None:
        if not caddy_marker:
            raise legacy_error
        try:
            _force_caddy_local_post_intent_maintenance(
                authority,
                boundary=boundary,
                store=store,
                caddy_entries=caddy_entries,
            )
        except BaseException as recovery:
            raise BaseExceptionGroup(
                "Legacy journal failed during Caddy prepare replay and maintenance recovery was incomplete",
                [legacy_error, recovery],
            ) from None
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_prepare_post_intent_maintenance"
        ) from legacy_error

    if caddy_error is not None:
        if not post_intent:
            raise caddy_error
        try:
            _force_post_intent_maintenance(
                authority,
                boundary=boundary,
                legacy_entries=legacy_entries,
            )
        except BaseException as recovery:
            raise BaseExceptionGroup(
                "Caddy state failed during prepare replay and maintenance recovery was incomplete",
                [caddy_error, recovery],
            ) from None
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_prepare_post_intent_maintenance"
        ) from caddy_error

    if caddy_marker and not post_intent:
        _force_caddy_local_post_intent_maintenance(
            authority,
            boundary=boundary,
            store=store,
            caddy_entries=caddy_entries,
        )
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_prepare_legacy_lineage_missing_after_commit_marker"
        )
    if post_intent:
        try:
            prepared_entry = _last(caddy_entries, "prepared")
            if prepared_entry is None:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_prepare_missing"
                )
            prepared = validate_prepare_receipt(
                prepared_entry.value["evidence"],
                plan=authority.plan,
            )
            (
                intent_receipt_sha256,
                intent_recorded_at_unix,
            ) = (
                _legacy_activation_intent_receipt_sha256(
                    authority, legacy_entries
                )
            )
            _ensure_caddy_post_intent_floor(
                authority,
                store=store,
                caddy_entries=caddy_entries,
                prepare_receipt=prepared,
                legacy_intent_receipt_sha256=intent_receipt_sha256,
                now_unix=now_unix,
                legacy_intent_recorded_at_unix=intent_recorded_at_unix,
            )
            caddy_entries = store.load(authority.plan.sha256)
            caddy_marker = _accepted_caddy_post_intent_marker(
                authority, caddy_entries
            )
            if not caddy_marker:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_post_intent_floor_invalid"
                )
            # A prepared receipt is pre-migration evidence.  Once the peer
            # intent is durable, maintenance must be live before that receipt
            # can be replayed to a caller.
            _force_caddy_local_post_intent_maintenance(
                authority,
                boundary=boundary,
                store=store,
                caddy_entries=caddy_entries,
            )
        except BaseException as primary:
            try:
                _recover_post_intent_maintenance(
                    authority,
                    boundary=boundary,
                    store=store,
                    legacy_entries=legacy_entries,
                    caddy_entries=caddy_entries,
                )
            except BaseException as recovery:
                raise BaseExceptionGroup(
                    "Caddy prepare could not establish the post-intent maintenance floor",
                    [primary, recovery],
                ) from None
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_prepare_post_intent_maintenance"
            ) from primary
    try:
        return _prepare_cutover_inner(
            authority,
            boundary=boundary,
            store=store,
            legacy_journal=legacy_journal,
            post_intent=post_intent,
            now_unix=now_unix,
        )
    except BaseException as primary:
        if not (post_intent or caddy_marker):
            raise
        try:
            _recover_post_intent_maintenance(
                authority,
                boundary=boundary,
                store=store,
                legacy_entries=legacy_entries,
                caddy_entries=caddy_entries,
            )
        except BaseException as recovery:
            raise BaseExceptionGroup(
                "Caddy prepare failed after activation intent and maintenance recovery was incomplete",
                [primary, recovery],
            ) from None
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_prepare_post_intent_maintenance"
        ) from primary


def _legacy_commit_lineage(
    authority: _Authority,
    *,
    journal: cutover.CutoverJournal,
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None] | None:
    entries = journal.load(authority.plan.sha256)
    intent_entries = [
        entry
        for entry in entries
        if entry.value["event"] == "activation_commit_intent"
    ]
    terminal_entries = [
        entry for entry in entries if entry.value["event"] == "terminal"
    ]
    intent = cutover._accepted_activation_commit_intent(entries, authority.plan)
    if intent is None:
        if intent_entries or terminal_entries:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_lineage_invalid"
            )
        return None
    if len(intent_entries) != 1:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_lineage_invalid"
        )
    if not terminal_entries:
        return intent, None
    if len(terminal_entries) != 1:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_lineage_invalid"
        )
    intent_entry = intent_entries[0]
    terminal_entry = terminal_entries[0]
    terminal = terminal_entry.value["evidence"]
    try:
        database = cutover._accepted_database_apply(entries, authority.plan)
        host = cutover._accepted_host_apply(entries, authority.plan)
        database_terminal = cutover._accepted_database_terminal(
            entries, authority.plan
        )
        boot = cutover._accepted_host_boot_commit(entries, authority.plan)
        capability_entry = cutover._last(
            entries, "capability_prerequisites_validated"
        )
        capability = (
            None
            if capability_entry is None
            else cutover._require_capability_prerequisite_acceptance(
                capability_entry.value["evidence"], plan=authority.plan
            )
        )
        gateway_entry = cutover._last(entries, "gateway_started")
    except BaseException as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_terminal_invalid"
        ) from exc
    gateway = None if gateway_entry is None else gateway_entry.value["evidence"]
    digest_fields = _LEGACY_TERMINAL_FIELDS - {
        "schema",
        "zero_canonical_database_mutation_observed",
        "direct_discord_disabled",
        "discord_dm_allowed",
        "rollback_used",
        "secret_material_recorded",
        "completed_at_unix",
    }
    if (
        not isinstance(terminal, Mapping)
        or set(terminal) != _LEGACY_TERMINAL_FIELDS
        or terminal.get("schema") != cutover.TERMINAL_SCHEMA
        or any(
            _SHA256.fullmatch(str(terminal.get(field))) is None
            for field in digest_fields
        )
        or terminal.get("plan_sha256") != authority.plan.sha256
        or terminal.get("freeze_plan_sha256") != authority.freeze.sha256
        or terminal.get("freeze_approval_sha256") != authority.approval_sha256
        or terminal.get("approval_sha256") != intent["approval_sha256"]
        or terminal.get("final_tail_receipt_sha256")
        != authority.plan.value["final_tail_receipt_sha256"]
        or capability is None
        or terminal.get("capability_prerequisite_receipt_sha256")
        != capability["prerequisite_receipt_sha256"]
        or terminal.get("capability_prerequisite_file_sha256")
        != capability["prerequisite_file_sha256"]
        or terminal.get("isolated_canary_goal_continuation_terminal_sha256")
        != capability["goal_continuation_terminal_sha256"]
        or terminal.get("isolated_canary_workspace_gateway_receipt_sha256")
        != capability["workspace_gateway_receipt_sha256"]
        or terminal.get("isolation_equivalence_projection_sha256")
        != capability["isolation_equivalence_projection_sha256"]
        or terminal.get("zero_canonical_database_mutation_observed")
        is not True
        or terminal.get("pre_db_zero_write_observation_sha256")
        != capability["pre_db_zero_write_observation_sha256"]
        or terminal.get("capability_topology_identity_sha256")
        != cutover.production_capability_topology_identity_sha256(
            authority.plan.value["capability_topology"]
        )
        or database is None
        or host is None
        or database_terminal is None
        or boot is None
        or terminal.get("database_apply_receipt_sha256")
        != database["receipt_sha256"]
        or terminal.get("host_apply_receipt_sha256")
        != host["receipt_sha256"]
        or terminal.get("host_boot_commit_receipt_sha256")
        != boot["receipt_sha256"]
        or terminal.get("activation_commit_intent_receipt_sha256")
        != intent["receipt_sha256"]
        or terminal.get("database_postflight_receipt_sha256")
        != database_terminal["receipt_sha256"]
        or not isinstance(gateway, Mapping)
        or set(gateway)
        != {
            "gateway_observation_sha256",
            "writer_observation_sha256",
            "connector_observation_sha256",
        }
        or terminal.get("gateway_observation_sha256")
        != gateway["gateway_observation_sha256"]
        or terminal.get("writer_observation_sha256")
        != gateway["writer_observation_sha256"]
        or terminal.get("connector_observation_sha256")
        != gateway["connector_observation_sha256"]
        or terminal.get("direct_discord_disabled") is not True
        or terminal.get("discord_dm_allowed") is not False
        or terminal.get("rollback_used") is not False
        or terminal.get("secret_material_recorded") is not False
        or type(terminal.get("completed_at_unix")) is not int
        or terminal["completed_at_unix"] != terminal_entry.value["recorded_at_unix"]
        or terminal_entry.value["sequence"] <= intent_entry.value["sequence"]
        or terminal_entry.value["recorded_at_unix"]
        < intent_entry.value["recorded_at_unix"]
        or terminal.get("receipt_sha256")
        != _sha256(
            _canonical(
            {key: item for key, item in terminal.items() if key != "receipt_sha256"}
            )
        )
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_terminal_invalid"
        )
    return intent, terminal


def _forward_recovery_maintenance(
    authority: _Authority,
    *,
    prepare_receipt: Mapping[str, Any],
    legacy_intent: Mapping[str, Any],
    boundary: CaddyBoundary,
    store: CaddyTransactionStore,
    payload: bytes,
    allowed_current_payloads: Sequence[bytes],
    now_unix: int,
) -> Mapping[str, Any]:
    intent_receipt_sha256 = legacy_intent.get("receipt_sha256")
    if _SHA256.fullmatch(str(intent_receipt_sha256)) is None:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_lineage_invalid"
        )
    _ensure_caddy_post_intent_floor(
        authority,
        store=store,
        caddy_entries=store.load(authority.plan.sha256),
        prepare_receipt=prepare_receipt,
        legacy_intent_receipt_sha256=str(intent_receipt_sha256),
        now_unix=now_unix,
    )
    boundary.validate_payload(payload, mode="maintenance")
    _replace_transaction_owned(
        boundary,
        payload,
        allowed_current_payloads=allowed_current_payloads,
    )
    boundary.reload()
    projection = boundary.observe(mode="maintenance")
    public = boundary.verify_public(expected_status=503)
    _require_public_verification(public, expected_status=503)
    unsigned = {
        "schema": MAINTENANCE_OBSERVATION_SCHEMA,
        "release_revision": authority.plan.value["release_revision"],
        "freeze_plan_sha256": authority.freeze.sha256,
        "cutover_plan_sha256": authority.plan.sha256,
        "freeze_approval_sha256": authority.approval_sha256,
        "authority_sha256": authority.sha256,
        "prepare_receipt_sha256": prepare_receipt["receipt_sha256"],
        "passkey_claim_entry_sha256": authority.claim_entry_sha256,
        "passkey_authorization_receipt_sha256": authority.claim[
            "authorization_receipt_sha256"
        ],
        "passkey_request_id": authority.claim["request_id"],
        "passkey_consume_attempt_id": authority.claim["consume_attempt_id"],
        "legacy_activation_commit_intent_receipt_sha256": legacy_intent[
            "receipt_sha256"
        ],
        "legacy_terminal_receipt_sha256": None,
        "outcome": "maintenance_active_forward_recovery_required",
        "public_status": 503,
        "active_route_projection_sha256": projection["projection_sha256"],
        "caddy_validated": True,
        "caddy_reloaded": True,
        "public_verified": True,
        "v1_route_restored": False,
        "forward_recovery_required": True,
        "rollback_mode": "post_migration_maintenance_only",
        "caller_selected_input_accepted": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "observed_at_unix": now_unix,
    }
    receipt = validate_maintenance_observation(
        {**unsigned, "receipt_sha256": _sha256(_canonical(unsigned))},
        plan=authority.plan,
        prepare_receipt=prepare_receipt,
    )
    store.append(
        authority.plan.sha256,
        "forward_recovery_maintenance",
        receipt,
        now_unix,
    )
    return receipt


def _terminal_unsigned(
    authority: _Authority,
    *,
    prepare_receipt: Mapping[str, Any],
    legacy_intent: Mapping[str, Any],
    legacy_terminal: Mapping[str, Any],
    outcome: str,
    public_status: int,
    route_projection_sha256: str,
    now_unix: int,
) -> dict[str, Any]:
    return {
        "schema": TERMINAL_RECEIPT_SCHEMA,
        "release_revision": authority.plan.value["release_revision"],
        "freeze_plan_sha256": authority.freeze.sha256,
        "cutover_plan_sha256": authority.plan.sha256,
        "freeze_approval_sha256": authority.approval_sha256,
        "authority_sha256": authority.sha256,
        "prepare_receipt_sha256": prepare_receipt["receipt_sha256"],
        "passkey_claim_entry_sha256": authority.claim_entry_sha256,
        "passkey_authorization_receipt_sha256": authority.claim[
            "authorization_receipt_sha256"
        ],
        "passkey_request_id": authority.claim["request_id"],
        "passkey_consume_attempt_id": authority.claim["consume_attempt_id"],
        "legacy_activation_commit_intent_receipt_sha256": legacy_intent[
            "receipt_sha256"
        ],
        "legacy_terminal_receipt_sha256": legacy_terminal["receipt_sha256"],
        "outcome": outcome,
        "public_status": public_status,
        "active_route_projection_sha256": route_projection_sha256,
        "caddy_validated": True,
        "caddy_reloaded": True,
        "public_verified": True,
        "v1_route_restored": False,
        "rollback_mode": "post_migration_maintenance_only",
        "caller_selected_input_accepted": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "completed_at_unix": now_unix,
    }


def _require_public_verification(
    value: Mapping[str, Any],
    *,
    expected_status: int,
) -> None:
    if not isinstance(value, Mapping) or expected_status not in {200, 503}:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_public_verify_failed"
        )
    if expected_status == 200:
        expected = {
            "status": 200,
            "body_size": len(PUBLIC_READY_BODY),
            "body_sha256": _sha256(PUBLIC_READY_BODY),
            "content_type": PUBLIC_READY_CONTENT_TYPE,
            "schema": PUBLIC_READY_SCHEMA,
            "service": "muncho-passkey-v2-web",
            "authority_ready": True,
            "tls_verified": True,
        }
    else:
        expected = {
            "status": 503,
            "body_size": len(PUBLIC_MAINTENANCE_BODY),
            "body_sha256": _sha256(PUBLIC_MAINTENANCE_BODY),
            "content_type": PUBLIC_MAINTENANCE_CONTENT_TYPE,
            "schema": "muncho-owner-gate-maintenance.v1",
            "service": "muncho-owner-gate-maintenance",
            "authority_ready": False,
            "tls_verified": True,
        }
    if dict(value) != expected:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_public_verify_failed"
        )


def _finish_mode(
    authority: _Authority,
    *,
    prepare_receipt: Mapping[str, Any],
    legacy_intent: Mapping[str, Any],
    legacy_terminal: Mapping[str, Any],
    boundary: CaddyBoundary,
    store: CaddyTransactionStore,
    payload: bytes,
    allowed_current_payloads: Sequence[bytes],
    mode: str,
    now_unix: int,
) -> Mapping[str, Any]:
    expected_status = 200 if mode == "private_v2" else 503
    outcome = "private_v2_active" if mode == "private_v2" else "maintenance_active"
    boundary.validate_payload(payload, mode=mode)
    _replace_transaction_owned(
        boundary,
        payload,
        allowed_current_payloads=allowed_current_payloads,
    )
    boundary.reload()
    projection = boundary.observe(mode=mode)
    public = boundary.verify_public(expected_status=expected_status)
    _require_public_verification(public, expected_status=expected_status)
    unsigned = _terminal_unsigned(
        authority,
        prepare_receipt=prepare_receipt,
        legacy_intent=legacy_intent,
        legacy_terminal=legacy_terminal,
        outcome=outcome,
        public_status=expected_status,
        route_projection_sha256=str(projection.get("projection_sha256")),
        now_unix=now_unix,
    )
    receipt = validate_terminal_receipt(
        {**unsigned, "receipt_sha256": _sha256(_canonical(unsigned))},
        plan=authority.plan,
        prepare_receipt=prepare_receipt,
    )
    store.append(authority.plan.sha256, "terminal", receipt, now_unix)
    return receipt


def _commit_cutover_inner(
    authority: _Authority,
    *,
    boundary: CaddyBoundary,
    store: CaddyTransactionStore,
    legacy_journal: cutover.CutoverJournal,
    now_unix: int,
) -> Mapping[str, Any]:
    """Commit private-v2 ingress or converge to post-migration maintenance."""

    entries = store.load(authority.plan.sha256)
    prepared_entry = _last(entries, "prepared")
    if prepared_entry is None:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_prepare_missing")
    prepared = validate_prepare_receipt(
        prepared_entry.value["evidence"], plan=authority.plan
    )
    if prepared["authority_sha256"] != authority.sha256:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_authority_replay_invalid")
    configs = _DerivedConfigs(
        store.read_artifact(authority.plan.sha256, "original.Caddyfile"),
        store.read_artifact(
            authority.plan.sha256, "approval-bridge.Caddyfile"
        ),
        store.read_artifact(authority.plan.sha256, "private-v2.Caddyfile"),
        store.read_artifact(authority.plan.sha256, "maintenance.Caddyfile"),
    )
    if configs != _derive_configs(
        configs.original, bridge_request_id=prepared["passkey_request_id"]
    ):
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_artifact_invalid")
    lineage = _legacy_commit_lineage(authority, journal=legacy_journal)
    if lineage is None:
        # This is the only edge that may restore v1.  It runs before the
        # irreversible legacy activation intent and therefore restores the
        # exact captured bytes even after a prior process crash.
        if boundary.stable_read().raw != configs.original:
            boundary.validate_payload(configs.original, mode="legacy")
            _replace_transaction_owned(
                boundary,
                configs.original,
                allowed_current_payloads=(
                    configs.approval_bridge,
                    configs.private_v2,
                    configs.maintenance,
                ),
            )
            boundary.reload()
            boundary.observe(mode="legacy")
        store.append(
            authority.plan.sha256,
            "pre_migration_exact_restore",
            {
                "authority_sha256": authority.sha256,
                "v1_route_restored": True,
                "rollback_mode": "pre_migration_exact_bytes",
            },
            now_unix,
        )
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_migration_not_committed"
        )
    legacy_intent, legacy_terminal = lineage
    if legacy_terminal is None:
        prior_recovery = _last(entries, "forward_recovery_maintenance")
        if prior_recovery is not None:
            receipt = validate_maintenance_observation(
                prior_recovery.value["evidence"],
                plan=authority.plan,
                prepare_receipt=prepared,
            )
            _ensure_caddy_post_intent_floor(
                authority,
                store=store,
                caddy_entries=store.load(authority.plan.sha256),
                prepare_receipt=prepared,
                legacy_intent_receipt_sha256=legacy_intent[
                    "receipt_sha256"
                ],
                now_unix=now_unix,
            )
            try:
                projection = boundary.observe(mode="maintenance")
                public = boundary.verify_public(expected_status=503)
                _require_public_verification(public, expected_status=503)
                if (
                    projection.get("projection_sha256")
                    != receipt["active_route_projection_sha256"]
                ):
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_maintenance_replay_drifted"
                    )
                return receipt
            except BaseException:
                pass
        return _forward_recovery_maintenance(
            authority,
            prepare_receipt=prepared,
            legacy_intent=legacy_intent,
            boundary=boundary,
            store=store,
            payload=configs.maintenance,
            allowed_current_payloads=(
                configs.original,
                configs.approval_bridge,
                configs.private_v2,
                configs.maintenance,
            ),
            now_unix=now_unix,
        )
    terminal_entry = _last(entries, "terminal")
    if terminal_entry is not None:
        terminal = validate_terminal_receipt(
            terminal_entry.value["evidence"],
            plan=authority.plan,
            prepare_receipt=prepared,
        )
        mode = (
            "private_v2"
            if terminal["outcome"] == "private_v2_active"
            else "maintenance"
        )
        payload = configs.private_v2 if mode == "private_v2" else configs.maintenance
        try:
            projection = boundary.observe(mode=mode)
            public = boundary.verify_public(expected_status=terminal["public_status"])
            _require_public_verification(
                public, expected_status=terminal["public_status"]
            )
            if (
                projection.get("projection_sha256")
                != terminal["active_route_projection_sha256"]
                or boundary.stable_read().raw != payload
            ):
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_terminal_replay_drifted"
                )
        except BaseException:
            # A replayed private terminal that has drifted is still after the
            # migration boundary.  Reconcile only to maintenance.
            return _finish_mode(
                authority,
                prepare_receipt=prepared,
                legacy_intent=legacy_intent,
                legacy_terminal=legacy_terminal,
                boundary=boundary,
                store=store,
                payload=configs.maintenance,
                allowed_current_payloads=(
                    configs.original,
                    configs.approval_bridge,
                    configs.private_v2,
                    configs.maintenance,
                ),
                mode="maintenance",
                now_unix=now_unix,
            )
        return terminal

    if _last(entries, "commit_started") is None:
        commit_started_unsigned = {
            "schema": CADDY_COMMIT_STARTED_SCHEMA,
            "cutover_plan_sha256": authority.plan.sha256,
            "authority_sha256": authority.sha256,
            "prepare_receipt_sha256": prepared["receipt_sha256"],
            "legacy_activation_commit_intent_receipt_sha256": (
                legacy_intent["receipt_sha256"]
            ),
            "legacy_terminal_receipt_sha256": legacy_terminal[
                "receipt_sha256"
            ],
            "rollback_mode": "post_migration_maintenance_only",
            "started_at_unix": now_unix,
        }
        store.append(
            authority.plan.sha256,
            "commit_started",
            {
                **commit_started_unsigned,
                "receipt_sha256": _sha256(
                    _canonical(commit_started_unsigned)
                ),
            },
            now_unix,
        )
    try:
        return _finish_mode(
            authority,
            prepare_receipt=prepared,
            legacy_intent=legacy_intent,
            legacy_terminal=legacy_terminal,
            boundary=boundary,
            store=store,
            payload=configs.private_v2,
            allowed_current_payloads=(
                configs.original,
                configs.approval_bridge,
                configs.private_v2,
                configs.maintenance,
            ),
            mode="private_v2",
            now_unix=now_unix,
        )
    except BaseException as primary:
        try:
            store.append(
                authority.plan.sha256,
                "maintenance_started",
                {
                    "authority_sha256": authority.sha256,
                    "v1_route_restored": False,
                    "rollback_mode": "post_migration_maintenance_only",
                },
                now_unix,
            )
            return _finish_mode(
                authority,
                prepare_receipt=prepared,
                legacy_intent=legacy_intent,
                legacy_terminal=legacy_terminal,
                boundary=boundary,
                store=store,
                payload=configs.maintenance,
                allowed_current_payloads=(
                    configs.original,
                    configs.approval_bridge,
                    configs.private_v2,
                    configs.maintenance,
                ),
                mode="maintenance",
                now_unix=now_unix,
            )
        except BaseException as recovery:
            raise BaseExceptionGroup(
                "Caddy commit failed and maintenance recovery was incomplete",
                [primary, recovery],
            ) from None


def _accepted_caddy_post_intent_marker(
    authority: _Authority,
    entries: Sequence[CaddyJournalEntry],
) -> bool:
    markers = [
        entry
        for entry in entries
        if entry.value["event"]
        in {
            "post_intent_maintenance_floor",
            "forward_recovery_maintenance",
            "commit_started",
            "terminal",
        }
    ]
    if not markers:
        return False
    for entry in markers:
        evidence = entry.value["evidence"]
        if entry.value["event"] == "post_intent_maintenance_floor":
            prepared_entries = [
                item
                for item in entries
                if item.value["event"] == "prepared"
            ]
            if len(prepared_entries) != 1:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_post_intent_floor_invalid"
                )
            prepared = validate_prepare_receipt(
                prepared_entries[0].value["evidence"],
                plan=authority.plan,
            )
            if (
                prepared["authority_sha256"] != authority.sha256
                or prepared_entries[0].value["recorded_at_unix"]
                != prepared["prepared_at_unix"]
                or entry.value["sequence"]
                <= prepared_entries[0].value["sequence"]
                or entry.value["recorded_at_unix"]
                < prepared_entries[0].value["recorded_at_unix"]
            ):
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_post_intent_floor_invalid"
                )
            _validate_caddy_post_intent_floor(
                authority,
                entry,
                prepare_receipt=prepared,
            )
        elif entry.value["event"] == "forward_recovery_maintenance":
            prepared_entries = [
                item
                for item in entries
                if item.value["event"] == "prepared"
            ]
            if len(prepared_entries) != 1:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_maintenance_observation_invalid"
                )
            prepared = validate_prepare_receipt(
                prepared_entries[0].value["evidence"],
                plan=authority.plan,
            )
            receipt = validate_maintenance_observation(
                evidence,
                plan=authority.plan,
                prepare_receipt=prepared,
            )
            if (
                prepared["authority_sha256"] != authority.sha256
                or prepared_entries[0].value["recorded_at_unix"]
                != prepared["prepared_at_unix"]
                or entry.value["sequence"]
                <= prepared_entries[0].value["sequence"]
                or entry.value["recorded_at_unix"]
                < prepared_entries[0].value["recorded_at_unix"]
                or entry.value["recorded_at_unix"]
                != receipt["observed_at_unix"]
            ):
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_maintenance_observation_invalid"
                )
        elif entry.value["event"] == "commit_started":
            raw = _hashed(
                evidence,
                _CADDY_COMMIT_STARTED_FIELDS,
                "commit_started",
            )
            if (
                raw["schema"] != CADDY_COMMIT_STARTED_SCHEMA
                or raw["cutover_plan_sha256"] != authority.plan.sha256
                or raw["authority_sha256"] != authority.sha256
                or any(
                    _SHA256.fullmatch(str(raw[field])) is None
                    for field in (
                        "prepare_receipt_sha256",
                        "legacy_activation_commit_intent_receipt_sha256",
                        "legacy_terminal_receipt_sha256",
                    )
                )
                or raw["rollback_mode"]
                != "post_migration_maintenance_only"
                or type(raw["started_at_unix"]) is not int
                or raw["started_at_unix"] != entry.value["recorded_at_unix"]
            ):
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_commit_marker_invalid"
                )
        else:
            if (
                not isinstance(evidence, Mapping)
                or set(evidence) != _TERMINAL_FIELDS
                or evidence.get("schema") != TERMINAL_RECEIPT_SCHEMA
                or evidence.get("cutover_plan_sha256")
                != authority.plan.sha256
                or evidence.get("authority_sha256") != authority.sha256
                or evidence.get("v1_route_restored") is not False
                or evidence.get("rollback_mode")
                != "post_migration_maintenance_only"
                or evidence.get("receipt_sha256")
                != _sha256(
                    _canonical(
                        {
                            key: item
                            for key, item in evidence.items()
                            if key != "receipt_sha256"
                        }
                    )
                )
            ):
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_commit_marker_invalid"
                )
    return True


def commit_cutover(
    authority: _Authority,
    *,
    boundary: CaddyBoundary,
    store: CaddyTransactionStore,
    legacy_journal: cutover.CutoverJournal,
    now_unix: int,
) -> Mapping[str, Any]:
    """Monotonic wrapper: any durable intent makes maintenance the floor."""

    caddy_entries: Sequence[CaddyJournalEntry] = ()
    caddy_marker = False
    caddy_error: BaseException | None = None
    try:
        caddy_entries = store.load(authority.plan.sha256)
        caddy_marker = _accepted_caddy_post_intent_marker(
            authority, caddy_entries
        )
    except BaseException as exc:
        caddy_error = exc

    legacy_entries: Sequence[cutover.JournalEntry] = ()
    legacy_marker = False
    legacy_error: BaseException | None = None
    try:
        legacy_entries = legacy_journal.load(authority.plan.sha256)
        legacy_marker = _has_legacy_activation_intent(legacy_entries)
    except BaseException as exc:
        legacy_error = exc

    if legacy_error is not None:
        if not caddy_marker:
            raise legacy_error
        try:
            _force_caddy_local_post_intent_maintenance(
                authority,
                boundary=boundary,
                store=store,
                caddy_entries=caddy_entries,
            )
        except BaseException as recovery:
            raise BaseExceptionGroup(
                "Legacy journal failed after Caddy commit marker and maintenance recovery was incomplete",
                [legacy_error, recovery],
            ) from None
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_post_intent_maintenance"
        ) from legacy_error

    if caddy_error is not None:
        if not legacy_marker:
            raise caddy_error
        try:
            _force_post_intent_maintenance(
                authority,
                boundary=boundary,
                legacy_entries=legacy_entries,
            )
        except BaseException as recovery:
            raise BaseExceptionGroup(
                "Caddy state validation failed after activation intent and maintenance recovery was incomplete",
                [caddy_error, recovery],
            ) from None
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_post_intent_maintenance"
        ) from caddy_error

    if caddy_marker and not legacy_marker:
        _force_caddy_local_post_intent_maintenance(
            authority,
            boundary=boundary,
            store=store,
            caddy_entries=caddy_entries,
        )
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_lineage_missing_after_commit_marker"
        )
    try:
        return _commit_cutover_inner(
            authority,
            boundary=boundary,
            store=store,
            legacy_journal=legacy_journal,
            now_unix=now_unix,
        )
    except BaseException as primary:
        if not (legacy_marker or caddy_marker):
            raise
        try:
            _recover_post_intent_maintenance(
                authority,
                boundary=boundary,
                store=store,
                legacy_entries=legacy_entries,
                caddy_entries=caddy_entries,
            )
        except BaseException as recovery:
            raise BaseExceptionGroup(
                "Caddy commit failed after activation intent and maintenance recovery was incomplete",
                [primary, recovery],
            ) from None
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_post_intent_maintenance"
        ) from primary


def _validate_pre_intent_restore_intent(
    value: Any,
    *,
    authority: _Authority,
    prepared: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw = _hashed(
        value,
        _PRE_INTENT_RESTORE_INTENT_FIELDS,
        "pre_intent_restore_intent",
    )
    if (
        raw["schema"] != cutover.CADDY_PRE_INTENT_RESTORE_INTENT_SCHEMA
        or raw["release_revision"]
        != authority.plan.value["release_revision"]
        or raw["freeze_plan_sha256"] != authority.freeze.sha256
        or raw["cutover_plan_sha256"] != authority.plan.sha256
        or raw["freeze_approval_sha256"] != authority.approval_sha256
        or raw["authority_sha256"] != authority.sha256
        or raw["caddy_prepare_receipt_sha256"]
        != prepared.get("receipt_sha256")
        or raw["maintenance_arm_receipt_sha256"]
        != arm.get("receipt_sha256")
        or _SHA256.fullmatch(str(raw["original_caddy_sha256"])) is None
        or raw["maintenance_caddy_sha256"]
        != arm.get("maintenance_caddy_sha256")
        or raw["active_route_projection_sha256"]
        != arm.get("active_route_projection_sha256")
        or raw["public_status"] != 503
        or raw["public_verified"] is not True
        or raw["v1_public_route_closed"] is not True
        or raw["exact_original_artifact_available"] is not True
        or raw["forward_apply_invalidated"] is not True
        or raw["recovery_basis"] not in {"freeze_abort", "cutover_rollback"}
        or (
            raw["recovery_basis"] == "freeze_abort"
            and raw["rollback_terminal_receipt_sha256"] is not None
        )
        or (
            raw["recovery_basis"] == "cutover_rollback"
            and _SHA256.fullmatch(
                str(raw["rollback_terminal_receipt_sha256"])
            )
            is None
        )
        or raw["rollback_mode"] != "pre_intent_exact_bytes"
        or raw["production_mutation_performed"] is not False
        or raw["caller_selected_input_accepted"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or type(raw["restore_started_at_unix"]) is not int
        or raw["restore_started_at_unix"] < arm.get("armed_at_unix", 0)
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_pre_intent_restore_intent_invalid"
        )
    return raw


def _publish_pre_intent_restore_intent(
    authority: _Authority,
    *,
    prepared: Mapping[str, Any],
    arm: Mapping[str, Any],
    original_caddy_sha256: str,
    maintenance_proof: Mapping[str, Any],
    recovery_basis: str,
    rollback_terminal_receipt_sha256: str | None,
    journal: cutover.CutoverJournal,
    now_unix: int,
) -> Mapping[str, Any]:
    entries = journal.load(authority.plan.sha256)
    matches = [
        entry
        for entry in entries
        if entry.value["event"] == "caddy_pre_intent_restore_intent"
    ]
    if len(matches) > 1:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_pre_intent_restore_intent_invalid"
        )
    started_at = (
        now_unix
        if not matches
        else matches[0].value["evidence"].get("restore_started_at_unix")
    )
    unsigned = {
        "schema": cutover.CADDY_PRE_INTENT_RESTORE_INTENT_SCHEMA,
        "release_revision": authority.plan.value["release_revision"],
        "freeze_plan_sha256": authority.freeze.sha256,
        "cutover_plan_sha256": authority.plan.sha256,
        "freeze_approval_sha256": authority.approval_sha256,
        "authority_sha256": authority.sha256,
        "caddy_prepare_receipt_sha256": prepared["receipt_sha256"],
        "maintenance_arm_receipt_sha256": arm["receipt_sha256"],
        "original_caddy_sha256": original_caddy_sha256,
        "maintenance_caddy_sha256": maintenance_proof[
            "maintenance_caddy_sha256"
        ],
        "active_route_projection_sha256": maintenance_proof[
            "active_route_projection_sha256"
        ],
        "public_status": 503,
        "public_verified": True,
        "v1_public_route_closed": True,
        "exact_original_artifact_available": True,
        "forward_apply_invalidated": True,
        "recovery_basis": recovery_basis,
        "rollback_terminal_receipt_sha256": (
            rollback_terminal_receipt_sha256
        ),
        "rollback_mode": "pre_intent_exact_bytes",
        "production_mutation_performed": False,
        "caller_selected_input_accepted": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "restore_started_at_unix": started_at,
    }
    expected = _validate_pre_intent_restore_intent(
        {**unsigned, "receipt_sha256": _sha256(_canonical(unsigned))},
        authority=authority,
        prepared=prepared,
        arm=arm,
    )
    if matches:
        observed = _validate_pre_intent_restore_intent(
            matches[0].value["evidence"],
            authority=authority,
            prepared=prepared,
            arm=arm,
        )
        if (
            matches[0].value["recorded_at_unix"]
            != observed["restore_started_at_unix"]
            or observed != expected
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_pre_intent_restore_intent_invalid"
            )
        try:
            gateway_observed = (
                cutover.require_caddy_pre_intent_restore_intent_dependency(
                    entries, authority.plan
                )
            )
        except (TypeError, ValueError, cutover.ProductionCutoverError) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_pre_intent_restore_intent_invalid"
            ) from exc
        if gateway_observed != observed:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_pre_intent_restore_intent_invalid"
            )
        return observed
    journal.append(
        authority.plan.sha256,
        "caddy_pre_intent_restore_intent",
        expected,
        expected["restore_started_at_unix"],
    )
    observed_entry = _last(
        journal.load(authority.plan.sha256),
        "caddy_pre_intent_restore_intent",
    )
    if observed_entry is None:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_pre_intent_restore_intent_invalid"
        )
    observed = _validate_pre_intent_restore_intent(
        observed_entry.value["evidence"],
        authority=authority,
        prepared=prepared,
        arm=arm,
    )
    if observed != expected:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_pre_intent_restore_intent_invalid"
        )
    try:
        gateway_observed = (
            cutover.require_caddy_pre_intent_restore_intent_dependency(
                journal.load(authority.plan.sha256), authority.plan
            )
        )
    except (TypeError, ValueError, cutover.ProductionCutoverError) as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_pre_intent_restore_intent_invalid"
        ) from exc
    if gateway_observed != observed:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_pre_intent_restore_intent_invalid"
        )
    return observed


def _publish_pre_intent_restore_dependency(
    authority: _Authority,
    *,
    prepared: Mapping[str, Any],
    arm: Mapping[str, Any],
    intent: Mapping[str, Any],
    original_caddy_sha256: str,
    projection_sha256: str,
    gateway_terminal_event: str,
    gateway_terminal_receipt_sha256: str,
    legacy_service_active_sha256: str,
    legacy_service_health_sha256: str,
    journal: cutover.CutoverJournal,
    now_unix: int,
) -> Mapping[str, Any]:
    entries = journal.load(authority.plan.sha256)
    matches = [
        entry
        for entry in entries
        if entry.value["event"] == "caddy_pre_intent_restored"
    ]
    if len(matches) > 1:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_pre_intent_restore_invalid"
        )
    restored_at = (
        now_unix
        if not matches
        else matches[0].value["evidence"].get("restored_at_unix")
    )
    unsigned = {
        "schema": cutover.CADDY_PRE_INTENT_RESTORE_SCHEMA,
        "release_revision": authority.plan.value["release_revision"],
        "freeze_plan_sha256": authority.freeze.sha256,
        "cutover_plan_sha256": authority.plan.sha256,
        "freeze_approval_sha256": authority.approval_sha256,
        "authority_sha256": authority.sha256,
        "caddy_prepare_receipt_sha256": prepared["receipt_sha256"],
        "maintenance_arm_receipt_sha256": arm["receipt_sha256"],
        "restore_intent_receipt_sha256": intent["receipt_sha256"],
        "original_caddy_sha256": original_caddy_sha256,
        "active_route_projection_sha256": projection_sha256,
        "exact_original_caddy_restored": True,
        "caddy_validated": True,
        "caddy_reloaded": True,
        "live_readback_verified": True,
        "v1_public_route_restored": True,
        "gateway_terminal_event": gateway_terminal_event,
        "gateway_terminal_receipt_sha256": gateway_terminal_receipt_sha256,
        "recovery_basis": intent["recovery_basis"],
        "legacy_service_active_sha256": legacy_service_active_sha256,
        "legacy_service_health_sha256": legacy_service_health_sha256,
        "rollback_mode": "pre_intent_exact_bytes",
        "production_mutation_performed": True,
        "caller_selected_input_accepted": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "restored_at_unix": restored_at,
    }
    expected = {**unsigned, "receipt_sha256": _sha256(_canonical(unsigned))}
    if matches:
        if matches[0].value["evidence"] != expected:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_pre_intent_restore_invalid"
            )
    else:
        journal.append(
            authority.plan.sha256,
            "caddy_pre_intent_restored",
            expected,
            restored_at,
        )
        entries = journal.load(authority.plan.sha256)
    try:
        observed = cutover.require_caddy_pre_intent_restore_dependency(
            entries, authority.plan
        )
    except (TypeError, ValueError, cutover.ProductionCutoverError) as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_pre_intent_restore_invalid"
        ) from exc
    if observed != expected:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_pre_intent_restore_invalid"
        )
    return observed


def _restore_pre_intent_caddy(
    authority: _Authority,
    *,
    boundary: CaddyBoundary,
    store: CaddyTransactionStore,
    legacy_journal: cutover.CutoverJournal,
    gateway_terminal_event: str,
    gateway_terminal: Mapping[str, Any],
    service: LegacyRetirementBoundary,
    now_unix: int,
) -> Mapping[str, Any] | None:
    """Restore exact captured ingress only while no activation intent exists."""

    legacy_entries = legacy_journal.load(authority.plan.sha256)
    if _has_legacy_activation_intent(legacy_entries):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_pre_intent_restore_forbidden"
        )
    local_entries = store.load(authority.plan.sha256)
    prepared_entry = _last(local_entries, "prepared")
    if prepared_entry is None:
        return None
    prepared = validate_prepare_receipt(
        prepared_entry.value["evidence"], plan=authority.plan
    )
    configs = _DerivedConfigs(
        store.read_artifact(authority.plan.sha256, "original.Caddyfile"),
        store.read_artifact(
            authority.plan.sha256, "approval-bridge.Caddyfile"
        ),
        store.read_artifact(authority.plan.sha256, "private-v2.Caddyfile"),
        store.read_artifact(authority.plan.sha256, "maintenance.Caddyfile"),
    )
    original_caddy_sha256 = _sha256(configs.original)
    arm_entry = _last(local_entries, "maintenance_armed")
    arm: Mapping[str, Any] | None = None
    restore_intent: Mapping[str, Any] | None = None
    if arm_entry is not None:
        arm = validate_maintenance_arm_receipt(
            arm_entry.value["evidence"],
            authority=authority,
            prepare_receipt=prepared,
        )
        try:
            restore_intent = (
                cutover.require_caddy_pre_intent_restore_intent_dependency(
                    legacy_entries, authority.plan
                )
            )
        except (TypeError, ValueError, cutover.ProductionCutoverError) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_pre_intent_restore_intent_invalid"
            ) from exc
        if restore_intent["maintenance_arm_receipt_sha256"] != arm[
            "receipt_sha256"
        ]:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_maintenance_arm_invalid"
            )
    if gateway_terminal_event == "freeze_aborted":
        try:
            validated_gateway_terminal = cutover._validate_freeze_abort_receipt(
                gateway_terminal, plan=authority.freeze
            )
        except (TypeError, ValueError, cutover.ProductionCutoverError) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_gateway_terminal_invalid"
            ) from exc
        if (
            validated_gateway_terminal["cutover_plan_sha256"]
            != authority.plan.sha256
            or validated_gateway_terminal["caddy_restore_required"]
            is not (restore_intent is not None)
            or validated_gateway_terminal[
                "caddy_restore_intent_receipt_sha256"
            ]
            != (
                None
                if restore_intent is None
                else restore_intent["receipt_sha256"]
            )
            or (
                restore_intent is not None
                and restore_intent["recovery_basis"] != "freeze_abort"
            )
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_gateway_terminal_invalid"
            )
    elif gateway_terminal_event == "rollback_terminal":
        rollback_entry = _last(legacy_entries, "rollback_terminal")
        try:
            validated_gateway_terminal = cutover._validate_rollback_terminal(
                gateway_terminal,
                plan=authority.plan,
                entries=legacy_entries,
            )
        except (TypeError, ValueError, cutover.ProductionCutoverError) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_gateway_terminal_invalid"
            ) from exc
        if (
            rollback_entry is None
            or rollback_entry.value["evidence"] != validated_gateway_terminal
            or restore_intent is None
            or restore_intent["recovery_basis"] != "cutover_rollback"
            or restore_intent["rollback_terminal_receipt_sha256"]
            != validated_gateway_terminal["receipt_sha256"]
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_gateway_terminal_invalid"
            )
    else:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_gateway_terminal_invalid"
        )
    active = service.observe_active()
    health = service.verify_local_v1()
    if (
        _SHA256.fullmatch(str(active.get("projection_sha256"))) is None
        or _SHA256.fullmatch(str(health.get("projection_sha256"))) is None
        or (
            arm is not None
            and active["projection_sha256"]
            != arm["legacy_service_active_sha256"]
        )
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_service_health_invalid"
        )
    boundary.validate_payload(configs.original, mode="legacy")
    _replace_transaction_owned(
        boundary,
        configs.original,
        allowed_current_payloads=(
            configs.original,
            configs.approval_bridge,
            configs.maintenance,
        ),
    )
    boundary.reload()
    projection = boundary.observe(mode="legacy")
    if _SHA256.fullmatch(str(projection.get("projection_sha256"))) is None:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_pre_intent_restore_invalid"
        )
    prior = _last(store.load(authority.plan.sha256), "pre_migration_exact_restore")
    if prior is not None:
        evidence = prior.value["evidence"]
        if (
            evidence.get("authority_sha256") != authority.sha256
            or evidence.get("v1_route_restored") is not True
            or evidence.get("rollback_mode")
            != "pre_migration_exact_bytes"
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_pre_intent_restore_invalid"
            )
        if (
            evidence.get("prepare_receipt_sha256")
            != prepared["receipt_sha256"]
            or evidence.get("active_route_projection_sha256")
            != projection["projection_sha256"]
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_pre_intent_restore_invalid"
            )
    else:
        evidence = {
            "authority_sha256": authority.sha256,
            "prepare_receipt_sha256": prepared["receipt_sha256"],
            "active_route_projection_sha256": projection[
                "projection_sha256"
            ],
            "v1_route_restored": True,
            "rollback_mode": "pre_migration_exact_bytes",
        }
        store.append(
            authority.plan.sha256,
            "pre_migration_exact_restore",
            evidence,
            now_unix,
        )
    if arm is None or restore_intent is None:
        return copy.deepcopy(dict(evidence))
    return _publish_pre_intent_restore_dependency(
        authority,
        prepared=prepared,
        arm=arm,
        intent=restore_intent,
        original_caddy_sha256=original_caddy_sha256,
        projection_sha256=projection["projection_sha256"],
        gateway_terminal_event=gateway_terminal_event,
        gateway_terminal_receipt_sha256=validated_gateway_terminal[
            "receipt_sha256"
        ],
        legacy_service_active_sha256=active["projection_sha256"],
        legacy_service_health_sha256=health["projection_sha256"],
        journal=legacy_journal,
        now_unix=now_unix,
    )


def validate_legacy_retirement_receipt(
    value: Any,
    *,
    authority: _Authority,
    maintenance_arm_receipt: Mapping[str, Any],
    legacy_terminal: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw = _hashed(value, _LEGACY_RETIREMENT_FIELDS, "legacy_retirement")
    if (
        raw["schema"] != LEGACY_RETIREMENT_SCHEMA
        or raw["release_revision"]
        != authority.plan.value["release_revision"]
        or raw["freeze_plan_sha256"] != authority.freeze.sha256
        or raw["cutover_plan_sha256"] != authority.plan.sha256
        or raw["authority_sha256"] != authority.sha256
        or raw["maintenance_arm_receipt_sha256"]
        != maintenance_arm_receipt.get("receipt_sha256")
        or raw["legacy_terminal_receipt_sha256"]
        != legacy_terminal.get("receipt_sha256")
        or raw["legacy_service_active_before_sha256"]
        != maintenance_arm_receipt.get("legacy_service_active_sha256")
        or any(
            _SHA256.fullmatch(str(raw[field])) is None
            for field in (
                "legacy_service_fragment_backup_sha256",
                "legacy_service_retired_sha256",
            )
        )
        or raw["legacy_service_fragment_backup_sha256"]
        != LEGACY_STEP_UP_FRAGMENT_SHA256
        or raw["unit"] != LEGACY_STEP_UP_UNIT
        or raw["fragment_path"] != str(LEGACY_STEP_UP_FRAGMENT)
        or raw["fragment_masked"] is not True
        or raw["service_inactive"] is not True
        or raw["permanent"] is not True
        or raw["v1_public_route_closed"] is not True
        or raw["rollback_mode"] != "forward_only_private_v2_or_maintenance"
        or raw["production_mutation_performed"] is not True
        or raw["caller_selected_input_accepted"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or type(raw["retired_at_unix"]) is not int
        or raw["retired_at_unix"] <= 0
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_retirement_invalid"
        )
    return raw


def retire_legacy_service(
    authority: _Authority,
    *,
    boundary: CaddyBoundary,
    store: CaddyTransactionStore,
    legacy_journal: cutover.CutoverJournal,
    service: LegacyRetirementBoundary,
    now_unix: int,
) -> Mapping[str, Any]:
    """Permanently mask exact legacy-v1 only after the durable intent."""

    caddy_entries = store.load(authority.plan.sha256)
    prepared_entry = _last(caddy_entries, "prepared")
    arm_entry = _last(caddy_entries, "maintenance_armed")
    if prepared_entry is None or arm_entry is None:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_retirement_invalid"
        )
    prepared = validate_prepare_receipt(
        prepared_entry.value["evidence"], plan=authority.plan
    )
    arm = validate_maintenance_arm_receipt(
        arm_entry.value["evidence"],
        authority=authority,
        prepare_receipt=prepared,
    )
    lineage = _legacy_commit_lineage(authority, journal=legacy_journal)
    if lineage is None or lineage[1] is None:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_retirement_before_terminal"
        )
    _legacy_intent, legacy_terminal = lineage
    prior = _last(caddy_entries, "legacy_v1_retired")
    if prior is not None:
        receipt = validate_legacy_retirement_receipt(
            prior.value["evidence"],
            authority=authority,
            maintenance_arm_receipt=arm,
            legacy_terminal=legacy_terminal,
        )
        retired = service.observe_retired()
        if (
            retired.get("projection_sha256")
            != receipt["legacy_service_retired_sha256"]
            or _sha256(
                store.read_artifact(
                    authority.plan.sha256, "legacy-stepup.service"
                )
            )
            != receipt["legacy_service_fragment_backup_sha256"]
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_retirement_invalid"
            )
        return receipt
    configs = _DerivedConfigs(
        store.read_artifact(authority.plan.sha256, "original.Caddyfile"),
        store.read_artifact(
            authority.plan.sha256, "approval-bridge.Caddyfile"
        ),
        store.read_artifact(authority.plan.sha256, "private-v2.Caddyfile"),
        store.read_artifact(authority.plan.sha256, "maintenance.Caddyfile"),
    )
    if boundary.stable_read().raw != configs.maintenance:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_retirement_without_maintenance"
        )
    projection = boundary.observe(mode="maintenance")
    public = boundary.verify_public(expected_status=503)
    _require_public_verification(public, expected_status=503)
    if (
        projection.get("projection_sha256")
        != arm["active_route_projection_sha256"]
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_retirement_without_maintenance"
        )
    intent_entry = _last(caddy_entries, "legacy_retirement_intent")
    if intent_entry is None:
        active = service.observe_active()
        if (
            not isinstance(active, Mapping)
            or _SHA256.fullmatch(
                str(active.get("projection_sha256"))
            )
            is None
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_identity_invalid"
            )
        fragment = service.snapshot_exact_fragment(active)
        if _sha256(fragment) != LEGACY_STEP_UP_FRAGMENT_SHA256:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_service_identity_invalid"
            )
        store.install_artifact(
            authority.plan.sha256, "legacy-stepup.service", fragment
        )
        intent_unsigned = {
            "schema": LEGACY_RETIREMENT_INTENT_SCHEMA,
            "release_revision": authority.plan.value["release_revision"],
            "freeze_plan_sha256": authority.freeze.sha256,
            "cutover_plan_sha256": authority.plan.sha256,
            "authority_sha256": authority.sha256,
            "maintenance_arm_receipt_sha256": arm["receipt_sha256"],
            "legacy_terminal_receipt_sha256": legacy_terminal[
                "receipt_sha256"
            ],
            "legacy_service_fragment_backup_sha256": _sha256(fragment),
            "rollback_mode": "forward_only_private_v2_or_maintenance",
            "recorded_at_unix": now_unix,
        }
        intent = {
            **intent_unsigned,
            "receipt_sha256": _sha256(_canonical(intent_unsigned)),
        }
        store.append(
            authority.plan.sha256,
            "legacy_retirement_intent",
            intent,
            now_unix,
        )
    else:
        intent = _hashed(
            intent_entry.value["evidence"],
            _LEGACY_RETIREMENT_INTENT_FIELDS,
            "legacy_retirement_intent",
        )
        if (
            intent["schema"] != LEGACY_RETIREMENT_INTENT_SCHEMA
            or intent["release_revision"]
            != authority.plan.value["release_revision"]
            or intent["freeze_plan_sha256"] != authority.freeze.sha256
            or intent["cutover_plan_sha256"] != authority.plan.sha256
            or intent["authority_sha256"] != authority.sha256
            or intent["maintenance_arm_receipt_sha256"]
            != arm["receipt_sha256"]
            or intent["legacy_terminal_receipt_sha256"]
            != legacy_terminal["receipt_sha256"]
            or intent["legacy_service_fragment_backup_sha256"]
            != LEGACY_STEP_UP_FRAGMENT_SHA256
            or intent["rollback_mode"]
            != "forward_only_private_v2_or_maintenance"
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_retirement_invalid"
            )
        backup = store.read_artifact(
            authority.plan.sha256, "legacy-stepup.service"
        )
        if _sha256(backup) != LEGACY_STEP_UP_FRAGMENT_SHA256:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_legacy_retirement_invalid"
            )
    retired = service.retire_exact()
    if (
        not isinstance(retired, Mapping)
        or _SHA256.fullmatch(
            str(retired.get("projection_sha256"))
        )
        is None
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_legacy_retirement_invalid"
        )
    unsigned = {
        "schema": LEGACY_RETIREMENT_SCHEMA,
        "release_revision": authority.plan.value["release_revision"],
        "freeze_plan_sha256": authority.freeze.sha256,
        "cutover_plan_sha256": authority.plan.sha256,
        "authority_sha256": authority.sha256,
        "maintenance_arm_receipt_sha256": arm["receipt_sha256"],
        "legacy_terminal_receipt_sha256": legacy_terminal[
            "receipt_sha256"
        ],
        "legacy_service_active_before_sha256": arm[
            "legacy_service_active_sha256"
        ],
        "legacy_service_fragment_backup_sha256": (
            LEGACY_STEP_UP_FRAGMENT_SHA256
        ),
        "legacy_service_retired_sha256": retired["projection_sha256"],
        "unit": LEGACY_STEP_UP_UNIT,
        "fragment_path": str(LEGACY_STEP_UP_FRAGMENT),
        "fragment_masked": True,
        "service_inactive": True,
        "permanent": True,
        "v1_public_route_closed": True,
        "rollback_mode": "forward_only_private_v2_or_maintenance",
        "production_mutation_performed": True,
        "caller_selected_input_accepted": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "retired_at_unix": now_unix,
    }
    receipt = validate_legacy_retirement_receipt(
        {**unsigned, "receipt_sha256": _sha256(_canonical(unsigned))},
        authority=authority,
        maintenance_arm_receipt=arm,
        legacy_terminal=legacy_terminal,
    )
    store.append(
        authority.plan.sha256,
        "legacy_v1_retired",
        receipt,
        now_unix,
    )
    return receipt


def validate_convergence_receipt(
    value: Any,
    *,
    authority: _Authority,
) -> Mapping[str, Any]:
    raw = _hashed(value, _CONVERGENCE_RECEIPT_FIELDS, "convergence")
    if (
        raw["schema"] != CONVERGENCE_RECEIPT_SCHEMA
        or raw["release_revision"]
        != authority.plan.value["release_revision"]
        or raw["freeze_plan_sha256"] != authority.freeze.sha256
        or raw["cutover_plan_sha256"] != authority.plan.sha256
        or any(
            _SHA256.fullmatch(str(raw[field])) is None
            for field in (
                "preflight_receipt_sha256",
                "caddy_prepare_receipt_sha256",
                "maintenance_arm_receipt_sha256",
                "cutover_terminal_receipt_sha256",
                "caddy_terminal_receipt_sha256",
                "legacy_service_retirement_receipt_sha256",
            )
        )
        or raw["caddy_outcome"]
        not in {"private_v2_active", "maintenance_active"}
        or raw["control_plane_mutation_performed"] is not True
        or raw["source_data_mutation_performed"] is not True
        or raw["production_host_mutation_performed"] is not True
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_convergence_receipt_invalid"
        )
    return raw


def _require_receipt_sha256(
    value: Any,
    *,
    plan_sha256: str,
    error_code: str,
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or value.get("plan_sha256") != plan_sha256
        or _SHA256.fullmatch(str(value.get("receipt_sha256"))) is None
    ):
        raise OwnerGateCaddyCutoverError(error_code)
    return copy.deepcopy(dict(value))


def converge_cutover(
    authority: _Authority,
    *,
    boundary_factory: Callable[[], CaddyBoundary],
    store_factory: Callable[[], CaddyTransactionStore],
    legacy_journal_factory: Callable[[], cutover.CutoverJournal],
    retirement_boundary_factory: Callable[[], LegacyRetirementBoundary],
    cutover_runner: Callable[[str], Mapping[str, Any]],
    lock: Callable[[], AbstractContextManager[Any]],
    now_unix: int,
) -> Mapping[str, Any]:
    """Converge the fixed staged cutover across every process crash point."""

    if not all(
        callable(item)
        for item in (
            boundary_factory,
            store_factory,
            legacy_journal_factory,
            retirement_boundary_factory,
            cutover_runner,
            lock,
        )
    ):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_convergence_boundary_invalid"
        )
    preflight: Mapping[str, Any] | None = None
    prepared: Mapping[str, Any] | None = None
    arm: Mapping[str, Any] | None = None
    terminal: Mapping[str, Any] | None = None
    retirement: Mapping[str, Any] | None = None
    caddy_terminal: Mapping[str, Any] | None = None
    legacy_journal = legacy_journal_factory()
    initial_legacy_entries = legacy_journal.load(authority.plan.sha256)
    existing_restore_intent = _last(
        initial_legacy_entries, "caddy_pre_intent_restore_intent"
    )
    if existing_restore_intent is not None:
        try:
            restore_handoff = (
                cutover.require_caddy_pre_intent_restore_intent_dependency(
                    initial_legacy_entries, authority.plan
                )
            )
        except (TypeError, ValueError, cutover.ProductionCutoverError) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_pre_intent_restore_intent_invalid"
            ) from exc
        if restore_handoff["recovery_basis"] == "freeze_abort":
            gateway_terminal_event = "freeze_aborted"
            gateway_terminal = cutover_runner("abort-freeze")
            completed_error = "owner_gate_caddy_pre_intent_aborted"
        else:
            rollback_entry = _last(initial_legacy_entries, "rollback_terminal")
            if rollback_entry is None:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_gateway_terminal_invalid"
                )
            gateway_terminal_event = "rollback_terminal"
            gateway_terminal = rollback_entry.value["evidence"]
            completed_error = "owner_gate_caddy_cutover_rolled_back_restored"
        with lock():
            _restore_pre_intent_caddy(
                authority,
                boundary=boundary_factory(),
                store=store_factory(),
                legacy_journal=legacy_journal,
                gateway_terminal_event=gateway_terminal_event,
                gateway_terminal=gateway_terminal,
                service=retirement_boundary_factory(),
                now_unix=now_unix,
            )
        raise OwnerGateCaddyCutoverError(completed_error)
    initial_freeze_entries = legacy_journal.load(authority.freeze.sha256)
    freeze_aborts = [
        entry
        for entry in initial_freeze_entries
        if entry.value["event"] == "freeze_aborted"
    ]
    if freeze_aborts:
        if (
            len(freeze_aborts) != 1
            or freeze_aborts[0].value["sequence"]
            != len(initial_freeze_entries) - 1
        ):
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_gateway_terminal_invalid"
            )
        try:
            freeze_abort = cutover._validate_freeze_abort_receipt(
                freeze_aborts[0].value["evidence"],
                plan=authority.freeze,
            )
        except (TypeError, ValueError, cutover.ProductionCutoverError) as exc:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_gateway_terminal_invalid"
            ) from exc
        if freeze_abort["caddy_restore_required"]:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_pre_intent_restore_intent_invalid"
            )
        if freeze_abort["cutover_plan_sha256"] not in {
            None,
            authority.plan.sha256,
        }:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_gateway_terminal_invalid"
            )
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_pre_intent_aborted"
        )
    try:
        with lock():
            boundary = boundary_factory()
            store = store_factory()
            initial_entries = store.load(authority.plan.sha256)
            convergence_entry = _last(
                initial_entries, "convergence_terminal"
            )
            preflight_entry = _last(
                initial_entries, "preflight_accepted"
            )
            if convergence_entry is not None:
                prior = validate_convergence_receipt(
                    convergence_entry.value["evidence"],
                    authority=authority,
                )
                prepared_entry = _last(initial_entries, "prepared")
                arm_entry = _last(initial_entries, "maintenance_armed")
                if (
                    prepared_entry is None
                    or arm_entry is None
                    or preflight_entry is None
                ):
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_convergence_receipt_invalid"
                    )
                prepared = validate_prepare_receipt(
                    prepared_entry.value["evidence"],
                    plan=authority.plan,
                )
                arm = validate_maintenance_arm_receipt(
                    arm_entry.value["evidence"],
                    authority=authority,
                    prepare_receipt=prepared,
                )
                preflight = _require_receipt_sha256(
                    preflight_entry.value["evidence"],
                    plan_sha256=authority.plan.sha256,
                    error_code=(
                        "owner_gate_caddy_preflight_receipt_invalid"
                    ),
                )
                retirement = retire_legacy_service(
                    authority,
                    boundary=boundary,
                    store=store,
                    legacy_journal=legacy_journal,
                    service=retirement_boundary_factory(),
                    now_unix=now_unix,
                )
                caddy_terminal = validate_terminal_receipt(
                    commit_cutover(
                        authority,
                        boundary=boundary,
                        store=store,
                        legacy_journal=legacy_journal,
                        now_unix=now_unix,
                    ),
                    plan=authority.plan,
                    prepare_receipt=prepared,
                )
                lineage = _legacy_commit_lineage(
                    authority, journal=legacy_journal
                )
                if (
                    lineage is None
                    or lineage[1] is None
                    or prior["preflight_receipt_sha256"]
                    != preflight["receipt_sha256"]
                    or prior["caddy_prepare_receipt_sha256"]
                    != prepared["receipt_sha256"]
                    or prior["maintenance_arm_receipt_sha256"]
                    != arm["receipt_sha256"]
                    or prior["cutover_terminal_receipt_sha256"]
                    != lineage[1]["receipt_sha256"]
                    or prior["caddy_terminal_receipt_sha256"]
                    != caddy_terminal["receipt_sha256"]
                    or prior["caddy_outcome"] != caddy_terminal["outcome"]
                    or prior[
                        "legacy_service_retirement_receipt_sha256"
                    ]
                    != retirement["receipt_sha256"]
                ):
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_convergence_receipt_invalid"
                    )
                return prior
        if preflight_entry is None:
            preflight = _require_receipt_sha256(
                cutover_runner("phase-b-preflight"),
                plan_sha256=authority.plan.sha256,
                error_code="owner_gate_caddy_preflight_receipt_invalid",
            )
            with lock():
                store = store_factory()
                observed = _last(
                    store.load(authority.plan.sha256),
                    "preflight_accepted",
                )
                if observed is None:
                    store.append(
                        authority.plan.sha256,
                        "preflight_accepted",
                        preflight,
                        now_unix,
                    )
                elif observed.value["evidence"] != preflight:
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_preflight_receipt_invalid"
                    )
        else:
            preflight = _require_receipt_sha256(
                preflight_entry.value["evidence"],
                plan_sha256=authority.plan.sha256,
                error_code="owner_gate_caddy_preflight_receipt_invalid",
            )
        with lock():
            boundary = boundary_factory()
            store = store_factory()
            caddy_entries = store.load(authority.plan.sha256)
            prepared_entry = _last(caddy_entries, "prepared")
            if prepared_entry is None:
                prepared = prepare_cutover(
                    authority,
                    boundary=boundary,
                    store=store,
                    legacy_journal=legacy_journal,
                    now_unix=now_unix,
                )
            else:
                prepared = validate_prepare_receipt(
                    prepared_entry.value["evidence"],
                    plan=authority.plan,
                )
            legacy_entries = legacy_journal.load(authority.plan.sha256)
            activation_intent = _has_legacy_activation_intent(
                legacy_entries
            )
            if not activation_intent:
                arm = arm_pre_intent_maintenance(
                    authority,
                    boundary=boundary,
                    store=store,
                    legacy_journal=legacy_journal,
                    service=retirement_boundary_factory(),
                    now_unix=now_unix,
                )
            else:
                arm_entry = _last(
                    store.load(authority.plan.sha256),
                    "maintenance_armed",
                )
                if arm_entry is None:
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_maintenance_arm_invalid"
                    )
                arm = validate_maintenance_arm_receipt(
                    arm_entry.value["evidence"],
                    authority=authority,
                    prepare_receipt=prepared,
                )
                cutover.require_caddy_maintenance_arm_dependency(
                    legacy_entries, authority.plan
                )
                if _last(caddy_entries, "terminal") is None:
                    configs = _DerivedConfigs(
                        store.read_artifact(
                            authority.plan.sha256, "original.Caddyfile"
                        ),
                        store.read_artifact(
                            authority.plan.sha256,
                            "approval-bridge.Caddyfile",
                        ),
                        store.read_artifact(
                            authority.plan.sha256,
                            "private-v2.Caddyfile",
                        ),
                        store.read_artifact(
                            authority.plan.sha256,
                            "maintenance.Caddyfile",
                        ),
                    )
                    boundary.validate_payload(
                        configs.maintenance, mode="maintenance"
                    )
                    _replace_transaction_owned(
                        boundary,
                        configs.maintenance,
                        allowed_current_payloads=(
                            configs.original,
                            configs.approval_bridge,
                            configs.private_v2,
                            configs.maintenance,
                        ),
                    )
                    boundary.reload()
                    boundary.observe(mode="maintenance")
                    public = boundary.verify_public(expected_status=503)
                    _require_public_verification(
                        public, expected_status=503
                    )
        terminal = _require_receipt_sha256(
            cutover_runner("apply-cutover"),
            plan_sha256=authority.plan.sha256,
            error_code="owner_gate_caddy_cutover_terminal_invalid",
        )
        with lock():
            boundary = boundary_factory()
            store = store_factory()
            configs = _DerivedConfigs(
                store.read_artifact(
                    authority.plan.sha256, "original.Caddyfile"
                ),
                store.read_artifact(
                    authority.plan.sha256, "approval-bridge.Caddyfile"
                ),
                store.read_artifact(
                    authority.plan.sha256, "private-v2.Caddyfile"
                ),
                store.read_artifact(
                    authority.plan.sha256, "maintenance.Caddyfile"
                ),
            )
            replay_terminal = _last(
                store.load(authority.plan.sha256), "terminal"
            )
            replay_retirement = _last(
                store.load(authority.plan.sha256), "legacy_v1_retired"
            )
            if (
                replay_terminal is None or replay_retirement is None
            ) and boundary.stable_read().raw != configs.maintenance:
                boundary.validate_payload(
                    configs.maintenance, mode="maintenance"
                )
                _replace_transaction_owned(
                    boundary,
                    configs.maintenance,
                    allowed_current_payloads=(
                        configs.original,
                        configs.approval_bridge,
                        configs.private_v2,
                        configs.maintenance,
                    ),
                )
                boundary.reload()
                boundary.observe(mode="maintenance")
                public = boundary.verify_public(expected_status=503)
                _require_public_verification(public, expected_status=503)
            retirement = retire_legacy_service(
                authority,
                boundary=boundary,
                store=store,
                legacy_journal=legacy_journal,
                service=retirement_boundary_factory(),
                now_unix=now_unix,
            )
            caddy_terminal = commit_cutover(
                authority,
                boundary=boundary,
                store=store,
                legacy_journal=legacy_journal,
                now_unix=now_unix,
            )
            caddy_terminal = validate_terminal_receipt(
                caddy_terminal,
                plan=authority.plan,
                prepare_receipt=prepared,
            )
            unsigned = {
                "schema": CONVERGENCE_RECEIPT_SCHEMA,
                "release_revision": authority.plan.value[
                    "release_revision"
                ],
                "freeze_plan_sha256": authority.freeze.sha256,
                "cutover_plan_sha256": authority.plan.sha256,
                "preflight_receipt_sha256": preflight["receipt_sha256"],
                "caddy_prepare_receipt_sha256": prepared[
                    "receipt_sha256"
                ],
                "maintenance_arm_receipt_sha256": arm[
                    "receipt_sha256"
                ],
                "cutover_terminal_receipt_sha256": terminal[
                    "receipt_sha256"
                ],
                "caddy_terminal_receipt_sha256": caddy_terminal[
                    "receipt_sha256"
                ],
                "caddy_outcome": caddy_terminal["outcome"],
                "legacy_service_retirement_receipt_sha256": retirement[
                    "receipt_sha256"
                ],
                "control_plane_mutation_performed": True,
                "source_data_mutation_performed": True,
                "production_host_mutation_performed": True,
                "secret_material_recorded": False,
                "secret_digest_recorded": False,
            }
            receipt = validate_convergence_receipt(
                {
                    **unsigned,
                    "receipt_sha256": _sha256(_canonical(unsigned)),
                },
                authority=authority,
            )
            prior = _last(
                store.load(authority.plan.sha256),
                "convergence_terminal",
            )
            if prior is None:
                store.append(
                    authority.plan.sha256,
                    "convergence_terminal",
                    receipt,
                    now_unix,
                )
            elif validate_convergence_receipt(
                prior.value["evidence"], authority=authority
            ) != receipt:
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_convergence_receipt_invalid"
                )
            return receipt
    except BaseException as primary:
        try:
            legacy_entries = legacy_journal.load(authority.plan.sha256)
            activation_intent = _has_legacy_activation_intent(
                legacy_entries
            )
        except BaseException as journal_error:
            raise BaseExceptionGroup(
                "Convergence failed and activation intent could not be reconciled",
                [primary, journal_error],
            ) from None
        if activation_intent:
            try:
                with lock():
                    _recover_post_intent_maintenance(
                        authority,
                        boundary=boundary_factory(),
                        store=store_factory(),
                        legacy_entries=legacy_entries,
                        caddy_entries=store_factory().load(
                            authority.plan.sha256
                        ),
                    )
            except BaseException as recovery:
                raise BaseExceptionGroup(
                    "Convergence failed after intent and maintenance recovery was incomplete",
                    [primary, recovery],
                ) from None
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_forward_recovery_required"
            ) from primary
        apply_intents = [
            entry
            for entry in legacy_entries
            if entry.value["event"] == "passkey_intent"
        ]
        rollback_terminal: Mapping[str, Any] | None = None
        if apply_intents:
            rollback_entries = [
                entry
                for entry in legacy_entries
                if entry.value["event"] == "rollback_terminal"
            ]
            restore_intent_entry = _last(
                legacy_entries, "caddy_pre_intent_restore_intent"
            )
            try:
                if (
                    len(apply_intents) != 1
                    or len(rollback_entries) != 1
                    or _last(legacy_entries, "terminal") is not None
                    or (
                        restore_intent_entry is None
                        and rollback_entries[0].value["sequence"]
                        != len(legacy_entries) - 1
                    )
                    or (
                        restore_intent_entry is not None
                        and restore_intent_entry.value["sequence"]
                        != rollback_entries[0].value["sequence"] + 1
                    )
                ):
                    raise OwnerGateCaddyCutoverError(
                        "owner_gate_caddy_rollback_terminal_missing"
                    )
                rollback_terminal = cutover._validate_rollback_terminal(
                    rollback_entries[0].value["evidence"],
                    plan=authority.plan,
                    entries=legacy_entries,
                )
            except BaseException as rollback_error:
                try:
                    with lock():
                        _recover_post_intent_maintenance(
                            authority,
                            boundary=boundary_factory(),
                            store=store_factory(),
                            legacy_entries=legacy_entries,
                            caddy_entries=store_factory().load(
                                authority.plan.sha256
                            ),
                        )
                except BaseException as recovery:
                    raise BaseExceptionGroup(
                        "Convergence failed with incomplete rollback and maintenance recovery was incomplete",
                        [primary, rollback_error, recovery],
                    ) from None
                raise OwnerGateCaddyCutoverError(
                    "owner_gate_caddy_forward_recovery_required"
                ) from primary
        recovery_basis = (
            "cutover_rollback" if rollback_terminal is not None else "freeze_abort"
        )
        recovery_errors: list[BaseException] = []
        try:
            with lock():
                boundary = boundary_factory()
                store = store_factory()
                local_entries = store.load(authority.plan.sha256)
                prepared_entry = _last(local_entries, "prepared")
                arm_entry = _last(local_entries, "maintenance_armed")
                if prepared_entry is not None and arm_entry is not None:
                    prepared = validate_prepare_receipt(
                        prepared_entry.value["evidence"],
                        plan=authority.plan,
                    )
                    arm = validate_maintenance_arm_receipt(
                        arm_entry.value["evidence"],
                        authority=authority,
                        prepare_receipt=prepared,
                    )
                    maintenance_proof = _force_caddy_local_post_intent_maintenance(
                        authority,
                        boundary=boundary,
                        store=store,
                        caddy_entries=local_entries,
                    )
                    _publish_pre_intent_restore_intent(
                        authority,
                        prepared=prepared,
                        arm=arm,
                        original_caddy_sha256=_sha256(
                            store.read_artifact(
                                authority.plan.sha256, "original.Caddyfile"
                            )
                        ),
                        maintenance_proof=maintenance_proof,
                        recovery_basis=recovery_basis,
                        rollback_terminal_receipt_sha256=(
                            None
                            if rollback_terminal is None
                            else rollback_terminal["receipt_sha256"]
                        ),
                        journal=legacy_journal,
                        now_unix=now_unix,
                    )
        except BaseException as exc:
            recovery_errors.append(exc)
        gateway_terminal_event = "rollback_terminal"
        gateway_terminal = rollback_terminal
        if rollback_terminal is None:
            gateway_terminal_event = "freeze_aborted"
            try:
                gateway_terminal = cutover_runner("abort-freeze")
            except BaseException as exc:
                recovery_errors.append(exc)
        if gateway_terminal is not None and not recovery_errors:
            try:
                with lock():
                    _restore_pre_intent_caddy(
                        authority,
                        boundary=boundary_factory(),
                        store=store_factory(),
                        legacy_journal=legacy_journal,
                        gateway_terminal_event=gateway_terminal_event,
                        gateway_terminal=gateway_terminal,
                        service=retirement_boundary_factory(),
                        now_unix=now_unix,
                    )
            except BaseException as exc:
                recovery_errors.append(exc)
        if recovery_errors:
            raise BaseExceptionGroup(
                "Convergence failed before intent and abort was incomplete",
                [primary, *recovery_errors],
            ) from None
        if rollback_terminal is not None:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_cutover_rolled_back_restored"
            ) from primary
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_pre_intent_aborted"
        ) from primary


def _require_release_runtime(authority: _Authority) -> None:
    cutover._require_production_runtime()
    expected = (
        cutover.PRODUCTION_RELEASE_BASE
        / f"hermes-agent-{authority.plan.value['release_revision'][:12]}"
        / "scripts/canary/owner_gate_caddy_cutover.py"
    )
    try:
        active = Path(__file__).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_release_runtime_invalid"
        ) from exc
    if active != expected or not active.is_file():
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_release_runtime_invalid"
        )


def _require_bridge_release_runtime(foundation: _BridgeFoundation) -> None:
    cutover._require_production_runtime()
    expected = (
        cutover.PRODUCTION_RELEASE_BASE
        / f"hermes-agent-{foundation.release_revision[:12]}"
        / "scripts/canary/owner_gate_caddy_cutover.py"
    )
    try:
        active = Path(__file__).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_release_runtime_invalid"
        ) from exc
    if active != expected or not active.is_file():
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_release_runtime_invalid"
        )


def execute_fixed_staged(
    phase: str,
    *,
    bridge_document: Mapping[str, Any] | None = None,
    now_unix: int | None = None,
    boundary_factory: Callable[[], CaddyBoundary] = ProductionCaddyBoundary,
    store_factory: Callable[[], CaddyTransactionStore] = CaddyTransactionStore,
    request_boundary_factory: Callable[[], LegacyRequestBoundary] = (
        ProductionLegacyRequestBoundary
    ),
    service_boundary_factory: Callable[[], LegacyServiceBoundary] = (
        ProductionLegacyServiceBoundary
    ),
    retirement_boundary_factory: Callable[[], LegacyRetirementBoundary] = (
        ProductionLegacyServiceBoundary
    ),
    legacy_journal_factory: Callable[[], cutover.CutoverJournal] = (
        cutover.RootCutoverJournal
    ),
    cutover_runner: Callable[[str], Mapping[str, Any]] = (
        cutover.execute_fixed_staged
    ),
    lock: Callable[[], AbstractContextManager[Any]] = activation._host_activation_lock,
    require_release_runtime: bool = True,
) -> Mapping[str, Any]:
    """Execute one allow-listed phase from only fixed staged authority."""

    if phase not in {
        "prepare-bridge",
        "activate-bridge",
        "prepare",
        "commit",
        "converge",
    }:
        raise OwnerGateCaddyCutoverError("owner_gate_caddy_phase_invalid")
    current = int(time.time()) if now_unix is None else now_unix
    if phase in {"prepare-bridge", "activate-bridge"}:
        if bridge_document is None:
            raise OwnerGateCaddyCutoverError(
                "owner_gate_caddy_bridge_input_required"
            )
        foundation = validate_bridge_bootstrap_input(bridge_document)
        bridge_clock = (
            time.time if now_unix is None else lambda: float(current)
        )
        if require_release_runtime:
            _require_bridge_release_runtime(foundation)
        with lock():
            boundary = boundary_factory()
            store = store_factory()
            if phase == "prepare-bridge":
                return prepare_bridge_bootstrap(
                    bridge_document,
                    boundary=boundary,
                    store=store,
                    request_boundary=request_boundary_factory(),
                    now_unix=current,
                    freshness_clock=bridge_clock,
                )
            return activate_bridge_bootstrap(
                bridge_document,
                boundary=boundary,
                store=store,
                service=service_boundary_factory(),
                now_unix=current,
                freshness_clock=bridge_clock,
            )
    if bridge_document is not None:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_input_forbidden"
        )
    legacy_journal = legacy_journal_factory()
    authority = _load_authority(
        now_unix=current,
        journal=legacy_journal,
        staged_loader=cutover._load_staged_json,
    )
    if require_release_runtime:
        _require_release_runtime(authority)
    if phase == "converge":
        return converge_cutover(
            authority,
            boundary_factory=boundary_factory,
            store_factory=store_factory,
            legacy_journal_factory=legacy_journal_factory,
            retirement_boundary_factory=retirement_boundary_factory,
            cutover_runner=cutover_runner,
            lock=lock,
            now_unix=current,
        )
    with lock():
        boundary = boundary_factory()
        store = store_factory()
        if phase == "prepare":
            return prepare_cutover(
                authority,
                boundary=boundary,
                store=store,
                legacy_journal=legacy_journal,
                now_unix=current,
            )
        retire_legacy_service(
            authority,
            boundary=boundary,
            store=store,
            legacy_journal=legacy_journal,
            service=retirement_boundary_factory(),
            now_unix=current,
        )
        return commit_cutover(
            authority,
            boundary=boundary,
            store=store,
            legacy_journal=legacy_journal,
            now_unix=current,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed owner-gate Caddy cutover phase"
    )
    parser.add_argument(
        "phase",
        choices=(
            "prepare-bridge",
            "activate-bridge",
            "prepare",
            "commit",
            "converge",
        ),
    )
    return parser


def _read_bridge_document() -> Mapping[str, Any]:
    try:
        raw = sys.stdin.buffer.read(MAX_JSON_BYTES + 1)
    except OSError as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_input_invalid"
        ) from exc
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_input_invalid"
        )
    try:
        value = json.loads(
            raw.decode("ascii", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_input_invalid"
        ) from exc
    if not isinstance(value, Mapping) or raw != _canonical(value):
        raise OwnerGateCaddyCutoverError(
            "owner_gate_caddy_bridge_input_invalid"
        )
    validate_bridge_bootstrap_input(value)
    return copy.deepcopy(dict(value))


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        document = (
            _read_bridge_document()
            if arguments.phase in {"prepare-bridge", "activate-bridge"}
            else None
        )
        receipt = execute_fixed_staged(
            arguments.phase, bridge_document=document
        )
    except BaseException as exc:
        code = (
            str(exc)
            if isinstance(exc, OwnerGateCaddyCutoverError)
            and re.fullmatch(r"[a-z0-9_]+", str(exc)) is not None
            else "owner_gate_caddy_cutover_failed"
        )
        print(
            json.dumps(
                {"ok": False, "error_code": code},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "BRIDGE_INPUT_SCHEMA",
    "BRIDGE_RECEIPT_SCHEMA",
    "BRIDGE_REQUEST_SCHEMA",
    "OwnerGateCaddyCutoverError",
    "MAINTENANCE_OBSERVATION_SCHEMA",
    "PREPARE_RECEIPT_SCHEMA",
    "TERMINAL_RECEIPT_SCHEMA",
    "activate_bridge_bootstrap",
    "arm_pre_intent_maintenance",
    "commit_cutover",
    "converge_cutover",
    "execute_fixed_staged",
    "main",
    "prepare_bridge_bootstrap",
    "prepare_cutover",
    "retire_legacy_service",
    "validate_bridge_bootstrap_input",
    "validate_bridge_bootstrap_receipt",
    "validate_bridge_bootstrap_request",
    "validate_bridge_receipt",
    "validate_bridge_request_receipt",
    "validate_prepare_receipt",
    "validate_maintenance_observation",
    "validate_maintenance_arm_receipt",
    "validate_legacy_retirement_receipt",
    "validate_convergence_receipt",
    "validate_terminal_receipt",
]


if __name__ == "__main__":  # pragma: no cover - installed runtime entry.
    raise SystemExit(main())
