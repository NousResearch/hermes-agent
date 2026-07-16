#!/usr/bin/env python3
"""Owner-side credential edge for the isolated full Cloud Muncho canary.

This module deliberately contains no model, routing, or task semantics.  Its
secret-free stopped-release action uses only fixed IAP argv to publish one
approved fork revision while every canary service remains inactive.  Its live
action does four mechanical things after an exact, fresh coordinator
authorization:

* reads the canary-only Discord credential from the owner's inherited stdin;
* asks the sealed remote coordinator to install that opaque credential;
* sends the exact plan-bound owner approval to that same coordinator process;
* accepts success only after stopped terminal truth and token retirement.

The separate explicit Phase-B action runs only behind the coordinator's exact
stopped-service gate. It creates approval-bound Cloud SQL credentials solely
in owner-process memory, streams them over IAP/SSH stdin, and reconciles their
terminal state before success. The normal live action never creates or
receives a database administrator or bootstrap credential.

Secret values are never accepted on argv, placed in the environment, written
to a file, logged, or included in a receipt. A temporary administrator is
not considered cleaned up until a post-delete users.list proves it absent.
The owner launcher is intentionally limited to Darwin and Linux and fails
closed on every other platform before reading a secret.
"""

from __future__ import annotations

import argparse
import base64
import copy
import ctypes
import errno
import getpass
import hashlib
import json
import os
import pwd
import re
import resource
import selectors
import secrets
import shlex
import shutil
import signal
import ssl
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Mapping, Protocol, Sequence

PROJECT = "adventico-ai-platform"
ZONE = "europe-west3-a"
VM_NAME = "muncho-canary-v2-01"
VM_INSTANCE_ID = "9153645328899914617"
OS_LOGIN_USERNAME = "lomliev_adventico_com"
OS_LOGIN_PROFILE_ID = "114674870412628413680"
STOPPED_RELEASE_SOURCE_REPOSITORY = "https://github.com/lomliev/hermes-agent.git"
STOPPED_RELEASE_SOURCE_BASE = "/opt/muncho-canary-source"
STOPPED_RELEASE_EVIDENCE_BASE = "/var/lib/muncho-canary-release-evidence"
STOPPED_RELEASE_PLAN_SCHEMA = "muncho-canary-stopped-release-plan.v1"
STOPPED_RELEASE_RECEIPT_SCHEMA = "muncho-canary-stopped-release-publication.v1"
COORDINATOR_INPUT_PUBLICATION_SCHEMA = (
    "muncho-full-canary-coordinator-input-publication.v1"
)
COORDINATOR_INPUT_PATH = "/etc/muncho/full-canary/coordinator-input.json"
COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH = (
    "/etc/muncho/full-canary/coordinator-input-publication.json"
)
STOPPED_RELEASE_HOST_RECEIPT_PATH = "/etc/muncho/full-canary/host-identity.json"
STOPPED_RELEASE_PYTHON_VERSION = "3.11.15"
HOST_RECEIPT_ROTATION_PLAN_SCHEMA = (
    "muncho-full-canary-host-identity-rotation-plan.v1"
)
HOST_RECEIPT_ROTATION_RECEIPT_SCHEMA = (
    "muncho-full-canary-host-identity-rotation-receipt.v1"
)
HOST_RECEIPT_ROTATION_ROOT = (
    "/etc/muncho/full-canary/host-identity-rotations"
)
HOST_RECEIPT_ROTATION_MODULE = (
    "scripts.canary.host_identity_receipt_rotation"
)
FIXTURE_PUBLICATION_PLAN_SCHEMA = "muncho-full-canary-fixture-publication-plan.v1"
FIXTURE_PUBLICATION_RECEIPT_SCHEMA = (
    "muncho-full-canary-fixture-publication-receipt.v1"
)
FIXTURE_PUBLICATION_MODULE = (
    "gateway.canonical_full_canary_fixture_publisher"
)
FIXTURE_PUBLICATION_OWNER_DISCORD_USER_ID = "1279454038731264061"
FIXTURE_PUBLICATION_GUILD_ID = "1282725267068157972"
FIXTURE_PUBLICATION_CHANNEL_ID = "1526858760100909066"
FIXTURE_PUBLICATION_PROMPT_SHA256 = (
    "5e89bcb3442002097955b6e7fdb0b9baf2a520bb0a873515ee10c4023c600b0c"
)
FIXTURE_PUBLICATION_ROOT = (
    "/var/lib/muncho-full-canary/fixture-publications"
)
FIXTURE_PUBLICATION_PATH = "/etc/muncho/full-canary/fixture.json"
FIXTURE_WRITER_PUBLIC_KEY_PATH = (
    "/etc/muncho/keys/writer-capability-public.pem"
)
FIXTURE_EDGE_PUBLIC_KEY_PATH = (
    "/etc/muncho/keys/discord-edge-receipt-public.pem"
)
WRITER_PREFLIGHT_PLAN_SCHEMA = "muncho-writer-preflight-publication-plan.v2"
WRITER_PREFLIGHT_RECEIPT_SCHEMA = "muncho-writer-preflight-publication.v3"
WRITER_PREFLIGHT_FAILURE_SCHEMA = (
    "muncho-writer-preflight-publication-failure.v2"
)
WRITER_PREFLIGHT_MODULE = "gateway.canonical_writer_preflight_publisher"
WRITER_PREFLIGHT_EVIDENCE_BASE = (
    "/var/lib/muncho-writer-canary-evidence/staged-publication"
)
WRITER_PREFLIGHT_OWNER_DISCORD_USER_ID = "1279454038731264061"
WRITER_PREFLIGHT_DATABASE_TLS_SERVER_NAME = (
    "14-0d81ef63-2cac-4a64-84ad-c4f58c0cfd56.europe-west3.sql.goog"
)
WRITER_ACTIVATION_BRIDGE_MODULE = (
    "gateway.canonical_writer_activation_bridge"
)
WRITER_ACTIVATION_MODULE = "gateway.canonical_writer_activation"
WRITER_PLANNER_MODULE = "gateway.canonical_writer_planner"
WRITER_AUTHORITY_FRAME_MAGIC = b"MWA1"
WRITER_AUTHORITY_FRAME_SCHEMA = "muncho-writer-owner-authority-frame.v1"
WRITER_AUTHORITY_STAGE_RECEIPT_SCHEMA = (
    "muncho-writer-owner-authority-stage-receipt.v1"
)
WRITER_ACTIVATION_OWNER_RECEIPT_SCHEMA = (
    "muncho-writer-stopped-owner-activation.v1"
)
WRITER_NATIVE_PLAN_PATH = (
    "/etc/muncho/writer-activation/native-observation-plan.json"
)
WRITER_STAGED_NATIVE_PLAN_PATH = (
    "/etc/muncho/writer-activation/staged/native-observation-plan.json"
)
WRITER_FINAL_PLAN_PATH = "/etc/muncho/writer-activation/activation-plan.json"
WRITER_STAGED_FINAL_PLAN_PATH = (
    "/etc/muncho/writer-activation/staged/activation-plan.json"
)
WRITER_STAGED_OWNER_APPROVAL_PATH = (
    "/etc/muncho/writer-activation/staged/owner-approval.json"
)
WRITER_STAGED_EXTERNAL_IAM_PATH = (
    "/etc/muncho/writer-activation/staged/external-iam-receipt.json"
)
WRITER_EXTERNAL_IAM_LIVE_PATH = (
    "/run/muncho-canonical-preflight/external-iam-receipt.json"
)
WRITER_OWNER_APPROVAL_ROOT = "/etc/muncho/writer-activation/approvals"
SQL_INSTANCE = "muncho-canary-pg18-v2"
TEMPORARY_ADMIN_AUTHORITY_RECEIPT_SCHEMA = (
    "muncho-cloud-sql-temporary-admin-authority.v2"
)
SCHEMA_RECONCILIATION_DATABASE_ROLES = (
    "canonical_brain_migration_owner",
    "canonical_brain_writer",
)
CANARY_BOOTSTRAP_LOGIN = "canonical_brain_canary_bootstrap_login"
CANARY_BOOTSTRAP_DATABASE_ROLE = "canonical_brain_canary_bootstrap"
CANARY_BOOTSTRAP_AUTHORITY_RECEIPT_SCHEMA = (
    "muncho-cloud-sql-bootstrap-login-authority.v1"
)
CANARY_BOOTSTRAP_ABSENCE_EVIDENCE_SCHEMA = (
    "muncho-cloud-sql-bootstrap-login-absence-evidence.v1"
)
DATABASE_HOST = "10.91.0.3"
DATABASE_PORT = 5432
DATABASE_NAME = "muncho_canary_brain"
OWNER_RECEIPT_SCHEMA = "muncho-full-canary-owner-launch-receipt.v2"
DISCORD_INSTALL_GATE_SCHEMA = "muncho-full-canary-discord-token-install-gate.v1"
DISCORD_INSTALL_RECEIPT_SCHEMA = "muncho-full-canary-discord-token-install-receipt.v1"
DISCORD_RETIREMENT_RECEIPT_SCHEMA = (
    "muncho-full-canary-discord-token-retirement-receipt.v1"
)
DISCORD_RETIREMENT_GATE_SCHEMA = "muncho-full-canary-discord-token-retirement-gate.v1"
DISCORD_RETIREMENT_ACK_SCHEMA = "muncho-full-canary-owner-discord-retirement-ack.v1"
RECOVERY_GATE_SCHEMA = "muncho-full-canary-recovery-takeover-gate.v1"
RECOVERY_ACK_SCHEMA = "muncho-full-canary-owner-recovery-takeover-ack.v1"
RECOVERY_WORKER_LEASE_SCHEMA = "muncho-full-canary-recovery-worker-lease.v1"
RECOVERY_SECRET_GATE_SCHEMA = "muncho-full-canary-recovery-admin-secret-gate.v1"
RECOVERY_WORKER_COMPLETION_SCHEMA = "muncho-full-canary-recovery-worker-completion.v1"
RECOVERY_CONCURRENT_LOSER_RECEIPT_SCHEMA = (
    "muncho-full-canary-recovery-worker-claim-lost-receipt.v1"
)
RECOVERY_FINALIZE_PENDING_RECEIPT_SCHEMA = (
    "muncho-full-canary-recovery-finalize-pending-receipt.v1"
)
RECOVERY_RECEIPT_SCHEMA = "muncho-full-canary-recovery-receipt.v2"
COORDINATOR_SECRET_GATE_SCHEMA = "muncho-full-canary-coordinator-secret-gate.v1"
COORDINATOR_RECEIPT_SCHEMA = "muncho-full-canary-coordinator-receipt.v1"
COORDINATOR_FAILURE_SCHEMA = (
    "muncho-full-canary-session-bound-coordinator-failure.v1"
)
BOOTSTRAP_INSTALL_COMPENSATION_PREFLIGHT_SCHEMA = (
    "muncho-full-canary-bootstrap-install-compensation-preflight.v1"
)
BOOTSTRAP_INSTALL_COMPENSATION_FRAME_SCHEMA = (
    "muncho-full-canary-owner-bootstrap-install-compensation-frame.v1"
)
BOOTSTRAP_INSTALL_COMPENSATION_FRAME_MAGIC = b"MCC1"
PHASE_B_OWNER_REQUEST_SCHEMA = "muncho-canonical-writer-phase-b-owner-request.v2"
PHASE_B_OWNER_RESPONSE_SCHEMA = "muncho-canonical-writer-phase-b-owner-response.v2"
PHASE_B_APPLY_RECEIPT_SCHEMA = "muncho-canonical-writer-phase-b-apply-receipt.v1"
PHASE_B_APPLY_GATE_SCHEMA = "muncho-canonical-writer-phase-b-owner-gate.v2"
PHASE_B_LIVE_GATE_SCHEMA = "muncho-full-canary-phase-b-live-gate.v1"
PHASE_B_OWNER_FRAME_MAGIC = b"MPB1"
PHASE_B_OWNER_FRAME_SCHEMA = "MPB1-u32be-json-u32be-opaque.v1"
PHASE_B_MAX_ROUNDS = 16
PHASE_B_MAX_REQUEST_BYTES = 512 * 1024
PHASE_B_MAX_RESPONSE_BYTES = 512 * 1024
PHASE_B_CREDENTIAL_BYTES = 64
PHASE_B_OWNER_PRIVATE_KEY_PATH = Path(
    "/Users/emillomliev/.ssh/skyvision_mac_ops_ed25519"
)
PHASE_B_OWNER_PUBLIC_KEY_PATH = Path(
    "/Users/emillomliev/.ssh/skyvision_mac_ops_ed25519.pub"
)
PHASE_B_OWNER_PUBLIC_KEY_COMMENT = "skyvision-mac-ops-emil-20260710"
PHASE_B_OWNER_PUBLIC_KEY_FINGERPRINT = (
    "SHA256:7Ea5WNys9ui7FL/p0FlOnL1ZLr6NPFuewekwqRw/rdw"
)
PHASE_B_PINNED_APPROVAL_SOURCE_SHA256 = hashlib.sha256(
    PHASE_B_OWNER_PUBLIC_KEY_FINGERPRINT.encode("ascii")
).hexdigest()
PHASE_B_OWNER_PUBLIC_KEY_UID = 501
PHASE_B_OWNER_PUBLIC_KEY_GID = 20
PHASE_B_APPROVAL_SSHSIG_NAMESPACE = (
    "muncho-canonical-writer-phase-b-owner-v2"
)
PHASE_B_SOURCE_AUTH_SSHSIG_NAMESPACE = (
    "muncho-canonical-writer-phase-b-source-auth-v1"
)
SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_SSHSIG_NAMESPACE = (
    "muncho-canonical-writer-schema-reconciliation-admin-preflight-owner-v2"
)
SCHEMA_RECONCILIATION_PREFLIGHT_AUTHORIZATION_SSHSIG_NAMESPACE = (
    "muncho-canonical-writer-schema-reconciliation-preflight-authorization-owner-v2"
)
SCHEMA_RECONCILIATION_ADMIN_CLEANUP_SSHSIG_NAMESPACE = (
    "muncho-canonical-writer-schema-reconciliation-admin-cleanup-owner-v2"
)
SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_MAGIC = b"MSA2"
SCHEMA_RECONCILIATION_PREFLIGHT_AUTHORIZATION_MAGIC = b"MSP2"
SCHEMA_RECONCILIATION_ADMIN_CLEANUP_MAGIC = b"MSC2"
SCHEMA_RECONCILIATION_CREDENTIAL_BYTES = 64
SCHEMA_RECONCILIATION_MIN_GATE_REMAINING_SECONDS = 900
_SCHEMA_RECONCILIATION_TRANSITIONS = {
    ("empty", "exact_old_missing_one_helper"): "reconcile_missing_helper",
    ("empty", "exact_target"): "adopt_existing_target",
    (
        "authorized_intent",
        "exact_old_missing_one_helper",
    ): "resume_durable_intent",
    ("authorized_intent", "exact_target"): "terminalize_durable_intent",
    ("terminal", "exact_target"): "reattest_terminal",
}
PHASE_B_OWNER_AUTHORITY_KIND = "skyvision_mac_ops_ed25519"
PHASE_B_OWNER_APPROVAL_ROOT = Path(
    "/Users/emillomliev/.hermes/owner-approvals/phase-b"
)
PHASE_B_SSH_KEYGEN = Path("/usr/bin/ssh-keygen")
PHASE_B_MAX_SSHSIG_BYTES = 4096
PHASE_B_MAX_PUBLIC_KEY_BYTES = 4096
PHASE_B_SIGN_TIMEOUT_SECONDS = 15.0
OWNER_APPROVAL_REQUEST_SCHEMA = "muncho-full-canary-owner-approval-request.v1"
SESSION_BOUND_APPROVAL_REQUEST_SCHEMA = (
    "muncho-full-canary-session-bound-owner-approval-request.v1"
)
SESSION_BOUND_COORDINATOR_RECEIPT_SCHEMA = (
    "muncho-full-canary-session-bound-coordinator-receipt.v1"
)
SESSION_BOUND_OWNER_RECEIPT_SCHEMA = (
    "muncho-full-canary-session-bound-owner-receipt.v1"
)
FINAL_APPROVAL_INSTALL_RECEIPT_SCHEMA = (
    "muncho-full-canary-final-approval-install-receipt.v1"
)
FINAL_APPROVAL_CANCEL_RECEIPT_SCHEMA = (
    "muncho-full-canary-final-approval-cancel-receipt.v2"
)
ADMIN_USERNAME_PREFIX = "muncho_canary_admin_"
ADMIN_FRAME_MAGIC = b"MCA2"
DISCORD_FRAME_MAGIC = b"DCT1"
OWNER_DISCORD_INPUT_MAGIC = b"MDO1"
DISCORD_TOKEN_PATH = "/etc/muncho/discord-edge-credentials/bot-token"
DISCORD_RETIREMENT_RECEIPT_PATH = (
    "/etc/muncho/full-canary/discord-token-retirement-receipt.json"
)
DISCORD_FRAME_SCHEMA = "DCT1-u32be-opaque-eof.v1"
ADMIN_FRAME_SCHEMA = "MCA2-u16be-u32be-utf8-eof.v1"
FINAL_APPROVAL_FRAME_MAGIC = b"MFA1"
FINAL_APPROVAL_FRAME_SCHEMA = "MFA1-u32be-canonical-json-eof.v1"
DISCORD_RETIREMENT_ACK_FRAME_MAGIC = b"DRA1"
DISCORD_RETIREMENT_ACK_FRAME_SCHEMA = "DRA1-u32be-canonical-json-eof.v1"
RECOVERY_ACK_FRAME_MAGIC = b"MRA1"
RECOVERY_ACK_FRAME_SCHEMA = "MRA1-u32be-canonical-json-no-secret.v1"
RECOVERY_ADMIN_FRAME_MAGIC = b"MRC2"
RECOVERY_ADMIN_FRAME_SCHEMA = "MRC2-gate-sha256-nonce-sha256-u16be-u32be-utf8-eof.v1"
APPROVAL_REQUEST_PATH = "/etc/muncho/full-canary/approval-request.json"
FINAL_APPROVAL_PATH = "/etc/muncho/full-canary/owner-approval.json"
TRUSTED_RUNTIME_BOOTSTRAP_RECEIPT_SCHEMA = (
    "muncho-full-canary-owner-trusted-runtime-bootstrap-receipt.v1"
)
TRUSTED_SDK_PUBLICATION_INTENT_SCHEMA = (
    "muncho-full-canary-owner-trusted-sdk-publication-intent.v1"
)

_ADMIN_PASSWORD_BYTES = 48
_ADMIN_PASSWORD_MIN_UTF8 = 24
_ADMIN_PASSWORD_MAX_UTF8 = 4096
_DISCORD_TOKEN_MAX_BYTES = 4096
_HTTP_RESPONSE_MAX_BYTES = 2 * 1024 * 1024
_CANARY_IAM_JSON_MAX_BYTES = 8 * 1024 * 1024
_HTTP_TIMEOUT_SECONDS = 30.0
_OPERATION_TIMEOUT_SECONDS = 180.0
_GATE_MAX_FUTURE_SECONDS = 900
FINAL_APPROVAL_MAX_WAIT_SECONDS = 240
_FINAL_APPROVAL_DELIVERY_RESERVE_SECONDS = 30
_FINAL_APPROVAL_TERMINAL_CLEANUP_GRACE_SECONDS = 300
_HBA_EXPIRY_SAFETY_MARGIN_SECONDS = 30
_SECRET_FRAME_TRANSMIT_MARGIN_SECONDS = 30
_MAX_JSON_LINE_BYTES = 256 * 1024
_WRITER_AUTHORITY_MAX_FRAME_BYTES = 192 * 1024
# Exact upper bound for the largest secret frame the transport may carry:
# MCB1's 76-byte header + 128 KiB authority receipt + 4 KiB password.
# Individual frame decoders retain their own tighter semantic bounds.
_MAX_REMOTE_SECRET_FRAME_BYTES = max(
    76 + 128 * 1024 + 4 * 1024,
    12 + PHASE_B_MAX_RESPONSE_BYTES + PHASE_B_CREDENTIAL_BYTES,
)
_STOPPED_RELEASE_ACTIVATION_PATHS = (
    "/etc/muncho/writer-activation/staged/writer.json",
    "/etc/muncho/writer-activation/staged/gateway.yaml",
    "/etc/muncho/writer-activation/staged/native-observation-plan.json",
    "/etc/muncho/writer-activation/staged/activation-plan.json",
    "/etc/muncho/writer-activation/staged/owner-approval.json",
    "/etc/muncho/writer-activation/staged/external-iam-receipt.json",
    "/etc/muncho/writer-activation/staged/muncho-canonical-writer.service",
    (
        "/etc/muncho/writer-activation/staged/"
        "muncho-canonical-writer-phase-b-readiness.service"
    ),
    "/etc/muncho/writer-activation/staged/hermes-cloud-gateway.service",
    "/etc/muncho/writer-activation/native-observation-plan.json",
    "/etc/muncho/writer-activation/activation-plan.json",
    "/etc/muncho/writer-activation/deployment-manifest.json",
    "/etc/systemd/system/muncho-canonical-writer.service",
    "/etc/systemd/system/muncho-canonical-writer-phase-b-readiness.service",
    "/etc/systemd/system/hermes-cloud-gateway.service",
    "/etc/systemd/system/muncho-canonical-writer-export.service",
    "/etc/tmpfiles.d/muncho-canonical-writer.conf",
    "/etc/muncho-canonical-writer/writer.json",
    "/etc/hermes/config.yaml",
)
_STOPPED_RELEASE_UNITS = (
    "muncho-canary-discord-edge.service",
    "muncho-discord-egress.service",
    "muncho-canonical-writer.service",
    "muncho-canonical-writer-phase-b-readiness.service",
    "muncho-canonical-writer-export.service",
    "muncho-canonical-writer-export.timer",
    "hermes-cloud-gateway.service",
)
_STOPPED_RELEASE_SERVICE_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "MainPID",
    "FragmentPath",
    "DropInPaths",
)
_PINNED_CA_CANDIDATES = (
    "/etc/ssl/cert.pem",
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
)
_FIXED_OWNER_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
_GCLOUD_ACTIVE_CONFIG_NAME = "adventico-ai-platform-admin"
_GCLOUD_CONFIG_RELATIVE = ".config/gcloud"
_GCLOUD_MAX_CONFIG_BYTES = 64 * 1024
_GCLOUD_MAX_SDK_ENTRIES = 100_000
_GCLOUD_MAX_SDK_BYTES = 2 * 1024 * 1024 * 1024
_GCLOUD_MAX_SDK_FILE_BYTES = 256 * 1024 * 1024
_GCLOUD_SDK_VERSION = "569.0.0"
_GCLOUD_SDK_ARCHIVE_URL = (
    "https://storage.googleapis.com/cloud-sdk-release/"
    "google-cloud-cli-569.0.0-darwin-arm.tar.gz"
)
_GCLOUD_SDK_ARCHIVE_BYTES = 60_511_521
_GCLOUD_SDK_ARCHIVE_SHA256 = (
    "2d4ab8eb0a9362a69feabade6df4163763cd989cb840dc3f7ced5ac24dde6e67"
)
_TRUSTED_SDK_RELATIVE = ".hermes/trusted/google-cloud-sdk-569.0.0"
_TRUSTED_SDK_PUBLICATION_INTENT_RELATIVE = (
    ".hermes/trusted/trusted-sdk-publication-569.0.0-"
    "2d4ab8eb0a9362a69feabade6df4163763cd989cb840dc3f7ced5ac24dde6e67.json"
)
_TRUSTED_PYTHON_RELATIVE = (
    ".local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11"
)
_TRUSTED_PYTHON_VERSION = "3.11.15"
_MAX_LAUNCHER_MODULE_BYTES = 2 * 1024 * 1024
_TRUSTED_PYTHON_DEPENDENCIES = (
    "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation",
    "/usr/lib/libSystem.B.dylib",
    "/usr/lib/libncurses.5.4.dylib",
    "/usr/lib/libpanel.5.4.dylib",
    "/System/Library/Frameworks/SystemConfiguration.framework/Versions/A/"
    "SystemConfiguration",
    "/usr/lib/libedit.3.dylib",
    "/usr/lib/libz.1.dylib",
)
_GCLOUD_PYTHON_ISOLATION_ARGS = (
    "-I",
    "-S",
    "-B",
    "-X",
    "pycache_prefix=/var/empty/muncho-canary",
)
_AMBIGUOUS_CLOUD_SQL_CREATE_ERRORS = frozenset({
    "cloud_sql_operation_failed",
    "cloud_sql_mutation_evidence_unconfirmed",
    "cloud_sql_operation_timeout",
    "google_api_ambiguous_status",
    "google_api_response_oversized",
    "google_api_unavailable",
    "invalid_cloud_sql_operation",
    "invalid_json",
})

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_SHA = re.compile(r"^[0-9a-f]{40}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_ADMIN_USERNAME = re.compile(r"^muncho_canary_admin_[0-9a-f]{16}$")
_OPERATION_NAME = re.compile(r"^[A-Za-z0-9._~-]{1,256}$")
_OPERATION_TYPE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_CLOUD_SQL_USER_OPERATION_TYPES = frozenset({
    "CREATE_USER",
    "UPDATE_USER",
    "DELETE_USER",
})
_PHASE_B_AUTHORITY_CONTEXT_KEYS = frozenset({
    "release_sha",
    "coordinator_input_sha256",
    "activation_plan_sha256",
    "writer_activation_receipt_sha256",
    "activation_owner_approval_sha256",
    "activation_approval_issued_at_unix",
    "activation_approval_expires_at_unix",
    "native_observation_plan_sha256",
    "native_observation_receipt_sha256",
    "native_observation_approval_sha256",
    "native_approval_issued_at_unix",
    "native_approval_expires_at_unix",
    "external_iam_policy_sha256",
    "external_iam_receipt_sha256",
    "config_collector_receipt_sha256",
    "gateway_config_intent_sha256",
    "edge_config_intent_sha256",
    "fixture_intent_sha256",
    "host_identity_receipt_sha256",
    "authority_sources",
    "authority_sources_sha256",
    "approval_source_sha256",
    "owner_subject_sha256",
    "owner_resume_public_key_ed25519_hex",
    "owner_resume_key_id",
    "owner_resume_public_key_file_sha256",
    "owner_resume_public_fingerprint",
})
_PHASE_B_AUTHORITY_SOURCE_LABELS = frozenset({
    "coordinator_input",
    "activation_plan",
    "activation_receipt",
    "activation_owner_approval",
    "native_plan",
    "native_receipt",
    "native_owner_approval",
    "external_iam_receipt",
    "config_collector_receipt",
    "gateway_config_intent",
    "edge_config_intent",
    "fixture_intent",
    "host_identity_receipt",
    "owner_resume_public_key",
})
_PHASE_B_AUTHORITY_SOURCE_KEYS = frozenset({
    "path",
    "file_sha256",
    "device",
    "inode",
    "uid",
    "gid",
    "mode",
    "size",
})
_PHASE_B_OWNER_RESUME_AUTHORITY_KEYS = frozenset({
    "public_key_ed25519_hex",
    "key_id",
    "public_key_file_sha256",
    "public_fingerprint",
    "public_key_source",
})
_PHASE_B_REQUEST_KEYS = frozenset({
    "schema",
    "frame_schema",
    "operation",
    "sequence",
    "previous_response_sha256",
    "authority_context_sha256",
    "authority_context",
    "phase_b_plan_sha256",
    "phase_b_approval_sha256",
    "boundary_kind",
    "boundary_ordinal",
    "credential_expected",
    "payload",
    "issued_at_unix",
    "expires_at_unix",
    "idempotency_key",
    "request_sha256",
})
_PHASE_B_OPERATIONS = frozenset({
    "authority_observe_initial",
    "authority_approve",
    "authority_resume_approve",
    "observe_initial",
    "observe_recovery",
    "temporary_create_or_rotate",
    "temporary_delete",
    "bootstrap_describe",
    "bootstrap_create_or_rotate",
    "observe_terminal",
})
_PHASE_B_APPLY_RECEIPT_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "phase_b_plan_sha256",
    "phase_b_approval_sha256",
    "phase_b_terminal_receipt_sha256",
    "phase_b_readiness_receipt_sha256",
    "safe_to_start",
    "completed_at_unix",
    "receipt_sha256",
})
_PHASE_B_APPLY_GATE_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "owner_subject_sha256",
    "approval_source_sha256",
    "authority_present",
    "phase_b_plan_sha256",
    "phase_b_approval_sha256",
    "phase_b_approval_sequence",
    "phase_b_incomplete_state",
    "phase_b_inspection_sha256",
    "phase_b_terminal",
    "phase_b_requires_reapproval",
    "issued_at_unix",
    "expires_at_unix",
    "gate_sha256",
})
_PHASE_B_LIVE_GATE_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "owner_subject_sha256",
    "approval_source_sha256",
    "phase_b_readiness_anchor",
    "phase_b_readiness_anchor_sha256",
    "issued_at_unix",
    "expires_at_unix",
    "gate_sha256",
})
_SESSION_BOUND_APPROVAL_REQUEST_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "full_canary_plan_sha256",
    "staged_plan_path",
    "staged_plan_file_sha256",
    "fixture_sha256",
    "phase_b_readiness_anchor_sha256",
    "phase_b_approval_sha256",
    "owner_subject_sha256",
    "approval_source_sha256",
    "requested_at_unix",
    "owner_input_cutoff_unix",
    "approval_deadline_unix",
    "approval_path",
    "final_approval_frame_schema",
    "request_sha256",
})
_SESSION_BOUND_COORDINATOR_RECEIPT_KEYS = frozenset({
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
    "live_driver_result",
    "live_driver_receipt_sha256",
    "services_stopped",
    "discord_token_retired",
    "temporary_admin_created",
    "bootstrap_credential_created",
    "completed_at_unix",
    "receipt_sha256",
})
_COORDINATOR_FAILURE_KEYS = frozenset({
    "schema",
    "ok",
    "phase",
    "command",
    "error_code",
    "release_sha",
    "coordinator_input_sha256",
    "cleanup_status",
    "discord_token_removed",
    "services_stopped",
    "obsolete_process_journal_absent",
    "completed_at_unix",
    "receipt_sha256",
})
_COORDINATOR_INPUT_PUBLICATION_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "coordinator_input_path",
    "coordinator_input_file_sha256",
    "publication_receipt_path",
    "owner_uid",
    "group_gid",
    "mode",
    "published_at_unix",
    "receipt_sha256",
})
_COORDINATOR_COMMANDS = frozenset({
    "publish-coordinator-input",
    "preflight-phase-b-apply",
    "preflight-phase-b-live-run",
    "phase-b-apply",
    "run",
    "install-discord-token",
    "stop-and-retire-discord-token",
})
_PHASE_B_READINESS_ANCHOR_KEYS = frozenset({
    "phase_b_release_revision",
    "phase_b_plan_sha256",
    "phase_b_approval_sha256",
    "phase_b_terminal_receipt_sha256",
    "phase_b_foundation_generation_sha256",
    "phase_b_readiness_receipt_sha256",
    "phase_b_readiness_handoff_file_sha256",
    "phase_b_readiness_sequence",
})
_PHASE_B_INCOMPLETE_HEAD_KEYS = frozenset({
    "schema",
    "plan_sha256",
    "intent_sha256",
    "owner_subject_sha256",
    "approval_source_sha256",
    "approval_sequence",
    "approval_sha256",
    "approval_expires_at_unix",
    "pending_source_sequence",
    "pending_source_authentication_sha256",
    "journal_entry_count",
    "journal_head_sha256",
    "journal_head_event",
    "journal_head_recorded_at_unix",
    "terminal",
    "incomplete_state",
    "resume_eligible",
    "fresh_head",
    "requires_reapproval",
    "mutation_authorized",
    "inspected_at_unix",
    "inspection_sha256",
})
_DISCORD_INSTALL_GATE_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "coordinator_input_sha256",
    "discord_token_install_approval_sha256",
    "owner_subject_sha256",
    "release_sha",
    "token_path",
    "edge_uid",
    "edge_gid",
    "expires_at_unix",
    "frame_schema",
    "gate_sha256",
})
_DISCORD_INSTALL_RECEIPT_KEYS = frozenset({
    "schema",
    "ok",
    "release_sha",
    "coordinator_input_sha256",
    "discord_token_install_approval_sha256",
    "owner_subject_sha256",
    "token_path",
    "device",
    "inode",
    "owner_uid",
    "group_gid",
    "mode",
    "size",
    "link_count",
    "content_or_digest_recorded",
    "installed_at_unix",
    "receipt_sha256",
})
_DISCORD_RETIREMENT_RECEIPT_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "discord_token_install_receipt_sha256",
    "token_path",
    "token_device",
    "token_inode",
    "services_stopped_proven",
    "services_enabled",
    "token_removed",
    "install_receipt_removed",
    "prepared_at_unix",
    "retired_at_unix",
    "receipt_sha256",
})
_DISCORD_RETIREMENT_GATE_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "owner_subject_sha256",
    "discord_token_install_receipt_sha256",
    "token_device",
    "token_inode",
    "process_lease_absent",
    "services_stopped_proven",
    "frame_schema",
    "expires_at_unix",
    "gate_sha256",
})
_DISCORD_RETIREMENT_ACK_KEYS = frozenset({
    "schema",
    "scope",
    "release_sha",
    "coordinator_input_sha256",
    "owner_subject_sha256",
    "retirement_gate_sha256",
    "discord_token_install_receipt_sha256",
    "token_device",
    "token_inode",
    "nonce_sha256",
    "approved_at_unix",
    "expires_at_unix",
    "ack_sha256",
})
_FINAL_APPROVAL_KEYS = frozenset({
    "schema",
    "scope",
    "plan_sha256",
    "authority_kind",
    "cryptographic_owner_proof",
    "owner_subject_sha256",
    "approval_source_sha256",
    "nonce_sha256",
    "approved_at_unix",
    "expires_at_unix",
})
_LIVE_DRIVER_RESULT_KEYS = frozenset({
    "schema",
    "ok",
    "release_sha",
    "full_canary_plan_sha256",
    "canary_run_id",
    "evidence_path",
    "evidence_sha256",
    "offline_invariant_receipt",
    "lifecycle_verification_receipt",
    "discord_ingress_claimed",
})
_STABLE_CODE = re.compile(r"^[a-z][a-z0-9_]{2,127}$")
_TLS_SERVER_NAME = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.europe-west3\.sql\.goog$"
)
_FORBIDDEN_RECEIPT_VALUE_KEYS = frozenset({
    "password",
    "token",
    "secret",
    "credential_value",
    "bot_token",
    "discord_token",
})
_ALLOWED_SECRET_STATUS_OR_BINDING_KEYS = frozenset({
    "discord_token_install_approval_sha256",
    "discord_token_removed",
})


class OwnerLauncherError(RuntimeError):
    """Stable secret-free failure used at the owner security boundary."""

    def __init__(self, code: str) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", code):
            code = "owner_launcher_failed"
        self.code = code
        super().__init__(code)


class CleanupBlocked(OwnerLauncherError):
    """Deletion was not proven; the canary must remain fail-closed."""

    def __init__(self, cause_code: str = "cleanup_blocked") -> None:
        super().__init__("cleanup_blocked")
        self.cause_code = (
            cause_code
            if _STABLE_CODE.fullmatch(cause_code) is not None
            else "cleanup_blocked"
        )


class CloudSqlCreateNotCommitted(OwnerLauncherError):
    """Cloud SQL explicitly rejected create before this launcher owned a user."""


class RemoteCommandFailed(OwnerLauncherError):
    """An exact remote terminal failure was validated through EOF and exit 2."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        code = receipt.get("error_code")
        super().__init__(code if isinstance(code, str) else "remote_command_failed")
        self.receipt = dict(receipt)


_CLEANUP_FAILURE_CODES_ATTRIBUTE = "_owner_cleanup_failure_codes"


def _stable_cleanup_failure_code(error: BaseException) -> str:
    if isinstance(error, OwnerLauncherError):
        return error.code
    return "remote_termination_unconfirmed"


def _cleanup_blocked_cause(error: BaseException) -> str | None:
    if isinstance(error, CleanupBlocked):
        return error.cause_code
    return None


def _attach_cleanup_failure(
    primary: BaseException,
    cleanup_error: BaseException,
) -> None:
    codes = list(getattr(primary, _CLEANUP_FAILURE_CODES_ATTRIBUTE, ()))
    code = _stable_cleanup_failure_code(cleanup_error)
    if code not in codes:
        codes.append(code)
    try:
        setattr(primary, _CLEANUP_FAILURE_CODES_ATTRIBUTE, tuple(codes))
    except BaseException:
        pass


def _attached_cleanup_failure_codes(error: BaseException) -> tuple[str, ...]:
    value = getattr(error, _CLEANUP_FAILURE_CODES_ATTRIBUTE, ())
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or _STABLE_CODE.fullmatch(item) is None
        for item in value
    ):
        return ()
    return value


def _close_session_preserving_primary(
    session: Any,
    primary: BaseException | None,
) -> None:
    try:
        session.close()
    except BaseException as cleanup_error:
        if primary is None:
            raise
        _attach_cleanup_failure(primary, cleanup_error)


@dataclass
class _OwnerLaunchSignal(BaseException):
    pass


class _OwnerSignalFence:
    """Raise before cleanup, then record and suppress all later signals."""

    def __init__(self) -> None:
        self._previous: dict[int, Any] = {}
        self.cleaning = False
        self.received = False

    def install(self) -> None:
        try:
            for signum in (
                signal.SIGINT,
                signal.SIGTERM,
                signal.SIGHUP,  # windows-footgun: ok
            ):
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
        except (OSError, RuntimeError, ValueError):
            self._restore_best_effort()
            raise OwnerLauncherError("owner_signal_guard_unavailable") from None

    def _handle(self, signum: int, _frame: Any) -> None:
        self.received = True
        if not self.cleaning:
            # Enter the non-reentrant cleanup state before raising.  A second
            # signal can arrive while Python is unwinding toward ``finally``;
            # it must not replace the first interruption or skip cleanup.
            self.cleaning = True
            raise _OwnerLaunchSignal(signum)

    def begin_cleanup(self) -> None:
        self.cleaning = True

    def _restore_best_effort(self) -> bool:
        numbers = set(self._previous)
        if not numbers or not hasattr(signal, "pthread_sigmask"):
            return not numbers
        try:
            prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, numbers)
        except (OSError, RuntimeError, ValueError):
            return False
        restored = True
        try:
            # Discard any termination signal queued during cleanup before
            # restoring caller handlers. Otherwise unmasking a pending signal
            # after restoring SIG_DFL can kill the process before its canonical
            # receipt is emitted.
            discardable = numbers.difference(set(prior_mask))
            for signum in numbers:
                try:
                    signal.signal(signum, signal.SIG_IGN)
                except (OSError, RuntimeError, ValueError):
                    restored = False
            if discardable and restored:
                try:
                    signal.pthread_sigmask(signal.SIG_UNBLOCK, discardable)
                    signal.pthread_sigmask(signal.SIG_BLOCK, discardable)
                except (OSError, RuntimeError, ValueError):
                    restored = False
            for signum, handler in self._previous.items():
                try:
                    signal.signal(signum, handler)
                except (OSError, RuntimeError, ValueError):
                    restored = False
        finally:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
            except (OSError, RuntimeError, ValueError):
                restored = False
        return restored

    def restore(self) -> None:
        if not self._restore_best_effort():
            raise OwnerLauncherError("owner_signal_guard_restore_failed")


def harden_owner_secret_process() -> None:
    """Fail closed unless this process is non-dumpable before secret input."""

    if sys.platform not in {"darwin", "linux"}:
        raise OwnerLauncherError("owner_secret_process_platform_unsupported")
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        core_limits = resource.getrlimit(resource.RLIMIT_CORE)
    except (OSError, ValueError):
        raise OwnerLauncherError("owner_secret_process_hardening_failed") from None
    if core_limits != (0, 0):
        raise OwnerLauncherError("owner_secret_process_hardening_failed")
    if sys.platform == "linux":
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            prctl = libc.prctl
            prctl.argtypes = [
                ctypes.c_int,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_ulong,
            ]
            prctl.restype = ctypes.c_int
            if prctl(4, 0, 0, 0, 0) != 0 or prctl(3, 0, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "prctl")
        except (AttributeError, OSError, TypeError, ValueError):
            raise OwnerLauncherError("owner_secret_process_hardening_failed") from None


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _wipe(value: bytearray | None) -> None:
    """Best-effort in-place clearing for caller-owned secret material."""

    if value is not None:
        value[:] = b"\x00" * len(value)


def build_discord_frame(token: bytearray) -> bytearray:
    if (
        not isinstance(token, bytearray)
        or not token
        or len(token) > _DISCORD_TOKEN_MAX_BYTES
        or b"\x00" in token
        or any(value < 0x20 or value == 0x7F for value in token)
    ):
        raise OwnerLauncherError("invalid_discord_token")
    frame = bytearray(DISCORD_FRAME_MAGIC)
    frame.extend(struct.pack(">I", len(token)))
    frame.extend(token)
    return frame


def _new_admin_password() -> bytearray:
    """Create a mutable high-entropy credential for the bounded Phase-B edge."""

    return bytearray(
        base64.urlsafe_b64encode(secrets.token_bytes(_ADMIN_PASSWORD_BYTES))
    )


def _ssh_wire_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def _read_ssh_wire_string(
    value: bytes,
    offset: int,
    *,
    code: str,
) -> tuple[bytes, int]:
    if offset < 0 or offset + 4 > len(value):
        raise OwnerLauncherError(code)
    length = struct.unpack(">I", value[offset : offset + 4])[0]
    start = offset + 4
    end = start + length
    if length > PHASE_B_MAX_PUBLIC_KEY_BYTES or end > len(value):
        raise OwnerLauncherError(code)
    return value[start:end], end


def _phase_b_key_file_identity(
    path: Path,
    *,
    mode: int,
    maximum: int,
    code: str,
) -> os.stat_result:
    """Attest one exact owner key file without opening private material."""

    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise OwnerLauncherError(code)
    try:
        parent = path.parent.lstat()
        item = path.lstat()
        resolved_parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError):
        raise OwnerLauncherError(code) from None
    current_uid = os.geteuid()  # windows-footgun: ok — macOS/Linux owner boundary
    if (
        resolved_parent != path.parent
        or not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid != current_uid
        or stat.S_IMODE(parent.st_mode) & 0o022
        or not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != current_uid
        or item.st_nlink != 1
        or stat.S_IMODE(item.st_mode) != mode
        or not 1 <= item.st_size <= maximum
    ):
        raise OwnerLauncherError(code)
    return item


def _phase_b_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@dataclass(frozen=True)
class _PhaseBOwnerPublicAuthority:
    public_key_ed25519_hex: str
    key_id: str
    public_key_file_sha256: str
    public_fingerprint: str
    public_key_source: Mapping[str, Any]
    _public_blob: bytes = field(repr=False, compare=True)
    _public_item_identity: tuple[int, ...] = field(repr=False, compare=True)
    _private_item_identity: tuple[int, ...] = field(repr=False, compare=True)

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "public_key_ed25519_hex": self.public_key_ed25519_hex,
            "key_id": self.key_id,
            "public_key_file_sha256": self.public_key_file_sha256,
            "public_fingerprint": self.public_fingerprint,
            "public_key_source": copy.deepcopy(dict(self.public_key_source)),
        }


class _PhaseBOwnerExternalSigner:
    """Use fixed OpenSSH signing without ever loading the private key in Python."""

    def __init__(
        self,
        *,
        private_key_path: Path = PHASE_B_OWNER_PRIVATE_KEY_PATH,
        public_key_path: Path = PHASE_B_OWNER_PUBLIC_KEY_PATH,
        expected_comment: str = PHASE_B_OWNER_PUBLIC_KEY_COMMENT,
        expected_fingerprint: str = PHASE_B_OWNER_PUBLIC_KEY_FINGERPRINT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            not isinstance(private_key_path, Path)
            or not isinstance(public_key_path, Path)
            or not isinstance(expected_comment, str)
            or not expected_comment
            or not isinstance(expected_fingerprint, str)
            or re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}", expected_fingerprint)
            is None
            or not callable(clock)
        ):
            raise OwnerLauncherError("phase_b_owner_signer_config_invalid")
        self._private_key_path = private_key_path
        self._public_key_path = public_key_path
        self._expected_comment = expected_comment
        self._expected_fingerprint = expected_fingerprint
        self._clock = clock

    def inspect(self) -> _PhaseBOwnerPublicAuthority:
        private_item = _phase_b_key_file_identity(
            self._private_key_path,
            mode=0o600,
            maximum=64 * 1024,
            code="phase_b_owner_private_key_untrusted",
        )
        public_item = _phase_b_key_file_identity(
            self._public_key_path,
            mode=0o600,
            maximum=PHASE_B_MAX_PUBLIC_KEY_BYTES,
            code="phase_b_owner_public_key_untrusted",
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        try:
            descriptor = os.open(self._public_key_path, flags)
            try:
                opened = os.fstat(descriptor)
                if _phase_b_stat_identity(opened) != _phase_b_stat_identity(
                    public_item
                ):
                    raise OwnerLauncherError(
                        "phase_b_owner_public_key_untrusted"
                    )
                raw = os.read(descriptor, PHASE_B_MAX_PUBLIC_KEY_BYTES + 1)
                if os.read(descriptor, 1):
                    raise OwnerLauncherError(
                        "phase_b_owner_public_key_untrusted"
                    )
            finally:
                os.close(descriptor)
        except OwnerLauncherError:
            raise
        except OSError:
            raise OwnerLauncherError("phase_b_owner_public_key_untrusted") from None
        if (
            not raw
            or len(raw) > PHASE_B_MAX_PUBLIC_KEY_BYTES
            or not raw.endswith(b"\n")
            or b"\r" in raw
            or raw.count(b"\n") != 1
        ):
            raise OwnerLauncherError("phase_b_owner_public_key_untrusted")
        try:
            line = raw[:-1].decode("ascii", errors="strict")
            pieces = line.split(" ")
            if len(pieces) != 3 or any(not piece for piece in pieces):
                raise ValueError("invalid public key line")
            key_type, encoded, comment = pieces
            public_blob = base64.b64decode(encoded, validate=True)
        except (UnicodeError, ValueError):
            raise OwnerLauncherError("phase_b_owner_public_key_untrusted") from None
        try:
            blob_type, offset = _read_ssh_wire_string(
                public_blob,
                0,
                code="phase_b_owner_public_key_untrusted",
            )
            public_bytes, offset = _read_ssh_wire_string(
                public_blob,
                offset,
                code="phase_b_owner_public_key_untrusted",
            )
        except OwnerLauncherError:
            raise
        if (
            key_type != "ssh-ed25519"
            or blob_type != b"ssh-ed25519"
            or offset != len(public_blob)
            or len(public_bytes) != 32
            or comment != self._expected_comment
        ):
            raise OwnerLauncherError("phase_b_owner_public_key_untrusted")
        fingerprint = "SHA256:" + base64.b64encode(
            hashlib.sha256(public_blob).digest()
        ).decode("ascii").rstrip("=")
        if fingerprint != self._expected_fingerprint:
            raise OwnerLauncherError("phase_b_owner_public_key_untrusted")
        file_sha256 = _sha256(raw)
        return _PhaseBOwnerPublicAuthority(
            public_key_ed25519_hex=public_bytes.hex(),
            key_id=_sha256(public_bytes),
            public_key_file_sha256=file_sha256,
            public_fingerprint=fingerprint,
            public_key_source={
                "path": str(self._public_key_path),
                "file_sha256": file_sha256,
                "device": public_item.st_dev,
                "inode": public_item.st_ino,
                "uid": public_item.st_uid,
                "gid": public_item.st_gid,
                "mode": f"{stat.S_IMODE(public_item.st_mode):04o}",
                "size": public_item.st_size,
            },
            _public_blob=public_blob,
            _public_item_identity=_phase_b_stat_identity(public_item),
            _private_item_identity=_phase_b_stat_identity(private_item),
        )

    def _run_signer(self, message: bytes, namespace: str) -> str:
        if (
            not isinstance(message, bytes)
            or not message
            or len(message) > 64 * 1024
            or namespace
            not in {
                PHASE_B_APPROVAL_SSHSIG_NAMESPACE,
                PHASE_B_SOURCE_AUTH_SSHSIG_NAMESPACE,
                SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_SSHSIG_NAMESPACE,
                SCHEMA_RECONCILIATION_PREFLIGHT_AUTHORIZATION_SSHSIG_NAMESPACE,
                SCHEMA_RECONCILIATION_ADMIN_CLEANUP_SSHSIG_NAMESPACE,
            }
        ):
            raise OwnerLauncherError("phase_b_owner_signing_request_invalid")
        try:
            executable = PHASE_B_SSH_KEYGEN.lstat()
        except OSError:
            raise OwnerLauncherError("phase_b_owner_signer_unavailable") from None
        if (
            not stat.S_ISREG(executable.st_mode)
            or stat.S_ISLNK(executable.st_mode)
            or executable.st_uid != 0
            or stat.S_IMODE(executable.st_mode) & 0o022
        ):
            raise OwnerLauncherError("phase_b_owner_signer_untrusted")
        argv = [
            str(PHASE_B_SSH_KEYGEN),
            "-Y",
            "sign",
            "-q",
            "-f",
            str(self._private_key_path),
            "-n",
            namespace,
        ]
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env={
                    "PATH": "/usr/bin:/bin",
                    "LANG": "C",
                    "LC_ALL": "C",
                },
                close_fds=True,
                shell=False,
            )
        except OSError:
            raise OwnerLauncherError("phase_b_owner_signer_unavailable") from None
        if process.stdin is None or process.stdout is None:
            process.kill()
            process.wait()
            raise OwnerLauncherError("phase_b_owner_signer_unavailable")
        selector = selectors.DefaultSelector()
        stdin_fd = process.stdin.fileno()
        stdout_fd = process.stdout.fileno()
        output = bytearray()
        input_offset = 0
        deadline = self._clock() + PHASE_B_SIGN_TIMEOUT_SECONDS
        try:
            os.set_blocking(stdin_fd, False)
            os.set_blocking(stdout_fd, False)
            selector.register(stdin_fd, selectors.EVENT_WRITE, "stdin")
            selector.register(stdout_fd, selectors.EVENT_READ, "stdout")
            stdout_open = True
            stdin_open = True
            while stdout_open or process.poll() is None:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise OwnerLauncherError("phase_b_owner_signer_timeout")
                events = selector.select(min(remaining, 0.25))
                for key, _mask in events:
                    if key.data == "stdin" and stdin_open:
                        try:
                            written = os.write(stdin_fd, message[input_offset:])
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            raise OwnerLauncherError(
                                "phase_b_owner_signer_failed"
                            ) from None
                        input_offset += written
                        if input_offset == len(message):
                            selector.unregister(stdin_fd)
                            process.stdin.close()
                            stdin_open = False
                    elif key.data == "stdout" and stdout_open:
                        try:
                            chunk = os.read(stdout_fd, 1024)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            selector.unregister(stdout_fd)
                            stdout_open = False
                            continue
                        output.extend(chunk)
                        if len(output) > PHASE_B_MAX_SSHSIG_BYTES:
                            raise OwnerLauncherError(
                                "phase_b_owner_signature_oversized"
                            )
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise OwnerLauncherError("phase_b_owner_signer_timeout")
            return_code = process.wait(timeout=remaining)
            if return_code != 0 or input_offset != len(message):
                raise OwnerLauncherError("phase_b_owner_signer_failed")
        except OwnerLauncherError:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        except (OSError, subprocess.SubprocessError):
            if process.poll() is None:
                process.kill()
            process.wait()
            raise OwnerLauncherError("phase_b_owner_signer_failed") from None
        finally:
            selector.close()
            if not process.stdin.closed:
                process.stdin.close()
            process.stdout.close()
        try:
            return bytes(output).decode("ascii", errors="strict")
        except UnicodeError:
            raise OwnerLauncherError("phase_b_owner_signature_invalid") from None

    def sign(
        self,
        message: bytes,
        *,
        namespace: str,
        expected_authority: _PhaseBOwnerPublicAuthority,
    ) -> str:
        if not isinstance(expected_authority, _PhaseBOwnerPublicAuthority):
            raise OwnerLauncherError("phase_b_owner_authority_unbound")
        before = self.inspect()
        if before != expected_authority:
            raise OwnerLauncherError("phase_b_owner_key_changed")
        signature = self._run_signer(message, namespace)
        after = self.inspect()
        if after != before:
            raise OwnerLauncherError("phase_b_owner_key_changed")
        from gateway import canonical_writer_foundation_phase_b as phase_b

        try:
            phase_b.verify_phase_b_sshsig(
                signature,
                message=message,
                public_key_ed25519_hex=before.public_key_ed25519_hex,
                namespace=namespace,
            )
        except (TypeError, ValueError) as exc:
            raise OwnerLauncherError("phase_b_owner_signature_invalid") from exc
        return signature


def _schema_reconciliation_time_window(
    gate: Mapping[str, Any],
    *,
    now_unix: int,
) -> tuple[int, int]:
    if (
        type(now_unix) is not int
        or type(gate.get("issued_at_unix")) is not int
        or type(gate.get("expires_at_unix")) is not int
        or not gate["issued_at_unix"] <= now_unix < gate["expires_at_unix"]
    ):
        raise OwnerLauncherError("schema_reconciliation_owner_frame_expired")
    expires_at = min(int(gate["expires_at_unix"]), now_unix + 300)
    if expires_at <= now_unix:
        raise OwnerLauncherError("schema_reconciliation_owner_frame_expired")
    return now_unix, expires_at


def _schema_reconciliation_nonce_sha256(
    nonce_factory: Callable[[int], bytes],
) -> str:
    try:
        nonce = nonce_factory(32)
    except BaseException:
        raise OwnerLauncherError("schema_reconciliation_owner_nonce_failed") from None
    if not isinstance(nonce, bytes) or len(nonce) != 32:
        raise OwnerLauncherError("schema_reconciliation_owner_nonce_failed")
    return _sha256(nonce)


def _schema_reconciliation_signed_claim(
    unsigned: Mapping[str, Any],
    *,
    digest_field: str,
    signature_payload: Callable[[Mapping[str, Any]], bytes],
    namespace: str,
    signer: _PhaseBOwnerExternalSigner,
    owner_authority: _PhaseBOwnerPublicAuthority,
) -> Mapping[str, Any]:
    if (
        not isinstance(unsigned, Mapping)
        or not isinstance(digest_field, str)
        or not digest_field
        or not callable(signature_payload)
    ):
        raise OwnerLauncherError("schema_reconciliation_owner_claim_invalid")
    template = {
        **copy.deepcopy(dict(unsigned)),
        digest_field: _sha256(_canonical_bytes(unsigned)),
        "signature_sshsig": "",
    }
    try:
        payload = signature_payload(template)
        signature = signer.sign(
            payload,
            namespace=namespace,
            expected_authority=owner_authority,
        )
    except OwnerLauncherError:
        raise
    except BaseException:
        raise OwnerLauncherError(
            "schema_reconciliation_owner_claim_invalid"
        ) from None
    if not isinstance(signature, str) or not signature:
        raise OwnerLauncherError("schema_reconciliation_owner_claim_invalid")
    return {**template, "signature_sshsig": signature}


def _schema_reconciliation_frame(
    magic: bytes,
    claim: Mapping[str, Any],
    *,
    credential: bytearray | None = None,
) -> bytearray:
    allowed = {
        SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_MAGIC,
        SCHEMA_RECONCILIATION_PREFLIGHT_AUTHORIZATION_MAGIC,
        SCHEMA_RECONCILIATION_ADMIN_CLEANUP_MAGIC,
    }
    if magic not in allowed or not isinstance(claim, Mapping):
        raise OwnerLauncherError("schema_reconciliation_owner_frame_invalid")
    expects_credential = magic == SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_MAGIC
    if expects_credential:
        if (
            not isinstance(credential, bytearray)
            or len(credential) != SCHEMA_RECONCILIATION_CREDENTIAL_BYTES
        ):
            raise OwnerLauncherError(
                "schema_reconciliation_owner_frame_invalid"
            )
    elif credential is not None:
        raise OwnerLauncherError("schema_reconciliation_owner_frame_invalid")
    payload = _canonical_bytes(claim)
    if not 2 <= len(payload) <= PHASE_B_MAX_RESPONSE_BYTES:
        raise OwnerLauncherError("schema_reconciliation_owner_frame_invalid")
    frame = bytearray(magic + struct.pack(">I", len(payload)) + payload)
    if credential is not None:
        frame.extend(credential)
    return frame


def build_schema_reconciliation_admin_preflight(
    *,
    gate: Mapping[str, Any],
    cloud_sql_authority_receipt: Mapping[str, Any],
    credential: bytearray,
    signer: _PhaseBOwnerExternalSigner,
    owner_authority: _PhaseBOwnerPublicAuthority,
    now_unix: int,
    nonce_factory: Callable[[int], bytes] = secrets.token_bytes,
) -> Mapping[str, Any]:
    from gateway import canonical_writer_schema_reconciliation_bootstrap as bootstrap

    issued_at, expires_at = _schema_reconciliation_time_window(
        gate,
        now_unix=now_unix,
    )
    if (
        not isinstance(credential, bytearray)
        or len(credential) != bootstrap.OPAQUE_CREDENTIAL_BYTES
        or bootstrap.OPAQUE_CREDENTIAL_BYTES
        != SCHEMA_RECONCILIATION_CREDENTIAL_BYTES
    ):
        raise OwnerLauncherError("schema_reconciliation_owner_frame_invalid")
    unsigned = {
        "schema": bootstrap.OWNER_ADMIN_PREFLIGHT_SCHEMA,
        "frame_schema": bootstrap.ADMIN_PREFLIGHT_FRAME_SCHEMA,
        "action": "authorize_temporary_admin_locked_preflight",
        "approved": True,
        "gate_sha256": gate["gate_sha256"],
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "temporary_admin_username_sha256": gate[
            "temporary_admin_username_sha256"
        ],
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "owner_key_id": gate["owner_key_id"],
        "cloud_sql_authority_receipt": copy.deepcopy(
            dict(cloud_sql_authority_receipt)
        ),
        "cloud_sql_authority_receipt_sha256": (
            cloud_sql_authority_receipt.get("receipt_sha256")
        ),
        "credential_present": True,
        "credential_length": len(credential),
        "issued_at_unix": issued_at,
        "expires_at_unix": expires_at,
        "nonce_sha256": _schema_reconciliation_nonce_sha256(nonce_factory),
        "secret_material_recorded": False,
    }
    return _schema_reconciliation_signed_claim(
        unsigned,
        digest_field="authority_claim_sha256",
        signature_payload=bootstrap.admin_preflight_signature_payload,
        namespace=SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_SSHSIG_NAMESPACE,
        signer=signer,
        owner_authority=owner_authority,
    )


def build_schema_reconciliation_preflight_authorization(
    *,
    gate: Mapping[str, Any],
    admin_preflight: Mapping[str, Any],
    challenge: Mapping[str, Any],
    signer: _PhaseBOwnerExternalSigner,
    owner_authority: _PhaseBOwnerPublicAuthority,
    now_unix: int,
    nonce_factory: Callable[[int], bytes] = secrets.token_bytes,
) -> Mapping[str, Any]:
    from gateway import canonical_writer_schema_reconciliation_bootstrap as bootstrap

    issued_at, expires_at = _schema_reconciliation_time_window(
        gate,
        now_unix=now_unix,
    )
    journal = gate.get("journal_head")
    preflight = challenge.get("preflight")
    if not isinstance(journal, Mapping) or not isinstance(preflight, Mapping):
        raise OwnerLauncherError("schema_reconciliation_owner_frame_invalid")
    mode = _SCHEMA_RECONCILIATION_TRANSITIONS.get(
        (str(journal.get("state")), str(preflight.get("state")))
    )
    if mode is None:
        raise OwnerLauncherError("schema_reconciliation_transition_invalid")
    unsigned = {
        "schema": bootstrap.OWNER_PREFLIGHT_AUTHORIZATION_SCHEMA,
        "frame_schema": bootstrap.PREFLIGHT_AUTHORIZATION_FRAME_SCHEMA,
        "action": "apply_schema_reconciliation",
        "approved": True,
        "gate_sha256": gate["gate_sha256"],
        "authority_claim_sha256": admin_preflight[
            "authority_claim_sha256"
        ],
        "preflight_challenge_sha256": challenge[
            "preflight_challenge_sha256"
        ],
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "journal_head_sha256": journal["head_sha256"],
        "execution_mode": mode,
        "preflight_sha256": preflight["preflight_sha256"],
        "preflight_state": preflight["state"],
        "observed_contract_sha256": preflight[
            "observed_contract_sha256"
        ],
        "truth_receipt_sha256": preflight["truth_receipt_sha256"],
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "owner_key_id": gate["owner_key_id"],
        "issued_at_unix": issued_at,
        "expires_at_unix": expires_at,
        "nonce_sha256": _schema_reconciliation_nonce_sha256(nonce_factory),
        "stored_authorized_intent_sha256": journal[
            "authorized_intent_sha256"
        ],
        "stored_terminal_receipt_sha256": journal[
            "terminal_receipt_sha256"
        ],
        "secret_material_recorded": False,
    }
    return _schema_reconciliation_signed_claim(
        unsigned,
        digest_field="preflight_authorization_claim_sha256",
        signature_payload=bootstrap.preflight_authorization_signature_payload,
        namespace=(
            SCHEMA_RECONCILIATION_PREFLIGHT_AUTHORIZATION_SSHSIG_NAMESPACE
        ),
        signer=signer,
        owner_authority=owner_authority,
    )


def build_schema_reconciliation_admin_cleanup(
    *,
    gate: Mapping[str, Any],
    admin_preflight: Mapping[str, Any],
    challenge: Mapping[str, Any],
    authorization: Mapping[str, Any],
    intermediate: Mapping[str, Any],
    cloud_sql_absence_receipt: Mapping[str, Any],
    signer: _PhaseBOwnerExternalSigner,
    owner_authority: _PhaseBOwnerPublicAuthority,
    now_unix: int,
    nonce_factory: Callable[[int], bytes] = secrets.token_bytes,
) -> Mapping[str, Any]:
    from gateway import canonical_writer_schema_reconciliation_bootstrap as bootstrap

    issued_at, expires_at = _schema_reconciliation_time_window(
        gate,
        now_unix=now_unix,
    )
    unsigned = {
        "schema": bootstrap.OWNER_ADMIN_CLEANUP_SCHEMA,
        "frame_schema": bootstrap.ADMIN_CLEANUP_FRAME_SCHEMA,
        "action": "confirm_temporary_admin_absence",
        "approved": True,
        "gate_sha256": gate["gate_sha256"],
        "authority_claim_sha256": admin_preflight[
            "authority_claim_sha256"
        ],
        "preflight_challenge_sha256": challenge[
            "preflight_challenge_sha256"
        ],
        "preflight_authorization_claim_sha256": authorization[
            "preflight_authorization_claim_sha256"
        ],
        "database_intermediate_sha256": intermediate[
            "database_intermediate_sha256"
        ],
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "temporary_admin_username_sha256": gate[
            "temporary_admin_username_sha256"
        ],
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "owner_key_id": gate["owner_key_id"],
        "cloud_sql_absence_receipt": copy.deepcopy(
            dict(cloud_sql_absence_receipt)
        ),
        "cloud_sql_absence_receipt_sha256": (
            cloud_sql_absence_receipt.get("evidence_sha256")
        ),
        "issued_at_unix": issued_at,
        "expires_at_unix": expires_at,
        "nonce_sha256": _schema_reconciliation_nonce_sha256(nonce_factory),
        "secret_material_recorded": False,
    }
    return _schema_reconciliation_signed_claim(
        unsigned,
        digest_field="cleanup_claim_sha256",
        signature_payload=bootstrap.admin_cleanup_signature_payload,
        namespace=SCHEMA_RECONCILIATION_ADMIN_CLEANUP_SSHSIG_NAMESPACE,
        signer=signer,
        owner_authority=owner_authority,
    )


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OwnerLauncherError("invalid_json")
        result[key] = value
    return result


def _decode_json_object(raw: bytes, *, maximum: int) -> Mapping[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > maximum:
        raise OwnerLauncherError("invalid_json")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                OwnerLauncherError("invalid_json")
            ),
        )
    except OwnerLauncherError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        raise OwnerLauncherError("invalid_json") from None
    if not isinstance(value, Mapping):
        raise OwnerLauncherError("invalid_json")
    return value


def _decode_json_value(raw: bytes, *, maximum: int) -> Any:
    if not isinstance(raw, bytes) or not raw or len(raw) > maximum:
        raise OwnerLauncherError("invalid_json")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                OwnerLauncherError("invalid_json")
            ),
        )
    except OwnerLauncherError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        raise OwnerLauncherError("invalid_json") from None


def _canary_iam_read_only_inventory() -> frozenset[tuple[str, ...]]:
    """Exact logical argv accepted from the three reviewed collectors."""

    project = f"--project={PROJECT}"
    service_account = (
        "muncho-canary-v2-runtime@adventico-ai-platform.iam.gserviceaccount.com"
    )
    network = "muncho-canary-vpc"
    subnet = "muncho-canary-europe-west3"
    return frozenset({
        ("gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=json"),
        ("gcloud", "iam", "service-accounts", "list", project, "--format=json"),
        ("gcloud", "iam", "roles", "list", project, "--format=json"),
        (
            "gcloud",
            "iam",
            "roles",
            "describe",
            "munchoCanaryCloudSqlReadinessV1",
            project,
            "--format=json",
        ),
        ("gcloud", "sql", "instances", "list", project, "--format=json"),
        ("gcloud", "compute", "networks", "list", project, "--format=json"),
        (
            "gcloud",
            "compute",
            "networks",
            "subnets",
            "list",
            project,
            "--format=json",
        ),
        (
            "gcloud",
            "compute",
            "addresses",
            "list",
            "--global",
            project,
            "--format=json",
        ),
        ("gcloud", "services", "list", "--enabled", project, "--format=json"),
        ("gcloud", "secrets", "list", project, "--format=json"),
        ("gcloud", "projects", "get-iam-policy", PROJECT, "--format=json"),
        (
            "gcloud",
            "compute",
            "networks",
            "describe",
            network,
            project,
            "--format=json",
        ),
        ("gcloud", "compute", "routes", "list", project, "--format=json"),
        (
            "gcloud",
            "services",
            "vpc-peerings",
            "list",
            f"--network={network}",
            "--service=servicenetworking.googleapis.com",
            project,
            "--format=json",
        ),
        (
            "gcloud",
            "compute",
            "networks",
            "subnets",
            "describe",
            subnet,
            "--region=europe-west3",
            project,
            "--format=json",
        ),
        (
            "gcloud",
            "iam",
            "service-accounts",
            "keys",
            "list",
            f"--iam-account={service_account}",
            project,
            "--format=json",
        ),
        (
            "gcloud",
            "sql",
            "instances",
            "describe",
            SQL_INSTANCE,
            project,
            "--format=json",
        ),
        (
            "gcloud",
            "sql",
            "databases",
            "list",
            f"--instance={SQL_INSTANCE}",
            project,
            "--format=json",
        ),
        (
            "gcloud",
            "iam",
            "service-accounts",
            "describe",
            service_account,
            project,
            "--format=json",
        ),
        (
            "gcloud",
            "compute",
            "firewall-rules",
            "list",
            project,
            "--format=json",
        ),
        (
            "gcloud",
            "compute",
            "networks",
            "get-effective-firewalls",
            network,
            project,
            "--format=json",
        ),
        ("gcloud", "compute", "instances", "list", project, "--format=json"),
        (
            "gcloud",
            "compute",
            "images",
            "describe",
            "debian-12-bookworm-v20260609",
            "--project=debian-cloud",
            "--format=json",
        ),
        (
            "gcloud",
            "compute",
            "instances",
            "describe",
            VM_NAME,
            f"--zone={ZONE}",
            project,
            "--format=json",
        ),
        (
            "gcloud",
            "compute",
            "disks",
            "describe",
            VM_NAME,
            f"--zone={ZONE}",
            project,
            "--format=json",
        ),
    })


def _require_sha256(value: object, code: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise OwnerLauncherError(code)
    return value


def _validate_self_digest(
    value: Mapping[str, Any],
    *,
    expected_keys: frozenset[str],
    digest_key: str,
    code: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise OwnerLauncherError(code)
    expected = _require_sha256(value.get(digest_key), code)
    unsigned = dict(value)
    del unsigned[digest_key]
    if _sha256(_canonical_bytes(unsigned)) != expected:
        raise OwnerLauncherError(code)
    return dict(value)


def _validate_receipt_time(value: object, *, now_unix: int, code: str) -> int:
    if (
        type(value) is not int
        or value < 0
        or value > now_unix + 30
        or now_unix - value > _GATE_MAX_FUTURE_SECONDS
    ):
        raise OwnerLauncherError(code)
    return value


def _reject_secret_echo(
    value: Any,
    *,
    active_secrets: Sequence[bytes | bytearray | memoryview],
    code: str,
) -> None:
    """Reject active secret bytes and explicit value-bearing secret fields."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = key.casefold() if isinstance(key, str) else ""
            derived_secret_field = (
                normalized not in _ALLOWED_SECRET_STATUS_OR_BINDING_KEYS
                and any(
                    marker in normalized for marker in ("password", "token", "secret")
                )
                and any(
                    marker in normalized
                    for marker in (
                        "sha",
                        "digest",
                        "hash",
                        "fingerprint",
                        "content",
                        "value",
                        "raw",
                        "bytes",
                    )
                )
            )
            if (
                not isinstance(key, str)
                or normalized in _FORBIDDEN_RECEIPT_VALUE_KEYS
                or derived_secret_field
            ):
                raise OwnerLauncherError(code)
            _reject_secret_echo(item, active_secrets=active_secrets, code=code)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_secret_echo(item, active_secrets=active_secrets, code=code)
        return
    candidate: bytes | None = None
    if isinstance(value, str):
        candidate = value.encode("utf-8", errors="strict")
    elif isinstance(value, bytes):
        candidate = value
    if candidate is not None and any(
        secret and candidate.find(secret) >= 0 for secret in active_secrets
    ):
        raise OwnerLauncherError(code)


def validate_discord_install_gate(
    gate: Mapping[str, Any],
    *,
    owner_gate: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    value = _validate_self_digest(
        gate,
        expected_keys=_DISCORD_INSTALL_GATE_KEYS,
        digest_key="gate_sha256",
        code="invalid_discord_install_gate",
    )
    expires = value.get("expires_at_unix")
    if (
        value.get("schema") != DISCORD_INSTALL_GATE_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "token_install_authorized"
        or value.get("coordinator_input_sha256")
        != owner_gate.get("coordinator_input_sha256")
        or value.get("owner_subject_sha256") != owner_gate.get("owner_subject_sha256")
        or value.get("release_sha") != owner_gate.get("release_sha")
        or value.get("token_path") != DISCORD_TOKEN_PATH
        or value.get("frame_schema") != DISCORD_FRAME_SCHEMA
        or type(value.get("edge_uid")) is not int
        or value["edge_uid"] <= 0
        or type(value.get("edge_gid")) is not int
        or value["edge_gid"] <= 0
        or type(expires) is not int
        or expires <= now_unix
        or expires - now_unix > _GATE_MAX_FUTURE_SECONDS
    ):
        raise OwnerLauncherError("invalid_discord_install_gate")
    _require_sha256(
        value.get("discord_token_install_approval_sha256"),
        "invalid_discord_install_gate",
    )
    _require_sha256(value.get("owner_subject_sha256"), "invalid_discord_install_gate")
    return value


def validate_discord_install_receipt(
    receipt: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    token: bytearray,
    now_unix: int,
) -> Mapping[str, Any]:
    _reject_secret_echo(
        receipt,
        active_secrets=(token,),
        code="invalid_discord_install_receipt",
    )
    value = _validate_self_digest(
        receipt,
        expected_keys=_DISCORD_INSTALL_RECEIPT_KEYS,
        digest_key="receipt_sha256",
        code="invalid_discord_install_receipt",
    )
    if (
        value.get("schema") != DISCORD_INSTALL_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("release_sha") != gate.get("release_sha")
        or value.get("coordinator_input_sha256") != gate.get("coordinator_input_sha256")
        or value.get("discord_token_install_approval_sha256")
        != gate.get("discord_token_install_approval_sha256")
        or value.get("owner_subject_sha256") != gate.get("owner_subject_sha256")
        or value.get("token_path") != DISCORD_TOKEN_PATH
        or type(value.get("device")) is not int
        or value["device"] <= 0
        or type(value.get("inode")) is not int
        or value["inode"] <= 0
        or value.get("owner_uid") != gate.get("edge_uid")
        or value.get("group_gid") != gate.get("edge_gid")
        or value.get("mode") != "0400"
        or value.get("size") != len(token)
        or value.get("link_count") != 1
        or value.get("content_or_digest_recorded") is not False
    ):
        raise OwnerLauncherError("invalid_discord_install_receipt")
    _validate_receipt_time(
        value.get("installed_at_unix"),
        now_unix=now_unix,
        code="invalid_discord_install_receipt",
    )
    return value


def validate_discord_retirement_receipt(
    receipt: Mapping[str, Any],
    *,
    owner_gate: Mapping[str, Any],
    install_receipt: Mapping[str, Any] | None,
    retirement_gate: Mapping[str, Any] | None = None,
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate the exact durable terminal token-lease retirement proof."""

    value = _validate_self_digest(
        receipt,
        expected_keys=_DISCORD_RETIREMENT_RECEIPT_KEYS,
        digest_key="receipt_sha256",
        code="invalid_discord_retirement_receipt",
    )
    prepared_at = value.get("prepared_at_unix")
    retired_at = value.get("retired_at_unix")
    expected_install_sha = (
        install_receipt.get("receipt_sha256")
        if install_receipt is not None
        else None
        if retirement_gate is None
        else retirement_gate.get("discord_token_install_receipt_sha256")
    )
    expected_device = (
        install_receipt.get("device")
        if install_receipt is not None
        else None
        if retirement_gate is None
        else retirement_gate.get("token_device")
    )
    expected_inode = (
        install_receipt.get("inode")
        if install_receipt is not None
        else None
        if retirement_gate is None
        else retirement_gate.get("token_inode")
    )
    expected_identity_valid = (expected_device is None and expected_inode is None) or (
        type(expected_device) is int
        and expected_device > 0
        and type(expected_inode) is int
        and expected_inode > 0
    )
    if (
        expected_install_sha is None
        or not expected_identity_valid
        or value.get("schema") != DISCORD_RETIREMENT_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "retired"
        or value.get("release_sha") != owner_gate.get("release_sha")
        or value.get("coordinator_input_sha256")
        != owner_gate.get("coordinator_input_sha256")
        or value.get("discord_token_install_receipt_sha256") != expected_install_sha
        or value.get("token_path") != DISCORD_TOKEN_PATH
        or value.get("token_device") != expected_device
        or value.get("token_inode") != expected_inode
        or value.get("services_stopped_proven") is not True
        or value.get("services_enabled") is not False
        or value.get("token_removed") is not True
        or value.get("install_receipt_removed") is not True
        or type(prepared_at) is not int
        or type(retired_at) is not int
        or prepared_at < 0
        or retired_at < prepared_at
    ):
        raise OwnerLauncherError("invalid_discord_retirement_receipt")
    _validate_receipt_time(
        retired_at,
        now_unix=now_unix,
        code="invalid_discord_retirement_receipt",
    )
    return value


def validate_discord_retirement_gate(
    gate: Mapping[str, Any],
    *,
    owner_gate: Mapping[str, Any],
    install_receipt: Mapping[str, Any] | None,
    now_unix: int,
) -> Mapping[str, Any]:
    value = _validate_self_digest(
        gate,
        expected_keys=_DISCORD_RETIREMENT_GATE_KEYS,
        digest_key="gate_sha256",
        code="invalid_discord_retirement_gate",
    )
    expires = value.get("expires_at_unix")
    if (
        value.get("schema") != DISCORD_RETIREMENT_GATE_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "awaiting_owner_discord_retirement_ack"
        or value.get("release_sha") != owner_gate.get("release_sha")
        or value.get("coordinator_input_sha256")
        != owner_gate.get("coordinator_input_sha256")
        or value.get("owner_subject_sha256") != owner_gate.get("owner_subject_sha256")
        or install_receipt is not None
        and value.get("discord_token_install_receipt_sha256")
        != install_receipt.get("receipt_sha256")
        or install_receipt is not None
        and value.get("token_device") != install_receipt.get("device")
        or install_receipt is not None
        and value.get("token_inode") != install_receipt.get("inode")
        or value.get("process_lease_absent") is not True
        or value.get("services_stopped_proven") is not True
        or value.get("frame_schema") != DISCORD_RETIREMENT_ACK_FRAME_SCHEMA
        or type(expires) is not int
        or expires <= now_unix
        or expires - now_unix > 300
    ):
        raise OwnerLauncherError("invalid_discord_retirement_gate")
    _require_sha256(
        value.get("discord_token_install_receipt_sha256"),
        "invalid_discord_retirement_gate",
    )
    token_device = value.get("token_device")
    token_inode = value.get("token_inode")
    if not (
        (token_device is None and token_inode is None)
        or (
            type(token_device) is int
            and token_device > 0
            and type(token_inode) is int
            and token_inode > 0
        )
    ):
        raise OwnerLauncherError("invalid_discord_retirement_gate")
    return value


def build_discord_retirement_ack(
    gate: Mapping[str, Any],
    *,
    now_unix: int,
    nonce: bytes | None = None,
) -> Mapping[str, Any]:
    nonce_value = secrets.token_bytes(32) if nonce is None else nonce
    if not isinstance(nonce_value, bytes) or len(nonce_value) < 16:
        raise OwnerLauncherError("invalid_discord_retirement_ack")
    expires = min(int(gate["expires_at_unix"]), now_unix + 300)
    if expires <= now_unix:
        raise OwnerLauncherError("invalid_discord_retirement_ack")
    unsigned = {
        "schema": DISCORD_RETIREMENT_ACK_SCHEMA,
        "scope": "stop_and_retire_full_canary_discord_token",
        "release_sha": gate["release_sha"],
        "coordinator_input_sha256": gate["coordinator_input_sha256"],
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "retirement_gate_sha256": gate["gate_sha256"],
        "discord_token_install_receipt_sha256": gate[
            "discord_token_install_receipt_sha256"
        ],
        "token_device": gate["token_device"],
        "token_inode": gate["token_inode"],
        "nonce_sha256": _sha256(nonce_value),
        "approved_at_unix": now_unix,
        "expires_at_unix": expires,
    }
    return {**unsigned, "ack_sha256": _sha256(_canonical_bytes(unsigned))}


def build_discord_retirement_ack_frame(ack: Mapping[str, Any]) -> bytes:
    if not isinstance(ack, Mapping) or set(ack) != _DISCORD_RETIREMENT_ACK_KEYS:
        raise OwnerLauncherError("invalid_discord_retirement_ack")
    expected = _require_sha256(ack.get("ack_sha256"), "invalid_discord_retirement_ack")
    unsigned = dict(ack)
    del unsigned["ack_sha256"]
    if expected != _sha256(_canonical_bytes(unsigned)):
        raise OwnerLauncherError("invalid_discord_retirement_ack")
    payload = _canonical_bytes(ack)
    if len(payload) > 128 * 1024:
        raise OwnerLauncherError("invalid_discord_retirement_ack")
    return (
        DISCORD_RETIREMENT_ACK_FRAME_MAGIC + struct.pack(">I", len(payload)) + payload
    )


def build_final_approval_frame(approval: Mapping[str, Any]) -> bytes:
    payload = _canonical_bytes(approval)
    if not payload or len(payload) > 128 * 1024:
        raise OwnerLauncherError("invalid_final_owner_approval")
    return FINAL_APPROVAL_FRAME_MAGIC + struct.pack(">I", len(payload)) + payload


def validate_terminal_first_failure(
    receipt: Mapping[str, Any],
    *,
    owner_gate: Mapping[str, Any] | None,
    expected_release_sha: str | None = None,
    active_secrets: Sequence[bytes | bytearray | memoryview] = (),
) -> Mapping[str, Any]:
    """Validate the session-bound coordinator's exact terminal failure."""

    _reject_secret_echo(
        receipt,
        active_secrets=active_secrets,
        code="invalid_terminal_first_failure",
    )
    value = _validate_self_digest(
        receipt,
        expected_keys=_COORDINATOR_FAILURE_KEYS,
        digest_key="receipt_sha256",
        code="invalid_terminal_first_failure",
    )
    phase = value.get("phase")
    command = value.get("command")
    error_code = value.get("error_code")
    cleanup_status = value.get("cleanup_status")
    discord_removed = value.get("discord_token_removed")
    services_stopped = value.get("services_stopped")
    obsolete_journal_absent = value.get("obsolete_process_journal_absent")
    release_sha = value.get("release_sha")
    coordinator_input_sha256 = value.get("coordinator_input_sha256")
    completed_at_unix = value.get("completed_at_unix")

    if owner_gate is not None:
        if (
            release_sha != owner_gate.get("release_sha")
            or coordinator_input_sha256
            != owner_gate.get("coordinator_input_sha256")
        ):
            raise OwnerLauncherError("invalid_terminal_first_failure")
    elif release_sha is not None:
        if (
            expected_release_sha is not None
            and release_sha != expected_release_sha
        ) or (
            not isinstance(release_sha, str)
            or _RELEASE_SHA.fullmatch(release_sha) is None
            or not isinstance(coordinator_input_sha256, str)
            or _SHA256.fullmatch(coordinator_input_sha256) is None
        ):
            raise OwnerLauncherError("invalid_terminal_first_failure")
    elif coordinator_input_sha256 is not None:
        raise OwnerLauncherError("invalid_terminal_first_failure")

    cleanup_complete = (
        discord_removed is True
        and services_stopped is True
        and obsolete_journal_absent is True
    )
    if (
        value.get("schema") != COORDINATOR_FAILURE_SCHEMA
        or value.get("ok") is not False
        or not isinstance(phase, str)
        or _STABLE_CODE.fullmatch(phase) is None
        or command not in _COORDINATOR_COMMANDS
        or not isinstance(error_code, str)
        or _STABLE_CODE.fullmatch(error_code) is None
        or cleanup_status not in {"complete", "cleanup_blocked"}
        or type(discord_removed) is not bool
        or type(services_stopped) is not bool
        or type(obsolete_journal_absent) is not bool
        or type(completed_at_unix) is not int
        or completed_at_unix <= 0
        or (cleanup_status == "complete") is not cleanup_complete
    ):
        raise OwnerLauncherError("invalid_terminal_first_failure")
    return value


def validate_coordinator_input_publication_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_release_sha: str,
) -> Mapping[str, Any]:
    """Validate the fixed-path, stopped-only coordinator input publication."""

    if _RELEASE_SHA.fullmatch(expected_release_sha) is None:
        raise OwnerLauncherError("invalid_release_sha")
    value = _validate_self_digest(
        receipt,
        expected_keys=_COORDINATOR_INPUT_PUBLICATION_KEYS,
        digest_key="receipt_sha256",
        code="invalid_coordinator_input_publication_receipt",
    )
    if (
        value.get("schema") != COORDINATOR_INPUT_PUBLICATION_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "published"
        or value.get("release_sha") != expected_release_sha
        or value.get("coordinator_input_path") != COORDINATOR_INPUT_PATH
        or value.get("publication_receipt_path")
        != COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH
        or value.get("owner_uid") != 0
        or value.get("group_gid") != 0
        or value.get("mode") != "0400"
        or type(value.get("published_at_unix")) is not int
        or value["published_at_unix"] < 0
    ):
        raise OwnerLauncherError("invalid_coordinator_input_publication_receipt")
    for name in (
        "coordinator_input_sha256",
        "coordinator_input_file_sha256",
    ):
        _require_sha256(
            value.get(name),
            "invalid_coordinator_input_publication_receipt",
        )
    _reject_secret_echo(
        value,
        active_secrets=(),
        code="invalid_coordinator_input_publication_receipt",
    )
    return value


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes = field(repr=False)


class HttpRequester(Protocol):
    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> HttpResponse: ...


def _default_http_request(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> HttpResponse:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    request = urllib.request.Request(
        url=url,
        data=body,
        headers=dict(headers),
        method=method,
    )
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            NoRedirect(),
            urllib.request.HTTPSHandler(context=_pinned_system_tls_context()),
        )
        with opener.open(request, timeout=timeout) as response:
            if response.geturl() != url:
                raise OwnerLauncherError("google_api_redirect_forbidden")
            payload = response.read(_HTTP_RESPONSE_MAX_BYTES + 1)
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        # Preserve only the secret-free status class. Never read or copy an
        # API error body: some APIs echo the submitted password.
        status = int(exc.code)
        payload = b""
        try:
            exc.close()
        except BaseException:
            pass
    except (urllib.error.URLError, TimeoutError, OSError):
        raise OwnerLauncherError("google_api_unavailable") from None
    if len(payload) > _HTTP_RESPONSE_MAX_BYTES:
        raise OwnerLauncherError("google_api_response_oversized")
    return HttpResponse(status=status, body=payload)


class AccessTokenProvider(Protocol):
    def __call__(self) -> str: ...


SubprocessRunner = Callable[..., subprocess.CompletedProcess[bytes]]


def _reject_custom_ca_environment() -> None:
    if any(os.environ.get(name) for name in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")):
        raise OwnerLauncherError("custom_ca_bundle_forbidden")


def _pinned_system_tls_context() -> ssl.SSLContext:
    """Build TLS trust from one fixed, root-owned system bundle only."""

    _reject_custom_ca_environment()
    selected: str | None = None
    for candidate in _PINNED_CA_CANDIDATES:
        try:
            metadata = os.stat(candidate, follow_symlinks=True)
        except OSError:
            continue
        if (
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == 0
            and metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH) == 0
        ):
            selected = candidate
            break
    if selected is None:
        raise OwnerLauncherError("trusted_ca_bundle_unavailable")
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        context.load_verify_locations(cafile=selected)
    except (OSError, ssl.SSLError):
        raise OwnerLauncherError("trusted_ca_bundle_unavailable") from None
    return context


def _canonical_owner_home(*, environment: Mapping[str, str] = os.environ) -> str:
    """Resolve the login home without consulting attacker-controlled ``HOME``."""

    try:
        account = pwd.getpwuid(os.getuid())  # windows-footgun: ok
        home = os.path.abspath(account.pw_dir)
        metadata = os.lstat(home)
    except (KeyError, OSError):
        raise OwnerLauncherError("canonical_owner_home_unavailable") from None
    if (
        not os.path.isabs(account.pw_dir)
        or os.path.realpath(home) != home
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()  # windows-footgun: ok
        or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise OwnerLauncherError("canonical_owner_home_invalid")
    ambient_home = environment.get("HOME")
    if ambient_home and os.path.abspath(ambient_home) != home:
        raise OwnerLauncherError("ambient_owner_home_forbidden")
    return home


def _read_pinned_regular_file(
    path: str,
    *,
    maximum: int,
    unavailable_code: str,
    invalid_code: str,
    changed_code: str,
    allowed_owners: frozenset[int],
    allow_empty: bool = False,
) -> tuple[tuple[Any, ...], bytes]:
    """Read one non-link regular file while binding its complete identity."""

    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid not in allowed_owners
            or before.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            or (before.st_size <= 0 and not allow_empty)
            or before.st_size > maximum
        ):
            raise OwnerLauncherError(invalid_code)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise OwnerLauncherError(changed_code)
            chunks: list[bytes] = []
            remaining = maximum + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OwnerLauncherError:
        raise
    except OSError:
        raise OwnerLauncherError(unavailable_code) from None
    identity = (
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_dev,
        before.st_ino,
        before.st_nlink,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_size,
    )
    after_identity = (
        after.st_mode,
        after.st_uid,
        after.st_gid,
        after.st_dev,
        after.st_ino,
        after.st_nlink,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_size,
    )
    if identity != after_identity or len(payload) != before.st_size:
        raise OwnerLauncherError(changed_code)
    return (*identity, _sha256(payload)), payload


class StableGcloudConfiguration(Protocol):
    @property
    def account(self) -> str: ...

    def environment_values(self) -> Mapping[str, str]: ...

    def assert_stable(self) -> None: ...


class PinnedGcloudConfiguration:
    """Parse and pin the one minimal human gcloud configuration without gcloud."""

    _ACCOUNT = re.compile(
        r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
        r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$"
    )

    def __init__(
        self,
        *,
        owner_home: str | os.PathLike[str] | None = None,
        environment: Mapping[str, str] = os.environ,
    ) -> None:
        if owner_home is None:
            home = _canonical_owner_home(environment=environment)
        else:
            home = os.path.abspath(os.fspath(owner_home))
            try:
                metadata = os.lstat(home)
            except OSError:
                raise OwnerLauncherError("canonical_owner_home_unavailable") from None
            if (
                os.path.realpath(home) != home
                or not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.getuid()  # windows-footgun: ok
                or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            ):
                raise OwnerLauncherError("canonical_owner_home_invalid")
            ambient_home = environment.get("HOME")
            if ambient_home and os.path.abspath(ambient_home) != home:
                raise OwnerLauncherError("ambient_owner_home_forbidden")
        self._home = home
        self._root = os.path.join(home, _GCLOUD_CONFIG_RELATIVE)
        ambient_config = environment.get("CLOUDSDK_CONFIG")
        if ambient_config and os.path.abspath(ambient_config) != self._root:
            raise OwnerLauncherError("ambient_gcloud_config_forbidden")
        self._active_path = os.path.join(self._root, "active_config")
        self._configuration_path = os.path.join(
            self._root,
            "configurations",
            f"config_{_GCLOUD_ACTIVE_CONFIG_NAME}",
        )
        self._fingerprint, self._account = self._capture()

    @staticmethod
    def _directory_identity(path: str) -> tuple[Any, ...]:
        try:
            metadata = os.lstat(path)
        except OSError:
            raise OwnerLauncherError("trusted_gcloud_config_unavailable") from None
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid()  # windows-footgun: ok
            or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise OwnerLauncherError("trusted_gcloud_config_invalid")
        return (
            metadata.st_mode,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_dev,
            metadata.st_ino,
        )

    @classmethod
    def _parse_configuration(cls, payload: bytes) -> str:
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeError:
            raise OwnerLauncherError("trusted_gcloud_config_invalid") from None
        current: str | None = None
        sections: list[str] = []
        values: dict[tuple[str, str], str] = {}
        allowed = {
            ("core", "account"),
            ("core", "project"),
            ("compute", "zone"),
        }
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                if section not in {"core", "compute"} or section in sections:
                    raise OwnerLauncherError("trusted_gcloud_config_invalid")
                current = section
                sections.append(section)
                continue
            if current is None or "=" not in line or line.startswith(("#", ";")):
                raise OwnerLauncherError("trusted_gcloud_config_invalid")
            key, value = (part.strip() for part in line.split("=", 1))
            pair = (current, key)
            if pair not in allowed or pair in values or not value:
                raise OwnerLauncherError("trusted_gcloud_config_invalid")
            values[pair] = value
        if set(values) != allowed or set(sections) != {"core", "compute"}:
            raise OwnerLauncherError("trusted_gcloud_config_invalid")
        account = values[("core", "account")]
        if (
            values[("core", "project")] != PROJECT
            or values[("compute", "zone")] != ZONE
            or cls._ACCOUNT.fullmatch(account) is None
            or account.casefold().endswith(".gserviceaccount.com")
        ):
            raise OwnerLauncherError("trusted_gcloud_config_invalid")
        return account

    def _capture(self) -> tuple[tuple[Any, ...], str]:
        private_config = self._directory_identity(os.path.join(self._home, ".config"))
        if private_config[0] & 0o077:
            raise OwnerLauncherError("trusted_gcloud_config_invalid")
        directory_fingerprint = (
            private_config,
            self._directory_identity(self._root),
            self._directory_identity(os.path.join(self._root, "configurations")),
        )
        active_fingerprint, active_payload = _read_pinned_regular_file(
            self._active_path,
            maximum=_GCLOUD_MAX_CONFIG_BYTES,
            unavailable_code="trusted_gcloud_config_unavailable",
            invalid_code="trusted_gcloud_config_invalid",
            changed_code="trusted_gcloud_config_changed",
            allowed_owners=frozenset({os.getuid()}),  # windows-footgun: ok
        )
        try:
            active_name = active_payload.decode("ascii", errors="strict")
        except UnicodeError:
            raise OwnerLauncherError("trusted_gcloud_config_invalid") from None
        if active_name not in {
            _GCLOUD_ACTIVE_CONFIG_NAME,
            f"{_GCLOUD_ACTIVE_CONFIG_NAME}\n",
        }:
            raise OwnerLauncherError("trusted_gcloud_config_invalid")
        configuration_fingerprint, configuration_payload = _read_pinned_regular_file(
            self._configuration_path,
            maximum=_GCLOUD_MAX_CONFIG_BYTES,
            unavailable_code="trusted_gcloud_config_unavailable",
            invalid_code="trusted_gcloud_config_invalid",
            changed_code="trusted_gcloud_config_changed",
            allowed_owners=frozenset({os.getuid()}),  # windows-footgun: ok
        )
        account = self._parse_configuration(configuration_payload)
        return (
            directory_fingerprint,
            active_fingerprint,
            configuration_fingerprint,
        ), account

    @property
    def account(self) -> str:
        self.assert_stable()
        return self._account

    def assert_stable(self) -> None:
        fingerprint, account = self._capture()
        if fingerprint != self._fingerprint or account != self._account:
            raise OwnerLauncherError("trusted_gcloud_config_changed")

    def environment_values(self) -> Mapping[str, str]:
        self.assert_stable()
        return {
            "HOME": self._home,
            "CLOUDSDK_CONFIG": self._root,
        }


class StableExecutable(Protocol):
    def trusted_command_prefix(self) -> tuple[str, ...]: ...


class StableKnownHosts(Protocol):
    def absolute_path(self) -> str: ...

    def private_key_path(self) -> str: ...

    def public_key_line(self) -> str: ...


class PinnedGoogleComputeKnownHosts:
    """Pin the exact owner SSH directory, host keys, and IAP identity key."""

    _MAX_BYTES = 2 * 1024 * 1024

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        private_key: str | os.PathLike[str] | None = None,
        public_key: str | os.PathLike[str] | None = None,
    ) -> None:
        ssh_root = os.path.join(_canonical_owner_home(), ".ssh")
        default = os.path.join(ssh_root, "google_compute_known_hosts")
        self._path = os.path.abspath(os.fspath(default if path is None else path))
        material_root = os.path.dirname(self._path)
        self._private_key = os.path.abspath(
            os.fspath(
                os.path.join(material_root, "google_compute_engine")
                if private_key is None
                else private_key
            )
        )
        self._public_key = os.path.abspath(
            os.fspath(
                os.path.join(material_root, "google_compute_engine.pub")
                if public_key is None
                else public_key
            )
        )
        self._ssh_root = material_root
        self._fingerprint = self._capture()

    @staticmethod
    def _validate_known_hosts_payload(payload: bytes) -> None:
        try:
            text = payload.decode("ascii", errors="strict")
        except UnicodeError:
            raise OwnerLauncherError("trusted_known_hosts_invalid") from None
        if not text.endswith("\n") or "\r" in text or "\x00" in text:
            raise OwnerLauncherError("trusted_known_hosts_invalid")
        hosts: set[str] = set()
        target_count = 0
        lines = text.splitlines()
        if not lines or len(lines) > 1_000:
            raise OwnerLauncherError("trusted_known_hosts_invalid")
        for line in lines:
            parts = line.split(" ")
            if len(parts) != 3 or any(not part for part in parts):
                raise OwnerLauncherError("trusted_known_hosts_invalid")
            host, algorithm, encoded = parts
            if (
                re.fullmatch(r"compute\.[1-9][0-9]*", host) is None
                or host in hosts
                or algorithm != "ssh-ed25519"
                or re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", encoded) is None
            ):
                raise OwnerLauncherError("trusted_known_hosts_invalid")
            try:
                blob = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError):
                raise OwnerLauncherError("trusted_known_hosts_invalid") from None
            algorithm_bytes = b"ssh-ed25519"
            if (
                len(blob) != 4 + len(algorithm_bytes) + 4 + 32
                or blob[:4] != struct.pack(">I", len(algorithm_bytes))
                or blob[4 : 4 + len(algorithm_bytes)] != algorithm_bytes
                or blob[4 + len(algorithm_bytes) : 8 + len(algorithm_bytes)]
                != struct.pack(">I", 32)
            ):
                raise OwnerLauncherError("trusted_known_hosts_invalid")
            hosts.add(host)
            if host == f"compute.{VM_INSTANCE_ID}":
                target_count += 1
        if target_count != 1:
            raise OwnerLauncherError("trusted_known_hosts_invalid")

    def _capture(self) -> tuple[Any, ...]:
        try:
            directory = os.lstat(self._ssh_root)
        except OwnerLauncherError:
            raise
        except OSError:
            raise OwnerLauncherError("trusted_known_hosts_unavailable") from None
        if (
            not stat.S_ISDIR(directory.st_mode)
            or stat.S_ISLNK(directory.st_mode)
            or directory.st_uid != os.getuid()  # windows-footgun: ok
            or stat.S_IMODE(directory.st_mode) != 0o700
            or os.path.dirname(self._private_key) != self._ssh_root
            or os.path.dirname(self._public_key) != self._ssh_root
        ):
            raise OwnerLauncherError("trusted_known_hosts_invalid")
        files: list[tuple[Any, ...]] = []
        for candidate, expected_mode in (
            (self._path, 0o644),
            (self._private_key, 0o600),
            (self._public_key, 0o644),
        ):
            try:
                metadata = os.lstat(candidate)
            except OSError:
                raise OwnerLauncherError("trusted_known_hosts_unavailable") from None
            if (
                metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != expected_mode
            ):
                raise OwnerLauncherError("trusted_known_hosts_invalid")
            fingerprint, payload = _read_pinned_regular_file(
                candidate,
                maximum=self._MAX_BYTES,
                unavailable_code="trusted_known_hosts_unavailable",
                invalid_code="trusted_known_hosts_invalid",
                changed_code="trusted_known_hosts_changed",
                allowed_owners=frozenset({os.getuid()}),  # windows-footgun: ok
            )
            if candidate == self._path:
                self._validate_known_hosts_payload(payload)
            files.append(fingerprint)
        return (
            (
                directory.st_mode,
                directory.st_uid,
                directory.st_gid,
                directory.st_dev,
                directory.st_ino,
                directory.st_mtime_ns,
                directory.st_ctime_ns,
            ),
            *files,
        )

    def _assert_stable(self) -> None:
        try:
            observed = self._capture()
        except OwnerLauncherError:
            raise OwnerLauncherError("trusted_known_hosts_changed") from None
        if observed != self._fingerprint:
            raise OwnerLauncherError("trusted_known_hosts_changed")

    def absolute_path(self) -> str:
        self._assert_stable()
        return self._path

    def private_key_path(self) -> str:
        self._assert_stable()
        return self._private_key

    def public_key_line(self) -> str:
        self._assert_stable()
        _fingerprint, payload = _read_pinned_regular_file(
            self._public_key,
            maximum=self._MAX_BYTES,
            unavailable_code="trusted_known_hosts_unavailable",
            invalid_code="trusted_known_hosts_invalid",
            changed_code="trusted_known_hosts_changed",
            allowed_owners=frozenset({os.getuid()}),  # windows-footgun: ok
        )
        try:
            line = payload.decode("ascii", errors="strict")
        except UnicodeError:
            raise OwnerLauncherError("trusted_public_key_invalid") from None
        if (
            not line.endswith("\n")
            or "\n" in line[:-1]
            or "\r" in line
            or "\x00" in line
            or len(line.split()) < 2
        ):
            raise OwnerLauncherError("trusted_public_key_invalid")
        return line[:-1]


class _PinnedExecutablePath:
    """Pin one absolute executable and every symlink/path component to it."""

    _MAX_LINKS = 16
    _MAX_EXECUTABLE_BYTES = 64 * 1024 * 1024

    def __init__(
        self,
        selected: str,
        *,
        invalid_code: str,
        changed_code: str,
    ) -> None:
        self._selected = selected
        self._invalid_code = invalid_code
        self._changed_code = changed_code
        self._fingerprint, self._resolved = self._capture(selected)

    def _capture(self, selected: str) -> tuple[tuple[Any, ...], str]:
        current = os.path.abspath(selected)
        chain: list[tuple[Any, ...]] = []
        seen: set[tuple[int, int]] = set()
        for _ in range(self._MAX_LINKS + 1):
            parts = Path(current).parts
            prefix = parts[0]
            followed_link = False
            for index, component in enumerate(parts[1:], start=1):
                prefix = os.path.join(prefix, component)
                try:
                    metadata = os.lstat(prefix)
                except OSError:
                    raise OwnerLauncherError(self._changed_code) from None
                identity = (metadata.st_dev, metadata.st_ino)
                if metadata.st_uid not in {0, os.getuid()}:  # windows-footgun: ok
                    raise OwnerLauncherError(self._invalid_code)
                if stat.S_ISLNK(metadata.st_mode):
                    if identity in seen:
                        raise OwnerLauncherError(self._invalid_code)
                    seen.add(identity)
                    try:
                        target = os.readlink(prefix)
                    except OSError:
                        raise OwnerLauncherError(self._changed_code) from None
                    chain.append((
                        "link",
                        prefix,
                        metadata.st_mode,
                        metadata.st_uid,
                        metadata.st_gid,
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                        target,
                    ))
                    target_path = (
                        target
                        if os.path.isabs(target)
                        else os.path.join(os.path.dirname(prefix), target)
                    )
                    current = os.path.abspath(
                        os.path.join(target_path, *parts[index + 1 :])
                    )
                    followed_link = True
                    break
                final_component = index == len(parts) - 1
                if not final_component:
                    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & (
                        stat.S_IWGRP | stat.S_IWOTH
                    ):
                        raise OwnerLauncherError(self._invalid_code)
                    chain.append((
                        "directory",
                        prefix,
                        metadata.st_mode,
                        metadata.st_uid,
                        metadata.st_gid,
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                    ))
                    continue
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                    or metadata.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                    == 0
                ):
                    raise OwnerLauncherError(self._invalid_code)
                file_fingerprint, _ = _read_pinned_regular_file(
                    prefix,
                    maximum=self._MAX_EXECUTABLE_BYTES,
                    unavailable_code=self._changed_code,
                    invalid_code=self._invalid_code,
                    changed_code=self._changed_code,
                    allowed_owners=frozenset({0, os.getuid()}),  # windows-footgun: ok
                )
                return (tuple(chain), ("file", prefix, *file_fingerprint)), prefix
            if followed_link:
                continue
            raise OwnerLauncherError(self._invalid_code)
        raise OwnerLauncherError(self._invalid_code)

    def absolute_path(self) -> str:
        fingerprint, resolved = self._capture(self._selected)
        if fingerprint != self._fingerprint or resolved != self._resolved:
            raise OwnerLauncherError(self._changed_code)
        return self._resolved


class TrustedGcloudExecutable:
    """Pin gcloud's wrapper, full SDK tree, and isolated Python runtime."""

    def __init__(
        self,
        *,
        candidates: Sequence[str] | None = None,
        python_candidates: Sequence[str] | None = None,
        release_sha: str | None = None,
    ) -> None:
        home_candidate = os.path.join(
            _canonical_owner_home(),
            ".hermes",
            "trusted",
            "google-cloud-sdk-569.0.0",
            "bin",
            "gcloud",
        )
        production_runtime = candidates is None and python_candidates is None
        choices = tuple(candidates or (home_candidate,))
        selected = next(
            (
                candidate
                for candidate in choices
                if isinstance(candidate, str)
                and os.path.isabs(candidate)
                and os.path.lexists(candidate)
            ),
            None,
        )
        if selected is None:
            raise OwnerLauncherError("trusted_gcloud_unavailable")
        fixed_uv_python = os.path.join(
            _canonical_owner_home(), _TRUSTED_PYTHON_RELATIVE
        )
        python_selected = next(
            (
                candidate
                for candidate in tuple(python_candidates or (fixed_uv_python,))
                if isinstance(candidate, str)
                and os.path.isabs(candidate)
                and os.path.lexists(candidate)
            ),
            None,
        )
        if python_selected is None:
            raise OwnerLauncherError("trusted_gcloud_python_unavailable")
        self._wrapper = _PinnedExecutablePath(
            selected,
            invalid_code="trusted_gcloud_invalid",
            changed_code="trusted_gcloud_changed",
        )
        self._python = _PinnedExecutablePath(
            python_selected,
            invalid_code="trusted_gcloud_python_invalid",
            changed_code="trusted_gcloud_python_changed",
        )
        wrapper_path = self._wrapper.absolute_path()
        if (
            os.path.basename(wrapper_path) != "gcloud"
            or Path(wrapper_path).parent.name != "bin"
        ):
            raise OwnerLauncherError("trusted_gcloud_invalid")
        self._sdk_root = str(Path(wrapper_path).parent.parent)
        self._gcloud_module = os.path.join(self._sdk_root, "lib", "gcloud.py")
        self._python_root = str(Path(self._python.absolute_path()).parent.parent)
        self._sdk_fingerprint = self._capture_tree(
            self._sdk_root,
            scope="sdk",
        )
        self._python_fingerprint = self._capture_tree(
            self._python_root,
            scope="python_tree",
        )
        self._production_runtime = production_runtime
        self._release_sha = release_sha
        self._otool: _PinnedExecutablePath | None = None
        self._python_version: str | None = None
        self._python_dependencies: tuple[str, ...] = ()
        self._launcher_sha256: str | None = None
        self._sdk_publication_fingerprint: tuple[int, int, str] | None = None
        self._publication_intent_fingerprint: tuple[Any, ...] | None = None
        self._publication_intent: Mapping[str, Any] | None = None
        self._bootstrap_receipt_fingerprint: tuple[Any, ...] | None = None
        if self._production_runtime:
            if release_sha is None or _RELEASE_SHA.fullmatch(release_sha) is None:
                raise OwnerLauncherError("trusted_runtime_release_unbound")
            self._validate_sdk_version()
            self._otool = _PinnedExecutablePath(
                "/usr/bin/otool",
                invalid_code="trusted_otool_invalid",
                changed_code="trusted_otool_changed",
            )
            self._python_version = self._capture_python_version()
            self._python_dependencies = self._capture_python_dependencies()
            self._launcher_sha256 = _current_launcher_sha256()
            self._sdk_publication_fingerprint = _capture_sdk_publication_tree(
                self._sdk_root
            )
            (
                self._publication_intent_fingerprint,
                self._publication_intent,
            ) = _validate_sdk_publication_intent(
                self._publication_intent_path(),
                destination=self._sdk_root,
                publication_tree=self._sdk_publication_fingerprint,
            )
            self._bootstrap_receipt_fingerprint = self._validate_bootstrap_receipt()

    @staticmethod
    def _feed_tree_entry(digest: Any, value: Sequence[Any]) -> None:
        encoded = json.dumps(
            list(value),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)

    @classmethod
    def _capture_tree(
        cls,
        root: str,
        *,
        scope: str,
    ) -> tuple[int, int, str]:
        invalid_code = f"trusted_gcloud_{scope}_invalid"
        changed_code = f"trusted_gcloud_{scope}_changed"
        oversized_code = f"trusted_gcloud_{scope}_oversized"
        digest = hashlib.sha256()
        entry_count = 0
        total_bytes = 0
        pending = [root]
        while pending:
            path = pending.pop()
            try:
                metadata = os.lstat(path)
            except OSError:
                raise OwnerLauncherError(changed_code) from None
            relative = os.path.relpath(path, root)
            if scope == "sdk" and (
                os.path.basename(path) == "__pycache__"
                or path.endswith((".pyc", ".pyo"))
            ):
                raise OwnerLauncherError("trusted_gcloud_sdk_bytecode_forbidden")
            if metadata.st_uid not in {0, os.getuid()}:  # windows-footgun: ok
                raise OwnerLauncherError(invalid_code)
            if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise OwnerLauncherError(invalid_code)
            common = (
                relative,
                metadata.st_mode,
                metadata.st_uid,
                metadata.st_gid,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_nlink,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
                metadata.st_size,
            )
            if stat.S_ISDIR(metadata.st_mode):
                cls._feed_tree_entry(digest, ("directory", *common))
                try:
                    children = sorted(
                        os.scandir(path),
                        key=lambda child: os.fsencode(child.name),
                        reverse=True,
                    )
                except OSError:
                    raise OwnerLauncherError(changed_code) from None
                pending.extend(child.path for child in children)
                try:
                    after = os.lstat(path)
                except OSError:
                    raise OwnerLauncherError(changed_code) from None
                if (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ) != (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                ):
                    raise OwnerLauncherError(changed_code)
            elif stat.S_ISREG(metadata.st_mode):
                file_fingerprint, payload = _read_pinned_regular_file(
                    path,
                    maximum=_GCLOUD_MAX_SDK_FILE_BYTES,
                    unavailable_code=changed_code,
                    invalid_code=invalid_code,
                    changed_code=changed_code,
                    allowed_owners=frozenset({0, os.getuid()}),  # windows-footgun: ok
                    allow_empty=True,
                )
                total_bytes += len(payload)
                cls._feed_tree_entry(
                    digest,
                    ("file", relative, *file_fingerprint),
                )
            elif stat.S_ISLNK(metadata.st_mode):
                try:
                    target = os.readlink(path)
                    after = os.lstat(path)
                except OSError:
                    raise OwnerLauncherError(changed_code) from None
                try:
                    target_path = os.path.realpath(path, strict=True)
                    inside_root = os.path.commonpath((root, target_path)) == root
                except (OSError, ValueError):
                    inside_root = False
                if not inside_root or (
                    after.st_dev,
                    after.st_ino,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ) != (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                ):
                    raise OwnerLauncherError(invalid_code)
                cls._feed_tree_entry(digest, ("symlink", *common, target))
            else:
                raise OwnerLauncherError(invalid_code)
            entry_count += 1
            if (
                entry_count > _GCLOUD_MAX_SDK_ENTRIES
                or total_bytes > _GCLOUD_MAX_SDK_BYTES
            ):
                raise OwnerLauncherError(oversized_code)
        return entry_count, total_bytes, digest.hexdigest()

    def _validate_sdk_version(self) -> None:
        _fingerprint, payload = _read_pinned_regular_file(
            os.path.join(self._sdk_root, "VERSION"),
            maximum=128,
            unavailable_code="trusted_gcloud_sdk_changed",
            invalid_code="trusted_gcloud_sdk_invalid",
            changed_code="trusted_gcloud_sdk_changed",
            allowed_owners=frozenset({0, os.getuid()}),  # windows-footgun: ok
        )
        if payload != f"{_GCLOUD_SDK_VERSION}\n".encode("ascii"):
            raise OwnerLauncherError("trusted_gcloud_sdk_version_invalid")

    def _capture_python_dependencies(self) -> tuple[str, ...]:
        if self._otool is None:
            raise OwnerLauncherError("trusted_otool_unavailable")
        otool = self._otool.absolute_path()
        python = self._python.absolute_path()
        try:
            completed = subprocess.run(
                (otool, "-L", python),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env={
                    "PATH": _FIXED_OWNER_PATH,
                    "LANG": "C",
                    "LC_ALL": "C",
                    "TMPDIR": "/tmp",
                },
                timeout=20.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            raise OwnerLauncherError(
                "trusted_python_dependencies_unavailable"
            ) from None
        finally:
            self._otool.absolute_path()
            self._python.absolute_path()
        if (
            completed.returncode != 0
            or not isinstance(completed.stdout, bytes)
            or not completed.stdout
            or len(completed.stdout) > 64 * 1024
        ):
            raise OwnerLauncherError("trusted_python_dependencies_invalid")
        try:
            lines = completed.stdout.decode("utf-8", errors="strict").splitlines()
        except UnicodeError:
            raise OwnerLauncherError("trusted_python_dependencies_invalid") from None
        if not lines or lines[0] != f"{python}:":
            raise OwnerLauncherError("trusted_python_dependencies_invalid")
        dependencies: list[str] = []
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped or " (" not in stripped:
                raise OwnerLauncherError("trusted_python_dependencies_invalid")
            dependency = stripped.split(" (", 1)[0]
            if not dependency.startswith(("/usr/lib/", "/System/Library/")):
                raise OwnerLauncherError("trusted_python_dependencies_invalid")
            dependencies.append(dependency)
        observed = tuple(dependencies)
        if observed != _TRUSTED_PYTHON_DEPENDENCIES:
            raise OwnerLauncherError("trusted_python_dependencies_invalid")
        return observed

    def _capture_python_version(self) -> str:
        python = self._python.absolute_path()
        try:
            completed = subprocess.run(
                (
                    python,
                    *_GCLOUD_PYTHON_ISOLATION_ARGS,
                    "-c",
                    "import sys;print('.'.join(map(str,sys.version_info[:3])))",
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env={
                    "PATH": _FIXED_OWNER_PATH,
                    "LANG": "C",
                    "LC_ALL": "C",
                    "TMPDIR": "/tmp",
                },
                timeout=20.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            raise OwnerLauncherError("trusted_python_version_unavailable") from None
        finally:
            self._python.absolute_path()
        expected = f"{_TRUSTED_PYTHON_VERSION}\n".encode("ascii")
        if completed.returncode != 0 or completed.stdout != expected:
            raise OwnerLauncherError("trusted_python_version_invalid")
        return _TRUSTED_PYTHON_VERSION

    def _publication_intent_path(self) -> str:
        home = _canonical_owner_home()
        return os.path.join(home, _TRUSTED_SDK_PUBLICATION_INTENT_RELATIVE)

    def _bootstrap_receipt_path(self) -> str:
        if self._release_sha is None:
            raise OwnerLauncherError("trusted_runtime_release_unbound")
        home = _canonical_owner_home()
        return os.path.join(
            home,
            ".hermes",
            "trusted",
            f"trusted-runtime-bootstrap-{self._release_sha}.json",
        )

    def _expected_bootstrap_receipt_fields(self) -> Mapping[str, Any]:
        if (
            self._release_sha is None
            or self._launcher_sha256 is None
            or self._publication_intent is None
            or self._sdk_publication_fingerprint is None
        ):
            raise OwnerLauncherError("trusted_runtime_release_unbound")
        return {
            "schema": TRUSTED_RUNTIME_BOOTSTRAP_RECEIPT_SCHEMA,
            "ok": True,
            "state": "trusted_runtime_ready",
            "release_sha": self._release_sha,
            "launcher_sha256": self._launcher_sha256,
            "sdk_archive_url": _GCLOUD_SDK_ARCHIVE_URL,
            "sdk_archive_bytes": _GCLOUD_SDK_ARCHIVE_BYTES,
            "sdk_archive_sha256": _GCLOUD_SDK_ARCHIVE_SHA256,
            "sdk_version": _GCLOUD_SDK_VERSION,
            "sdk_root": self._sdk_root,
            "sdk_tree_entries": self._sdk_fingerprint[0],
            "sdk_tree_bytes": self._sdk_fingerprint[1],
            "sdk_tree_sha256": self._sdk_fingerprint[2],
            "sdk_publication_release_sha": self._publication_intent[
                "publication_release_sha"
            ],
            "sdk_publication_intent_sha256": self._publication_intent["intent_sha256"],
            "sdk_publication_tree_entries": self._sdk_publication_fingerprint[0],
            "sdk_publication_tree_bytes": self._sdk_publication_fingerprint[1],
            "sdk_publication_tree_sha256": self._sdk_publication_fingerprint[2],
            "python_root": self._python_root,
            "python_executable": self._python.absolute_path(),
            "python_version": self._python_version,
            "python_tree_entries": self._python_fingerprint[0],
            "python_tree_bytes": self._python_fingerprint[1],
            "python_tree_sha256": self._python_fingerprint[2],
            "python_dependencies": list(self._python_dependencies),
        }

    def _validate_bootstrap_receipt(self) -> tuple[Any, ...]:
        fingerprint, payload = _read_pinned_regular_file(
            self._bootstrap_receipt_path(),
            maximum=_GCLOUD_MAX_CONFIG_BYTES,
            unavailable_code="trusted_runtime_bootstrap_receipt_unavailable",
            invalid_code="trusted_runtime_bootstrap_receipt_invalid",
            changed_code="trusted_runtime_bootstrap_receipt_changed",
            allowed_owners=frozenset({os.getuid()}),  # windows-footgun: ok
        )
        if stat.S_IMODE(int(fingerprint[0])) != 0o600 or int(fingerprint[5]) != 1:
            raise OwnerLauncherError("trusted_runtime_bootstrap_receipt_invalid")
        try:
            value = _decode_json_object(payload, maximum=_GCLOUD_MAX_CONFIG_BYTES)
        except OwnerLauncherError:
            raise OwnerLauncherError(
                "trusted_runtime_bootstrap_receipt_invalid"
            ) from None
        if payload != _canonical_bytes(value) + b"\n":
            raise OwnerLauncherError("trusted_runtime_bootstrap_receipt_invalid")
        expected = self._expected_bootstrap_receipt_fields()
        expected_keys = set(expected) | {"created_at_unix", "receipt_sha256"}
        created = value.get("created_at_unix")
        receipt_sha = value.get("receipt_sha256")
        unsigned = dict(value)
        unsigned.pop("receipt_sha256", None)
        if (
            set(value) != expected_keys
            or any(value.get(key) != item for key, item in expected.items())
            or type(created) is not int
            or created < 0
            or not isinstance(receipt_sha, str)
            or receipt_sha != _sha256(_canonical_bytes(unsigned))
        ):
            raise OwnerLauncherError("trusted_runtime_bootstrap_receipt_invalid")
        return fingerprint

    def trusted_command_prefix(self) -> tuple[str, ...]:
        self._wrapper.absolute_path()
        python = self._python.absolute_path()
        if self._capture_tree(self._sdk_root, scope="sdk") != self._sdk_fingerprint:
            raise OwnerLauncherError("trusted_gcloud_sdk_changed")
        if (
            self._capture_tree(self._python_root, scope="python_tree")
            != self._python_fingerprint
        ):
            raise OwnerLauncherError("trusted_gcloud_python_tree_changed")
        if self._production_runtime:
            self._validate_sdk_version()
            if _current_launcher_sha256() != self._launcher_sha256:
                raise OwnerLauncherError("local_launcher_changed")
            if self._capture_python_version() != self._python_version:
                raise OwnerLauncherError("trusted_python_version_changed")
            if self._capture_python_dependencies() != self._python_dependencies:
                raise OwnerLauncherError("trusted_python_dependencies_changed")
            publication_fingerprint = _capture_sdk_publication_tree(self._sdk_root)
            if publication_fingerprint != self._sdk_publication_fingerprint:
                raise OwnerLauncherError("trusted_runtime_publication_tree_changed")
            intent_fingerprint, intent = _validate_sdk_publication_intent(
                self._publication_intent_path(),
                destination=self._sdk_root,
                publication_tree=publication_fingerprint,
            )
            if (
                intent_fingerprint != self._publication_intent_fingerprint
                or intent != self._publication_intent
            ):
                raise OwnerLauncherError("trusted_runtime_publication_intent_changed")
            if (
                self._validate_bootstrap_receipt()
                != self._bootstrap_receipt_fingerprint
            ):
                raise OwnerLauncherError("trusted_runtime_bootstrap_receipt_changed")
        try:
            module = os.path.realpath(self._gcloud_module, strict=True)
        except OSError:
            raise OwnerLauncherError("trusted_gcloud_sdk_changed") from None
        if module != self._gcloud_module:
            raise OwnerLauncherError("trusted_gcloud_sdk_invalid")
        return (python, *_GCLOUD_PYTHON_ISOLATION_ARGS, module)


def _owner_gcloud_environment(
    configuration: StableGcloudConfiguration,
    python_path: str,
) -> Mapping[str, str]:
    """Return a closed, non-secret gcloud environment for the pinned runtime."""

    _reject_custom_ca_environment()
    if not os.path.isabs(python_path):
        raise OwnerLauncherError("trusted_gcloud_python_invalid")
    result = dict(configuration.environment_values())
    result.update({
        "PATH": _FIXED_OWNER_PATH,
        "TMPDIR": "/tmp",
        "LANG": "C",
        "LC_ALL": "C",
        "CLOUDSDK_CORE_PROJECT": PROJECT,
        "CLOUDSDK_COMPUTE_ZONE": ZONE,
        "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
        "CLOUDSDK_CORE_LOG_HTTP": "0",
        "CLOUDSDK_CORE_LOG_HTTP_SHOW_REQUEST_BODY": "0",
        "CLOUDSDK_CORE_LOG_HTTP_REDACT_TOKEN": "1",
        "CLOUDSDK_CORE_DISABLE_FILE_LOGGING": "1",
        "CLOUDSDK_CORE_DISABLE_USAGE_REPORTING": "1",
        "CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK": "1",
        "CLOUDSDK_CORE_VERBOSITY": "error",
        "CLOUDSDK_PYTHON": python_path,
        "CLOUDSDK_PYTHON_ARGS": " ".join(_GCLOUD_PYTHON_ISOLATION_ARGS),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return result


def _require_private_owner_directory(
    path: str,
    *,
    create: bool,
) -> None:
    if create and not os.path.lexists(path):
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        except OSError:
            raise OwnerLauncherError("trusted_runtime_directory_unavailable") from None
    try:
        metadata = os.lstat(path)
    except OSError:
        raise OwnerLauncherError("trusted_runtime_directory_unavailable") from None
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()  # windows-footgun: ok
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise OwnerLauncherError("trusted_runtime_directory_invalid")


def _darwin_rename_no_replace(
    source: str,
    destination: str,
    *,
    exists_code: str,
    failed_code: str,
) -> None:
    """Atomically publish one Darwin pathname without replacing any object."""

    if sys.platform != "darwin":
        raise OwnerLauncherError("atomic_no_replace_unavailable")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renamex = libc.renamex_np
        renamex.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex.restype = ctypes.c_int
    except (AttributeError, OSError):
        raise OwnerLauncherError("atomic_no_replace_unavailable") from None
    ctypes.set_errno(0)
    if renamex(os.fsencode(source), os.fsencode(destination), 0x00000004) != 0:
        error = ctypes.get_errno()
        if error in {
            errno.EEXIST,
            errno.ENOTEMPTY,
        }:
            raise OwnerLauncherError(exists_code)
        raise OwnerLauncherError(failed_code)


def _rename_directory_no_replace(source: str, destination: str) -> None:
    """Darwin atomic directory publication with an explicit exclusion flag."""

    _darwin_rename_no_replace(
        source,
        destination,
        exists_code="trusted_runtime_destination_exists",
        failed_code="trusted_runtime_publish_failed",
    )


def _publish_regular_no_replace(
    source: str,
    destination: str,
    *,
    exists_code: str = "trusted_runtime_bootstrap_receipt_exists",
    failed_code: str = "trusted_runtime_bootstrap_receipt_failed",
    cleanup_code: str = "trusted_runtime_bootstrap_staging_cleanup_failed",
) -> None:
    """Atomically add one regular file name without a two-link crash state."""

    del cleanup_code
    _darwin_rename_no_replace(
        source,
        destination,
        exists_code=exists_code,
        failed_code=failed_code,
    )


def _fsync_directory(path: str, *, error_code: str) -> None:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        raise OwnerLauncherError(error_code) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_owner_file_no_replace(
    destination: str,
    payload: bytes,
    *,
    parent: str,
    temporary_prefix: str,
    exists_code: str,
    failed_code: str,
) -> None:
    """Durably publish one owner-only regular file without replacing a name."""

    temporary = os.path.join(parent, f"{temporary_prefix}.{secrets.token_hex(8)}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short owner file write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _publish_regular_no_replace(
            temporary,
            destination,
            exists_code=exists_code,
            failed_code=failed_code,
            cleanup_code=f"{failed_code}_staging_cleanup",
        )
        _fsync_directory(parent, error_code=failed_code)
    except OwnerLauncherError:
        raise
    except OSError:
        raise OwnerLauncherError(failed_code) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.lexists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _current_launcher_sha256() -> str:
    module_path = os.path.abspath(__file__)
    try:
        if os.path.realpath(module_path, strict=True) != module_path:
            raise OwnerLauncherError("local_launcher_path_invalid")
    except OSError:
        raise OwnerLauncherError("local_launcher_unavailable") from None
    fingerprint, _payload = _read_pinned_regular_file(
        module_path,
        maximum=_MAX_LAUNCHER_MODULE_BYTES,
        unavailable_code="local_launcher_unavailable",
        invalid_code="local_launcher_invalid",
        changed_code="local_launcher_changed",
        allowed_owners=frozenset({os.getuid()}),  # windows-footgun: ok
    )
    return str(fingerprint[-1])


def _capture_sdk_publication_tree(root: str) -> tuple[int, int, str]:
    """Return a deterministic complete SDK fingerprint across re-extraction."""

    digest = hashlib.sha256()
    entry_count = 0
    total_bytes = 0
    pending = [root]
    while pending:
        path = pending.pop()
        try:
            metadata = os.lstat(path)
        except OSError:
            raise OwnerLauncherError(
                "trusted_runtime_publication_tree_changed"
            ) from None
        relative = os.path.relpath(path, root)
        if os.path.basename(path) == "__pycache__" or path.endswith((".pyc", ".pyo")):
            raise OwnerLauncherError("trusted_gcloud_sdk_bytecode_forbidden")
        if metadata.st_uid not in {
            0,
            os.getuid(),  # windows-footgun: ok
        } or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise OwnerLauncherError("trusted_runtime_publication_tree_invalid")
        if stat.S_ISDIR(metadata.st_mode):
            TrustedGcloudExecutable._feed_tree_entry(
                digest,
                ("directory", relative, stat.S_IMODE(metadata.st_mode)),
            )
            try:
                with os.scandir(path) as entries:
                    children = sorted(
                        (entry.path for entry in entries),
                        key=os.fsencode,
                        reverse=True,
                    )
            except OSError:
                raise OwnerLauncherError(
                    "trusted_runtime_publication_tree_changed"
                ) from None
            pending.extend(children)
        elif stat.S_ISREG(metadata.st_mode):
            _fingerprint, payload = _read_pinned_regular_file(
                path,
                maximum=_GCLOUD_MAX_SDK_FILE_BYTES,
                unavailable_code="trusted_runtime_publication_tree_changed",
                invalid_code="trusted_runtime_publication_tree_invalid",
                changed_code="trusted_runtime_publication_tree_changed",
                allowed_owners=frozenset({0, os.getuid()}),  # windows-footgun: ok
                allow_empty=True,
            )
            total_bytes += len(payload)
            TrustedGcloudExecutable._feed_tree_entry(
                digest,
                (
                    "file",
                    relative,
                    stat.S_IMODE(metadata.st_mode),
                    len(payload),
                    _sha256(payload),
                ),
            )
        elif stat.S_ISLNK(metadata.st_mode):
            try:
                target = os.readlink(path)
                resolved = os.path.realpath(path, strict=True)
                inside = os.path.commonpath((root, resolved)) == root
            except (OSError, ValueError):
                inside = False
                target = ""
            if not inside:
                raise OwnerLauncherError("trusted_runtime_publication_tree_invalid")
            TrustedGcloudExecutable._feed_tree_entry(
                digest,
                ("symlink", relative, target),
            )
        else:
            raise OwnerLauncherError("trusted_runtime_publication_tree_invalid")
        entry_count += 1
        if entry_count > _GCLOUD_MAX_SDK_ENTRIES or total_bytes > _GCLOUD_MAX_SDK_BYTES:
            raise OwnerLauncherError("trusted_runtime_publication_tree_oversized")
    return entry_count, total_bytes, digest.hexdigest()


def _validate_sdk_publication_intent(
    intent_path: str,
    *,
    destination: str,
    publication_tree: tuple[int, int, str],
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    fingerprint, payload = _read_pinned_regular_file(
        intent_path,
        maximum=_GCLOUD_MAX_CONFIG_BYTES,
        unavailable_code="trusted_runtime_publication_intent_unavailable",
        invalid_code="trusted_runtime_publication_intent_invalid",
        changed_code="trusted_runtime_publication_intent_changed",
        allowed_owners=frozenset({os.getuid()}),  # windows-footgun: ok
    )
    if stat.S_IMODE(int(fingerprint[0])) != 0o600 or int(fingerprint[5]) != 1:
        raise OwnerLauncherError("trusted_runtime_publication_intent_invalid")
    try:
        value = _decode_json_object(payload, maximum=_GCLOUD_MAX_CONFIG_BYTES)
    except OwnerLauncherError:
        raise OwnerLauncherError("trusted_runtime_publication_intent_invalid") from None
    if payload != _canonical_bytes(value) + b"\n":
        raise OwnerLauncherError("trusted_runtime_publication_intent_invalid")
    expected = {
        "schema": TRUSTED_SDK_PUBLICATION_INTENT_SCHEMA,
        "ok": True,
        "state": "trusted_sdk_publication_prepared",
        "sdk_archive_url": _GCLOUD_SDK_ARCHIVE_URL,
        "sdk_archive_bytes": _GCLOUD_SDK_ARCHIVE_BYTES,
        "sdk_archive_sha256": _GCLOUD_SDK_ARCHIVE_SHA256,
        "sdk_version": _GCLOUD_SDK_VERSION,
        "sdk_root": destination,
        "sdk_tree_entries": publication_tree[0],
        "sdk_tree_bytes": publication_tree[1],
        "sdk_tree_sha256": publication_tree[2],
    }
    publication_release = value.get("publication_release_sha")
    launcher_sha = value.get("launcher_sha256")
    prepared = value.get("prepared_at_unix")
    intent_sha = value.get("intent_sha256")
    unsigned = dict(value)
    unsigned.pop("intent_sha256", None)
    if (
        set(value)
        != set(expected)
        | {
            "publication_release_sha",
            "launcher_sha256",
            "prepared_at_unix",
            "intent_sha256",
        }
        or any(value.get(key) != item for key, item in expected.items())
        or not isinstance(publication_release, str)
        or _RELEASE_SHA.fullmatch(publication_release) is None
        or not isinstance(launcher_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", launcher_sha) is None
        or type(prepared) is not int
        or prepared < 0
        or not isinstance(intent_sha, str)
        or intent_sha != _sha256(_canonical_bytes(unsigned))
    ):
        raise OwnerLauncherError("trusted_runtime_publication_intent_invalid")
    return fingerprint, value


def _download_pinned_gcloud_archive(destination: str) -> None:
    """Download the one reviewed Google archive without proxy or redirect."""

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    context = _pinned_system_tls_context()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        NoRedirect(),
        urllib.request.HTTPSHandler(context=context),
    )
    request = urllib.request.Request(
        _GCLOUD_SDK_ARCHIVE_URL,
        method="GET",
        headers={"Accept": "application/octet-stream"},
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    digest = hashlib.sha256()
    count = 0
    try:
        descriptor = os.open(destination, flags, 0o600)
        with opener.open(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
            length = response.headers.get("Content-Length")
            if (
                int(response.status) != 200
                or response.geturl() != _GCLOUD_SDK_ARCHIVE_URL
                or length != str(_GCLOUD_SDK_ARCHIVE_BYTES)
            ):
                raise OwnerLauncherError("trusted_runtime_archive_rejected")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > _GCLOUD_SDK_ARCHIVE_BYTES:
                    raise OwnerLauncherError("trusted_runtime_archive_oversized")
                digest.update(chunk)
                view = memoryview(chunk)
                offset = 0
                try:
                    while offset < len(view):
                        written = os.write(descriptor, view[offset:])
                        if written <= 0:
                            raise OSError("short archive write")
                        offset += written
                finally:
                    view.release()
        os.fsync(descriptor)
    except OwnerLauncherError:
        raise
    except (OSError, urllib.error.URLError, TimeoutError, ValueError):
        raise OwnerLauncherError("trusted_runtime_archive_unavailable") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        count != _GCLOUD_SDK_ARCHIVE_BYTES
        or digest.hexdigest() != _GCLOUD_SDK_ARCHIVE_SHA256
    ):
        raise OwnerLauncherError("trusted_runtime_archive_digest_mismatch")


def _safe_extract_gcloud_archive(archive_path: str, staging_root: str) -> str:
    """Extract reviewed tar members with exclusive files and no path traversal."""

    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError):
        raise OwnerLauncherError("trusted_runtime_archive_invalid") from None
    root = os.path.join(staging_root, "google-cloud-sdk")
    seen: set[str] = set()
    directories: set[str] = {root}
    files: list[tuple[tarfile.TarInfo, str]] = []
    links: list[tuple[tarfile.TarInfo, str]] = []
    total_bytes = 0
    try:
        members = archive.getmembers()
        if len(members) > _GCLOUD_MAX_SDK_ENTRIES:
            raise OwnerLauncherError("trusted_runtime_archive_oversized")
        for member in members:
            name = member.name.rstrip("/")
            pure = PurePosixPath(name)
            if (
                not name
                or pure.is_absolute()
                or not pure.parts
                or pure.parts[0] != "google-cloud-sdk"
                or any(part in {"", ".", ".."} for part in pure.parts)
                or name in seen
                or "\x00" in name
                or "__pycache__" in pure.parts
                or name.endswith((".pyc", ".pyo"))
            ):
                raise OwnerLauncherError("trusted_runtime_archive_member_invalid")
            seen.add(name)
            destination = os.path.join(staging_root, *pure.parts)
            parent = os.path.dirname(destination)
            while parent != staging_root:
                directories.add(parent)
                parent = os.path.dirname(parent)
            if member.isdir():
                directories.add(destination)
            elif member.isreg():
                if member.size < 0 or member.size > _GCLOUD_MAX_SDK_FILE_BYTES:
                    raise OwnerLauncherError("trusted_runtime_archive_oversized")
                total_bytes += member.size
                files.append((member, destination))
            elif member.issym():
                target = member.linkname
                target_path = os.path.abspath(
                    os.path.join(os.path.dirname(destination), target)
                )
                try:
                    contained = os.path.commonpath((root, target_path)) == root
                except ValueError:
                    contained = False
                if (
                    not target
                    or os.path.isabs(target)
                    or not contained
                    or "\x00" in target
                ):
                    raise OwnerLauncherError("trusted_runtime_archive_link_invalid")
                links.append((member, destination))
            else:
                raise OwnerLauncherError("trusted_runtime_archive_member_invalid")
        if total_bytes > _GCLOUD_MAX_SDK_BYTES:
            raise OwnerLauncherError("trusted_runtime_archive_oversized")
        non_directories = {destination for _member, destination in (*files, *links)}
        if directories & non_directories:
            raise OwnerLauncherError("trusted_runtime_archive_member_invalid")
        for destination in sorted(
            directories,
            key=lambda item: len(Path(item).parts),
        ):
            try:
                os.mkdir(destination, 0o700)
            except FileExistsError:
                metadata = os.lstat(destination)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise OwnerLauncherError("trusted_runtime_archive_member_invalid")
        if not os.path.isdir(root):
            raise OwnerLauncherError("trusted_runtime_archive_root_missing")
        for member, destination in files:
            parent = os.path.dirname(destination)
            if not os.path.isdir(parent):
                raise OwnerLauncherError("trusted_runtime_archive_member_invalid")
            source = archive.extractfile(member)
            if source is None:
                raise OwnerLauncherError("trusted_runtime_archive_member_invalid")
            mode = 0o700 if member.mode & 0o111 else 0o600
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = -1
            copied = 0
            try:
                descriptor = os.open(destination, flags, mode)
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > member.size:
                        raise OwnerLauncherError(
                            "trusted_runtime_archive_member_invalid"
                        )
                    view = memoryview(chunk)
                    offset = 0
                    try:
                        while offset < len(view):
                            written = os.write(descriptor, view[offset:])
                            if written <= 0:
                                raise OSError("short member write")
                            offset += written
                    finally:
                        view.release()
                os.fsync(descriptor)
            except OwnerLauncherError:
                raise
            except OSError:
                raise OwnerLauncherError(
                    "trusted_runtime_archive_extract_failed"
                ) from None
            finally:
                source.close()
                if descriptor >= 0:
                    os.close(descriptor)
            if copied != member.size:
                raise OwnerLauncherError("trusted_runtime_archive_member_invalid")
        for member, destination in links:
            if not os.path.isdir(os.path.dirname(destination)):
                raise OwnerLauncherError("trusted_runtime_archive_member_invalid")
            try:
                os.symlink(member.linkname, destination)
            except OSError:
                raise OwnerLauncherError(
                    "trusted_runtime_archive_link_invalid"
                ) from None
        for _member, destination in links:
            try:
                resolved = os.path.realpath(destination, strict=True)
                if os.path.commonpath((root, resolved)) != root:
                    raise OwnerLauncherError("trusted_runtime_archive_link_invalid")
            except OwnerLauncherError:
                raise
            except (OSError, ValueError):
                raise OwnerLauncherError(
                    "trusted_runtime_archive_link_invalid"
                ) from None
        for directory in sorted(
            directories,
            key=lambda item: len(Path(item).parts),
            reverse=True,
        ):
            try:
                descriptor = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                raise OwnerLauncherError(
                    "trusted_runtime_archive_fsync_failed"
                ) from None
    finally:
        archive.close()
    return root


def _fixed_python_runtime_snapshot() -> tuple[
    _PinnedExecutablePath,
    str,
    tuple[int, int, str],
    str,
    tuple[str, ...],
]:
    home = _canonical_owner_home()
    python_path = os.path.join(home, _TRUSTED_PYTHON_RELATIVE)
    python = _PinnedExecutablePath(
        python_path,
        invalid_code="trusted_gcloud_python_invalid",
        changed_code="trusted_gcloud_python_changed",
    )
    resolved = python.absolute_path()
    python_root = str(Path(resolved).parent.parent)
    tree = TrustedGcloudExecutable._capture_tree(
        python_root,
        scope="python_tree",
    )
    probe = object.__new__(TrustedGcloudExecutable)
    probe._python = python
    probe._python_root = python_root
    probe._otool = _PinnedExecutablePath(
        "/usr/bin/otool",
        invalid_code="trusted_otool_invalid",
        changed_code="trusted_otool_changed",
    )
    version = probe._capture_python_version()
    dependencies = probe._capture_python_dependencies()
    return python, python_root, tree, version, dependencies


def bootstrap_trusted_gcloud_runtime(
    release_sha: str,
    *,
    now_unix: int | None = None,
    launcher_sha256: str | None = None,
    archive_downloader: Callable[[str], None] = _download_pinned_gcloud_archive,
    python_snapshot: Callable[
        [],
        tuple[
            _PinnedExecutablePath,
            str,
            tuple[int, int, str],
            str,
            tuple[str, ...],
        ],
    ] = _fixed_python_runtime_snapshot,
    runtime_validator: Callable[[str], None] | None = None,
) -> Mapping[str, Any]:
    """Publish the reviewed SDK snapshot and one release-bound receipt."""

    if _RELEASE_SHA.fullmatch(release_sha) is None:
        raise OwnerLauncherError("invalid_release_sha")
    current_launcher_sha256 = _current_launcher_sha256()
    if launcher_sha256 is not None and launcher_sha256 != current_launcher_sha256:
        raise OwnerLauncherError("local_launcher_changed")
    home = _canonical_owner_home()
    hermes_root = os.path.join(home, ".hermes")
    trusted_root = os.path.join(hermes_root, "trusted")
    _require_private_owner_directory(hermes_root, create=False)
    _require_private_owner_directory(trusted_root, create=True)
    destination = os.path.join(home, _TRUSTED_SDK_RELATIVE)
    intent_path = os.path.join(
        home,
        _TRUSTED_SDK_PUBLICATION_INTENT_RELATIVE,
    )
    receipt_path = os.path.join(
        trusted_root,
        f"trusted-runtime-bootstrap-{release_sha}.json",
    )
    validate_runtime = runtime_validator or (
        lambda exact_release: TrustedGcloudExecutable(
            release_sha=exact_release
        ).trusted_command_prefix()
    )
    if os.path.lexists(receipt_path):
        validate_runtime(release_sha)
        _fingerprint, payload = _read_pinned_regular_file(
            receipt_path,
            maximum=_GCLOUD_MAX_CONFIG_BYTES,
            unavailable_code="trusted_runtime_bootstrap_receipt_unavailable",
            invalid_code="trusted_runtime_bootstrap_receipt_invalid",
            changed_code="trusted_runtime_bootstrap_receipt_changed",
            allowed_owners=frozenset({os.getuid()}),  # windows-footgun: ok
        )
        return _decode_json_object(payload, maximum=_GCLOUD_MAX_CONFIG_BYTES)

    created = int(time.time()) if now_unix is None else now_unix
    if type(created) is not int or created < 0:
        raise OwnerLauncherError("trusted_runtime_bootstrap_clock_invalid")

    if os.path.lexists(destination):
        publication_tree = _capture_sdk_publication_tree(destination)
        _validate_sdk_publication_intent(
            intent_path,
            destination=destination,
            publication_tree=publication_tree,
        )
    else:
        stage_parent = tempfile.mkdtemp(prefix=".gcloud-bootstrap-", dir=trusted_root)
        os.chmod(stage_parent, 0o700)
        archive_path = os.path.join(stage_parent, "sdk.tar.gz")
        try:
            archive_downloader(archive_path)
            archive_fingerprint, archive_payload = _read_pinned_regular_file(
                archive_path,
                maximum=_GCLOUD_SDK_ARCHIVE_BYTES,
                unavailable_code="trusted_runtime_archive_unavailable",
                invalid_code="trusted_runtime_archive_invalid",
                changed_code="trusted_runtime_archive_changed",
                allowed_owners=frozenset({os.getuid()}),  # windows-footgun: ok
            )
            if (
                archive_fingerprint[-2] != _GCLOUD_SDK_ARCHIVE_BYTES
                or len(archive_payload) != _GCLOUD_SDK_ARCHIVE_BYTES
                or archive_fingerprint[-1] != _GCLOUD_SDK_ARCHIVE_SHA256
            ):
                raise OwnerLauncherError("trusted_runtime_archive_digest_mismatch")
            extracted = _safe_extract_gcloud_archive(archive_path, stage_parent)
            os.unlink(archive_path)
            version_path = os.path.join(extracted, "VERSION")
            _version_fingerprint, version = _read_pinned_regular_file(
                version_path,
                maximum=128,
                unavailable_code="trusted_runtime_archive_invalid",
                invalid_code="trusted_runtime_archive_invalid",
                changed_code="trusted_runtime_archive_changed",
                allowed_owners=frozenset({os.getuid()}),  # windows-footgun: ok
            )
            if version != f"{_GCLOUD_SDK_VERSION}\n".encode("ascii"):
                raise OwnerLauncherError("trusted_gcloud_sdk_version_invalid")
            TrustedGcloudExecutable._capture_tree(extracted, scope="sdk")
            extracted_publication_tree = _capture_sdk_publication_tree(extracted)
            if os.path.lexists(intent_path):
                _validate_sdk_publication_intent(
                    intent_path,
                    destination=destination,
                    publication_tree=extracted_publication_tree,
                )
            else:
                intent_unsigned = {
                    "schema": TRUSTED_SDK_PUBLICATION_INTENT_SCHEMA,
                    "ok": True,
                    "state": "trusted_sdk_publication_prepared",
                    "publication_release_sha": release_sha,
                    "launcher_sha256": current_launcher_sha256,
                    "sdk_archive_url": _GCLOUD_SDK_ARCHIVE_URL,
                    "sdk_archive_bytes": _GCLOUD_SDK_ARCHIVE_BYTES,
                    "sdk_archive_sha256": _GCLOUD_SDK_ARCHIVE_SHA256,
                    "sdk_version": _GCLOUD_SDK_VERSION,
                    "sdk_root": destination,
                    "sdk_tree_entries": extracted_publication_tree[0],
                    "sdk_tree_bytes": extracted_publication_tree[1],
                    "sdk_tree_sha256": extracted_publication_tree[2],
                    "prepared_at_unix": created,
                }
                intent = {
                    **intent_unsigned,
                    "intent_sha256": _sha256(_canonical_bytes(intent_unsigned)),
                }
                try:
                    _write_owner_file_no_replace(
                        intent_path,
                        _canonical_bytes(intent) + b"\n",
                        parent=trusted_root,
                        temporary_prefix=".trusted-sdk-publication-intent",
                        exists_code="trusted_runtime_publication_intent_exists",
                        failed_code="trusted_runtime_publication_intent_failed",
                    )
                except OwnerLauncherError as exc:
                    if exc.code != "trusted_runtime_publication_intent_exists":
                        raise
            # Whether this process or a concurrent process published the
            # intent, re-read its durable exact authority before the SDK move.
            _validate_sdk_publication_intent(
                intent_path,
                destination=destination,
                publication_tree=extracted_publication_tree,
            )
            try:
                _rename_directory_no_replace(extracted, destination)
            except OwnerLauncherError as exc:
                if exc.code != "trusted_runtime_destination_exists":
                    raise
                winning_tree = _capture_sdk_publication_tree(destination)
                _validate_sdk_publication_intent(
                    intent_path,
                    destination=destination,
                    publication_tree=winning_tree,
                )
                if winning_tree != extracted_publication_tree:
                    raise OwnerLauncherError("trusted_runtime_destination_mismatch")
            _fsync_directory(
                trusted_root,
                error_code="trusted_runtime_publish_failed",
            )
            published_tree = _capture_sdk_publication_tree(destination)
            if published_tree != extracted_publication_tree:
                raise OwnerLauncherError("trusted_runtime_destination_mismatch")
            _validate_sdk_publication_intent(
                intent_path,
                destination=destination,
                publication_tree=published_tree,
            )
        finally:
            shutil.rmtree(stage_parent, ignore_errors=True)

    python, python_root, python_tree, python_version, dependencies = python_snapshot()
    if python_version != _TRUSTED_PYTHON_VERSION:
        raise OwnerLauncherError("trusted_python_version_invalid")
    sdk_tree = TrustedGcloudExecutable._capture_tree(destination, scope="sdk")
    publication_tree = _capture_sdk_publication_tree(destination)
    _intent_fingerprint, publication_intent = _validate_sdk_publication_intent(
        intent_path,
        destination=destination,
        publication_tree=publication_tree,
    )
    wrapper = _PinnedExecutablePath(
        os.path.join(destination, "bin", "gcloud"),
        invalid_code="trusted_gcloud_invalid",
        changed_code="trusted_gcloud_changed",
    )
    wrapper.absolute_path()
    unsigned = {
        "schema": TRUSTED_RUNTIME_BOOTSTRAP_RECEIPT_SCHEMA,
        "ok": True,
        "state": "trusted_runtime_ready",
        "release_sha": release_sha,
        "launcher_sha256": current_launcher_sha256,
        "sdk_archive_url": _GCLOUD_SDK_ARCHIVE_URL,
        "sdk_archive_bytes": _GCLOUD_SDK_ARCHIVE_BYTES,
        "sdk_archive_sha256": _GCLOUD_SDK_ARCHIVE_SHA256,
        "sdk_version": _GCLOUD_SDK_VERSION,
        "sdk_root": destination,
        "sdk_tree_entries": sdk_tree[0],
        "sdk_tree_bytes": sdk_tree[1],
        "sdk_tree_sha256": sdk_tree[2],
        "sdk_publication_release_sha": publication_intent["publication_release_sha"],
        "sdk_publication_intent_sha256": publication_intent["intent_sha256"],
        "sdk_publication_tree_entries": publication_tree[0],
        "sdk_publication_tree_bytes": publication_tree[1],
        "sdk_publication_tree_sha256": publication_tree[2],
        "python_root": python_root,
        "python_executable": python.absolute_path(),
        "python_version": python_version,
        "python_tree_entries": python_tree[0],
        "python_tree_bytes": python_tree[1],
        "python_tree_sha256": python_tree[2],
        "python_dependencies": list(dependencies),
        "created_at_unix": created,
    }
    receipt = {**unsigned, "receipt_sha256": _sha256(_canonical_bytes(unsigned))}
    try:
        _write_owner_file_no_replace(
            receipt_path,
            _canonical_bytes(receipt) + b"\n",
            parent=trusted_root,
            temporary_prefix=f".trusted-runtime-bootstrap-{release_sha}",
            exists_code="trusted_runtime_bootstrap_receipt_exists",
            failed_code="trusted_runtime_bootstrap_receipt_failed",
        )
    except OwnerLauncherError as exc:
        if exc.code != "trusted_runtime_bootstrap_receipt_exists":
            raise
        validate_runtime(release_sha)
        _fingerprint, payload = _read_pinned_regular_file(
            receipt_path,
            maximum=_GCLOUD_MAX_CONFIG_BYTES,
            unavailable_code="trusted_runtime_bootstrap_receipt_unavailable",
            invalid_code="trusted_runtime_bootstrap_receipt_invalid",
            changed_code="trusted_runtime_bootstrap_receipt_changed",
            allowed_owners=frozenset({os.getuid()}),  # windows-footgun: ok
        )
        return _decode_json_object(payload, maximum=_GCLOUD_MAX_CONFIG_BYTES)
    validate_runtime(release_sha)
    return receipt


class LocalLauncherProvenance:
    """Bind this local secret-bearing launcher to the exact release commit."""

    _RELATIVE_MODULE = "scripts/canary/full_canary_owner_launcher.py"
    _MAX_MODULE_BYTES = 2 * 1024 * 1024

    def __init__(
        self,
        *,
        module_path: str | os.PathLike[str] = __file__,
        git_executable: str = "/usr/bin/git",
        runner: SubprocessRunner = subprocess.run,
    ) -> None:
        self._module_path = Path(module_path).absolute()
        self._git_executable = git_executable
        self._runner = runner

    def _git_identity(self) -> tuple[Any, ...]:
        if not os.path.isabs(self._git_executable):
            raise OwnerLauncherError("trusted_git_invalid")
        try:
            metadata = os.stat(self._git_executable, follow_symlinks=False)
        except OSError:
            raise OwnerLauncherError("trusted_git_unavailable") from None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            or metadata.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) == 0
            or metadata.st_size <= 0
            or metadata.st_size > 64 * 1024 * 1024
        ):
            raise OwnerLauncherError("trusted_git_invalid")
        try:
            with open(self._git_executable, "rb") as stream:
                payload = stream.read(64 * 1024 * 1024 + 1)
        except OSError:
            raise OwnerLauncherError("trusted_git_unavailable") from None
        if len(payload) != metadata.st_size:
            raise OwnerLauncherError("trusted_git_changed")
        return (
            metadata.st_mode,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mtime_ns,
            metadata.st_size,
            _sha256(payload),
        )

    def _module_snapshot(self) -> tuple[tuple[Any, ...], bytes]:
        try:
            metadata = os.lstat(self._module_path)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size <= 0
                or metadata.st_size > self._MAX_MODULE_BYTES
            ):
                raise OwnerLauncherError("local_launcher_invalid")
            payload = self._module_path.read_bytes()
        except OwnerLauncherError:
            raise
        except OSError:
            raise OwnerLauncherError("local_launcher_unavailable") from None
        if len(payload) != metadata.st_size:
            raise OwnerLauncherError("local_launcher_changed")
        return (
            (
                metadata.st_mode,
                metadata.st_uid,
                metadata.st_gid,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mtime_ns,
                metadata.st_size,
                _sha256(payload),
            ),
            payload,
        )

    def _run_git(self, args: Sequence[str]) -> bytes:
        try:
            completed = self._runner(
                (self._git_executable, *args),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env={
                    "HOME": os.path.expanduser("~"),
                    "LANG": "C",
                    "LC_ALL": "C",
                    "PATH": _FIXED_OWNER_PATH,
                },
                timeout=20.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            raise OwnerLauncherError("local_launcher_git_failed") from None
        if completed.returncode != 0 or not isinstance(completed.stdout, bytes):
            raise OwnerLauncherError("local_launcher_untracked")
        if len(completed.stdout) > self._MAX_MODULE_BYTES + 1024:
            raise OwnerLauncherError("local_launcher_git_failed")
        return completed.stdout

    def __call__(self, release_sha: str) -> str:
        if not _RELEASE_SHA.fullmatch(release_sha):
            raise OwnerLauncherError("invalid_release_sha")
        repository = self._module_path.parents[2]
        expected_module = repository / self._RELATIVE_MODULE
        if self._module_path != expected_module:
            raise OwnerLauncherError("local_launcher_path_invalid")
        git_before = self._git_identity()
        module_before, payload = self._module_snapshot()
        head = self._run_git(("-C", str(repository), "rev-parse", "--verify", "HEAD"))
        try:
            head_sha = head.decode("ascii", errors="strict").strip()
        except UnicodeError:
            raise OwnerLauncherError("local_launcher_git_failed") from None
        if head_sha != release_sha:
            raise OwnerLauncherError("local_launcher_release_mismatch")
        committed = self._run_git((
            "-C",
            str(repository),
            "show",
            f"{release_sha}:{self._RELATIVE_MODULE}",
        ))
        if committed != payload:
            raise OwnerLauncherError("local_launcher_dirty")
        module_after, _ = self._module_snapshot()
        if module_after != module_before or self._git_identity() != git_before:
            raise OwnerLauncherError("local_launcher_changed")
        return _sha256(payload)


def require_local_launcher_provenance(release_sha: str) -> str:
    return LocalLauncherProvenance()(release_sha)


def _validate_owner_interpreter_invocation(python_path: str) -> None:
    try:
        executable = os.path.realpath(sys.executable, strict=True)
        expected = os.path.realpath(python_path, strict=True)
        module_path = os.path.abspath(__file__)
        module_metadata = os.lstat(module_path)
    except OSError:
        raise OwnerLauncherError("trusted_owner_interpreter_unavailable") from None
    flags = sys.flags
    if (
        executable != expected
        or not os.path.isabs(__file__)
        or os.path.realpath(module_path) != module_path
        or not stat.S_ISREG(module_metadata.st_mode)
        or stat.S_ISLNK(module_metadata.st_mode)
        or module_metadata.st_uid != os.getuid()  # windows-footgun: ok
        or flags.isolated != 1
        or flags.no_site != 1
        or flags.dont_write_bytecode != 1
        or flags.no_user_site != 1
        or flags.ignore_environment != 1
        or getattr(flags, "safe_path", False) is not True
        or sys.pycache_prefix != "/var/empty/muncho-canary"
    ):
        raise OwnerLauncherError("trusted_owner_interpreter_invalid")


def require_trusted_bootstrap_interpreter() -> None:
    python, _root, _tree, _version, _dependencies = _fixed_python_runtime_snapshot()
    _validate_owner_interpreter_invocation(python.absolute_path())


def require_trusted_owner_runtime(release_sha: str) -> TrustedGcloudExecutable:
    runtime = TrustedGcloudExecutable(release_sha=release_sha)
    command_prefix = runtime.trusted_command_prefix()
    _validate_owner_interpreter_invocation(command_prefix[0])
    return runtime


def require_owner_runtime_and_launcher_provenance(release_sha: str) -> None:
    require_trusted_owner_runtime(release_sha)
    require_local_launcher_provenance(release_sha)


class GcloudOwnerAccessToken:
    """Use exactly one already-active human gcloud identity; never a key file."""

    _ACCOUNT = re.compile(
        r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
        r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$"
    )

    def __init__(
        self,
        *,
        gcloud_executable: StableExecutable | None = None,
        gcloud_configuration: StableGcloudConfiguration | None = None,
        runner: SubprocessRunner = subprocess.run,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._runner = runner
        self._gcloud_executable = gcloud_executable or TrustedGcloudExecutable()
        self._gcloud_configuration = gcloud_configuration or PinnedGcloudConfiguration()
        self._timeout_seconds = timeout_seconds
        self.owner_subject_sha256: str | None = None
        self._pinned_account: str | None = None
        self._approved = False

    @property
    def gcloud_configuration(self) -> StableGcloudConfiguration:
        return self._gcloud_configuration

    def _run(self, arguments: Sequence[str], *, code: str) -> bytes:
        if not arguments or any(
            not isinstance(item, str) or not item for item in arguments
        ):
            raise OwnerLauncherError("invalid_gcloud_argv")
        command_prefix = self._gcloud_executable.trusted_command_prefix()
        if (
            len(command_prefix) != len(_GCLOUD_PYTHON_ISOLATION_ARGS) + 2
            or command_prefix[1:-1] != _GCLOUD_PYTHON_ISOLATION_ARGS
        ):
            raise OwnerLauncherError("invalid_gcloud_command_prefix")
        environment = _owner_gcloud_environment(
            self._gcloud_configuration,
            command_prefix[0],
        )
        argv = (*command_prefix, *arguments)
        try:
            try:
                completed = self._runner(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    env=dict(environment),
                    timeout=self._timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                raise OwnerLauncherError(code) from None
        finally:
            # A same-UID mutation during gcloud execution is still a failed
            # provenance check, even if the subprocess itself returned zero.
            self._gcloud_executable.trusted_command_prefix()
            self._gcloud_configuration.assert_stable()
        output = completed.stdout
        if (
            completed.returncode != 0
            or not isinstance(output, bytes)
            or not output
            or len(output) > 32 * 1024
        ):
            raise OwnerLauncherError(code)
        return output

    def _active_account(self) -> str:
        accounts_raw = self._run(
            (
                "auth",
                "list",
                "--filter=status:ACTIVE",
                "--format=value(account)",
                "--limit=2",
                "--quiet",
            ),
            code="active_owner_identity_unavailable",
        )
        try:
            accounts = [
                value.strip()
                for value in accounts_raw.decode("utf-8", errors="strict").splitlines()
                if value.strip()
            ]
        except UnicodeError:
            raise OwnerLauncherError("active_owner_identity_unavailable") from None
        if (
            len(accounts) != 1
            or self._ACCOUNT.fullmatch(accounts[0]) is None
            or accounts[0].casefold().endswith(".gserviceaccount.com")
            or accounts[0] != self._gcloud_configuration.account
        ):
            raise OwnerLauncherError("active_owner_identity_unavailable")
        return accounts[0]

    def run_canary_iam_read_only_json(self, argv: Sequence[str]) -> Any:
        """Run only the exact list/describe inventory used by IAM collectors."""

        logical = tuple(argv)
        if logical not in _canary_iam_read_only_inventory():
            raise OwnerLauncherError("canary_iam_command_forbidden")
        account = self.account_for_read_only_preflight()
        command_prefix = self._gcloud_executable.trusted_command_prefix()
        if (
            len(command_prefix) != len(_GCLOUD_PYTHON_ISOLATION_ARGS) + 2
            or command_prefix[1:-1] != _GCLOUD_PYTHON_ISOLATION_ARGS
        ):
            raise OwnerLauncherError("invalid_gcloud_command_prefix")
        arguments = list(logical[1:])
        if logical[1:3] != ("auth", "list"):
            arguments.append(f"--account={account}")
        arguments.append("--quiet")
        environment = _owner_gcloud_environment(
            self._gcloud_configuration,
            command_prefix[0],
        )
        try:
            try:
                completed = self._runner(
                    (*command_prefix, *arguments),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    env=dict(environment),
                    timeout=60.0,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                raise OwnerLauncherError("canary_iam_collection_failed") from None
        finally:
            self._gcloud_executable.trusted_command_prefix()
            self._gcloud_configuration.assert_stable()
        output = completed.stdout
        if (
            completed.returncode != 0
            or not isinstance(output, bytes)
            or not output
            or len(output) > _CANARY_IAM_JSON_MAX_BYTES
        ):
            raise OwnerLauncherError("canary_iam_collection_failed")
        try:
            value = _decode_json_value(
                output,
                maximum=_CANARY_IAM_JSON_MAX_BYTES,
            )
        except OwnerLauncherError:
            raise OwnerLauncherError("canary_iam_collection_invalid") from None
        self.require_stable()
        return value

    def bind_approved_subject(self, expected_sha256: str) -> None:
        expected = _require_sha256(expected_sha256, "approved_owner_identity_invalid")
        account = self._active_account()
        observed = _sha256(account.encode("utf-8"))
        if observed != expected:
            raise OwnerLauncherError("approved_owner_identity_mismatch")
        if self._pinned_account is not None and self._pinned_account != account:
            raise OwnerLauncherError("approved_owner_identity_changed")
        self._pinned_account = account
        self.owner_subject_sha256 = observed
        self._approved = True

    def account_for_read_only_preflight(self) -> str:
        if self._pinned_account is None:
            account = self._active_account()
            self._pinned_account = account
            self.owner_subject_sha256 = _sha256(account.encode("utf-8"))
        return self._pinned_account

    @property
    def approved_account(self) -> str:
        if not self._approved or self._pinned_account is None:
            raise OwnerLauncherError("approved_owner_identity_unbound")
        return self._pinned_account

    def require_stable(self) -> None:
        if (
            self._pinned_account is None
            or self.owner_subject_sha256 is None
        ):
            raise OwnerLauncherError("approved_owner_identity_unbound")
        account = self._active_account()
        if (
            account != self._pinned_account
            or _sha256(account.encode("utf-8")) != self.owner_subject_sha256
        ):
            raise OwnerLauncherError("approved_owner_identity_changed")

    def __call__(self) -> str:
        if not self._approved or self._pinned_account is None:
            raise OwnerLauncherError("approved_owner_identity_unbound")
        token_raw = self._run(
            (
                "auth",
                "print-access-token",
                f"--account={self._pinned_account}",
                "--quiet",
            ),
            code="owner_access_token_failed",
        )
        try:
            token = token_raw.decode("ascii", errors="strict").strip()
        except UnicodeError:
            raise OwnerLauncherError("owner_access_token_failed") from None
        if (
            not token
            or len(token) > 16 * 1024
            or any(
                ord(character) < 0x21 or ord(character) > 0x7E for character in token
            )
        ):
            raise OwnerLauncherError("owner_access_token_failed")
        return token


class GoogleRestClient:
    """Small fixed-host JSON REST client with injected owner bearer tokens."""

    def __init__(
        self,
        token_provider: AccessTokenProvider,
        *,
        requester: HttpRequester = _default_http_request,
        timeout_seconds: float = _HTTP_TIMEOUT_SECONDS,
    ) -> None:
        self._token_provider = token_provider
        self._requester = requester
        self._timeout_seconds = timeout_seconds

    def request_json(
        self,
        method: str,
        url: str,
        *,
        body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        _reject_custom_ca_environment()
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.hostname != "sqladmin.googleapis.com"
            or parsed.fragment
        ):
            raise OwnerLauncherError("forbidden_google_api_url")
        try:
            token = self._token_provider()
        except OwnerLauncherError:
            raise
        except Exception:
            raise OwnerLauncherError("owner_access_token_failed") from None
        if (
            not isinstance(token, str)
            or not token
            or len(token) > 16 * 1024
            or any(
                ord(character) < 0x21 or ord(character) > 0x7E for character in token
            )
        ):
            raise OwnerLauncherError("invalid_owner_access_token")
        encoded = None if body is None else _canonical_bytes(body)
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if encoded is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        try:
            response = self._requester(
                method,
                url,
                headers,
                encoded,
                self._timeout_seconds,
            )
        except OwnerLauncherError:
            raise
        except Exception:
            raise OwnerLauncherError("google_api_unavailable") from None
        if response.status < 200 or response.status >= 300:
            if response.status not in {400, 401, 403, 404, 409}:
                # The request may have reached Cloud SQL even though a
                # transient HTTP response was returned. Never treat these
                # classes as proof that a user mutation did not commit.
                raise OwnerLauncherError("google_api_ambiguous_status")
            raise OwnerLauncherError("google_api_rejected")
        return _decode_json_object(response.body, maximum=_HTTP_RESPONSE_MAX_BYTES)


class CloudSqlTemporaryAdmin:
    """Cloud SQL Admin REST lifecycle for one approval-derived BUILT_IN user."""

    _BASE = f"https://sqladmin.googleapis.com/sql/v1beta4/projects/{PROJECT}"
    _SETTLE_ATTEMPTS = 5

    def __init__(
        self,
        client: GoogleRestClient,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        operation_timeout_seconds: float = _OPERATION_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._operation_timeout_seconds = operation_timeout_seconds
        self._mutation_operation_baseline: frozenset[str] | None = None
        self._mutation_relevant_baseline: (
            Mapping[str, tuple[str, str, str, bool]] | None
        ) = None
        self._mutation_known_operations: set[str] = set()
        self._mutation_authority_known_operations: set[str] = set()
        self._confirmed_authority_operation_name: str | None = None
        self._confirmed_authority_operation_type: str | None = None
        self._expected_owner_subject_sha256: str | None = None
        self._expected_mutation_context_sha256: str | None = None
        self._mutation_ambiguous = False
        self._mutation_ambiguity_observed = False
        self._reconciliation_proven = False
        self._reconciliation_evidence_sha256: str | None = None
        self._reconciliation_receipt: Mapping[str, Any] | None = None
        self._reconciliation_quiet_window_seconds: float | None = None
        self._reconciliation_response_known_candidate_observed: bool | None = None
        self._reconciliation_post_baseline_authority_operation_count: int | None = None

    @property
    def _users_url(self) -> str:
        return f"{self._BASE}/instances/{SQL_INSTANCE}/users"

    def _operations_url(self, page_token: str | None = None) -> str:
        query_values: dict[str, object] = {
            "instance": SQL_INSTANCE,
            "maxResults": 100,
        }
        if page_token is not None:
            query_values["pageToken"] = page_token
        query = urllib.parse.urlencode(query_values)
        return f"{self._BASE}/operations?{query}"

    def _instance_operations(self) -> Mapping[str, tuple[str, str, str, bool]]:
        operations: dict[str, tuple[str, str, str, bool]] = {}
        page_token: str | None = None
        visited_tokens: set[str] = set()
        first_page: Mapping[str, Any] | None = None
        for page in range(100):
            payload = self._client.request_json("GET", self._operations_url(page_token))
            if page == 0:
                first_page = payload
            if payload.get("kind") != "sql#operationsList" or any(
                payload.get(name) not in (None, [], {})
                for name in ("warning", "warnings")
            ):
                raise OwnerLauncherError("cloud_sql_operations_evidence_incomplete")
            items = payload.get("items", [])
            if not isinstance(items, list):
                raise OwnerLauncherError("invalid_cloud_sql_operations")
            for item in items:
                if not isinstance(item, Mapping):
                    raise OwnerLauncherError("invalid_cloud_sql_operations")
                name = item.get("name")
                operation_type = item.get("operationType")
                status = item.get("status")
                if (
                    not isinstance(name, str)
                    or _OPERATION_NAME.fullmatch(name) is None
                    or not isinstance(operation_type, str)
                    or _OPERATION_TYPE.fullmatch(operation_type) is None
                    or status not in {"PENDING", "RUNNING", "DONE"}
                    or name in operations
                ):
                    raise OwnerLauncherError("invalid_cloud_sql_operations")
                if operation_type in _CLOUD_SQL_USER_OPERATION_TYPES:
                    error = item.get("error")
                    actor = item.get("user")
                    expected_self_link = (
                        f"{self._BASE}/operations/{urllib.parse.quote(name, safe='')}"
                    )
                    expected_target_link = f"{self._BASE}/instances/{SQL_INSTANCE}"
                    if (
                        item.get("kind") != "sql#operation"
                        or item.get("targetProject") != PROJECT
                        or item.get("targetId") != SQL_INSTANCE
                        or item.get("selfLink") != expected_self_link
                        or item.get("targetLink") != expected_target_link
                        or not isinstance(actor, str)
                        or not actor
                        or len(actor) > 320
                        or any(
                            ord(character) < 0x21 or ord(character) > 0x7E
                            for character in actor
                        )
                        or item.get("apiWarning") is not None
                        or (
                            error is not None
                            and (
                                status != "DONE"
                                or not isinstance(error, Mapping)
                                or error.get("kind") != "sql#operationErrors"
                                or not isinstance(error.get("errors"), list)
                                or not error["errors"]
                                or any(
                                    not isinstance(entry, Mapping)
                                    or entry.get("kind") != "sql#operationError"
                                    or not isinstance(entry.get("code"), str)
                                    or not entry["code"]
                                    for entry in error["errors"]
                                )
                            )
                        )
                    ):
                        raise OwnerLauncherError("invalid_cloud_sql_operations")
                    actor_sha256 = _sha256(actor.encode("ascii"))
                    # A non-terminal operation has not succeeded yet, even
                    # when Cloud SQL has not attached an error object.
                    operation_succeeded = status == "DONE" and error is None
                else:
                    actor_sha256 = "0" * 64
                    operation_succeeded = status == "DONE"
                operations[name] = (
                    operation_type,
                    status,
                    actor_sha256,
                    operation_succeeded,
                )
            next_token = payload.get("nextPageToken")
            if next_token is None:
                if first_page is None:
                    raise OwnerLauncherError("cloud_sql_operations_evidence_incomplete")
                refetched_first_page = self._client.request_json(
                    "GET", self._operations_url()
                )
                if _canonical_bytes(refetched_first_page) != _canonical_bytes(
                    first_page
                ):
                    raise OwnerLauncherError("cloud_sql_operations_evidence_incomplete")
                return operations
            if (
                not isinstance(next_token, str)
                or not next_token
                or len(next_token) > 4096
                or next_token in visited_tokens
                or next_token == page_token
            ):
                raise OwnerLauncherError("cloud_sql_operations_evidence_incomplete")
            visited_tokens.add(next_token)
            page_token = next_token
        raise OwnerLauncherError("cloud_sql_operations_evidence_incomplete")

    def _stable_instance_operations(
        self,
    ) -> Mapping[str, tuple[str, str, str, bool]]:
        first = self._instance_operations()
        second = self._instance_operations()
        if first != second:
            raise OwnerLauncherError("cloud_sql_operations_evidence_incomplete")
        return second

    @staticmethod
    def _relevant_user_operations(
        operations: Mapping[str, tuple[str, str, str, bool]],
    ) -> Mapping[str, tuple[str, str, str, bool]]:
        return {
            name: value
            for name, value in operations.items()
            if value[0] in _CLOUD_SQL_USER_OPERATION_TYPES
        }

    def begin_mutation_observation(
        self,
        *,
        expected_owner_subject_sha256: str | None = None,
        expected_mutation_context_sha256: str | None = None,
    ) -> None:
        """Capture the complete operation namespace before a user mutation."""

        operations = self._stable_instance_operations()
        relevant = self._relevant_user_operations(operations)
        if expected_owner_subject_sha256 is not None:
            _require_sha256(
                expected_owner_subject_sha256,
                "invalid_owner_subject_sha256",
            )
        if expected_mutation_context_sha256 is not None:
            _require_sha256(
                expected_mutation_context_sha256,
                "invalid_mutation_context_sha256",
            )
        if any(value[1] != "DONE" for value in relevant.values()):
            raise OwnerLauncherError("cloud_sql_user_operations_not_quiescent")
        self._expected_owner_subject_sha256 = expected_owner_subject_sha256
        self._expected_mutation_context_sha256 = expected_mutation_context_sha256
        self._mutation_operation_baseline = frozenset(operations)
        self._mutation_relevant_baseline = dict(relevant)
        self._mutation_known_operations.clear()
        self._mutation_authority_known_operations.clear()
        self._confirmed_authority_operation_name = None
        self._confirmed_authority_operation_type = None
        self._mutation_ambiguous = False
        self._mutation_ambiguity_observed = False
        self._reconciliation_proven = False
        self._reconciliation_evidence_sha256 = None
        self._reconciliation_receipt = None
        self._reconciliation_quiet_window_seconds = None
        self._reconciliation_response_known_candidate_observed = None
        self._reconciliation_post_baseline_authority_operation_count = None

    def mutation_reconciliation_required(self) -> bool:
        return self._mutation_ambiguous

    def reconciliation_evidence(self) -> Mapping[str, Any]:
        return {
            "mutation_ambiguity_observed": self._mutation_ambiguity_observed,
            "reconciliation_proven": self._reconciliation_proven,
            "reconciliation_evidence_sha256": (self._reconciliation_evidence_sha256),
            "quiet_window_seconds": self._reconciliation_quiet_window_seconds,
            "response_known_candidate_observed": (
                self._reconciliation_response_known_candidate_observed
            ),
            "post_baseline_authority_operation_count": (
                self._reconciliation_post_baseline_authority_operation_count
            ),
        }

    def reconciliation_receipt(self) -> Mapping[str, Any]:
        """Return the complete, secret-free DELETE/absence operation receipt."""

        receipt = self._reconciliation_receipt
        if not self._reconciliation_proven or receipt is None:
            raise OwnerLauncherError("cloud_sql_reconciliation_evidence_unconfirmed")
        value = copy.deepcopy(dict(receipt))
        digest = value.get("evidence_sha256")
        unsigned = {
            name: item for name, item in value.items() if name != "evidence_sha256"
        }
        if (
            not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or _sha256(_canonical_bytes(unsigned)) != digest
        ):
            raise OwnerLauncherError("cloud_sql_reconciliation_evidence_invalid")
        return value

    def _ensure_mutation_observation(self) -> None:
        if self._mutation_operation_baseline is None:
            self.begin_mutation_observation()

    def _record_operation_name(
        self,
        operation: Mapping[str, Any],
        *,
        authority_candidate: bool = False,
    ) -> str:
        operation_name = operation.get("name")
        if not isinstance(operation_name, str) or not _OPERATION_NAME.fullmatch(
            operation_name
        ):
            raise OwnerLauncherError("invalid_cloud_sql_operation")
        if (
            self._mutation_operation_baseline is not None
            and operation_name in self._mutation_operation_baseline
        ):
            raise OwnerLauncherError("invalid_cloud_sql_operation")
        self._mutation_known_operations.add(operation_name)
        if authority_candidate:
            self._mutation_authority_known_operations.add(operation_name)
        return operation_name

    def _confirm_direct_mutation_operation(
        self,
        operation_name: str,
        *,
        expected_operation_type: str,
    ) -> None:
        baseline = self._mutation_operation_baseline
        baseline_relevant = self._mutation_relevant_baseline
        if baseline is None or baseline_relevant is None:
            raise OwnerLauncherError("cloud_sql_mutation_evidence_unconfirmed")
        operations = self._relevant_user_operations(self._stable_instance_operations())
        observed_baseline = {
            name: operations[name] for name in baseline_relevant if name in operations
        }
        delta = {
            name: value for name, value in operations.items() if name not in baseline
        }
        expected = delta.get(operation_name)
        if (
            observed_baseline != baseline_relevant
            or set(delta) != {operation_name}
            or expected is None
            or expected[0] != expected_operation_type
            or expected[1] != "DONE"
            or expected[3] is not True
            or (
                self._expected_owner_subject_sha256 is not None
                and expected[2] != self._expected_owner_subject_sha256
            )
        ):
            raise OwnerLauncherError("cloud_sql_mutation_evidence_unconfirmed")
        self._confirmed_authority_operation_name = operation_name
        self._confirmed_authority_operation_type = expected_operation_type

    def require_current_authority(self, username: str) -> None:
        """Revalidate exact mutation authority at the first-byte boundary.

        The canary SQL instance is dedicated to this flow and has no approved
        concurrent user mutator. A page-fenced exact user read is enclosed by
        two identical stable operation-ledger snapshots, so the proof ends on
        an operation fence. The dedicated-instance invariant is the boundary
        for a mutation accepted after that final fence.
        """

        try:
            if not self._valid_target_username(username):
                raise OwnerLauncherError("invalid_admin_username")
            baseline = self._mutation_operation_baseline
            baseline_relevant = self._mutation_relevant_baseline
            operation_name = self._confirmed_authority_operation_name
            operation_type = self._confirmed_authority_operation_type
            if (
                self._mutation_ambiguous
                or baseline is None
                or baseline_relevant is None
                or operation_name is None
                or operation_type not in {"CREATE_USER", "UPDATE_USER"}
            ):
                raise OwnerLauncherError("cloud_sql_mutation_evidence_unconfirmed")

            def validate_authority_snapshot(
                operations: Mapping[str, tuple[str, str, str, bool]],
            ) -> Mapping[str, tuple[str, str, str, bool]]:
                relevant = self._relevant_user_operations(operations)
                observed_baseline = {
                    name: relevant[name]
                    for name in baseline_relevant
                    if name in relevant
                }
                delta = {
                    name: value
                    for name, value in relevant.items()
                    if name not in baseline
                }
                expected = delta.get(operation_name)
                if (
                    observed_baseline != baseline_relevant
                    or set(delta) != {operation_name}
                    or expected is None
                    or expected[0] != operation_type
                    or expected[1] != "DONE"
                    or expected[3] is not True
                    or (
                        self._expected_owner_subject_sha256 is not None
                        and expected[2] != self._expected_owner_subject_sha256
                    )
                ):
                    raise OwnerLauncherError("cloud_sql_mutation_evidence_unconfirmed")
                return relevant

            first = validate_authority_snapshot(self._instance_operations())
            users = self._user_names(exact_admin_username=username)
            second = validate_authority_snapshot(self._instance_operations())
            if first != second or username not in users:
                raise OwnerLauncherError("cloud_sql_mutation_evidence_unconfirmed")
        except OwnerLauncherError:
            self._mutation_ambiguous = True
            self._mutation_ambiguity_observed = True
            raise OwnerLauncherError(
                "cloud_sql_mutation_evidence_unconfirmed"
            ) from None

    def temporary_admin_authority_receipt(
        self,
        username: str,
    ) -> Mapping[str, Any]:
        """Seal current owner authority before opening SQL or issuing DELETE."""

        self.require_current_authority(username)
        baseline = self._mutation_operation_baseline
        baseline_relevant = self._mutation_relevant_baseline
        operation_name = self._confirmed_authority_operation_name
        operation_type = self._confirmed_authority_operation_type
        if (
            baseline is None
            or baseline_relevant is None
            or operation_name is None
            or operation_type not in {"CREATE_USER", "UPDATE_USER"}
        ):
            raise OwnerLauncherError("cloud_sql_mutation_evidence_unconfirmed")
        operations = self._relevant_user_operations(
            self._stable_instance_operations()
        )
        operation = operations.get(operation_name)
        if (
            operation is None
            or operation[0] != operation_type
            or operation[1] != "DONE"
            or operation[3] is not True
        ):
            raise OwnerLauncherError("cloud_sql_mutation_evidence_unconfirmed")
        receipt = {
            "schema": TEMPORARY_ADMIN_AUTHORITY_RECEIPT_SCHEMA,
            "project": PROJECT,
            "instance": SQL_INSTANCE,
            "username_sha256": _sha256(username.encode("utf-8")),
            "host": "",
            "type": "BUILT_IN",
            "user_present": True,
            "owner_subject_sha256": self._expected_owner_subject_sha256,
            "mutation_context_sha256": self._expected_mutation_context_sha256,
            "baseline_operation_names": sorted(baseline),
            "baseline_user_operations": [
                [name, *value]
                for name, value in sorted(baseline_relevant.items())
            ],
            "authority_operation": [operation_name, *operation],
        }
        self.require_current_authority(username)
        return {
            **receipt,
            "receipt_sha256": _sha256(_canonical_bytes(receipt)),
        }

    def _wait_operation(
        self,
        operation_name: object,
        *,
        expected_operation_type: str,
        deadline: float | None = None,
    ) -> None:
        if not isinstance(operation_name, str) or not _OPERATION_NAME.fullmatch(
            operation_name
        ):
            raise OwnerLauncherError("invalid_cloud_sql_operation")
        if deadline is None:
            deadline = self._monotonic() + self._operation_timeout_seconds
        url = f"{self._BASE}/operations/{urllib.parse.quote(operation_name, safe='')}"
        target_link = f"{self._BASE}/instances/{SQL_INSTANCE}"
        while True:
            operation = self._client.request_json("GET", url)
            actor = operation.get("user")
            if (
                operation.get("name") != operation_name
                or operation.get("kind") != "sql#operation"
                or operation.get("targetProject") != PROJECT
                or operation.get("targetId") != SQL_INSTANCE
                or operation.get("selfLink") != url
                or operation.get("targetLink") != target_link
                or operation.get("operationType") != expected_operation_type
                or not isinstance(actor, str)
                or not actor
                or len(actor) > 320
                or any(
                    ord(character) < 0x21 or ord(character) > 0x7E
                    for character in actor
                )
                or (
                    self._expected_owner_subject_sha256 is not None
                    and _sha256(actor.encode("ascii"))
                    != self._expected_owner_subject_sha256
                )
                or operation.get("apiWarning") is not None
            ):
                raise OwnerLauncherError("invalid_cloud_sql_operation")
            status = operation.get("status")
            if status == "DONE":
                error = operation.get("error")
                if error is not None:
                    if (
                        not isinstance(error, Mapping)
                        or error.get("kind") != "sql#operationErrors"
                        or not isinstance(error.get("errors"), list)
                        or not error["errors"]
                        or any(
                            not isinstance(entry, Mapping)
                            or entry.get("kind") != "sql#operationError"
                            or not isinstance(entry.get("code"), str)
                            or not entry["code"]
                            for entry in error["errors"]
                        )
                    ):
                        raise OwnerLauncherError("invalid_cloud_sql_operation")
                    raise OwnerLauncherError("cloud_sql_operation_failed")
                return
            if status not in {"PENDING", "RUNNING"}:
                raise OwnerLauncherError("invalid_cloud_sql_operation")
            if self._monotonic() >= deadline:
                raise OwnerLauncherError("cloud_sql_operation_timeout")
            self._sleeper(1.0)

    def _valid_target_username(self, username: object) -> bool:
        return (
            isinstance(username, str)
            and _ADMIN_USERNAME.fullmatch(username) is not None
        )

    def _validate_exact_user_resource(
        self,
        item: Mapping[str, Any],
        *,
        username: str,
    ) -> Mapping[str, Any]:
        if (
            item.get("kind") != "sql#user"
            or item.get("name") != username
            or item.get("project") != PROJECT
            or item.get("instance") != SQL_INSTANCE
            or ("type" in item and item.get("type") != "BUILT_IN")
        ):
            raise OwnerLauncherError("invalid_cloud_sql_users")
        normalized = dict(item)
        normalized["type"] = "BUILT_IN"
        return normalized

    def _create_user_body(self, username: str, password: str) -> Mapping[str, Any]:
        return {
            "instance": SQL_INSTANCE,
            "name": username,
            "password": password,
            "project": PROJECT,
            "type": "BUILT_IN",
        }

    def _update_user_body(self, username: str, password: str) -> Mapping[str, Any]:
        return {
            "name": username,
            "password": password,
            "type": "BUILT_IN",
        }

    def _update_user_query_values(self, username: str) -> Mapping[str, object]:
        return {"host": "", "name": username}

    def _users_and_exact_resource(
        self,
        *,
        exact_admin_username: str | None = None,
    ) -> tuple[frozenset[str], Mapping[str, Any] | None]:
        exact_resource: Mapping[str, Any] | None = None
        names: set[str] = set()
        page_token: str | None = None
        visited_tokens: set[str] = set()
        pages = 0
        first_page: Mapping[str, Any] | None = None
        while True:
            pages += 1
            if pages > 100:
                raise OwnerLauncherError("cloud_sql_users_pagination_failed")
            query = ""
            if page_token is not None:
                query = "?" + urllib.parse.urlencode({"pageToken": page_token})
            payload = self._client.request_json("GET", self._users_url + query)
            if pages == 1:
                first_page = payload
            if payload.get("kind") != "sql#usersList" or any(
                payload.get(name) not in (None, [], {})
                for name in ("warning", "warnings")
            ):
                raise OwnerLauncherError("invalid_cloud_sql_users")
            items = payload.get("items", [])
            if not isinstance(items, list):
                raise OwnerLauncherError("invalid_cloud_sql_users")
            for item in items:
                if not isinstance(item, Mapping):
                    raise OwnerLauncherError("invalid_cloud_sql_users")
                name = item.get("name")
                if not isinstance(name, str) or name in names:
                    raise OwnerLauncherError("invalid_cloud_sql_users")
                if name == exact_admin_username:
                    exact_resource = self._validate_exact_user_resource(
                        item,
                        username=name,
                    )
                names.add(name)
            next_token = payload.get("nextPageToken")
            if next_token is None:
                if first_page is None:
                    raise OwnerLauncherError("invalid_cloud_sql_users")
                refetched_first_page = self._client.request_json("GET", self._users_url)
                if _canonical_bytes(refetched_first_page) != _canonical_bytes(
                    first_page
                ):
                    raise OwnerLauncherError("invalid_cloud_sql_users")
                return frozenset(names), exact_resource
            if (
                not isinstance(next_token, str)
                or not next_token
                or len(next_token) > 4096
                or next_token in visited_tokens
                or next_token == page_token
            ):
                raise OwnerLauncherError("invalid_cloud_sql_users")
            visited_tokens.add(next_token)
            page_token = next_token

    def _user_names(self, *, exact_admin_username: str | None = None) -> frozenset[str]:
        names, _resource = self._users_and_exact_resource(
            exact_admin_username=exact_admin_username
        )
        return names

    def require_absent(self, username: str) -> None:
        if not self._valid_target_username(username):
            raise OwnerLauncherError("invalid_admin_username")
        if username in self._user_names(exact_admin_username=username):
            raise OwnerLauncherError("temporary_admin_already_exists")

    def _settle_user_presence(self, username: str, *, expected: bool) -> bool:
        """Boundedly prove user presence/absence after an ambiguous operation."""

        for attempt in range(self._SETTLE_ATTEMPTS):
            try:
                if (
                    username in self._user_names(exact_admin_username=username)
                ) is expected:
                    return True
            except OwnerLauncherError:
                pass
            if attempt + 1 < self._SETTLE_ATTEMPTS:
                self._sleeper(1.0)
        return False

    def create(self, username: str, password: str) -> None:
        if not self._valid_target_username(username):
            raise OwnerLauncherError("invalid_admin_username")
        encoded_password = password.encode("utf-8")
        if (
            not _ADMIN_PASSWORD_MIN_UTF8
            <= len(encoded_password)
            <= _ADMIN_PASSWORD_MAX_UTF8
            or password != password.strip()
            or any(
                ord(character) < 0x20 or ord(character) == 0x7F
                for character in password
            )
        ):
            raise OwnerLauncherError("invalid_admin_password")
        self._ensure_mutation_observation()
        self._mutation_ambiguous = True
        self._mutation_ambiguity_observed = True
        try:
            operation = self._client.request_json(
                "POST",
                self._users_url,
                body=self._create_user_body(username, password),
            )
        except OwnerLauncherError as exc:
            if exc.code == "google_api_rejected":
                self._mutation_ambiguous = False
                self._mutation_ambiguity_observed = False
                raise CloudSqlCreateNotCommitted(exc.code) from None
            # Only lost/invalid response classes are ambiguous about whether
            # Cloud SQL committed the POST. An explicit API or operation
            # rejection must never become success merely because a same-named
            # account appeared concurrently. After an ambiguous response,
            # confirm a password reset so authority over the account is known.
            if exc.code not in _AMBIGUOUS_CLOUD_SQL_CREATE_ERRORS:
                self._mutation_ambiguous = False
                self._mutation_ambiguity_observed = False
                raise
            raise
        try:
            operation_name = self._record_operation_name(
                operation,
                authority_candidate=True,
            )
            self._wait_operation(
                operation_name,
                expected_operation_type="CREATE_USER",
            )
        except OwnerLauncherError as exc:
            if exc.code not in _AMBIGUOUS_CLOUD_SQL_CREATE_ERRORS:
                self._mutation_ambiguous = False
                self._mutation_ambiguity_observed = False
                raise
            raise
        if not self._settle_user_presence(username, expected=True):
            raise OwnerLauncherError("temporary_admin_create_unconfirmed")
        self._confirm_direct_mutation_operation(
            operation_name,
            expected_operation_type="CREATE_USER",
        )
        self._mutation_ambiguous = False
        self._mutation_ambiguity_observed = False

    def rotate_existing(self, username: str, password: str) -> None:
        """Reset one exact stale BUILT_IN admin to a fresh in-memory secret."""

        self._rotate_existing(
            username,
            password,
        )

    def _rotate_existing(
        self,
        username: str,
        password: str,
    ) -> None:

        if not self._valid_target_username(username):
            raise OwnerLauncherError("invalid_admin_username")
        encoded_password = password.encode("utf-8")
        if (
            not _ADMIN_PASSWORD_MIN_UTF8
            <= len(encoded_password)
            <= _ADMIN_PASSWORD_MAX_UTF8
            or password != password.strip()
            or any(
                ord(character) < 0x20 or ord(character) == 0x7F
                for character in password
            )
        ):
            raise OwnerLauncherError("invalid_admin_password")
        self._ensure_mutation_observation()
        if not self._settle_user_presence(username, expected=True):
            raise OwnerLauncherError("temporary_admin_recovery_user_missing")
        query = urllib.parse.urlencode(
            self._update_user_query_values(username),
            doseq=True,
        )
        self._mutation_ambiguous = True
        self._mutation_ambiguity_observed = True
        try:
            operation = self._client.request_json(
                "PUT",
                f"{self._users_url}?{query}",
                body=self._update_user_body(username, password),
            )
            operation_name = self._record_operation_name(
                operation,
                authority_candidate=True,
            )
            self._wait_operation(
                operation_name,
                expected_operation_type="UPDATE_USER",
            )
            self._confirm_direct_mutation_operation(
                operation_name,
                expected_operation_type="UPDATE_USER",
            )
        except OwnerLauncherError as exc:
            if exc.code not in _AMBIGUOUS_CLOUD_SQL_CREATE_ERRORS:
                self._mutation_ambiguous = False
                self._mutation_ambiguity_observed = False
            raise
        self._mutation_ambiguous = False
        self._mutation_ambiguity_observed = False

    def create_or_rotate_recovery(self, username: str, password: str) -> None:
        """Provision the exact recovery admin for either valid stale-user state."""

        if not self._valid_target_username(username):
            raise OwnerLauncherError("invalid_admin_username")
        self._ensure_mutation_observation()
        present = username in self._user_names(exact_admin_username=username)
        if present:
            self.rotate_existing(username, password)
        else:
            self.create(username, password)

    @staticmethod
    def _track_operation_snapshot(
        relevant: Mapping[str, tuple[str, str, str, bool]],
        *,
        known_operations: dict[str, tuple[str, str, str, bool]],
        pending_seen: set[str],
        done_seen: set[str],
    ) -> set[str]:
        newly_seen: set[str] = set()
        for name, current in relevant.items():
            operation_type, status, actor_sha256, operation_succeeded = current
            previous = known_operations.get(name)
            if previous is None:
                newly_seen.add(name)
            else:
                (
                    previous_type,
                    previous_status,
                    previous_actor_sha256,
                    previous_succeeded,
                ) = previous
                valid_statuses = {
                    "PENDING": {"PENDING", "RUNNING", "DONE"},
                    "RUNNING": {"RUNNING", "DONE"},
                    "DONE": {"DONE"},
                }
                if (
                    previous_type != operation_type
                    or previous_actor_sha256 != actor_sha256
                    or status not in valid_statuses[previous_status]
                    or (
                        previous_status == "DONE"
                        and previous_succeeded is not operation_succeeded
                    )
                ):
                    raise CleanupBlocked("cloud_sql_operation_ledger_drifted")
            known_operations[name] = current
            if status == "DONE":
                done_seen.add(name)
                pending_seen.discard(name)
            else:
                pending_seen.add(name)
        if any(name not in relevant for name in pending_seen | done_seen):
            # Every operation observed in this reconciliation horizon must
            # remain in each complete snapshot. Losing either an active or a
            # terminal row would let pagination/history drift erase causal
            # evidence before the absence receipt is sealed.
            raise CleanupBlocked("cloud_sql_operation_ledger_incomplete")
        return newly_seen

    def _cleanup_snapshot(
        self,
        username: str,
    ) -> tuple[Mapping[str, tuple[str, str, str, bool]], bool]:
        try:
            first_operations = self._instance_operations()
            users = self._user_names(exact_admin_username=username)
            second_operations = self._instance_operations()
            if first_operations != second_operations:
                raise OwnerLauncherError("cloud_sql_operations_evidence_incomplete")
        except OwnerLauncherError as exc:
            raise CleanupBlocked(exc.code) from None
        return (
            self._relevant_user_operations(second_operations),
            username in users,
        )

    def _delete_user_once(self, username: str, *, deadline: float) -> bool:
        """Attempt one DELETE_USER; return whether its terminality is ambiguous."""

        query = urllib.parse.urlencode({"host": "", "name": username})
        try:
            operation = self._client.request_json(
                "DELETE", f"{self._users_url}?{query}"
            )
            operation_name = self._record_operation_name(operation)
            self._wait_operation(
                operation_name,
                expected_operation_type="DELETE_USER",
                deadline=deadline,
            )
        except BaseException:
            # The response may be lost after Cloud SQL accepted DELETE_USER.
            # The complete operation ledger and users.list decide terminality.
            self._mutation_ambiguity_observed = True
            return True
        return False

    def _record_absence_reconciliation(
        self,
        *,
        username: str,
        observed_relevant: Mapping[str, tuple[str, str, str, bool]],
        baseline: frozenset[str],
        baseline_relevant: Mapping[str, tuple[str, str, str, bool]],
        ambiguity_observed: bool,
    ) -> None:
        post_baseline_authority = {
            name: value
            for name, value in observed_relevant.items()
            if name not in baseline and value[0] in {"CREATE_USER", "UPDATE_USER"}
        }
        response_known_candidate_observed = bool(
            set(observed_relevant) & self._mutation_authority_known_operations
        )
        response_known_delete_names = sorted(
            name
            for name in self._mutation_known_operations
            if name in observed_relevant
            and observed_relevant[name][0] == "DELETE_USER"
        )
        evidence = {
            **self._absence_evidence_identity(),
            "project": PROJECT,
            "instance": SQL_INSTANCE,
            "username_sha256": _sha256(username.encode("utf-8")),
            "owner_subject_sha256": self._expected_owner_subject_sha256,
            "mutation_context_sha256": self._expected_mutation_context_sha256,
            "user_absent": True,
            "baseline_operation_names": sorted(baseline),
            "baseline_user_operations": [
                [
                    name,
                    operation_type,
                    status,
                    actor_sha256,
                    operation_succeeded,
                ]
                for name, (
                    operation_type,
                    status,
                    actor_sha256,
                    operation_succeeded,
                ) in sorted(baseline_relevant.items())
            ],
            "known_operation_names": sorted(self._mutation_known_operations),
            "response_known_authority_operation_names": sorted(
                self._mutation_authority_known_operations
            ),
            "response_known_delete_operation_names": (
                response_known_delete_names
            ),
            "post_baseline_authority_operations": [
                [
                    name,
                    operation_type,
                    status,
                    actor_sha256,
                    operation_succeeded,
                ]
                for name, (
                    operation_type,
                    status,
                    actor_sha256,
                    operation_succeeded,
                ) in sorted(post_baseline_authority.items())
            ],
            "response_known_candidate_observed": (response_known_candidate_observed),
            "post_baseline_authority_operation_count": len(post_baseline_authority),
            "terminal_user_operations": [
                [
                    name,
                    operation_type,
                    status,
                    actor_sha256,
                    operation_succeeded,
                ]
                for name, (
                    operation_type,
                    status,
                    actor_sha256,
                    operation_succeeded,
                ) in sorted(observed_relevant.items())
            ],
            "mutation_ambiguity_observed": ambiguity_observed,
            "quiet_window_seconds": self._operation_timeout_seconds,
        }
        evidence_sha256 = _sha256(_canonical_bytes(evidence))
        self._reconciliation_receipt = json.loads(
            _canonical_bytes(
                {
                    **evidence,
                    "evidence_sha256": evidence_sha256,
                }
            ).decode("utf-8")
        )
        self._reconciliation_evidence_sha256 = evidence_sha256
        self._reconciliation_quiet_window_seconds = self._operation_timeout_seconds
        self._reconciliation_response_known_candidate_observed = (
            response_known_candidate_observed
        )
        self._reconciliation_post_baseline_authority_operation_count = len(
            post_baseline_authority
        )
        self._reconciliation_proven = True
        self._mutation_ambiguity_observed = bool(
            self._mutation_ambiguity_observed or ambiguity_observed
        )

    def _absence_evidence_identity(self) -> Mapping[str, Any]:
        return {
            "schema": "muncho-cloud-sql-admin-absence-evidence.v1",
            "temporary_admin_absent": True,
        }

    def _reconcile_absence(
        self,
        username: str,
        *,
        baseline: frozenset[str],
        baseline_relevant: Mapping[str, tuple[str, str, str, bool]],
        ambiguity_observed: bool,
    ) -> None:
        quiet_window = self._operation_timeout_seconds
        started = self._monotonic()
        hard_deadline = started + max(quiet_window * 2.0, quiet_window + 1.0)
        quiet_since: float | None = None
        previous_signature: (
            tuple[tuple[str, tuple[str, str, str, bool]], ...] | None
        ) = None
        previous_present: bool | None = None
        known_operations: dict[str, tuple[str, str, str, bool]] = dict(
            baseline_relevant
        )
        pending_seen: set[str] = set()
        done_seen: set[str] = {
            name for name, value in baseline_relevant.items() if value[1] == "DONE"
        }
        pending_seen.update(
            name for name, value in baseline_relevant.items() if value[1] != "DONE"
        )
        delete_attempts = 0
        polls = 0
        maximum_polls = max(8, int(hard_deadline - started) + 8)
        poll_interval = min(5.0, max(0.1, quiet_window / 2.0))
        while polls < maximum_polls:
            polls += 1
            relevant, present = self._cleanup_snapshot(username)
            self._track_operation_snapshot(
                relevant,
                known_operations=known_operations,
                pending_seen=pending_seen,
                done_seen=done_seen,
            )
            signature = tuple(sorted(relevant.items()))
            current = self._monotonic()
            if signature != previous_signature or present != previous_present:
                quiet_since = current
            previous_signature = signature
            previous_present = present
            active = any(value[1] != "DONE" for value in relevant.values())
            if not active and present:
                if delete_attempts >= self._SETTLE_ATTEMPTS:
                    raise CleanupBlocked("cloud_sql_delete_attempts_exhausted")
                delete_attempts += 1
                ambiguity_observed = bool(
                    self._delete_user_once(username, deadline=hard_deadline)
                    or ambiguity_observed
                )
                quiet_since = self._monotonic()
                previous_signature = None
                previous_present = None
            elif (
                not active
                and not present
                and quiet_since is not None
                and current - quiet_since >= quiet_window
            ):
                final_relevant, final_present = self._cleanup_snapshot(username)
                self._track_operation_snapshot(
                    final_relevant,
                    known_operations=known_operations,
                    pending_seen=pending_seen,
                    done_seen=done_seen,
                )
                final_signature = tuple(sorted(final_relevant.items()))
                if (
                    final_present
                    or final_signature != signature
                    or any(value[1] != "DONE" for value in final_relevant.values())
                ):
                    quiet_since = self._monotonic()
                    previous_signature = final_signature
                    previous_present = final_present
                else:
                    self._record_absence_reconciliation(
                        username=username,
                        observed_relevant=known_operations,
                        baseline=baseline,
                        baseline_relevant=baseline_relevant,
                        ambiguity_observed=ambiguity_observed,
                    )
                    self._mutation_ambiguous = False
                    return
            if self._monotonic() >= hard_deadline:
                raise CleanupBlocked("cloud_sql_quiet_window_timeout")
            self._sleeper(poll_interval)
        raise CleanupBlocked("cloud_sql_quiet_window_timeout")

    def delete_and_confirm_absent(self, username: str) -> None:
        """Drain user operations and require one full continuous quiet window."""

        if not self._valid_target_username(username):
            raise CleanupBlocked()
        try:
            initial_operations = self._stable_instance_operations()
        except OwnerLauncherError as exc:
            raise CleanupBlocked(exc.code) from None
        initial_relevant = self._relevant_user_operations(initial_operations)
        causal_baseline = (
            self._mutation_operation_baseline
            if self._mutation_operation_baseline is not None
            else frozenset(initial_operations)
        )
        causal_baseline_relevant = (
            self._mutation_relevant_baseline
            if self._mutation_relevant_baseline is not None
            else initial_relevant
        )
        self._reconcile_absence(
            username,
            baseline=causal_baseline,
            baseline_relevant=causal_baseline_relevant,
            ambiguity_observed=False,
        )

    def reconcile_ambiguous_mutation_and_confirm_absent(
        self,
        username: str,
    ) -> None:
        """Observe the full operation horizon before resolving lost mutation truth."""

        if (
            not self._valid_target_username(username)
            or self._mutation_operation_baseline is None
            or not self._mutation_ambiguous
        ):
            raise CleanupBlocked()
        baseline = self._mutation_operation_baseline
        baseline_relevant = self._mutation_relevant_baseline
        if baseline_relevant is None:
            raise CleanupBlocked()
        self._reconcile_absence(
            username,
            baseline=baseline,
            baseline_relevant=baseline_relevant,
            ambiguity_observed=True,
        )


class CloudSqlSchemaReconciliationAdmin(CloudSqlTemporaryAdmin):
    """Exact owner-approved Cloud SQL login for one schema repair.

    Cloud SQL grants custom PostgreSQL roles with INHERIT but without SET.
    The migration-owner membership is full ephemeral owner-equivalent
    authority over Canonical Brain objects; it is not ordinary writer access.
    This boundary never pretends the login can SET ROLE.  It binds that exact
    authority to the sealed owner-owned trampoline and explicitly revokes
    every stale role when recovering a deterministic prior login.
    """

    _DATABASE_ROLES = SCHEMA_RECONCILIATION_DATABASE_ROLES

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fixed_update_etag: str | None = None

    @staticmethod
    def _valid_etag(value: object) -> bool:
        return (
            isinstance(value, str)
            and 1 <= len(value) <= 1024
            and all(0x21 <= ord(character) <= 0x7E for character in value)
        )

    def _validate_exact_user_resource(
        self,
        item: Mapping[str, Any],
        *,
        username: str,
    ) -> Mapping[str, Any]:
        if (
            not self._valid_target_username(username)
            or item.get("kind") != "sql#user"
            or item.get("name") != username
            or item.get("project") != PROJECT
            or item.get("instance") != SQL_INSTANCE
            or item.get("host") not in (None, "")
        ):
            raise OwnerLauncherError("invalid_cloud_sql_users")
        # The real Cloud SQL users.list response omits databaseRoles and may
        # carry an ETag that differs from users.get.  Treat this projection as
        # presence-only; every authority-bearing field is read from users.get.
        return {
            "instance": SQL_INSTANCE,
            "name": username,
            "project": PROJECT,
        }

    def _validate_described_resource(
        self,
        item: Mapping[str, Any],
        *,
        username: str,
        require_exact_roles: bool,
    ) -> Mapping[str, Any]:
        roles = item.get("databaseRoles")
        etag = item.get("etag")
        if (
            not self._valid_target_username(username)
            or item.get("kind") != "sql#user"
            or item.get("name") != username
            or item.get("project") != PROJECT
            or item.get("instance") != SQL_INSTANCE
            or item.get("type") != "BUILT_IN"
            or item.get("host") != ""
            or not self._valid_etag(etag)
            or "password" in item
            or type(roles) is not list
            or any(not isinstance(role, str) for role in roles)
            or len(roles) != len(set(roles))
            or (
                require_exact_roles
                and sorted(roles) != list(self._DATABASE_ROLES)
            )
        ):
            raise OwnerLauncherError(
                "cloud_sql_schema_reconciliation_admin_resource_invalid"
            )
        return {
            "databaseRoles": sorted(roles),
            "etag": etag,
            "host": "",
            "instance": SQL_INSTANCE,
            "name": username,
            "project": PROJECT,
            "type": "BUILT_IN",
        }

    def _user_url(self, username: str) -> str:
        if not self._valid_target_username(username):
            raise OwnerLauncherError("invalid_admin_username")
        encoded_username = urllib.parse.quote(username, safe="")
        query = urllib.parse.urlencode({"host": ""})
        return f"{self._users_url}/{encoded_username}?{query}"

    def _description_snapshot(
        self,
        username: str,
        *,
        require_exact_roles: bool,
    ) -> Mapping[str, Any] | None:
        before = self._instance_operations()
        names, _presence = self._users_and_exact_resource(
            exact_admin_username=username
        )
        resource: Mapping[str, Any] | None = None
        if username in names:
            try:
                payload = self._client.request_json(
                    "GET",
                    self._user_url(username),
                )
            except OwnerLauncherError:
                raise OwnerLauncherError(
                    "cloud_sql_schema_reconciliation_admin_resource_drifted"
                ) from None
            resource = self._validate_described_resource(
                payload,
                username=username,
                require_exact_roles=require_exact_roles,
            )
        after = self._instance_operations()
        if before != after:
            raise OwnerLauncherError(
                "cloud_sql_schema_reconciliation_admin_resource_drifted"
            )
        return resource

    def _role_bound_resource(
        self,
        username: str,
        *,
        require_exact_roles: bool,
    ) -> Mapping[str, Any] | None:
        first = self._description_snapshot(
            username,
            require_exact_roles=require_exact_roles,
        )
        second = self._description_snapshot(
            username,
            require_exact_roles=require_exact_roles,
        )
        if first != second:
            raise OwnerLauncherError(
                "cloud_sql_schema_reconciliation_admin_resource_drifted"
            )
        return second

    def _create_user_body(self, username: str, password: str) -> Mapping[str, Any]:
        return {
            **super()._create_user_body(username, password),
            "databaseRoles": list(self._DATABASE_ROLES),
        }

    def _update_user_query_values(self, username: str) -> Mapping[str, object]:
        return {
            "host": "",
            "name": username,
            "databaseRoles": list(self._DATABASE_ROLES),
            "revokeExistingRoles": "true",
        }

    def _update_user_body(self, username: str, password: str) -> Mapping[str, Any]:
        if self._fixed_update_etag is None:
            raise OwnerLauncherError(
                "cloud_sql_schema_reconciliation_admin_resource_invalid"
            )
        return {
            "databaseRoles": list(self._DATABASE_ROLES),
            "etag": self._fixed_update_etag,
            "name": username,
            "password": password,
            "revokeExistingRoles": True,
            "type": "BUILT_IN",
        }

    def create_or_rotate_recovery(self, username: str, password: str) -> None:
        if not self._valid_target_username(username):
            raise OwnerLauncherError("invalid_admin_username")
        self._ensure_mutation_observation()
        resource = self._role_bound_resource(
            username,
            require_exact_roles=False,
        )
        if resource is None:
            self.create(username, password)
        else:
            etag = resource.get("etag")
            if not self._valid_etag(etag):
                raise OwnerLauncherError(
                    "cloud_sql_schema_reconciliation_admin_resource_invalid"
                )
            self._fixed_update_etag = str(etag)
            try:
                self.rotate_existing(username, password)
            finally:
                self._fixed_update_etag = None
        self.require_current_authority(username)

    def require_current_authority(self, username: str) -> None:
        super().require_current_authority(username)
        resource = self._role_bound_resource(username, require_exact_roles=True)
        if resource is None:
            self._mutation_ambiguous = True
            self._mutation_ambiguity_observed = True
            raise OwnerLauncherError(
                "cloud_sql_schema_reconciliation_admin_authority_unconfirmed"
            )
        super().require_current_authority(username)

    def temporary_admin_authority_receipt(
        self,
        username: str,
    ) -> Mapping[str, Any]:
        base = dict(super().temporary_admin_authority_receipt(username))
        base.pop("receipt_sha256", None)
        resource = self._role_bound_resource(username, require_exact_roles=True)
        if resource is None:
            raise OwnerLauncherError(
                "cloud_sql_schema_reconciliation_admin_authority_unconfirmed"
            )
        self.require_current_authority(username)
        unsigned = {
            **base,
            "database_roles": list(self._DATABASE_ROLES),
            "cloudsqlsuperuser_absent": True,
            "resource_etag_sha256": _sha256(
                str(resource["etag"]).encode("ascii")
            ),
        }
        return {
            **unsigned,
            "receipt_sha256": _sha256(_canonical_bytes(unsigned)),
        }


class CloudSqlCanaryBootstrapLogin(CloudSqlTemporaryAdmin):
    """Fixed Cloud SQL boundary for the canary bootstrap login.

    The target project, instance, login, host, type, and sole database role are
    constants.  Callers provide only the owner identity digest at construction
    and an in-memory provisional password at mutation time.
    """

    def __init__(
        self,
        client: GoogleRestClient,
        *,
        expected_owner_subject_sha256: str,
        expected_mutation_context_sha256: str | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        operation_timeout_seconds: float = _OPERATION_TIMEOUT_SECONDS,
    ) -> None:
        _require_sha256(
            expected_owner_subject_sha256,
            "invalid_owner_subject_sha256",
        )
        if expected_mutation_context_sha256 is not None:
            _require_sha256(
                expected_mutation_context_sha256,
                "invalid_mutation_context_sha256",
            )
        super().__init__(
            client,
            monotonic=monotonic,
            sleeper=sleeper,
            operation_timeout_seconds=operation_timeout_seconds,
        )
        self._fixed_owner_subject_sha256 = expected_owner_subject_sha256
        self._fixed_mutation_context_sha256 = expected_mutation_context_sha256
        self._fixed_update_etag: str | None = None
        self._fixed_delete_authorized = False
        self._fixed_delete_resource: Mapping[str, Any] | None = None
        self._fixed_delete_authority_operation_name: str | None = None

    def _valid_target_username(self, username: object) -> bool:
        return username == CANARY_BOOTSTRAP_LOGIN

    def _validate_exact_user_resource(
        self,
        item: Mapping[str, Any],
        *,
        username: str,
    ) -> Mapping[str, Any]:
        if (
            username != CANARY_BOOTSTRAP_LOGIN
            or item.get("kind") != "sql#user"
            or item.get("name") != CANARY_BOOTSTRAP_LOGIN
            or item.get("project") != PROJECT
            or item.get("instance") != SQL_INSTANCE
            or item.get("host") not in (None, "")
        ):
            raise OwnerLauncherError("cloud_sql_bootstrap_login_resource_invalid")
        # Cloud SQL users.list omits security-bearing fields on the real
        # canary.  This projection is intentionally presence-only; describe()
        # obtains and validates the complete User resource separately.
        return {
            "instance": SQL_INSTANCE,
            "name": CANARY_BOOTSTRAP_LOGIN,
            "project": PROJECT,
        }

    @staticmethod
    def _validate_described_resource(
        item: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        etag = item.get("etag")
        database_roles = item.get("databaseRoles")
        if (
            item.get("kind") != "sql#user"
            or item.get("name") != CANARY_BOOTSTRAP_LOGIN
            or item.get("project") != PROJECT
            or item.get("instance") != SQL_INSTANCE
            or ("type" in item and item.get("type") != "BUILT_IN")
            or item.get("host") != ""
            or not isinstance(etag, str)
            or not 1 <= len(etag) <= 1024
            or any(ord(character) < 0x21 or ord(character) > 0x7E for character in etag)
            or type(database_roles) is not list
            or database_roles != [CANARY_BOOTSTRAP_DATABASE_ROLE]
            or any(
                isinstance(role, str) and role.casefold() == "cloudsqlsuperuser"
                for role in database_roles
            )
            or "password" in item
        ):
            raise OwnerLauncherError("cloud_sql_bootstrap_login_resource_invalid")
        return {
            "databaseRoles": [CANARY_BOOTSTRAP_DATABASE_ROLE],
            "etag": etag,
            "host": "",
            "instance": SQL_INSTANCE,
            "name": CANARY_BOOTSTRAP_LOGIN,
            "project": PROJECT,
            "type": "BUILT_IN",
        }

    @property
    def _fixed_user_url(self) -> str:
        encoded_login = urllib.parse.quote(CANARY_BOOTSTRAP_LOGIN, safe="")
        query = urllib.parse.urlencode({"host": ""})
        return f"{self._users_url}/{encoded_login}?{query}"

    def _create_user_body(self, username: str, password: str) -> Mapping[str, Any]:
        if username != CANARY_BOOTSTRAP_LOGIN:
            raise OwnerLauncherError("cloud_sql_bootstrap_login_target_invalid")
        return {
            "databaseRoles": [CANARY_BOOTSTRAP_DATABASE_ROLE],
            "host": "",
            "instance": SQL_INSTANCE,
            "name": CANARY_BOOTSTRAP_LOGIN,
            "password": password,
            "project": PROJECT,
            "type": "BUILT_IN",
        }

    def _update_user_body(self, username: str, password: str) -> Mapping[str, Any]:
        if username != CANARY_BOOTSTRAP_LOGIN or self._fixed_update_etag is None:
            raise OwnerLauncherError("cloud_sql_bootstrap_login_target_invalid")
        return {
            "databaseRoles": [CANARY_BOOTSTRAP_DATABASE_ROLE],
            "etag": self._fixed_update_etag,
            "host": "",
            "name": CANARY_BOOTSTRAP_LOGIN,
            "password": password,
            "type": "BUILT_IN",
        }

    def _update_user_query_values(self, username: str) -> Mapping[str, object]:
        if username != CANARY_BOOTSTRAP_LOGIN:
            raise OwnerLauncherError("cloud_sql_bootstrap_login_target_invalid")
        return {
            "host": "",
            "name": CANARY_BOOTSTRAP_LOGIN,
            "databaseRoles": [CANARY_BOOTSTRAP_DATABASE_ROLE],
            "revokeExistingRoles": "true",
        }

    def begin_mutation_observation(self) -> None:
        super().begin_mutation_observation(
            expected_owner_subject_sha256=self._fixed_owner_subject_sha256,
            expected_mutation_context_sha256=(
                self._fixed_mutation_context_sha256
            ),
        )

    def _description_snapshot(self) -> Mapping[str, Any] | None:
        first_operations = self._instance_operations()
        names, _presence = self._users_and_exact_resource(
            exact_admin_username=CANARY_BOOTSTRAP_LOGIN
        )
        resource: Mapping[str, Any] | None = None
        if CANARY_BOOTSTRAP_LOGIN in names:
            try:
                payload = self._client.request_json("GET", self._fixed_user_url)
            except OwnerLauncherError:
                raise OwnerLauncherError(
                    "cloud_sql_bootstrap_login_resource_drifted"
                ) from None
            resource = self._validate_described_resource(payload)
        second_operations = self._instance_operations()
        if first_operations != second_operations:
            raise OwnerLauncherError("cloud_sql_bootstrap_login_resource_drifted")
        return resource

    def describe(self) -> Mapping[str, Any] | None:
        """Return a stable, password-free projection of the fixed user."""

        first = self._description_snapshot()
        second = self._description_snapshot()
        if first != second:
            raise OwnerLauncherError("cloud_sql_bootstrap_login_resource_drifted")
        return second

    def _require_observation_unchanged(self) -> None:
        baseline = self._mutation_relevant_baseline
        if baseline is None:
            raise OwnerLauncherError("cloud_sql_mutation_evidence_unconfirmed")
        current = self._relevant_user_operations(self._stable_instance_operations())
        if current != baseline:
            raise OwnerLauncherError("cloud_sql_mutation_evidence_unconfirmed")

    def require_absent(self) -> None:
        if self.describe() is not None:
            raise OwnerLauncherError("cloud_sql_bootstrap_login_already_exists")

    def create(self, provisional_password: str) -> None:
        """Create only the fixed login with its exact sole database role."""

        self._fixed_delete_authorized = False
        self.begin_mutation_observation()
        self.require_absent()
        self._require_observation_unchanged()
        super().create(CANARY_BOOTSTRAP_LOGIN, provisional_password)
        try:
            if self.describe() is None:
                raise OwnerLauncherError("cloud_sql_bootstrap_login_create_unconfirmed")
        except OwnerLauncherError:
            self._mutation_ambiguous = True
            self._mutation_ambiguity_observed = True
            raise
        self._fixed_delete_authorized = True

    def _rotate_from_resource(
        self,
        provisional_password: str,
        resource: Mapping[str, Any],
    ) -> None:
        self._fixed_delete_authorized = False
        self.begin_mutation_observation()
        current = self.describe()
        if current is None:
            raise OwnerLauncherError("cloud_sql_bootstrap_login_missing")
        if current != resource:
            raise OwnerLauncherError("cloud_sql_bootstrap_login_resource_drifted")
        self._require_observation_unchanged()
        etag = current.get("etag")
        if not isinstance(etag, str):
            raise OwnerLauncherError("cloud_sql_bootstrap_login_resource_invalid")
        self._fixed_update_etag = etag
        try:
            super().rotate_existing(CANARY_BOOTSTRAP_LOGIN, provisional_password)
        finally:
            self._fixed_update_etag = None
        try:
            if self.describe() is None:
                raise OwnerLauncherError("cloud_sql_bootstrap_login_rotate_unconfirmed")
        except OwnerLauncherError:
            self._mutation_ambiguous = True
            self._mutation_ambiguity_observed = True
            raise
        self._fixed_delete_authorized = True

    def rotate_existing(self, provisional_password: str) -> None:
        resource = self.describe()
        if resource is None:
            raise OwnerLauncherError("cloud_sql_bootstrap_login_missing")
        self._rotate_from_resource(provisional_password, resource)

    def create_or_rotate_recovery(self, provisional_password: str) -> None:
        """Recover authority without changing or broadening database roles."""

        resource = self.describe()
        if resource is None:
            self.create(provisional_password)
        else:
            self._rotate_from_resource(provisional_password, resource)

    def require_current_authority(self) -> None:
        super().require_current_authority(CANARY_BOOTSTRAP_LOGIN)

    def authority_receipt(self) -> Mapping[str, Any]:
        """Return fixed authority and resource truth without password material."""

        if not self._fixed_delete_authorized:
            raise OwnerLauncherError("cloud_sql_mutation_evidence_unconfirmed")
        self.require_current_authority()
        resource = self.describe()
        if resource is None:
            raise OwnerLauncherError("cloud_sql_mutation_evidence_unconfirmed")
        self.require_current_authority()
        operation_name = self._confirmed_authority_operation_name
        operation_type = self._confirmed_authority_operation_type
        if operation_name is None or operation_type not in {
            "CREATE_USER",
            "UPDATE_USER",
        }:
            raise OwnerLauncherError("cloud_sql_mutation_evidence_unconfirmed")
        receipt = {
            "schema": CANARY_BOOTSTRAP_AUTHORITY_RECEIPT_SCHEMA,
            "project": PROJECT,
            "instance": SQL_INSTANCE,
            "name": CANARY_BOOTSTRAP_LOGIN,
            "host": "",
            "type": "BUILT_IN",
            "database_roles": [CANARY_BOOTSTRAP_DATABASE_ROLE],
            "etag": resource["etag"],
            "resource_projection_sha256": _sha256(_canonical_bytes(resource)),
            "operation_name": operation_name,
            "operation_type": operation_type,
            "owner_subject_sha256": self._fixed_owner_subject_sha256,
            "mutation_context_sha256": self._fixed_mutation_context_sha256,
        }
        return {
            **receipt,
            "receipt_sha256": _sha256(_canonical_bytes(receipt)),
        }

    def _absence_evidence_identity(self) -> Mapping[str, Any]:
        return {
            "schema": CANARY_BOOTSTRAP_ABSENCE_EVIDENCE_SCHEMA,
            "bootstrap_login_absent": True,
            "bootstrap_login": CANARY_BOOTSTRAP_LOGIN,
            "database_roles": [CANARY_BOOTSTRAP_DATABASE_ROLE],
        }

    def _confirm_absent_without_delete(self, *, ambiguity_observed: bool) -> None:
        """Boundedly prove absence without ever deleting an unowned resource."""

        try:
            initial_operations = self._stable_instance_operations()
        except OwnerLauncherError as exc:
            raise CleanupBlocked(exc.code) from None
        initial_relevant = self._relevant_user_operations(initial_operations)
        baseline = (
            self._mutation_operation_baseline
            if self._mutation_operation_baseline is not None
            else frozenset(initial_operations)
        )
        baseline_relevant = (
            self._mutation_relevant_baseline
            if self._mutation_relevant_baseline is not None
            else initial_relevant
        )
        if baseline_relevant is None:
            raise CleanupBlocked("cloud_sql_mutation_evidence_unconfirmed")
        quiet_window = self._operation_timeout_seconds
        started = self._monotonic()
        hard_deadline = started + max(quiet_window * 2.0, quiet_window + 1.0)
        quiet_since: float | None = None
        previous_signature: tuple[tuple[str, tuple[str, str, str, bool]], ...] | None = None
        known_operations: dict[str, tuple[str, str, str, bool]] = dict(
            baseline_relevant
        )
        pending_seen = {
            name for name, value in baseline_relevant.items() if value[1] != "DONE"
        }
        done_seen = {
            name for name, value in baseline_relevant.items() if value[1] == "DONE"
        }
        polls = 0
        maximum_polls = max(8, int(hard_deadline - started) + 8)
        poll_interval = min(5.0, max(0.1, quiet_window / 2.0))
        while polls < maximum_polls:
            polls += 1
            relevant, present = self._cleanup_snapshot(CANARY_BOOTSTRAP_LOGIN)
            self._track_operation_snapshot(
                relevant,
                known_operations=known_operations,
                pending_seen=pending_seen,
                done_seen=done_seen,
            )
            if present:
                raise CleanupBlocked("cloud_sql_bootstrap_login_ownership_unproven")
            signature = tuple(sorted(relevant.items()))
            current = self._monotonic()
            if signature != previous_signature:
                quiet_since = current
            previous_signature = signature
            if (
                not any(value[1] != "DONE" for value in relevant.values())
                and quiet_since is not None
                and current - quiet_since >= quiet_window
            ):
                final_relevant, final_present = self._cleanup_snapshot(
                    CANARY_BOOTSTRAP_LOGIN
                )
                self._track_operation_snapshot(
                    final_relevant,
                    known_operations=known_operations,
                    pending_seen=pending_seen,
                    done_seen=done_seen,
                )
                if final_present:
                    raise CleanupBlocked(
                        "cloud_sql_bootstrap_login_ownership_unproven"
                    )
                if (
                    tuple(sorted(final_relevant.items())) == signature
                    and not any(
                        value[1] != "DONE" for value in final_relevant.values()
                    )
                ):
                    self._record_absence_reconciliation(
                        username=CANARY_BOOTSTRAP_LOGIN,
                        observed_relevant=known_operations,
                        baseline=baseline,
                        baseline_relevant=baseline_relevant,
                        ambiguity_observed=ambiguity_observed,
                    )
                    self._mutation_ambiguous = False
                    return
            if self._monotonic() >= hard_deadline:
                raise CleanupBlocked("cloud_sql_quiet_window_timeout")
            self._sleeper(poll_interval)
        raise CleanupBlocked("cloud_sql_quiet_window_timeout")

    def _validate_fixed_delete_ledger(
        self,
        relevant: Mapping[str, tuple[str, str, str, bool]],
    ) -> None:
        baseline = self._mutation_operation_baseline
        baseline_relevant = self._mutation_relevant_baseline
        authority_name = self._fixed_delete_authority_operation_name
        authority_type = self._confirmed_authority_operation_type
        if (
            baseline is None
            or baseline_relevant is None
            or authority_name is None
            or authority_type not in {"CREATE_USER", "UPDATE_USER"}
        ):
            raise CleanupBlocked("cloud_sql_mutation_evidence_unconfirmed")
        observed_baseline = {
            name: relevant[name] for name in baseline_relevant if name in relevant
        }
        authority_delta = {
            name: value
            for name, value in relevant.items()
            if name not in baseline and value[0] in {"CREATE_USER", "UPDATE_USER"}
        }
        expected_authority = authority_delta.get(authority_name)
        post_baseline_deletes = {
            name: value
            for name, value in relevant.items()
            if name not in baseline and value[0] == "DELETE_USER"
        }
        if (
            observed_baseline != baseline_relevant
            or set(authority_delta) != {authority_name}
            or expected_authority is None
            or expected_authority[0] != authority_type
            or expected_authority[1] != "DONE"
            or expected_authority[2] != self._fixed_owner_subject_sha256
            or expected_authority[3] is not True
            or any(
                value[2] != self._fixed_owner_subject_sha256
                for value in post_baseline_deletes.values()
            )
        ):
            raise CleanupBlocked("cloud_sql_bootstrap_login_ownership_unproven")

    def _cleanup_snapshot(
        self,
        username: str,
    ) -> tuple[Mapping[str, tuple[str, str, str, bool]], bool]:
        relevant, present = super()._cleanup_snapshot(username)
        if self._fixed_delete_resource is not None:
            self._validate_fixed_delete_ledger(relevant)
        return relevant, present

    def _delete_user_once(self, username: str, *, deadline: float) -> bool:
        bound_resource = self._fixed_delete_resource
        if username != CANARY_BOOTSTRAP_LOGIN or bound_resource is None:
            raise CleanupBlocked("cloud_sql_bootstrap_login_ownership_unproven")
        try:
            current = self.describe()
            current_relevant = self._relevant_user_operations(
                self._stable_instance_operations()
            )
        except OwnerLauncherError as exc:
            raise CleanupBlocked(exc.code) from None
        self._validate_fixed_delete_ledger(current_relevant)
        if current is None:
            return False
        if current != bound_resource:
            raise CleanupBlocked("cloud_sql_bootstrap_login_ownership_unproven")
        return super()._delete_user_once(username, deadline=deadline)

    def delete_and_confirm_absent(self) -> None:
        """Delete only after this object has confirmed current mutation authority."""

        resource = self.describe()
        if resource is None:
            self._confirm_absent_without_delete(ambiguity_observed=False)
            return
        if not self._fixed_delete_authorized:
            raise CleanupBlocked("cloud_sql_bootstrap_login_ownership_unproven")
        try:
            super().require_current_authority(CANARY_BOOTSTRAP_LOGIN)
        except OwnerLauncherError as exc:
            raise CleanupBlocked(exc.code) from None
        authority_operation_name = self._confirmed_authority_operation_name
        if authority_operation_name is None:
            raise CleanupBlocked("cloud_sql_mutation_evidence_unconfirmed")
        self._fixed_delete_resource = dict(resource)
        self._fixed_delete_authority_operation_name = authority_operation_name
        try:
            super().delete_and_confirm_absent(CANARY_BOOTSTRAP_LOGIN)
            try:
                final_resource = self.describe()
            except OwnerLauncherError as exc:
                raise CleanupBlocked(exc.code) from None
            if final_resource is not None:
                cause_code = (
                    "cloud_sql_bootstrap_login_delete_unconfirmed"
                    if final_resource == resource
                    else "cloud_sql_bootstrap_login_ownership_unproven"
                )
                raise CleanupBlocked(cause_code)
        except BaseException:
            self._fixed_delete_authorized = False
            raise
        finally:
            self._fixed_delete_resource = None
            self._fixed_delete_authority_operation_name = None
        self._fixed_delete_authorized = False

    def reconcile_ambiguous_mutation_and_confirm_absent(self) -> None:
        """Reconcile lost truth without blind-deleting a same-named account."""

        if self._mutation_operation_baseline is None or not self._mutation_ambiguous:
            raise CleanupBlocked("cloud_sql_mutation_evidence_unconfirmed")
        self._confirm_absent_without_delete(ambiguity_observed=True)


def _phase_b_secret_free(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = key.casefold() if isinstance(key, str) else ""
            declarative_absence = lowered in {
                "secret_material_recorded",
                "password_or_digest_recorded",
                "content_or_digest_recorded",
            } and item is False
            if not isinstance(key, str) or (
                not declarative_absence
                and any(
                    marker in lowered
                    for marker in (
                        "password",
                        "secret",
                        "secret_digest",
                        "credential_digest",
                        "credential_value",
                    )
                )
            ):
                raise OwnerLauncherError("phase_b_secret_field_forbidden")
            _phase_b_secret_free(item)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for item in value:
            _phase_b_secret_free(item)


def _phase_b_operation_rows(
    boundary: CloudSqlTemporaryAdmin,
) -> list[list[Any]]:
    operations = boundary._relevant_user_operations(
        boundary._stable_instance_operations()
    )
    rows = [
        [name, operation_type, status, actor_sha256, succeeded]
        for name, (
            operation_type,
            status,
            actor_sha256,
            succeeded,
        ) in sorted(operations.items())
    ]
    if any(row[2] != "DONE" or row[4] is not True for row in rows):
        raise OwnerLauncherError("cloud_sql_user_operations_not_quiescent")
    return rows


def _phase_b_initial_cloud_observation(
    boundary: CloudSqlTemporaryAdmin,
) -> Mapping[str, Any]:
    first = _phase_b_operation_rows(boundary)
    names = sorted(boundary._user_names())
    second = _phase_b_operation_rows(boundary)
    if first != second:
        raise OwnerLauncherError("cloud_sql_operations_evidence_incomplete")
    temporary = sorted(name for name in names if _ADMIN_USERNAME.fullmatch(name))
    if names != ["muncho_canary_writer_login", "postgres"] or temporary:
        raise OwnerLauncherError("phase_b_cloud_not_pristine")
    return {
        "project": PROJECT,
        "instance": SQL_INSTANCE,
        "visible_users": names,
        "user_inventory_sha256": _sha256(_canonical_bytes(names)),
        "bootstrap_login_absent": True,
        "temporary_admin_users": temporary,
        "user_operations_quiescent": True,
        "operation_ledger_sha256": _sha256(_canonical_bytes(second)),
    }


def _phase_b_recovery_cloud_observation(
    boundary: CloudSqlTemporaryAdmin,
    bootstrap: CloudSqlCanaryBootstrapLogin,
    *,
    temporary_admin_username: str,
) -> Mapping[str, Any]:
    first = _phase_b_operation_rows(boundary)
    names = sorted(boundary._user_names())
    resource = bootstrap.describe()
    second = _phase_b_operation_rows(boundary)
    if first != second:
        raise OwnerLauncherError("cloud_sql_operations_evidence_incomplete")
    temporary = sorted(name for name in names if _ADMIN_USERNAME.fullmatch(name))
    if any(name != temporary_admin_username for name in temporary):
        raise OwnerLauncherError("phase_b_cloud_unapproved_temporary_admin")
    return {
        "project": PROJECT,
        "instance": SQL_INSTANCE,
        "bootstrap_resource": resource,
        "temporary_admin_users": temporary,
        "user_inventory_sha256": _sha256(_canonical_bytes(names)),
        "user_operations_quiescent": True,
        "operation_ledger_sha256": _sha256(_canonical_bytes(second)),
    }


def _phase_b_complete_user_resource(
    boundary: CloudSqlTemporaryAdmin,
    username: str,
    *,
    expected_roles: list[str],
) -> Mapping[str, Any]:
    if username not in {"muncho_canary_writer_login", "postgres"}:
        raise OwnerLauncherError("phase_b_cloud_user_target_invalid")
    encoded = urllib.parse.quote(username, safe="")
    query = urllib.parse.urlencode({"host": ""})
    url = f"{boundary._users_url}/{encoded}?{query}"

    def one() -> Mapping[str, Any]:
        before = boundary._stable_instance_operations()
        names = boundary._user_names(exact_admin_username=username)
        if username not in names:
            raise OwnerLauncherError("phase_b_cloud_user_missing")
        raw = boundary._client.request_json("GET", url)
        after = boundary._stable_instance_operations()
        if before != after:
            raise OwnerLauncherError("cloud_sql_operations_evidence_incomplete")
        roles = raw.get("databaseRoles", [])
        normalized = {
            "databaseRoles": roles,
            "etag": raw.get("etag"),
            "host": raw.get("host", ""),
            "instance": raw.get("instance"),
            "name": raw.get("name"),
            "project": raw.get("project"),
            "type": raw.get("type", "BUILT_IN"),
        }
        if (
            normalized["databaseRoles"] != expected_roles
            or not isinstance(normalized["etag"], str)
            or not normalized["etag"]
            or normalized["host"] != ""
            or normalized["instance"] != SQL_INSTANCE
            or normalized["name"] != username
            or normalized["project"] != PROJECT
            or normalized["type"] != "BUILT_IN"
            or "password" in raw
        ):
            raise OwnerLauncherError("phase_b_cloud_user_invalid")
        return normalized

    first = one()
    second = one()
    if first != second:
        raise OwnerLauncherError("phase_b_cloud_user_drifted")
    return second


def _phase_b_terminal_cloud_observation(
    boundary: CloudSqlTemporaryAdmin,
    bootstrap: CloudSqlCanaryBootstrapLogin,
    *,
    temporary_admin_username: str,
    expected_bootstrap_resource: Mapping[str, Any],
) -> Mapping[str, Any]:
    first_rows = _phase_b_operation_rows(boundary)
    names = sorted(boundary._user_names())
    bootstrap_resource = bootstrap.describe()
    if not isinstance(bootstrap_resource, Mapping):
        raise OwnerLauncherError("phase_b_terminal_cloud_invalid")
    writer_resource = _phase_b_complete_user_resource(
        boundary,
        "muncho_canary_writer_login",
        expected_roles=[],
    )
    postgres_resource = _phase_b_complete_user_resource(
        boundary,
        "postgres",
        expected_roles=[],
    )
    second_rows = _phase_b_operation_rows(boundary)
    if (
        first_rows != second_rows
        or bootstrap_resource != expected_bootstrap_resource
        or names
        != sorted([
            "muncho_canary_writer_login",
            "postgres",
            CANARY_BOOTSTRAP_LOGIN,
        ])
        or any(_ADMIN_USERNAME.fullmatch(name) for name in names)
    ):
        raise OwnerLauncherError("phase_b_terminal_cloud_invalid")
    inventory = sorted(
        [writer_resource, postgres_resource, dict(bootstrap_resource)],
        key=lambda item: str(item["name"]),
    )
    return {
        "project": PROJECT,
        "instance": SQL_INSTANCE,
        "bootstrap_resource": dict(bootstrap_resource),
        "temporary_admin_absent": True,
        "temporary_admin_username_sha256": _sha256(
            temporary_admin_username.encode("ascii")
        ),
        "user_inventory": inventory,
        "user_inventory_sha256": _sha256(_canonical_bytes(inventory)),
        "user_operations_quiescent": True,
        "relevant_user_operations": second_rows,
        "operation_ledger_sha256": _sha256(_canonical_bytes(second_rows)),
        "observed_at_unix": int(time.time()),
    }


def _phase_b_owner_source_directory(plan_sha256: str) -> int:
    _require_sha256(plan_sha256, "phase_b_owner_source_path_invalid")
    hermes_root = PHASE_B_OWNER_APPROVAL_ROOT.parents[1]
    try:
        root_item = hermes_root.lstat()
        if (
            hermes_root.resolve(strict=True) != hermes_root
            or not stat.S_ISDIR(root_item.st_mode)
            or stat.S_ISLNK(root_item.st_mode)
            or root_item.st_uid != os.geteuid()  # windows-footgun: ok — macOS/Linux owner boundary
            or stat.S_IMODE(root_item.st_mode) & 0o077
        ):
            raise OwnerLauncherError("phase_b_owner_source_root_untrusted")
        descriptor = os.open(
            hermes_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        for component in ("owner-approvals", "phase-b", plan_sha256):
            try:
                os.mkdir(component, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            item = os.fstat(child)
            if (
                not stat.S_ISDIR(item.st_mode)
                or item.st_uid != os.geteuid()  # windows-footgun: ok — macOS/Linux owner boundary
                or stat.S_IMODE(item.st_mode) != 0o700
            ):
                os.close(child)
                raise OwnerLauncherError("phase_b_owner_source_root_untrusted")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OwnerLauncherError:
        raise
    except OSError:
        raise OwnerLauncherError("phase_b_owner_source_root_untrusted") from None


def _phase_b_owner_source_receipt(
    *,
    plan_sha256: str,
    sequence: int,
    candidate: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    if type(sequence) is not int or not 0 <= sequence <= 32:
        raise OwnerLauncherError("phase_b_owner_source_path_invalid")
    directory = _phase_b_owner_source_directory(plan_sha256)
    name = f"{sequence:08d}.source.json"
    try:
        try:
            before = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            before = None
        if before is not None:
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or before.st_uid != os.geteuid()  # windows-footgun: ok — macOS/Linux owner boundary
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o400
                or not 1 <= before.st_size <= 64 * 1024
            ):
                raise OwnerLauncherError("phase_b_owner_source_untrusted")
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory,
            )
            try:
                opened = os.fstat(descriptor)
                raw = os.read(descriptor, 64 * 1024 + 1)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            reachable = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if (
                len(raw) != before.st_size
                or _phase_b_stat_identity(before)
                != _phase_b_stat_identity(opened)
                or _phase_b_stat_identity(before)
                != _phase_b_stat_identity(after)
                or _phase_b_stat_identity(before)
                != _phase_b_stat_identity(reachable)
                or not raw.endswith(b"\n")
            ):
                raise OwnerLauncherError("phase_b_owner_source_untrusted")
            loaded = _decode_json_object(raw[:-1], maximum=64 * 1024)
            if raw != _canonical_bytes(loaded) + b"\n":
                raise OwnerLauncherError("phase_b_owner_source_untrusted")
            if candidate is not None and dict(loaded) != dict(candidate):
                raise OwnerLauncherError("phase_b_owner_source_conflict")
            return copy.deepcopy(dict(loaded))
        if candidate is None:
            return None
        payload = _canonical_bytes(candidate) + b"\n"
        if len(payload) > 64 * 1024:
            raise OwnerLauncherError("phase_b_owner_source_oversized")
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
            dir_fd=directory,
        )
        try:
            os.fchmod(descriptor, 0o400)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("short owner source write")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(directory)
        return _phase_b_owner_source_receipt(
            plan_sha256=plan_sha256,
            sequence=sequence,
            candidate=candidate,
        )
    except OwnerLauncherError:
        raise
    except OSError:
        raise OwnerLauncherError("phase_b_owner_source_write_failed") from None
    finally:
        os.close(directory)


def _author_phase_b_signed_pair(
    *,
    plan: Any,
    approval_source_sha256: str,
    purpose: str,
    sequence: int,
    previous_approval_sha256: str | None,
    issued_at_unix: int,
    expires_at_unix: int,
    signer: _PhaseBOwnerExternalSigner,
    owner_authority: _PhaseBOwnerPublicAuthority,
    existing_source: Mapping[str, Any] | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    from gateway import canonical_writer_foundation_phase_b as phase_b

    if (
        purpose != ("initial_apply" if sequence == 0 else "resume_incomplete")
        or (sequence == 0 and previous_approval_sha256 is not None)
        or (sequence > 0 and not isinstance(previous_approval_sha256, str))
        or not issued_at_unix < expires_at_unix
        or expires_at_unix - issued_at_unix > 3600
        or plan.value["owner_resume_public_key_ed25519_hex"]
        != owner_authority.public_key_ed25519_hex
        or plan.value["owner_resume_key_id"] != owner_authority.key_id
        or plan.value["owner_resume_public_key_file_sha256"]
        != owner_authority.public_key_file_sha256
    ):
        raise OwnerLauncherError("phase_b_owner_approval_invalid")
    if previous_approval_sha256 is not None:
        _require_sha256(
            previous_approval_sha256,
            "phase_b_owner_approval_invalid",
        )
    source_common = {
        "schema": phase_b.PHASE_B_SOURCE_AUTH_SCHEMA,
        "authority_kind": PHASE_B_OWNER_AUTHORITY_KIND,
        "purpose": purpose,
        "sequence": sequence,
        "previous_approval_sha256": previous_approval_sha256,
        "plan_sha256": plan.sha256,
        "intent_sha256": plan.value["intent_sha256"],
        "owner_subject_sha256": plan.owner_subject_sha256,
        "approval_source_sha256": approval_source_sha256,
        "owner_key_id": owner_authority.key_id,
        "requested_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
    }
    if existing_source is None:
        source_template = {
            **source_common,
            "nonce_sha256": _sha256(secrets.token_bytes(32)),
            "signature_sshsig": "",
            "receipt_sha256": "0" * 64,
        }
        source_signature = signer.sign(
            phase_b.phase_b_source_authentication_signature_payload(
                source_template
            ),
            namespace=PHASE_B_SOURCE_AUTH_SSHSIG_NAMESPACE,
            expected_authority=owner_authority,
        )
        source_unsigned = {
            **{
                key: copy.deepcopy(value)
                for key, value in source_template.items()
                if key != "receipt_sha256"
            },
            "signature_sshsig": source_signature,
        }
        source = {
            **source_unsigned,
            "receipt_sha256": _sha256(_canonical_bytes(source_unsigned)),
        }
    else:
        source = copy.deepcopy(dict(existing_source))
        if any(source.get(key) != value for key, value in source_common.items()):
            raise OwnerLauncherError("phase_b_owner_source_conflict")
    approval_template = {
        "schema": phase_b.PHASE_B_APPROVAL_SCHEMA,
        "purpose": purpose,
        "sequence": sequence,
        "previous_approval_sha256": previous_approval_sha256,
        "plan_sha256": plan.sha256,
        "intent_sha256": plan.value["intent_sha256"],
        "owner_subject_sha256": plan.owner_subject_sha256,
        "approval_source_sha256": approval_source_sha256,
        "owner_public_key_ed25519_hex": owner_authority.public_key_ed25519_hex,
        "owner_key_id": owner_authority.key_id,
        "owner_public_key_file_sha256": owner_authority.public_key_file_sha256,
        "source_authentication_sha256": source["receipt_sha256"],
        "approved": True,
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
        "secret_material_recorded": False,
        "signature_sshsig": "",
        "approval_sha256": "0" * 64,
    }
    approval_signature = signer.sign(
        phase_b.phase_b_approval_signature_payload(approval_template),
        namespace=PHASE_B_APPROVAL_SSHSIG_NAMESPACE,
        expected_authority=owner_authority,
    )
    approval_unsigned = {
        **{
            key: copy.deepcopy(value)
            for key, value in approval_template.items()
            if key != "approval_sha256"
        },
        "signature_sshsig": approval_signature,
    }
    try:
        approval = phase_b.PhaseBApproval.from_mapping(
            {
                **approval_unsigned,
                "approval_sha256": _sha256(
                    _canonical_bytes(approval_unsigned)
                ),
            },
            plan=plan,
            now_unix=issued_at_unix,
        )
        validated_source = phase_b.validate_phase_b_source_authentication(
            source,
            plan=plan,
            approval=approval,
        )
    except (phase_b.PhaseBError, TypeError, ValueError) as exc:
        raise OwnerLauncherError("phase_b_owner_approval_invalid") from exc
    return approval, validated_source


def _read_phase_b_owner_source_at(
    directory: int,
    name: str,
) -> Mapping[str, Any]:
    try:
        before = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != os.geteuid()  # windows-footgun: ok — macOS/Linux owner boundary
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o400
            or not 1 <= before.st_size <= 64 * 1024
        ):
            raise OwnerLauncherError("phase_b_owner_source_untrusted")
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory,
        )
        try:
            opened = os.fstat(descriptor)
            raw = os.read(descriptor, 64 * 1024 + 1)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        reachable = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except OwnerLauncherError:
        raise
    except OSError:
        raise OwnerLauncherError("phase_b_owner_source_untrusted") from None
    if (
        len(raw) != before.st_size
        or _phase_b_stat_identity(before) != _phase_b_stat_identity(opened)
        or _phase_b_stat_identity(before) != _phase_b_stat_identity(after)
        or _phase_b_stat_identity(before) != _phase_b_stat_identity(reachable)
        or not raw.endswith(b"\n")
    ):
        raise OwnerLauncherError("phase_b_owner_source_untrusted")
    value = _decode_json_object(raw[:-1], maximum=64 * 1024)
    if raw != _canonical_bytes(value) + b"\n":
        raise OwnerLauncherError("phase_b_owner_source_untrusted")
    return copy.deepcopy(dict(value))


def _phase_b_owner_resume_source_receipts(
    *,
    plan_sha256: str,
    sequence: int,
    candidate: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Append/list signed resume attempts without overwriting stale attempts."""

    if type(sequence) is not int or not 1 <= sequence <= 32:
        raise OwnerLauncherError("phase_b_owner_source_path_invalid")
    directory = _phase_b_owner_source_directory(plan_sha256)
    pattern = re.compile(
        rf"^{sequence:08d}\.([0-9a-f]{{64}})\.source\.json$"
    )
    try:
        names = sorted(name for name in os.listdir(directory) if pattern.fullmatch(name))
        if candidate is not None:
            receipt_sha = _require_sha256(
                candidate.get("receipt_sha256"),
                "phase_b_owner_source_invalid",
            )
            name = f"{sequence:08d}.{receipt_sha}.source.json"
            if name not in names:
                payload = _canonical_bytes(candidate) + b"\n"
                if len(payload) > 64 * 1024:
                    raise OwnerLauncherError("phase_b_owner_source_oversized")
                descriptor = os.open(
                    name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o400,
                    dir_fd=directory,
                )
                try:
                    os.fchmod(descriptor, 0o400)
                    offset = 0
                    while offset < len(payload):
                        written = os.write(descriptor, payload[offset:])
                        if written <= 0:
                            raise OSError("short owner source write")
                        offset += written
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                os.fsync(directory)
                names.append(name)
                names.sort()
        values = tuple(_read_phase_b_owner_source_at(directory, name) for name in names)
        if candidate is not None and not any(
            dict(value) == dict(candidate) for value in values
        ):
            raise OwnerLauncherError("phase_b_owner_source_write_failed")
        return values
    except OwnerLauncherError:
        raise
    except OSError:
        raise OwnerLauncherError("phase_b_owner_source_write_failed") from None
    finally:
        os.close(directory)


class _PhaseBOwnerProtocol:
    """Owner half of the exact, bounded Phase-B operation state machine."""

    _MAX_COUNTS = {
        "authority_observe_initial": 1,
        "authority_approve": 1,
        # At most one expired source-only crash residue may be completed as
        # history, followed immediately by one fresh successor.  Neither step
        # is repeatable beyond this exact two-frame recovery shape.
        "authority_resume_approve": 2,
        "observe_initial": 1,
        "observe_recovery": 1,
        "temporary_create_or_rotate": 4,
        "temporary_delete": 1,
        "bootstrap_describe": 1,
        "bootstrap_create_or_rotate": 4,
        "observe_terminal": 1,
    }
    _ALLOWED_AFTER = {
        None: {
            "authority_observe_initial",
            "authority_resume_approve",
            "observe_initial",
            "observe_recovery",
        },
        "authority_observe_initial": {"authority_approve"},
        "authority_approve": {"observe_initial", "observe_recovery"},
        "authority_resume_approve": {
            "authority_resume_approve",
            "observe_initial",
            "observe_recovery",
        },
        "observe_initial": {
            "temporary_create_or_rotate",
            "bootstrap_create_or_rotate",
            "bootstrap_describe",
        },
        "observe_recovery": {
            "temporary_create_or_rotate",
            "bootstrap_create_or_rotate",
            "bootstrap_describe",
        },
        "temporary_create_or_rotate": {
            "temporary_create_or_rotate",
            "bootstrap_create_or_rotate",
            "temporary_delete",
        },
        "bootstrap_create_or_rotate": {
            "bootstrap_create_or_rotate",
            "temporary_create_or_rotate",
            "bootstrap_describe",
        },
        "temporary_delete": {"bootstrap_describe"},
        "bootstrap_describe": {"observe_terminal"},
        "observe_terminal": set(),
    }

    def __init__(
        self,
        *,
        gate: Mapping[str, Any],
        cloud_sql_client: GoogleRestClient,
        password_factory: Callable[[], bytearray] | None = None,
        clock: Callable[[], float] = time.time,
        authority_guard: Callable[[], None] = lambda: None,
        owner_signer: _PhaseBOwnerExternalSigner | None = None,
    ) -> None:
        if not callable(authority_guard):
            raise OwnerLauncherError("invalid_phase_b_authority_guard")
        self._gate = dict(gate)
        self._client = cloud_sql_client
        # Resolve the secret generator at construction time.  The helper is
        # intentionally colocated with the other secret functions later in
        # this module, so a class-definition default would make import order a
        # hidden authority dependency.
        self._password_factory = password_factory or _new_admin_password
        self._clock = clock
        self._authority_guard = authority_guard
        self._owner_signer = owner_signer or _PhaseBOwnerExternalSigner()
        self._owner_resume_authority: _PhaseBOwnerPublicAuthority | None = None
        self._context: Mapping[str, Any] | None = None
        self._context_sha256: str | None = None
        self._sequence = 0
        self._previous_response_sha256: str | None = None
        self._previous_operation: str | None = None
        self._counts = {name: 0 for name in _PHASE_B_OPERATIONS}
        self._plan: Any = None
        self._approval: Any = None
        self._initial_cloud: Mapping[str, Any] | None = None
        self._temporary: dict[int, dict[str, Any]] = {}
        self._bootstrap: dict[int, dict[str, Any]] = {}

    def _validate_context(
        self,
        value: Any,
        *,
        allow_unbound_owner_authority: bool,
    ) -> Mapping[str, Any]:
        if not isinstance(value, Mapping) or set(value) != _PHASE_B_AUTHORITY_CONTEXT_KEYS:
            raise OwnerLauncherError("invalid_phase_b_authority_context")
        context = copy.deepcopy(dict(value))
        if (
            context["release_sha"] != self._gate["release_sha"]
            or context["coordinator_input_sha256"]
            != self._gate["coordinator_input_sha256"]
            or context["owner_subject_sha256"]
            != self._gate["owner_subject_sha256"]
            or context["approval_source_sha256"]
            != self._gate["approval_source_sha256"]
        ):
            raise OwnerLauncherError("invalid_phase_b_authority_context")
        owner_authority_names = (
            "owner_resume_public_key_ed25519_hex",
            "owner_resume_key_id",
            "owner_resume_public_key_file_sha256",
            "owner_resume_public_fingerprint",
        )
        sources = context["authority_sources"]
        if not isinstance(sources, Mapping):
            raise OwnerLauncherError("invalid_phase_b_authority_context")
        owner_source = sources.get("owner_resume_public_key")
        owner_unbound = (
            all(context[name] is None for name in owner_authority_names)
            and owner_source is None
        )
        if owner_unbound is not allow_unbound_owner_authority:
            raise OwnerLauncherError("invalid_phase_b_authority_context")
        for name, item in context.items():
            if (
                name.endswith("_sha256")
                and name not in {"authority_sources_sha256"}
                and item is not None
            ):
                _require_sha256(item, "invalid_phase_b_authority_context")
        for name in (
            "activation_approval_issued_at_unix",
            "activation_approval_expires_at_unix",
            "native_approval_issued_at_unix",
            "native_approval_expires_at_unix",
        ):
            if type(context[name]) is not int or context[name] < 0:
                raise OwnerLauncherError("invalid_phase_b_authority_context")
        if (
            set(sources) != _PHASE_B_AUTHORITY_SOURCE_LABELS
            or context["authority_sources_sha256"]
            != _sha256(_canonical_bytes(sources))
        ):
            raise OwnerLauncherError("invalid_phase_b_authority_context")
        for label, source in sources.items():
            if label == "owner_resume_public_key" and owner_unbound:
                continue
            if (
                not isinstance(source, Mapping)
                or set(source) != _PHASE_B_AUTHORITY_SOURCE_KEYS
                or not isinstance(source["path"], str)
                or not os.path.isabs(source["path"])
                or "\x00" in source["path"]
                or not isinstance(source["mode"], str)
                or re.fullmatch(r"0[0-7]{3}", source["mode"]) is None
                or any(
                    type(source[name]) is not int or source[name] < 0
                    for name in ("device", "inode", "uid", "gid", "size")
                )
            ):
                raise OwnerLauncherError("invalid_phase_b_authority_context")
            _require_sha256(
                source["file_sha256"],
                "invalid_phase_b_authority_context",
            )
        if not owner_unbound:
            public_hex = context["owner_resume_public_key_ed25519_hex"]
            if (
                not isinstance(public_hex, str)
                or re.fullmatch(r"[0-9a-f]{64}", public_hex) is None
                or context["owner_resume_key_id"]
                != _sha256(bytes.fromhex(public_hex))
                or context["owner_resume_public_key_file_sha256"]
                != owner_source["file_sha256"]
                or owner_source["path"] != str(PHASE_B_OWNER_PUBLIC_KEY_PATH)
                or owner_source["uid"] != PHASE_B_OWNER_PUBLIC_KEY_UID
                or owner_source["gid"] != PHASE_B_OWNER_PUBLIC_KEY_GID
                or owner_source["mode"] != "0600"
                or not 1 <= owner_source["size"] <= PHASE_B_MAX_PUBLIC_KEY_BYTES
            ):
                raise OwnerLauncherError("invalid_phase_b_authority_context")
            public_blob = _ssh_wire_string(b"ssh-ed25519") + _ssh_wire_string(
                bytes.fromhex(public_hex)
            )
            fingerprint = "SHA256:" + base64.b64encode(
                hashlib.sha256(public_blob).digest()
            ).decode("ascii").rstrip("=")
            if (
                context["owner_resume_public_fingerprint"] != fingerprint
                or fingerprint != PHASE_B_OWNER_PUBLIC_KEY_FINGERPRINT
            ):
                raise OwnerLauncherError("invalid_phase_b_authority_context")
        expected_paths = {
            "coordinator_input": "/etc/muncho/full-canary/coordinator-input.json",
            "activation_plan": "/etc/muncho/writer-activation/activation-plan.json",
            "activation_receipt": (
                "/var/lib/muncho-writer-activation/plans/"
                f"{context['release_sha']}/{context['activation_plan_sha256']}"
                "/success/activation.json"
            ),
            "activation_owner_approval": (
                "/etc/muncho/writer-activation/approvals/activation/"
                f"{context['activation_plan_sha256']}/"
                f"{context['activation_owner_approval_sha256']}.json"
            ),
            "native_plan": (
                "/etc/muncho/writer-activation/native-observation-plan.json"
            ),
            "native_receipt": (
                "/var/lib/muncho-writer-canary-evidence/"
                f"{context['release_sha']}/"
                f"{context['native_observation_plan_sha256']}"
                "/native-observation.json"
            ),
            "native_owner_approval": (
                "/etc/muncho/writer-activation/approvals/native_observation/"
                f"{context['native_observation_plan_sha256']}/"
                f"{context['native_observation_approval_sha256']}.json"
            ),
            "external_iam_receipt": (
                "/var/lib/muncho-writer-canary-evidence/"
                f"{context['release_sha']}/"
                f"{context['native_observation_plan_sha256']}/external-iam/"
                f"{context['external_iam_receipt_sha256']}.json"
            ),
            "config_collector_receipt": (
                "/var/lib/muncho-writer-canary-evidence/config-collector/"
                f"{context['release_sha']}/"
                f"{context['config_collector_receipt_sha256']}.json"
            ),
            "gateway_config_intent": (
                "/etc/muncho/full-canary/staged/gateway.yaml"
            ),
            "edge_config_intent": (
                "/etc/muncho/full-canary/staged/discord-edge.json"
            ),
            "fixture_intent": "/etc/muncho/full-canary/fixture.json",
            "host_identity_receipt": (
                "/etc/muncho/full-canary/host-identity.json"
            ),
        }
        if any(
            sources[label]["path"] != expected_path
            for label, expected_path in expected_paths.items()
        ) or any(
            sources[label]["file_sha256"] != context[digest_name]
            for label, digest_name in (
                ("gateway_config_intent", "gateway_config_intent_sha256"),
                ("edge_config_intent", "edge_config_intent_sha256"),
            )
        ):
            raise OwnerLauncherError("invalid_phase_b_authority_context")
        if not (
            context["activation_approval_issued_at_unix"]
            < context["activation_approval_expires_at_unix"]
            and context["native_approval_issued_at_unix"]
            < context["native_approval_expires_at_unix"]
        ):
            raise OwnerLauncherError("invalid_phase_b_authority_context")
        _phase_b_secret_free(context)
        return context

    def validate_request(self, request: Any) -> Mapping[str, Any]:
        if not isinstance(request, Mapping) or set(request) != _PHASE_B_REQUEST_KEYS:
            raise OwnerLauncherError("invalid_phase_b_request")
        value = copy.deepcopy(dict(request))
        unsigned = {key: item for key, item in value.items() if key != "request_sha256"}
        operation = value["operation"]
        if (
            value["schema"] != PHASE_B_OWNER_REQUEST_SCHEMA
            or value["frame_schema"] != PHASE_B_OWNER_FRAME_SCHEMA
            or operation not in _PHASE_B_OPERATIONS
            or value["sequence"] != self._sequence
            or value["previous_response_sha256"] != self._previous_response_sha256
            or value["request_sha256"] != _sha256(_canonical_bytes(unsigned))
            or type(value["issued_at_unix"]) is not int
            or type(value["expires_at_unix"]) is not int
            or not value["issued_at_unix"] <= int(self._clock()) < value[
                "expires_at_unix"
            ]
            or value["expires_at_unix"] > self._gate["expires_at_unix"]
            or operation not in self._ALLOWED_AFTER[self._previous_operation]
            or self._counts[operation] >= self._MAX_COUNTS[operation]
            or type(value["credential_expected"]) is not bool
            or value["credential_expected"]
            is not (operation in {
                "temporary_create_or_rotate",
                "bootstrap_create_or_rotate",
            })
            or not isinstance(value["payload"], Mapping)
        ):
            raise OwnerLauncherError("invalid_phase_b_request")
        context = self._validate_context(
            value["authority_context"],
            allow_unbound_owner_authority=(
                operation == "authority_observe_initial"
                and self._context is None
            ),
        )
        context_sha = _require_sha256(
            value["authority_context_sha256"],
            "invalid_phase_b_request",
        )
        if context_sha != _sha256(_canonical_bytes(context)):
            raise OwnerLauncherError("invalid_phase_b_request")
        if self._context is None:
            self._context = context
            self._context_sha256 = context_sha
        elif context != self._context or context_sha != self._context_sha256:
            if (
                operation != "authority_approve"
                or self._previous_operation != "authority_observe_initial"
                or self._owner_resume_authority is None
            ):
                raise OwnerLauncherError("phase_b_authority_context_changed")
            expected = copy.deepcopy(dict(self._context))
            authority = self._owner_resume_authority.to_mapping()
            expected_sources = copy.deepcopy(dict(expected["authority_sources"]))
            expected_sources["owner_resume_public_key"] = authority[
                "public_key_source"
            ]
            expected.update({
                "owner_resume_public_key_ed25519_hex": authority[
                    "public_key_ed25519_hex"
                ],
                "owner_resume_key_id": authority["key_id"],
                "owner_resume_public_key_file_sha256": authority[
                    "public_key_file_sha256"
                ],
                "owner_resume_public_fingerprint": authority[
                    "public_fingerprint"
                ],
                "authority_sources": dict(sorted(expected_sources.items())),
            })
            expected["authority_sources_sha256"] = _sha256(
                _canonical_bytes(expected["authority_sources"])
            )
            if context != expected:
                raise OwnerLauncherError("phase_b_authority_context_changed")
            self._context = context
            self._context_sha256 = context_sha
        expected_boundary = (
            "temporary"
            if operation.startswith("temporary_")
            else "bootstrap"
            if operation.startswith("bootstrap_")
            else None
        )
        if (
            value["boundary_kind"] != expected_boundary
            or (
                expected_boundary is None
                and value["boundary_ordinal"] is not None
                or expected_boundary is not None
                and (
                    type(value["boundary_ordinal"]) is not int
                    or not 0 <= value["boundary_ordinal"] <= 7
                )
            )
        ):
            raise OwnerLauncherError("invalid_phase_b_request")
        idempotency_projection = {
            "authority_context_sha256": context_sha,
            "phase_b_plan_sha256": value["phase_b_plan_sha256"],
            "phase_b_approval_sha256": value["phase_b_approval_sha256"],
            "operation": operation,
            "sequence": value["sequence"],
            "previous_response_sha256": value["previous_response_sha256"],
            "payload": value["payload"],
        }
        if value["idempotency_key"] != _sha256(
            _canonical_bytes(idempotency_projection)
        ):
            raise OwnerLauncherError("invalid_phase_b_request")
        for name in ("phase_b_plan_sha256", "phase_b_approval_sha256"):
            if value[name] is not None:
                _require_sha256(value[name], "invalid_phase_b_request")
        _phase_b_secret_free(value)
        self._counts[operation] += 1
        return value

    def _observer(self) -> CloudSqlTemporaryAdmin:
        return CloudSqlTemporaryAdmin(self._client)

    def _bootstrap_boundary(self, ordinal: int) -> CloudSqlCanaryBootstrapLogin:
        if self._plan is None:
            raise OwnerLauncherError("phase_b_plan_missing")
        state = self._bootstrap.get(ordinal)
        if state is None:
            state = {
                "boundary": CloudSqlCanaryBootstrapLogin(
                    self._client,
                    expected_owner_subject_sha256=self._plan.owner_subject_sha256,
                    expected_mutation_context_sha256=self._plan.sha256,
                ),
                "successful_mutations": 0,
            }
            self._bootstrap[ordinal] = state
        return state["boundary"]

    def _adopt_plan_approval(self, payload: Mapping[str, Any]) -> None:
        from gateway import canonical_writer_foundation_phase_b as phase_b

        if set(payload) != {"phase_b_plan", "phase_b_approval"}:
            raise OwnerLauncherError("invalid_phase_b_observation_request")
        plan = phase_b.PhaseBPlan.from_mapping(payload["phase_b_plan"])
        approval = phase_b.PhaseBApproval.from_mapping(
            payload["phase_b_approval"],
            plan=plan,
            now_unix=int(self._clock()),
        )
        if (
            plan.owner_subject_sha256 != self._context["owner_subject_sha256"]
            or approval.value["approval_source_sha256"]
            != self._context["approval_source_sha256"]
        ):
            raise OwnerLauncherError("invalid_phase_b_observation_request")
        if self._plan is not None and self._plan.to_mapping() != plan.to_mapping():
            raise OwnerLauncherError("phase_b_plan_changed")
        if self._approval is not None and self._approval.to_mapping() != approval.to_mapping():
            raise OwnerLauncherError("phase_b_approval_changed")
        self._plan = plan
        self._approval = approval

    def execute(
        self,
        request: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], bytearray | None, bool]:
        from gateway import canonical_writer_foundation_phase_b as phase_b

        operation = request["operation"]
        payload = request["payload"]
        ordinal = request["boundary_ordinal"]
        self._authority_guard()
        if operation == "authority_observe_initial":
            if set(payload) != {"local_preflight"} or self._initial_cloud is not None:
                raise OwnerLauncherError("invalid_phase_b_authority_request")
            local = payload["local_preflight"]
            if not isinstance(local, Mapping):
                raise OwnerLauncherError("invalid_phase_b_authority_request")
            self._owner_resume_authority = self._owner_signer.inspect()
            self._initial_cloud = _phase_b_initial_cloud_observation(self._observer())
            return {
                "cloud_observation": self._initial_cloud,
                "owner_resume_authority": (
                    self._owner_resume_authority.to_mapping()
                ),
            }, None, False
        if operation == "authority_approve":
            if (
                set(payload) != {"phase_b_plan"}
                or self._initial_cloud is None
                or self._owner_resume_authority is None
            ):
                raise OwnerLauncherError("invalid_phase_b_authority_request")
            plan = phase_b.PhaseBPlan.from_mapping(payload["phase_b_plan"])
            owner_authority = self._owner_resume_authority
            if (
                plan.owner_subject_sha256 != self._context["owner_subject_sha256"]
                or plan.revision != self._context["release_sha"]
                or plan.preflight.value["cloud_sql"] != self._initial_cloud
                or plan.value["owner_resume_public_key_ed25519_hex"]
                != owner_authority.public_key_ed25519_hex
                or plan.value["owner_resume_key_id"] != owner_authority.key_id
                or plan.value["owner_resume_public_key_file_sha256"]
                != owner_authority.public_key_file_sha256
                or request["phase_b_plan_sha256"] != plan.sha256
                or request["phase_b_approval_sha256"] is not None
            ):
                raise OwnerLauncherError("invalid_phase_b_authority_request")
            now = int(self._clock())
            expires = min(request["expires_at_unix"], now + 3600)
            source_template = {
                "schema": phase_b.PHASE_B_SOURCE_AUTH_SCHEMA,
                "authority_kind": PHASE_B_OWNER_AUTHORITY_KIND,
                "purpose": "initial_apply",
                "sequence": 0,
                "previous_approval_sha256": None,
                "plan_sha256": plan.sha256,
                "intent_sha256": plan.value["intent_sha256"],
                "owner_subject_sha256": plan.owner_subject_sha256,
                "approval_source_sha256": self._context[
                    "approval_source_sha256"
                ],
                "owner_key_id": owner_authority.key_id,
                "requested_at_unix": now,
                "expires_at_unix": expires,
                "nonce_sha256": _sha256(secrets.token_bytes(32)),
                "signature_sshsig": "",
                "receipt_sha256": "0" * 64,
            }
            try:
                source_signature_payload = (
                    phase_b.phase_b_source_authentication_signature_payload(
                        source_template
                    )
                )
            except (phase_b.PhaseBError, TypeError, ValueError) as exc:
                raise OwnerLauncherError(
                    "phase_b_owner_source_auth_invalid"
                ) from exc
            source_signature = self._owner_signer.sign(
                source_signature_payload,
                namespace=PHASE_B_SOURCE_AUTH_SSHSIG_NAMESPACE,
                expected_authority=owner_authority,
            )
            source_unsigned = {
                **{
                    key: copy.deepcopy(value)
                    for key, value in source_template.items()
                    if key != "receipt_sha256"
                },
                "signature_sshsig": source_signature,
            }
            source_authentication = {
                **source_unsigned,
                "receipt_sha256": _sha256(_canonical_bytes(source_unsigned)),
            }
            approval_template = {
                "schema": phase_b.PHASE_B_APPROVAL_SCHEMA,
                "purpose": "initial_apply",
                "sequence": 0,
                "previous_approval_sha256": None,
                "plan_sha256": plan.sha256,
                "intent_sha256": plan.value["intent_sha256"],
                "owner_subject_sha256": plan.owner_subject_sha256,
                "approval_source_sha256": self._context["approval_source_sha256"],
                "owner_public_key_ed25519_hex": (
                    owner_authority.public_key_ed25519_hex
                ),
                "owner_key_id": owner_authority.key_id,
                "owner_public_key_file_sha256": (
                    owner_authority.public_key_file_sha256
                ),
                "source_authentication_sha256": source_authentication[
                    "receipt_sha256"
                ],
                "approved": True,
                "issued_at_unix": now,
                "expires_at_unix": expires,
                "secret_material_recorded": False,
                "signature_sshsig": "",
                "approval_sha256": "0" * 64,
            }
            try:
                approval_signature_payload = (
                    phase_b.phase_b_approval_signature_payload(
                        approval_template
                    )
                )
            except (phase_b.PhaseBError, TypeError, ValueError) as exc:
                raise OwnerLauncherError("phase_b_owner_approval_invalid") from exc
            approval_signature = self._owner_signer.sign(
                approval_signature_payload,
                namespace=PHASE_B_APPROVAL_SSHSIG_NAMESPACE,
                expected_authority=owner_authority,
            )
            approval_unsigned = {
                **{
                    key: copy.deepcopy(value)
                    for key, value in approval_template.items()
                    if key != "approval_sha256"
                },
                "signature_sshsig": approval_signature,
            }
            try:
                approval = phase_b.PhaseBApproval.from_mapping(
                    {
                        **approval_unsigned,
                        "approval_sha256": _sha256(
                            _canonical_bytes(approval_unsigned)
                        ),
                    },
                    plan=plan,
                    now_unix=now,
                )
                validated_source = phase_b.validate_phase_b_source_authentication(
                    source_authentication,
                    plan=plan,
                    approval=approval,
                )
            except (phase_b.PhaseBError, TypeError, ValueError) as exc:
                raise OwnerLauncherError("phase_b_owner_approval_invalid") from exc
            _phase_b_owner_source_receipt(
                plan_sha256=plan.sha256,
                sequence=0,
                candidate=validated_source,
            )
            self._plan = plan
            self._approval = approval
            return {
                "phase_b_approval": approval.to_mapping(),
                "phase_b_approval_source": validated_source,
            }, None, False
        if operation == "authority_resume_approve":
            if set(payload) != {
                "phase_b_plan",
                "phase_b_approval_chain",
                "phase_b_incomplete_head",
            }:
                raise OwnerLauncherError("invalid_phase_b_resume_request")
            plan = phase_b.PhaseBPlan.from_mapping(payload["phase_b_plan"])
            raw_chain = payload["phase_b_approval_chain"]
            inspection = payload["phase_b_incomplete_head"]
            if (
                not isinstance(raw_chain, list)
                or not raw_chain
                or any(not isinstance(item, Mapping) for item in raw_chain)
                or not isinstance(inspection, Mapping)
                or set(inspection) != _PHASE_B_INCOMPLETE_HEAD_KEYS
            ):
                raise OwnerLauncherError("invalid_phase_b_resume_request")
            inspection_unsigned = {
                key: copy.deepcopy(value)
                for key, value in inspection.items()
                if key != "inspection_sha256"
            }
            if inspection["inspection_sha256"] != _sha256(
                _canonical_bytes(inspection_unsigned)
            ):
                raise OwnerLauncherError("invalid_phase_b_resume_request")
            try:
                approvals = phase_b.validate_phase_b_approval_chain(
                    raw_chain,
                    plan=plan,
                )
            except (phase_b.PhaseBError, TypeError, ValueError) as exc:
                raise OwnerLauncherError("invalid_phase_b_resume_request") from exc
            head = approvals[-1]
            if (
                request["phase_b_plan_sha256"] != plan.sha256
                or request["phase_b_approval_sha256"] != head.sha256
                or plan.owner_subject_sha256
                != self._context["owner_subject_sha256"]
                or plan.revision != self._context["release_sha"]
                or head.value["approval_source_sha256"]
                != self._context["approval_source_sha256"]
                or inspection["plan_sha256"] != plan.sha256
                or inspection["intent_sha256"] != plan.value["intent_sha256"]
                or inspection["owner_subject_sha256"]
                != plan.owner_subject_sha256
                or inspection["approval_source_sha256"]
                != head.value["approval_source_sha256"]
                or inspection["approval_sequence"] != head.sequence
                or inspection["approval_sha256"] != head.sha256
                or inspection["terminal"] is not False
                or inspection["resume_eligible"] is not True
                or inspection["requires_reapproval"] is not True
                or inspection["mutation_authorized"] is not False
                or inspection["incomplete_state"]
                not in {
                    "authority_published_no_intent",
                    "journal_incomplete",
                }
            ):
                raise OwnerLauncherError("invalid_phase_b_resume_request")
            owner_authority = self._owner_signer.inspect()
            owner_mapping = owner_authority.to_mapping()
            if (
                plan.value["owner_resume_public_key_ed25519_hex"]
                != owner_authority.public_key_ed25519_hex
                or plan.value["owner_resume_key_id"] != owner_authority.key_id
                or plan.value["owner_resume_public_key_file_sha256"]
                != owner_authority.public_key_file_sha256
                or self._context["owner_resume_public_key_ed25519_hex"]
                != owner_mapping["public_key_ed25519_hex"]
                or self._context["owner_resume_key_id"]
                != owner_mapping["key_id"]
                or self._context["owner_resume_public_key_file_sha256"]
                != owner_mapping["public_key_file_sha256"]
                or self._context["owner_resume_public_fingerprint"]
                != owner_mapping["public_fingerprint"]
                or self._context["authority_sources"][
                    "owner_resume_public_key"
                ]
                != owner_mapping["public_key_source"]
            ):
                raise OwnerLauncherError("invalid_phase_b_resume_request")
            self._owner_resume_authority = owner_authority
            sequence = head.sequence + 1
            pending_sha = inspection[
                "pending_source_authentication_sha256"
            ]
            pending_sequence = inspection["pending_source_sequence"]
            candidates = _phase_b_owner_resume_source_receipts(
                plan_sha256=plan.sha256,
                sequence=sequence,
            )
            existing_source: Mapping[str, Any] | None = None
            if pending_sha is not None:
                if pending_sequence != sequence:
                    raise OwnerLauncherError("invalid_phase_b_resume_request")
                existing_source = next(
                    (
                        source
                        for source in candidates
                        if source.get("receipt_sha256") == pending_sha
                    ),
                    None,
                )
                if existing_source is None:
                    raise OwnerLauncherError(
                        "phase_b_owner_pending_source_missing"
                    )
            elif pending_sequence is not None:
                raise OwnerLauncherError("invalid_phase_b_resume_request")
            else:
                now_unix = int(self._clock())
                existing_source = next(
                    (
                        source
                        for source in reversed(candidates)
                        if source.get("previous_approval_sha256") == head.sha256
                        and source.get("requested_at_unix", now_unix + 1)
                        <= now_unix
                        < source.get("expires_at_unix", -1)
                    ),
                    None,
                )
            if existing_source is None:
                issued_at = int(self._clock())
                expires_at = min(
                    request["expires_at_unix"],
                    issued_at + 3600,
                )
            else:
                issued_at = existing_source.get("requested_at_unix")
                expires_at = existing_source.get("expires_at_unix")
                if type(issued_at) is not int or type(expires_at) is not int:
                    raise OwnerLauncherError("phase_b_owner_source_invalid")
            approval, source = _author_phase_b_signed_pair(
                plan=plan,
                approval_source_sha256=head.value[
                    "approval_source_sha256"
                ],
                purpose="resume_incomplete",
                sequence=sequence,
                previous_approval_sha256=head.sha256,
                issued_at_unix=issued_at,
                expires_at_unix=expires_at,
                signer=self._owner_signer,
                owner_authority=owner_authority,
                existing_source=existing_source,
            )
            _phase_b_owner_resume_source_receipts(
                plan_sha256=plan.sha256,
                sequence=sequence,
                candidate=source,
            )
            self._plan = plan
            self._approval = approval
            return {
                "phase_b_approval": approval.to_mapping(),
                "phase_b_approval_source": source,
            }, None, False
        if operation in {"observe_initial", "observe_recovery"}:
            self._adopt_plan_approval(payload)
            if (
                request["phase_b_plan_sha256"] != self._plan.sha256
                or request["phase_b_approval_sha256"] != self._approval.sha256
            ):
                raise OwnerLauncherError("invalid_phase_b_observation_request")
            if operation == "observe_initial":
                observation = _phase_b_initial_cloud_observation(self._observer())
            else:
                observation = _phase_b_recovery_cloud_observation(
                    self._observer(),
                    self._bootstrap_boundary(7),
                    temporary_admin_username=self._plan.temporary_admin_username,
                )
            return {"cloud_observation": observation}, None, False
        if self._plan is None or self._approval is None:
            raise OwnerLauncherError("phase_b_authority_missing")
        if (
            request["phase_b_plan_sha256"] != self._plan.sha256
            or request["phase_b_approval_sha256"] != self._approval.sha256
        ):
            raise OwnerLauncherError("phase_b_authority_changed")
        if operation == "temporary_create_or_rotate":
            if set(payload) != {
                "username",
                "expected_owner_subject_sha256",
                "expected_mutation_context_sha256",
            } or payload != {
                "username": self._plan.temporary_admin_username,
                "expected_owner_subject_sha256": self._plan.owner_subject_sha256,
                "expected_mutation_context_sha256": self._plan.sha256,
            }:
                raise OwnerLauncherError("invalid_phase_b_temporary_request")
            state = self._temporary.get(ordinal)
            if state is None:
                state = {
                    "boundary": CloudSqlTemporaryAdmin(self._client),
                    "successful_mutations": 0,
                }
                self._temporary[ordinal] = state
            boundary = state["boundary"]
            if boundary.mutation_reconciliation_required():
                boundary.reconcile_ambiguous_mutation_and_confirm_absent(
                    self._plan.temporary_admin_username
                )
                boundary = CloudSqlTemporaryAdmin(self._client)
                state["boundary"] = boundary
            if state["successful_mutations"] != 0:
                raise OwnerLauncherError("phase_b_temporary_mutation_replay")
            secret = self._password_factory()
            if not isinstance(secret, bytearray) or len(secret) != PHASE_B_CREDENTIAL_BYTES:
                _wipe(secret if isinstance(secret, bytearray) else None)
                raise OwnerLauncherError("invalid_phase_b_credential")
            try:
                boundary.begin_mutation_observation(
                    expected_owner_subject_sha256=self._plan.owner_subject_sha256,
                    expected_mutation_context_sha256=self._plan.sha256,
                )
                boundary.create_or_rotate_recovery(
                    self._plan.temporary_admin_username,
                    secret.decode("ascii"),
                )
                authority = boundary.temporary_admin_authority_receipt(
                    self._plan.temporary_admin_username
                )
                self._authority_guard()
            except BaseException:
                _wipe(secret)
                raise
            state["successful_mutations"] += 1
            state["authority"] = authority
            return {"authority_receipt": authority}, secret, False
        if operation == "temporary_delete":
            if set(payload) != {"username", "authority_receipt_sha256"}:
                raise OwnerLauncherError("invalid_phase_b_temporary_request")
            state = self._temporary.get(ordinal)
            if (
                state is None
                or payload["username"] != self._plan.temporary_admin_username
                or not isinstance(state.get("authority"), Mapping)
                or payload["authority_receipt_sha256"]
                != state["authority"].get("receipt_sha256")
            ):
                raise OwnerLauncherError("invalid_phase_b_temporary_request")
            boundary = state["boundary"]
            boundary.delete_and_confirm_absent(self._plan.temporary_admin_username)
            absence = boundary.reconciliation_receipt()
            self._authority_guard()
            return {"absence_receipt": absence}, None, False
        if operation == "bootstrap_create_or_rotate":
            if payload != {}:
                raise OwnerLauncherError("invalid_phase_b_bootstrap_request")
            boundary = self._bootstrap_boundary(ordinal)
            state = self._bootstrap[ordinal]
            if boundary.mutation_reconciliation_required():
                boundary.reconcile_ambiguous_mutation_and_confirm_absent()
                boundary = CloudSqlCanaryBootstrapLogin(
                    self._client,
                    expected_owner_subject_sha256=self._plan.owner_subject_sha256,
                    expected_mutation_context_sha256=self._plan.sha256,
                )
                state["boundary"] = boundary
            if state["successful_mutations"] >= 2:
                raise OwnerLauncherError("phase_b_bootstrap_mutation_replay")
            secret = self._password_factory()
            if not isinstance(secret, bytearray) or len(secret) != PHASE_B_CREDENTIAL_BYTES:
                _wipe(secret if isinstance(secret, bytearray) else None)
                raise OwnerLauncherError("invalid_phase_b_credential")
            try:
                boundary.create_or_rotate_recovery(secret.decode("ascii"))
                authority = boundary.authority_receipt()
                self._authority_guard()
            except BaseException:
                _wipe(secret)
                raise
            state["successful_mutations"] += 1
            state["authority"] = authority
            return {"authority_receipt": authority}, secret, False
        if operation == "bootstrap_describe":
            if payload != {}:
                raise OwnerLauncherError("invalid_phase_b_bootstrap_request")
            return {"resource": self._bootstrap_boundary(ordinal).describe()}, None, False
        if operation == "observe_terminal":
            if set(payload) != {"bootstrap_resource", "absence_receipt"}:
                raise OwnerLauncherError("invalid_phase_b_terminal_request")
            observation = _phase_b_terminal_cloud_observation(
                self._observer(),
                self._bootstrap_boundary(7),
                temporary_admin_username=self._plan.temporary_admin_username,
                expected_bootstrap_resource=payload["bootstrap_resource"],
            )
            return {"cloud_observation": observation}, None, True
        raise AssertionError("unreachable Phase-B owner operation")

    def response_frame(
        self,
        request: Mapping[str, Any],
    ) -> tuple[bytearray, bool]:
        credential: bytearray | None = None
        terminal = False
        try:
            try:
                result, credential, terminal = self.execute(request)
                ok = True
                error_code = None
            except BaseException as exc:
                result = {
                    "mutation_reconciliation_required": bool(
                        request["operation"] in {
                            "temporary_create_or_rotate",
                            "bootstrap_create_or_rotate",
                        }
                        and self._mutation_reconciliation_required(request)
                    )
                }
                ok = False
                error_code = (
                    exc.code
                    if isinstance(exc, OwnerLauncherError)
                    else "phase_b_owner_operation_failed"
                )
            if credential is not None and (
                not ok or len(credential) != PHASE_B_CREDENTIAL_BYTES
            ):
                raise OwnerLauncherError("invalid_phase_b_credential")
            if ok and (credential is not None) is not request["credential_expected"]:
                raise OwnerLauncherError("invalid_phase_b_credential")
            unsigned = {
                "schema": PHASE_B_OWNER_RESPONSE_SCHEMA,
                "frame_schema": PHASE_B_OWNER_FRAME_SCHEMA,
                "ok": ok,
                "operation": request["operation"],
                "sequence": request["sequence"],
                "request_sha256": request["request_sha256"],
                "idempotency_key": request["idempotency_key"],
                "authority_context_sha256": request["authority_context_sha256"],
                "phase_b_plan_sha256": request["phase_b_plan_sha256"],
                "phase_b_approval_sha256": request["phase_b_approval_sha256"],
                "credential_present": credential is not None,
                "credential_length": 0 if credential is None else len(credential),
                "result": copy.deepcopy(dict(result)),
                "error_code": error_code,
                "completed_at_unix": int(self._clock()),
            }
            receipt = {
                **unsigned,
                "response_sha256": _sha256(_canonical_bytes(unsigned)),
            }
            _phase_b_secret_free(receipt)
            encoded = _canonical_bytes(receipt)
            if len(encoded) > PHASE_B_MAX_RESPONSE_BYTES:
                raise OwnerLauncherError("phase_b_owner_response_oversized")
            frame = bytearray(
                PHASE_B_OWNER_FRAME_MAGIC
                + struct.pack(">II", len(encoded), 0 if credential is None else len(credential))
                + encoded
            )
            if credential is not None:
                frame.extend(credential)
            self._sequence += 1
            self._previous_response_sha256 = receipt["response_sha256"]
            self._previous_operation = request["operation"]
            return frame, terminal
        finally:
            _wipe(credential)

    def _mutation_reconciliation_required(
        self,
        request: Mapping[str, Any],
    ) -> bool:
        ordinal = request["boundary_ordinal"]
        state = (
            self._temporary.get(ordinal)
            if request["operation"] == "temporary_create_or_rotate"
            else self._bootstrap.get(ordinal)
        )
        boundary = None if state is None else state.get("boundary")
        return bool(
            boundary is not None
            and boundary.mutation_reconciliation_required()
        )


class DiscordTokenSource(Protocol):
    def read_discord_token(self) -> bytearray: ...


class FinalApprovalSource(Protocol):
    def read_final_approval(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    if type(length) is not int or length < 0:
        raise OwnerLauncherError("invalid_owner_discord_frame")
    chunks = bytearray()
    while len(chunks) < length:
        part = stream.read(length - len(chunks))
        if not isinstance(part, bytes) or not part:
            raise OwnerLauncherError("invalid_owner_discord_frame")
        chunks.extend(part)
    return bytes(chunks)


class OwnerStdinDiscordTokenReader:
    """Read one length-delimited credential from the owner's fixed FD 0."""

    def __init__(self, stream: BinaryIO | None = None) -> None:
        self._stream = sys.stdin.buffer if stream is None else stream

    def read_discord_token(self) -> bytearray:
        if bool(getattr(self._stream, "isatty", lambda: False)()):
            raise OwnerLauncherError("owner_discord_stdin_is_tty")
        header = _read_exact(self._stream, 8)
        if header[:4] != OWNER_DISCORD_INPUT_MAGIC:
            raise OwnerLauncherError("invalid_owner_discord_frame")
        length = struct.unpack(">I", header[4:])[0]
        if not 0 < length <= _DISCORD_TOKEN_MAX_BYTES:
            raise OwnerLauncherError("invalid_owner_discord_frame")
        token = bytearray(_read_exact(self._stream, length))
        trailing = self._stream.read(1)
        if trailing != b"":
            _wipe(token)
            raise OwnerLauncherError("invalid_owner_discord_frame")
        if b"\x00" in token or any(value < 0x20 or value == 0x7F for value in token):
            _wipe(token)
            raise OwnerLauncherError("invalid_owner_discord_frame")
        return token


class OwnerDiscordTokenReader:
    """Select masked TTY input or the exact framed non-TTY FD0 protocol."""

    def __init__(
        self,
        stream: Any | None = None,
        *,
        masked_reader: Callable[[str], str] = getpass.getpass,
    ) -> None:
        self._stream = sys.stdin if stream is None else stream
        self._masked_reader = masked_reader

    def read_discord_token(self) -> bytearray:
        if bool(getattr(self._stream, "isatty", lambda: False)()):
            try:
                raw = self._masked_reader("Canary Discord bot token: ")
            except (EOFError, KeyboardInterrupt, OSError):
                raise OwnerLauncherError("owner_discord_masked_input_failed") from None
            if not isinstance(raw, str):
                raise OwnerLauncherError("owner_discord_masked_input_failed")
            try:
                token = bytearray(raw.encode("utf-8", errors="strict"))
            except UnicodeError:
                raise OwnerLauncherError("invalid_discord_token") from None
            finally:
                raw = ""
            try:
                build_discord_frame(token)
            except BaseException:
                _wipe(token)
                raise
            return token

        framed_stream = getattr(self._stream, "buffer", self._stream)
        return OwnerStdinDiscordTokenReader(framed_stream).read_discord_token()


class FixedLocalFinalApprovalFile:
    """Wait for one exact owner-only local approval artifact, then consume it."""

    def __init__(
        self,
        path: os.PathLike[str] | str | None = None,
        *,
        now: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        default = os.path.join(
            os.path.expanduser("~"),
            ".hermes",
            "approvals",
            "muncho-full-canary-final-approval.json",
        )
        self.path = os.fspath(default if path is None else path)
        self._now = now
        self._sleeper = sleeper

    def read_final_approval(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        path = os.path.abspath(self.path)
        if path != self.path or ".." in PurePosixPath(path).parts:
            raise OwnerLauncherError("local_final_approval_path_invalid")
        deadline = request.get("approval_deadline_unix")
        delivery_deadline = request.get("owner_input_cutoff_unix")
        margin = request.get("final_approval_transmit_margin_seconds")
        legacy_window = (
            margin == _FINAL_APPROVAL_DELIVERY_RESERVE_SECONDS
            and delivery_deadline == deadline - margin
        )
        session_bound_window = (
            request.get("schema") == SESSION_BOUND_APPROVAL_REQUEST_SCHEMA
            and margin is None
            and type(deadline) is int
            and type(delivery_deadline) is int
            and delivery_deadline == deadline - 5
        )
        if (
            type(deadline) is not int
            or type(delivery_deadline) is not int
            or not (legacy_window or session_bound_window)
        ):
            raise OwnerLauncherError("invalid_owner_approval_request")
        if delivery_deadline < 0:
            raise OwnerLauncherError("final_approval_delivery_window_exhausted")
        while self._now() <= delivery_deadline:
            try:
                before = os.lstat(path)
            except FileNotFoundError:
                self._sleeper(0.2)
                continue
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or before.st_nlink != 1
                or before.st_uid != os.getuid()  # windows-footgun: ok
                or stat.S_IMODE(before.st_mode) != 0o600
                or not 0 < before.st_size <= 128 * 1024
            ):
                raise OwnerLauncherError("local_final_approval_identity_invalid")
            flags = (
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                raw = bytearray()
                while len(raw) <= 128 * 1024:
                    chunk = os.read(
                        descriptor, min(64 * 1024, 128 * 1024 + 1 - len(raw))
                    )
                    if not chunk:
                        break
                    raw.extend(chunk)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)

            def identity(item: os.stat_result) -> tuple[int, ...]:
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

            reachable = os.lstat(path)
            if (
                len(raw) != before.st_size
                or len(raw) > 128 * 1024
                or identity(before) != identity(opened)
                or identity(before) != identity(after)
                or identity(before) != identity(reachable)
            ):
                raise OwnerLauncherError("local_final_approval_replaced")
            value = _decode_json_object(bytes(raw), maximum=128 * 1024)
            if bytes(raw) != _canonical_bytes(value):
                raise OwnerLauncherError("local_final_approval_noncanonical")
            try:
                os.unlink(path)
            except OSError:
                raise OwnerLauncherError("local_final_approval_not_consumed") from None
            return value
        raise OwnerLauncherError("local_final_approval_timeout")


class ApprovedOwnerIdentity(Protocol):
    def bind_approved_subject(self, expected_sha256: str) -> None: ...

    def require_stable(self) -> None: ...


class RemoteSecretSession(Protocol):
    @property
    def termination_proven(self) -> bool: ...

    @property
    def terminal_receipt_validated(self) -> bool: ...

    def read_gate(self) -> Mapping[str, Any]: ...

    def exchange(self, frame: bytes | bytearray | memoryview) -> Mapping[str, Any]: ...

    def exchange_before(
        self,
        frame: bytes | bytearray | memoryview,
        *,
        write_guard: Callable[[], None],
        on_first_write: Callable[[], None],
    ) -> Mapping[str, Any]: ...

    def phase_b_exchange(
        self,
        frame: bytes | bytearray | memoryview,
        *,
        terminal: bool,
    ) -> Mapping[str, Any]: ...

    def finish(self, frame: bytes | bytearray | memoryview) -> Mapping[str, Any]: ...

    def finish_before(
        self,
        frame: bytes | bytearray | memoryview,
        *,
        write_guard: Callable[[], None],
        on_first_write: Callable[[], None],
    ) -> Mapping[str, Any]: ...

    def cancel_no_secret(self) -> Mapping[str, Any]: ...

    def read_next(self) -> Mapping[str, Any]: ...

    def read_next_bounded(self, timeout_seconds: float) -> Mapping[str, Any]: ...

    def mark_validated(self, receipt: Mapping[str, Any]) -> None: ...

    def abort_and_prove_terminated(self) -> None: ...

    def close(self) -> None: ...


class CoordinatorTransport(Protocol):
    def preflight_phase_b_apply(self, release_sha: str) -> Mapping[str, Any]: ...

    def preflight_phase_b_live_run(self, release_sha: str) -> Mapping[str, Any]: ...

    def open_phase_b_apply(self, release_sha: str) -> RemoteSecretSession: ...

    def open_discord_install(self, release_sha: str) -> RemoteSecretSession: ...

    def open_run(self, release_sha: str) -> RemoteSecretSession: ...

    def open_discord_retirement(self, release_sha: str) -> RemoteSecretSession: ...


def _validated_input_gate(
    session: RemoteSecretSession,
    raw: Mapping[str, Any],
    *,
    owner_gate: Mapping[str, Any] | None,
    active_secrets: Sequence[bytes | bytearray | memoryview] = (),
) -> Mapping[str, Any]:
    if raw.get("schema") != COORDINATOR_FAILURE_SCHEMA:
        return raw
    terminal = validate_terminal_first_failure(
        raw,
        owner_gate=owner_gate,
        active_secrets=active_secrets,
    )
    session.mark_validated(terminal)
    primary = RemoteCommandFailed(terminal)
    _close_session_preserving_primary(session, primary)
    raise primary


class RemoteTerminationUnconfirmed(OwnerLauncherError):
    def __init__(self) -> None:
        super().__init__("remote_termination_unconfirmed")


class _IapRemoteSession:
    """One fixed remote coordinator process with bounded canonical NDJSON I/O."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        environment: Mapping[str, str],
        popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        gate_timeout_seconds: float = 300.0,
        post_frame_timeout_seconds: float = 300.0,
        terminal_timeout_seconds: float = 2_400.0,
        termination_timeout_seconds: float = 15.0,
        maximum_line_bytes: int = _MAX_JSON_LINE_BYTES,
        postflight_guard: Callable[[], None] | None = None,
        authority_guard: Callable[[], None] | None = None,
    ) -> None:
        if (
            len(argv) < len(_GCLOUD_PYTHON_ISOLATION_ARGS) + 3
            or not isinstance(argv[0], str)
            or not os.path.isabs(argv[0])
            or tuple(argv[1 : 1 + len(_GCLOUD_PYTHON_ISOLATION_ARGS)])
            != _GCLOUD_PYTHON_ISOLATION_ARGS
            or not isinstance(argv[len(_GCLOUD_PYTHON_ISOLATION_ARGS) + 1], str)
            or not os.path.isabs(argv[len(_GCLOUD_PYTHON_ISOLATION_ARGS) + 1])
            or os.path.basename(argv[len(_GCLOUD_PYTHON_ISOLATION_ARGS) + 1])
            != "gcloud.py"
            or any(not isinstance(item, str) or not item for item in argv)
        ):
            raise OwnerLauncherError("invalid_iap_ssh_argv")
        self._gate_timeout = gate_timeout_seconds
        self._post_frame_timeout = post_frame_timeout_seconds
        self._terminal_timeout = terminal_timeout_seconds
        self._termination_timeout = termination_timeout_seconds
        if (
            type(maximum_line_bytes) is not int
            or not _MAX_JSON_LINE_BYTES <= maximum_line_bytes <= PHASE_B_MAX_RESPONSE_BYTES
        ):
            raise OwnerLauncherError("remote_ndjson_limit_invalid")
        self._maximum_line_bytes = maximum_line_bytes
        self._postflight_guard = postflight_guard or (lambda: None)
        self._authority_guard = authority_guard or self._postflight_guard
        if not callable(self._postflight_guard) or not callable(
            self._authority_guard
        ):
            raise OwnerLauncherError("iap_ssh_authority_guard_invalid")
        self._postflight_completed = False
        try:
            self._process = popen_factory(
                tuple(argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=dict(environment),
                shell=False,
                start_new_session=True,
                bufsize=0,
            )
        except (OSError, subprocess.SubprocessError):
            self._run_postflight()
            raise OwnerLauncherError("iap_ssh_start_failed") from None
        if self._process.stdin is None or self._process.stdout is None:
            self._force_local_stop()
            self._run_postflight()
            raise RemoteTerminationUnconfirmed()
        self._stdin_closed = False
        self._stdout_eof = False
        self._buffer = bytearray()
        self._messages_read = 0
        self._frames_written = 0
        self._last_mapping: Mapping[str, Any] | None = None
        self._validated_terminal = False
        self._termination_proven = False

    @property
    def termination_proven(self) -> bool:
        return self._termination_proven

    @property
    def terminal_receipt_validated(self) -> bool:
        return self._validated_terminal

    def require_current_authority(self) -> None:
        """Recheck the same pinned IAP/OS Login authorization snapshot."""

        try:
            self._authority_guard()
        except OwnerLauncherError:
            raise
        except BaseException:
            raise OwnerLauncherError("iap_ssh_authorization_changed") from None

    def _read_line(self, timeout_seconds: float) -> Mapping[str, Any]:
        if not 0 < timeout_seconds <= 2_400:
            raise OwnerLauncherError("iap_ssh_timeout_invalid")
        deadline = time.monotonic() + timeout_seconds
        descriptor = self._process.stdout.fileno()
        selector = selectors.DefaultSelector()
        try:
            selector.register(descriptor, selectors.EVENT_READ)
            while True:
                newline = self._buffer.find(b"\n")
                if newline >= 0:
                    raw = bytes(self._buffer[:newline])
                    del self._buffer[: newline + 1]
                    if not raw or raw.endswith(b"\r"):
                        raise OwnerLauncherError("remote_ndjson_invalid")
                    value = _decode_json_object(raw, maximum=self._maximum_line_bytes)
                    if raw != _canonical_bytes(value):
                        raise OwnerLauncherError("remote_ndjson_noncanonical")
                    self._messages_read += 1
                    self._last_mapping = dict(value)
                    return dict(value)
                if len(self._buffer) > self._maximum_line_bytes:
                    raise OwnerLauncherError("remote_ndjson_oversized")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OwnerLauncherError("iap_ssh_read_timeout")
                if not selector.select(min(remaining, 1.0)):
                    if self._process.poll() is not None:
                        raise OwnerLauncherError("iap_ssh_ended_without_receipt")
                    continue
                try:
                    chunk = os.read(descriptor, 64 * 1024)
                except OSError:
                    raise OwnerLauncherError("iap_ssh_read_failed") from None
                if not chunk:
                    self._stdout_eof = True
                    raise OwnerLauncherError("iap_ssh_ended_without_receipt")
                self._buffer.extend(chunk)
        finally:
            selector.close()

    def read_gate(self) -> Mapping[str, Any]:
        if self._messages_read != 0:
            raise OwnerLauncherError("remote_gate_replay_forbidden")
        return self._read_line(self._gate_timeout)

    def _close_stdin(self) -> None:
        if not self._stdin_closed:
            try:
                self._process.stdin.close()
            except OSError:
                raise OwnerLauncherError("iap_ssh_stdin_close_failed") from None
            self._stdin_closed = True

    def _write_frame(
        self,
        frame: bytes | bytearray | memoryview,
        *,
        close_stdin: bool,
        write_guard: Callable[[], None] | None = None,
        on_first_write: Callable[[], None] | None = None,
        on_write_complete: Callable[[], None] | None = None,
    ) -> Mapping[str, Any]:
        if self._stdin_closed:
            raise OwnerLauncherError("remote_frame_state_invalid")
        if not isinstance(frame, (bytes, bytearray, memoryview)):
            raise OwnerLauncherError("remote_frame_invalid")
        source_view: memoryview | None = None
        try:
            source_view = memoryview(frame)
            if not source_view.c_contiguous:
                raise TypeError("non-contiguous frame")
            view = source_view.cast("B")
        except (TypeError, ValueError):
            if source_view is not None:
                source_view.release()
            raise OwnerLauncherError("remote_frame_invalid") from None
        if not view.nbytes or view.nbytes > _MAX_REMOTE_SECRET_FRAME_BYTES:
            view.release()
            source_view.release()
            raise OwnerLauncherError("remote_frame_invalid")
        descriptor = self._process.stdin.fileno()
        offset = 0
        ambiguity_marked = False
        restore_blocking = False
        selector: selectors.BaseSelector | None = None
        try:
            if write_guard is not None:
                try:
                    if os.get_blocking(descriptor):
                        os.set_blocking(descriptor, False)
                        restore_blocking = True
                except OSError:
                    raise OwnerLauncherError("iap_ssh_secret_write_failed") from None
                selector = selectors.DefaultSelector()
                selector.register(descriptor, selectors.EVENT_WRITE)
            while offset < view.nbytes:
                if selector is not None and not selector.select(0.05):
                    if self._process.poll() is not None:
                        raise OwnerLauncherError("iap_ssh_secret_write_failed")
                    continue
                if offset == 0:
                    # A BlockingIOError proves that no byte crossed, so a
                    # fresh guard is required before retrying.  Once a
                    # positive partial write occurs, disclosure is irreversible
                    # and the remaining bytes must complete without a new
                    # expiry decision.
                    if write_guard is not None:
                        write_guard()
                    if not ambiguity_marked:
                        if on_first_write is not None:
                            on_first_write()
                        ambiguity_marked = True
                try:
                    written = os.write(descriptor, view[offset:])
                except BlockingIOError:
                    if selector is None:
                        if self._process.poll() is not None:
                            raise OwnerLauncherError("iap_ssh_secret_write_failed")
                    continue
                if written <= 0:
                    raise OSError("short write")
                offset += written
            if close_stdin:
                self._close_stdin()
        except (BrokenPipeError, OSError):
            raise OwnerLauncherError("iap_ssh_secret_write_failed") from None
        finally:
            if selector is not None:
                selector.close()
            if restore_blocking and not self._stdin_closed:
                try:
                    os.set_blocking(descriptor, True)
                except OSError:
                    pass
            view.release()
            source_view.release()
        if on_write_complete is not None:
            try:
                on_write_complete()
            except OwnerLauncherError:
                raise
            except BaseException:
                raise OwnerLauncherError(
                    "iap_ssh_post_write_cleanup_failed"
                ) from None
        self._frames_written += 1
        return self._read_line(self._post_frame_timeout)

    def exchange(self, frame: bytes | bytearray | memoryview) -> Mapping[str, Any]:
        if self._messages_read != 1 or self._frames_written != 0:
            raise OwnerLauncherError("remote_frame_state_invalid")
        return self._write_frame(frame, close_stdin=False)

    def exchange_before(
        self,
        frame: bytes | bytearray | memoryview,
        *,
        write_guard: Callable[[], None],
        on_first_write: Callable[[], None],
    ) -> Mapping[str, Any]:
        if self._messages_read != 1 or self._frames_written != 0:
            raise OwnerLauncherError("remote_frame_state_invalid")
        return self._write_frame(
            frame,
            close_stdin=False,
            write_guard=write_guard,
            on_first_write=on_first_write,
        )

    def phase_b_exchange(
        self,
        frame: bytes | bytearray | memoryview,
        *,
        terminal: bool,
    ) -> Mapping[str, Any]:
        """Exchange one exact MPB1 frame in the bounded Phase-B dialogue."""

        if (
            type(terminal) is not bool
            or self._stdin_closed
            or self._frames_written >= PHASE_B_MAX_ROUNDS
            or self._messages_read != self._frames_written + 1
        ):
            raise OwnerLauncherError("phase_b_remote_frame_state_invalid")
        try:
            view = memoryview(frame).cast("B")
        except (TypeError, ValueError):
            raise OwnerLauncherError("phase_b_remote_frame_invalid") from None
        try:
            if view.nbytes < 12 or view.nbytes > 12 + PHASE_B_MAX_RESPONSE_BYTES + PHASE_B_CREDENTIAL_BYTES:
                raise OwnerLauncherError("phase_b_remote_frame_invalid")
            magic, receipt_size, credential_size = struct.unpack(">4sII", view[:12])
            if (
                magic != PHASE_B_OWNER_FRAME_MAGIC
                or not 2 <= receipt_size <= PHASE_B_MAX_RESPONSE_BYTES
                or credential_size not in {0, PHASE_B_CREDENTIAL_BYTES}
                or view.nbytes != 12 + receipt_size + credential_size
            ):
                raise OwnerLauncherError("phase_b_remote_frame_invalid")
        finally:
            view.release()
        return self._write_frame(frame, close_stdin=terminal)

    def schema_reconciliation_exchange(
        self,
        frame: bytes | bytearray | memoryview,
        *,
        terminal: bool,
    ) -> Mapping[str, Any]:
        """Exchange one frame in the fixed three-frame reconciliation v2 flow."""

        if (
            type(terminal) is not bool
            or self._stdin_closed
            or self._frames_written >= 3
            or self._messages_read != self._frames_written + 1
            or terminal is not (self._frames_written == 2)
        ):
            raise OwnerLauncherError(
                "schema_reconciliation_remote_frame_state_invalid"
            )
        self._validate_schema_reconciliation_frame(
            frame,
            sequence=self._frames_written,
        )
        return self._write_frame(frame, close_stdin=terminal)

    @staticmethod
    def _validate_schema_reconciliation_frame(
        frame: bytes | bytearray | memoryview,
        *,
        sequence: int,
    ) -> None:
        if type(sequence) is not int or sequence not in {0, 1, 2}:
            raise OwnerLauncherError(
                "schema_reconciliation_remote_frame_invalid"
            )
        try:
            view = memoryview(frame).cast("B")
        except (TypeError, ValueError):
            raise OwnerLauncherError(
                "schema_reconciliation_remote_frame_invalid"
            ) from None
        try:
            if view.nbytes < 10:
                raise OwnerLauncherError(
                    "schema_reconciliation_remote_frame_invalid"
                )
            magic = bytes(view[:4])
            payload_size = struct.unpack(">I", view[4:8])[0]
            expected_magic = (
                SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_MAGIC,
                SCHEMA_RECONCILIATION_PREFLIGHT_AUTHORIZATION_MAGIC,
                SCHEMA_RECONCILIATION_ADMIN_CLEANUP_MAGIC,
            )[sequence]
            credential_size = (
                SCHEMA_RECONCILIATION_CREDENTIAL_BYTES if sequence == 0 else 0
            )
            if (
                magic != expected_magic
                or not 2 <= payload_size <= PHASE_B_MAX_RESPONSE_BYTES
                or view.nbytes != 8 + payload_size + credential_size
            ):
                raise OwnerLauncherError(
                    "schema_reconciliation_remote_frame_invalid"
                )
        finally:
            view.release()

    def schema_reconciliation_exchange_before(
        self,
        frame: bytes | bytearray | memoryview,
        *,
        write_guard: Callable[[], None],
        on_first_write: Callable[[], None],
        on_write_complete: Callable[[], None],
    ) -> Mapping[str, Any]:
        """Send only A1 after one last owner/Cloud authority recheck."""

        if (
            self._stdin_closed
            or self._frames_written != 0
            or self._messages_read != 1
            or not callable(write_guard)
            or not callable(on_first_write)
            or not callable(on_write_complete)
        ):
            raise OwnerLauncherError(
                "schema_reconciliation_remote_frame_state_invalid"
            )
        self._validate_schema_reconciliation_frame(frame, sequence=0)
        return self._write_frame(
            frame,
            close_stdin=False,
            write_guard=write_guard,
            on_first_write=on_first_write,
            on_write_complete=on_write_complete,
        )

    def finish(self, frame: bytes | bytearray | memoryview) -> Mapping[str, Any]:
        if (
            self._frames_written == 0
            and self._messages_read != 1
            or self._frames_written == 1
            and self._messages_read != 2
            or self._frames_written not in {0, 1}
        ):
            raise OwnerLauncherError("remote_frame_state_invalid")
        return self._write_frame(frame, close_stdin=True)

    def finish_before(
        self,
        frame: bytes | bytearray | memoryview,
        *,
        write_guard: Callable[[], None],
        on_first_write: Callable[[], None],
    ) -> Mapping[str, Any]:
        if (
            self._frames_written == 0
            and self._messages_read != 1
            or self._frames_written == 1
            and self._messages_read != 2
            or self._frames_written not in {0, 1}
        ):
            raise OwnerLauncherError("remote_frame_state_invalid")
        return self._write_frame(
            frame,
            close_stdin=True,
            write_guard=write_guard,
            on_first_write=on_first_write,
        )

    def cancel_no_secret(self) -> Mapping[str, Any]:
        if self._messages_read != 1 or self._stdin_closed:
            raise OwnerLauncherError("remote_cancel_state_invalid")
        self._close_stdin()
        return self._read_line(self._post_frame_timeout)

    def read_next(self) -> Mapping[str, Any]:
        if self._messages_read < 2 or not self._stdin_closed:
            raise OwnerLauncherError("remote_terminal_state_invalid")
        return self._read_line(self._terminal_timeout)

    def read_next_bounded(self, timeout_seconds: float) -> Mapping[str, Any]:
        if self._messages_read < 2 or not self._stdin_closed:
            raise OwnerLauncherError("remote_terminal_state_invalid")
        return self._read_line(timeout_seconds)

    def _wait_exact_exit(self, timeout_seconds: float) -> int:
        deadline = time.monotonic() + timeout_seconds
        descriptor = self._process.stdout.fileno()
        selector = selectors.DefaultSelector()
        try:
            selector.register(descriptor, selectors.EVENT_READ)
            while not self._stdout_eof:
                if self._buffer:
                    raise OwnerLauncherError("remote_ndjson_trailing_output")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OwnerLauncherError("iap_ssh_termination_timeout")
                if not selector.select(min(remaining, 1.0)):
                    continue
                chunk = os.read(descriptor, 1)
                if chunk:
                    raise OwnerLauncherError("remote_ndjson_trailing_output")
                self._stdout_eof = True
        finally:
            selector.close()
        try:
            return self._process.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise OwnerLauncherError("iap_ssh_termination_timeout") from None

    def complete_read_only(self) -> None:
        self._close_stdin()
        if self._wait_exact_exit(self._termination_timeout) != 0:
            raise OwnerLauncherError("iap_ssh_read_only_failed")
        self._termination_proven = True
        self._run_postflight()

    def mark_validated(self, receipt: Mapping[str, Any]) -> None:
        if self._last_mapping != receipt:
            raise OwnerLauncherError("remote_terminal_receipt_mismatch")
        self._close_stdin()
        returncode = self._wait_exact_exit(self._termination_timeout)
        expected_returncode = 0 if receipt.get("ok") is True else 2
        if returncode != expected_returncode:
            raise OwnerLauncherError("remote_terminal_exit_mismatch")
        self._validated_terminal = True
        self._termination_proven = True
        self._run_postflight()

    def abort_and_prove_terminated(self) -> None:
        """Close stdin and require the remote fail-closed process to exit 2."""

        if self._termination_proven:
            self._run_postflight()
            return
        deadline = time.monotonic() + self._termination_timeout
        selector: selectors.BaseSelector | None = None
        drained = len(self._buffer)
        self._buffer.clear()
        try:
            self._close_stdin()
            descriptor = self._process.stdout.fileno()
            selector = selectors.DefaultSelector()
            selector.register(descriptor, selectors.EVENT_READ)
            while not self._stdout_eof:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OwnerLauncherError("iap_ssh_termination_timeout")
                if not selector.select(min(remaining, 1.0)):
                    continue
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    self._stdout_eof = True
                    break
                drained += len(chunk)
                if drained > _HTTP_RESPONSE_MAX_BYTES:
                    raise OwnerLauncherError("remote_ndjson_oversized")
            try:
                returncode = self._process.wait(
                    max(0.1, deadline - time.monotonic())
                )
            except subprocess.TimeoutExpired:
                raise OwnerLauncherError(
                    "iap_ssh_termination_timeout"
                ) from None
            if returncode != 2:
                raise OwnerLauncherError("iap_ssh_abort_exit_mismatch")
            self._termination_proven = True
            self._run_postflight()
        except BaseException:
            self._force_local_stop()
            try:
                self._run_postflight()
            except BaseException:
                pass
            raise RemoteTerminationUnconfirmed() from None
        finally:
            if selector is not None:
                selector.close()

    def _run_postflight(self) -> None:
        if self._postflight_completed:
            return
        self._postflight_guard()
        self._postflight_completed = True

    def _force_local_stop(self) -> None:
        try:
            if getattr(self, "_process", None) is None:
                return
            if self._process.stdin is not None and not self._process.stdin.closed:
                self._process.stdin.close()
            if self._process.poll() is None:
                try:
                    os.killpg(self._process.pid, signal.SIGTERM)  # windows-footgun: ok
                except (OSError, ProcessLookupError):
                    pass
                try:
                    self._process.wait(self._termination_timeout)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(  # windows-footgun: ok
                            self._process.pid,
                            signal.SIGKILL,  # windows-footgun: ok
                        )
                    except (OSError, ProcessLookupError):
                        pass
                    try:
                        self._process.wait(self._termination_timeout)
                    except subprocess.TimeoutExpired:
                        pass
        except BaseException:
            pass

    def close(self) -> None:
        if self._validated_terminal and self._termination_proven:
            self._run_postflight()
            return
        if self._termination_proven and self._messages_read == 1:
            self._run_postflight()
            return
        self._force_local_stop()
        self._run_postflight()
        raise RemoteTerminationUnconfirmed()


class IapCoordinatorTransport:
    """Pinned IAP/SSH transport to the one isolated canary VM and release."""

    _RELEASE_BASE = "/opt/muncho-canary-releases"
    _MODULE = "gateway.canonical_full_canary_coordinator"
    _COMMANDS = frozenset({
        "publish-coordinator-input",
        "preflight-phase-b-apply",
        "preflight-phase-b-live-run",
        "install-discord-token",
        "phase-b-apply",
        "run",
        "stop-and-retire-discord-token",
    })

    def __init__(
        self,
        owner_identity: GcloudOwnerAccessToken,
        *,
        gcloud_executable: StableExecutable | None = None,
        gcloud_configuration: StableGcloudConfiguration | None = None,
        known_hosts: StableKnownHosts | None = None,
        popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        preflight_runner: SubprocessRunner = subprocess.run,
        preflight_timeout_seconds: float = 30.0,
    ) -> None:
        self._owner_identity = owner_identity
        self._gcloud_executable = gcloud_executable or TrustedGcloudExecutable()
        identity_configuration = getattr(
            owner_identity,
            "gcloud_configuration",
            None,
        )
        if (
            gcloud_configuration is not None
            and identity_configuration is not None
            and gcloud_configuration is not identity_configuration
        ):
            raise OwnerLauncherError("gcloud_configuration_not_shared")
        self._gcloud_configuration = (
            gcloud_configuration
            or identity_configuration
            or PinnedGcloudConfiguration()
        )
        self._known_hosts = known_hosts or PinnedGoogleComputeKnownHosts()
        self._popen_factory = popen_factory
        self._preflight_runner = preflight_runner
        self._preflight_timeout_seconds = preflight_timeout_seconds

    @staticmethod
    def _ssh_flags(known_hosts: str, private_key: str) -> tuple[str, ...]:
        return (
            "--ssh-flag=-F/dev/null",
            "--ssh-flag=-T",
            f"--ssh-flag=-i{private_key}",
            "--ssh-flag=-oBatchMode=yes",
            "--ssh-flag=-oIdentitiesOnly=yes",
            "--ssh-flag=-oIdentityAgent=none",
            "--ssh-flag=-oCertificateFile=none",
            "--ssh-flag=-oPreferredAuthentications=publickey",
            "--ssh-flag=-oPubkeyAuthentication=yes",
            "--ssh-flag=-oPasswordAuthentication=no",
            "--ssh-flag=-oKbdInteractiveAuthentication=no",
            "--ssh-flag=-oGSSAPIAuthentication=no",
            "--ssh-flag=-oHostbasedAuthentication=no",
            "--ssh-flag=-oPermitLocalCommand=no",
            "--ssh-flag=-oClearAllForwardings=yes",
            "--ssh-flag=-oControlMaster=no",
            "--ssh-flag=-oControlPath=none",
            "--ssh-flag=-oKnownHostsCommand=none",
            "--ssh-flag=-oCanonicalizeHostname=no",
            "--ssh-flag=-oForwardAgent=no",
            "--ssh-flag=-oEscapeChar=none",
            "--ssh-flag=-oRequestTTY=no",
            "--ssh-flag=-oStrictHostKeyChecking=yes",
            f"--ssh-flag=-oUserKnownHostsFile={known_hosts}",
            "--ssh-flag=-oGlobalKnownHostsFile=none",
            "--ssh-flag=-oUpdateHostKeys=no",
            "--ssh-flag=-oVerifyHostKeyDNS=no",
            "--ssh-flag=-oServerAliveInterval=15",
            "--ssh-flag=-oServerAliveCountMax=4",
        )

    def _argv(
        self, release_sha: str, command: str, *, approved: bool
    ) -> tuple[str, ...]:
        if not _RELEASE_SHA.fullmatch(release_sha):
            raise OwnerLauncherError("invalid_release_sha")
        if command not in self._COMMANDS:
            raise OwnerLauncherError("invalid_coordinator_command")
        command_prefix = self._gcloud_executable.trusted_command_prefix()
        if (
            len(command_prefix) != len(_GCLOUD_PYTHON_ISOLATION_ARGS) + 2
            or command_prefix[1:-1] != _GCLOUD_PYTHON_ISOLATION_ARGS
        ):
            raise OwnerLauncherError("invalid_gcloud_command_prefix")
        self._gcloud_configuration.assert_stable()
        known_hosts = self._known_hosts.absolute_path()
        private_key = self._known_hosts.private_key_path()
        self._known_hosts.public_key_line()
        account = (
            self._owner_identity.approved_account
            if approved
            else self._owner_identity.account_for_read_only_preflight()
        )
        interpreter = f"{self._RELEASE_BASE}/{release_sha}/venv/bin/python"
        remote_command = shlex.join((
            "/usr/bin/sudo",
            "--non-interactive",
            "--",
            interpreter,
            "-B",
            "-I",
            "-m",
            self._MODULE,
            command,
        ))
        return (
            *command_prefix,
            "compute",
            "ssh",
            f"{OS_LOGIN_USERNAME}@{VM_NAME}",
            f"--project={PROJECT}",
            f"--zone={ZONE}",
            f"--account={account}",
            "--plain",
            "--tunnel-through-iap",
            "--quiet",
            f"--command={remote_command}",
            *self._ssh_flags(known_hosts, private_key),
        )

    def _run_read_only_gcloud_json(
        self,
        args: Sequence[str],
    ) -> Mapping[str, Any]:
        command_prefix = self._gcloud_executable.trusted_command_prefix()
        environment = _owner_gcloud_environment(
            self._gcloud_configuration,
            command_prefix[0],
        )
        try:
            completed = self._preflight_runner(
                (*command_prefix, *args),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=dict(environment),
                shell=False,
                timeout=self._preflight_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            self._postflight()
            raise OwnerLauncherError("iap_ssh_preflight_unavailable") from None
        self._postflight()
        if (
            completed.returncode != 0
            or not isinstance(completed.stdout, bytes)
            or not completed.stdout
            or len(completed.stdout) > _HTTP_RESPONSE_MAX_BYTES
        ):
            raise OwnerLauncherError("iap_ssh_preflight_invalid")
        try:
            return _decode_json_object(
                completed.stdout,
                maximum=_HTTP_RESPONSE_MAX_BYTES,
            )
        except OwnerLauncherError:
            raise OwnerLauncherError("iap_ssh_preflight_invalid") from None

    @staticmethod
    def _metadata_items(
        value: Mapping[str, Any],
        field: str,
    ) -> tuple[tuple[str, str], ...]:
        container = value.get(field)
        if not isinstance(container, Mapping):
            raise OwnerLauncherError("iap_ssh_authorization_invalid")
        items = container.get("items")
        if not isinstance(items, list):
            raise OwnerLauncherError("iap_ssh_authorization_invalid")
        result: dict[str, str] = {}
        for item in items:
            if (
                not isinstance(item, Mapping)
                or set(item) != {"key", "value"}
                or not isinstance(item.get("key"), str)
                or not isinstance(item.get("value"), str)
                or item["key"] in result
            ):
                raise OwnerLauncherError("iap_ssh_authorization_invalid")
            result[str(item["key"])] = str(item["value"])
        return tuple(sorted(result.items()))

    def _authorization_snapshot(self, account: str) -> tuple[str, str, str]:
        instance = self._run_read_only_gcloud_json((
            "compute",
            "instances",
            "describe",
            VM_NAME,
            f"--project={PROJECT}",
            f"--zone={ZONE}",
            f"--account={account}",
            "--format=json(id,name,zone,metadata.items)",
            "--quiet",
        ))
        instance_metadata = self._metadata_items(instance, "metadata")
        if (
            set(instance) != {"id", "name", "zone", "metadata"}
            or instance.get("id") != VM_INSTANCE_ID
            or instance.get("name") != VM_NAME
            or instance.get("zone")
            != f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}"
            or dict(instance_metadata).get("enable-oslogin") != "TRUE"
            or "ssh-keys" in dict(instance_metadata)
        ):
            raise OwnerLauncherError("iap_ssh_authorization_invalid")

        project = self._run_read_only_gcloud_json((
            "compute",
            "project-info",
            "describe",
            f"--project={PROJECT}",
            f"--account={account}",
            "--format=json(name,commonInstanceMetadata.items)",
            "--quiet",
        ))
        self._metadata_items(project, "commonInstanceMetadata")
        if (
            set(project) != {"name", "commonInstanceMetadata"}
            or project.get("name") != PROJECT
        ):
            raise OwnerLauncherError("iap_ssh_authorization_invalid")

        profile = self._run_read_only_gcloud_json((
            "compute",
            "os-login",
            "describe-profile",
            f"--project={PROJECT}",
            f"--account={account}",
            "--format=json",
            "--quiet",
        ))
        posix_accounts = profile.get("posixAccounts")
        ssh_keys = profile.get("sshPublicKeys")
        public_key = self._known_hosts.public_key_line()
        if (
            set(profile) != {"name", "posixAccounts", "sshPublicKeys"}
            or profile.get("name") != OS_LOGIN_PROFILE_ID
            or not isinstance(posix_accounts, list)
            or not isinstance(ssh_keys, Mapping)
        ):
            raise OwnerLauncherError("iap_ssh_authorization_invalid")
        matching_accounts = [
            item
            for item in posix_accounts
            if isinstance(item, Mapping)
            and item.get("username") == OS_LOGIN_USERNAME
            and item.get("primary") is True
            and item.get("operatingSystemType") == "LINUX"
            and item.get("homeDirectory") == f"/home/{OS_LOGIN_USERNAME}"
        ]
        matching_keys = [
            item
            for fingerprint, item in ssh_keys.items()
            if isinstance(fingerprint, str)
            and re.fullmatch(r"[0-9a-f]{64}", fingerprint) is not None
            and isinstance(item, Mapping)
            and item.get("fingerprint") == fingerprint
            and item.get("key") == public_key
        ]
        if len(matching_accounts) != 1 or len(matching_keys) != 1:
            raise OwnerLauncherError("iap_ssh_authorization_invalid")
        if any(not isinstance(item, Mapping) for item in posix_accounts) or any(
            not isinstance(key, str) or not isinstance(item, Mapping)
            for key, item in ssh_keys.items()
        ):
            raise OwnerLauncherError("iap_ssh_authorization_invalid")
        normalized_instance = {
            "id": VM_INSTANCE_ID,
            "name": VM_NAME,
            "zone": instance["zone"],
            "metadata": [
                {"key": key, "value": value} for key, value in instance_metadata
            ],
        }
        project_metadata = self._metadata_items(project, "commonInstanceMetadata")
        normalized_project = {
            "name": PROJECT,
            "metadata": [
                {"key": key, "value": value} for key, value in project_metadata
            ],
        }
        normalized_profile = {
            "name": OS_LOGIN_PROFILE_ID,
            "posixAccounts": sorted(
                (dict(item) for item in posix_accounts),
                key=_canonical_bytes,
            ),
            "sshPublicKeys": [
                {"fingerprint": fingerprint, "value": dict(item)}
                for fingerprint, item in sorted(ssh_keys.items())
            ],
        }
        self._postflight()
        return (
            _sha256(_canonical_bytes(normalized_instance)),
            _sha256(_canonical_bytes(normalized_project)),
            _sha256(_canonical_bytes(normalized_profile)),
        )

    def _validate_dry_run(self, argv: Sequence[str]) -> None:
        command_prefix = self._gcloud_executable.trusted_command_prefix()
        environment = _owner_gcloud_environment(
            self._gcloud_configuration,
            command_prefix[0],
        )
        try:
            completed = self._preflight_runner(
                (*argv, "--dry-run"),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=dict(environment),
                shell=False,
                timeout=self._preflight_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            self._postflight()
            raise OwnerLauncherError("iap_ssh_dry_run_unavailable") from None
        self._postflight()
        if (
            completed.returncode != 0
            or not isinstance(completed.stdout, bytes)
            or not completed.stdout.endswith(b"\n")
            or b"\n" in completed.stdout[:-1]
            or len(completed.stdout) > 256 * 1024
        ):
            raise OwnerLauncherError("iap_ssh_dry_run_invalid")
        try:
            rendered = completed.stdout[:-1].decode("utf-8", errors="strict")
            observed = tuple(shlex.split(rendered, posix=True))
        except (UnicodeError, ValueError):
            raise OwnerLauncherError("iap_ssh_dry_run_invalid") from None
        known_hosts = self._known_hosts.absolute_path()
        private_key = self._known_hosts.private_key_path()
        self._known_hosts.public_key_line()
        remote = next(
            (item.split("=", 1)[1] for item in argv if item.startswith("--command=")),
            None,
        )
        proxy = "ProxyCommand " + " ".join((
            *command_prefix,
            "compute",
            "start-iap-tunnel",
            VM_NAME,
            "%p",
            "--listen-on-stdin",
            f"--project={PROJECT}",
            f"--zone={ZONE}",
            "--verbosity=error",
        ))
        ssh_options = tuple(
            item.removeprefix("--ssh-flag=")
            for item in self._ssh_flags(known_hosts, private_key)
        )
        expected = (
            "/usr/bin/ssh",
            "-T",
            "-o",
            proxy,
            "-o",
            "ProxyUseFdpass=no",
            *ssh_options,
            f"{OS_LOGIN_USERNAME}@compute.{VM_INSTANCE_ID}",
            "--",
            *(() if remote is None else remote.split(" ")),
        )
        if remote is None or observed != expected:
            raise OwnerLauncherError("iap_ssh_dry_run_invalid")

    def _open(
        self,
        release_sha: str,
        command: str,
        *,
        approved: bool,
        gate_timeout_seconds: float = 300.0,
        post_frame_timeout_seconds: float = 300.0,
        maximum_line_bytes: int = _MAX_JSON_LINE_BYTES,
    ) -> _IapRemoteSession:
        argv = self._argv(release_sha, command, approved=approved)
        account = next(
            item.split("=", 1)[1] for item in argv if item.startswith("--account=")
        )
        authorization_before = self._authorization_snapshot(account)
        self._validate_dry_run(argv)
        if self._authorization_snapshot(account) != authorization_before:
            raise OwnerLauncherError("iap_ssh_authorization_changed")
        command_prefix = self._gcloud_executable.trusted_command_prefix()

        def postflight() -> None:
            self._postflight()
            if self._authorization_snapshot(account) != authorization_before:
                raise OwnerLauncherError("iap_ssh_authorization_changed")

        return _IapRemoteSession(
            argv,
            environment=_owner_gcloud_environment(
                self._gcloud_configuration,
                command_prefix[0],
            ),
            popen_factory=self._popen_factory,
            gate_timeout_seconds=gate_timeout_seconds,
            post_frame_timeout_seconds=post_frame_timeout_seconds,
            maximum_line_bytes=maximum_line_bytes,
            postflight_guard=postflight,
            authority_guard=postflight,
        )

    def _postflight(self) -> None:
        self._gcloud_executable.trusted_command_prefix()
        self._gcloud_configuration.assert_stable()
        self._known_hosts.absolute_path()
        self._known_hosts.private_key_path()
        self._known_hosts.public_key_line()

    def publish_coordinator_input(self, release_sha: str) -> Mapping[str, Any]:
        """Publish the fixed coordinator input after packaged activation truth."""

        session = self._open(
            release_sha,
            "publish-coordinator-input",
            approved=False,
        )
        primary: BaseException | None = None
        try:
            value = session.read_gate()
            if value.get("schema") == COORDINATOR_FAILURE_SCHEMA:
                terminal = validate_terminal_first_failure(
                    value,
                    owner_gate=None,
                    expected_release_sha=release_sha,
                )
                session.mark_validated(terminal)
                raise RemoteCommandFailed(terminal)
            receipt = validate_coordinator_input_publication_receipt(
                value,
                expected_release_sha=release_sha,
            )
            session.mark_validated(receipt)
            self._owner_identity.require_stable()
            return receipt
        except BaseException as exc:
            primary = exc
            raise
        finally:
            _close_session_preserving_primary(session, primary)

    def preflight_phase_b_apply(self, release_sha: str) -> Mapping[str, Any]:
        session = self._open(
            release_sha,
            "preflight-phase-b-apply",
            approved=False,
        )
        primary: BaseException | None = None
        try:
            value = session.read_gate()
            if value.get("schema") == COORDINATOR_FAILURE_SCHEMA:
                terminal = validate_terminal_first_failure(
                    value,
                    owner_gate=None,
                    expected_release_sha=release_sha,
                )
                session.mark_validated(terminal)
                raise RemoteCommandFailed(terminal)
            session.complete_read_only()
            return value
        except BaseException as exc:
            primary = exc
            raise
        finally:
            _close_session_preserving_primary(session, primary)

    def preflight_phase_b_live_run(self, release_sha: str) -> Mapping[str, Any]:
        session = self._open(
            release_sha,
            "preflight-phase-b-live-run",
            approved=False,
        )
        primary: BaseException | None = None
        try:
            value = session.read_gate()
            if value.get("schema") == COORDINATOR_FAILURE_SCHEMA:
                terminal = validate_terminal_first_failure(
                    value,
                    owner_gate=None,
                    expected_release_sha=release_sha,
                )
                session.mark_validated(terminal)
                raise RemoteCommandFailed(terminal)
            session.complete_read_only()
            return value
        except BaseException as exc:
            primary = exc
            raise
        finally:
            _close_session_preserving_primary(session, primary)

    def open_discord_install(self, release_sha: str) -> RemoteSecretSession:
        return self._open(release_sha, "install-discord-token", approved=True)

    def open_phase_b_apply(self, release_sha: str) -> RemoteSecretSession:
        return self._open(
            release_sha,
            "phase-b-apply",
            approved=True,
            post_frame_timeout_seconds=2_400.0,
            maximum_line_bytes=PHASE_B_MAX_REQUEST_BYTES,
        )

    def open_run(self, release_sha: str) -> RemoteSecretSession:
        return self._open(release_sha, "run", approved=True)

    def open_discord_retirement(self, release_sha: str) -> RemoteSecretSession:
        return self._open(
            release_sha,
            "stop-and-retire-discord-token",
            approved=True,
            post_frame_timeout_seconds=2_400.0,
        )


class IapSchemaReconciliationTransport(IapCoordinatorTransport):
    """Pinned transport for the packaged reconciliation v2 state machine."""

    _MODULE = "gateway.canonical_writer_schema_reconciliation_bootstrap"
    _COMMANDS = frozenset({"run"})

    def open_reconciliation(self, release_sha: str) -> _IapRemoteSession:
        return self._open(
            release_sha,
            "run",
            # G0 is secret-free and carries the exact owner subject.  The
            # launcher binds that subject before creating the Cloud SQL user
            # and rechecks this same IAP snapshot at A1 first byte.
            approved=False,
            post_frame_timeout_seconds=2_400.0,
            maximum_line_bytes=PHASE_B_MAX_RESPONSE_BYTES,
        )


def validate_phase_b_apply_gate(
    value: Any,
    *,
    expected_release_sha: str,
    now_unix: int,
) -> Mapping[str, Any]:
    gate = _validate_self_digest(
        value,
        expected_keys=_PHASE_B_APPLY_GATE_KEYS,
        digest_key="gate_sha256",
        code="invalid_phase_b_apply_gate",
    )
    authority_present = gate.get("authority_present")
    state = gate.get("state")
    if (
        gate.get("schema") != PHASE_B_APPLY_GATE_SCHEMA
        or gate.get("ok") is not True
        or state
        not in {"initial_apply_ready", "same_plan_resume_or_replay"}
        or gate.get("release_sha") != expected_release_sha
        or type(authority_present) is not bool
        or authority_present is not (state == "same_plan_resume_or_replay")
        or type(gate.get("phase_b_terminal")) is not bool
        or type(gate.get("phase_b_requires_reapproval")) is not bool
        or type(gate.get("issued_at_unix")) is not int
        or not gate["issued_at_unix"] <= now_unix
        or type(gate.get("expires_at_unix")) is not int
        or not now_unix < gate["expires_at_unix"]
        or gate["expires_at_unix"] - gate["issued_at_unix"] > 3600
        or gate["expires_at_unix"] - now_unix > 3600
    ):
        raise OwnerLauncherError("invalid_phase_b_apply_gate")
    for name in (
        "coordinator_input_sha256",
        "owner_subject_sha256",
        "approval_source_sha256",
    ):
        _require_sha256(gate.get(name), "invalid_phase_b_apply_gate")
    if not authority_present:
        if (
            any(
                gate.get(name) is not None
                for name in (
                    "phase_b_plan_sha256",
                    "phase_b_approval_sha256",
                    "phase_b_approval_sequence",
                    "phase_b_incomplete_state",
                    "phase_b_inspection_sha256",
                )
            )
            or gate["phase_b_terminal"] is not False
            or gate["phase_b_requires_reapproval"] is not False
        ):
            raise OwnerLauncherError("invalid_phase_b_apply_gate")
    else:
        for name in (
            "phase_b_plan_sha256",
            "phase_b_approval_sha256",
            "phase_b_inspection_sha256",
        ):
            _require_sha256(gate.get(name), "invalid_phase_b_apply_gate")
        if (
            type(gate.get("phase_b_approval_sequence")) is not int
            or gate["phase_b_approval_sequence"] < 0
            or gate.get("phase_b_incomplete_state")
            not in {
                "authority_published_no_intent",
                "journal_incomplete",
                "terminal",
            }
            or gate["phase_b_terminal"]
            is not (gate["phase_b_incomplete_state"] == "terminal")
            or gate["phase_b_terminal"]
            and gate["phase_b_requires_reapproval"]
        ):
            raise OwnerLauncherError("invalid_phase_b_apply_gate")
    _phase_b_secret_free(gate)
    return gate


def validate_phase_b_apply_receipt(
    value: Any,
    *,
    owner_gate: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate the terminal, secret-free Phase-B handoff receipt."""

    if (
        not isinstance(value, Mapping)
        or set(value) != _PHASE_B_APPLY_RECEIPT_KEYS
        or value.get("schema") != PHASE_B_APPLY_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "terminal_ready"
        or value.get("release_sha") != owner_gate.get("release_sha")
        or value.get("coordinator_input_sha256")
        != owner_gate.get("coordinator_input_sha256")
        or value.get("safe_to_start") is not True
    ):
        raise OwnerLauncherError("invalid_phase_b_apply_receipt")
    for name in (
        "phase_b_plan_sha256",
        "phase_b_approval_sha256",
        "phase_b_terminal_receipt_sha256",
        "phase_b_readiness_receipt_sha256",
    ):
        _require_sha256(value.get(name), "invalid_phase_b_apply_receipt")
    _validate_receipt_time(
        value.get("completed_at_unix"),
        now_unix=now_unix,
        code="invalid_phase_b_apply_receipt",
    )
    expected = _require_sha256(
        value.get("receipt_sha256"),
        "invalid_phase_b_apply_receipt",
    )
    unsigned = dict(value)
    del unsigned["receipt_sha256"]
    if expected != _sha256(_canonical_bytes(unsigned)):
        raise OwnerLauncherError("invalid_phase_b_apply_receipt")
    _phase_b_secret_free(value)
    return copy.deepcopy(dict(value))


def validate_phase_b_live_gate(
    value: Any,
    *,
    expected_release_sha: str,
    expected_phase_b_apply_receipt: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate the read-only terminal-Phase-B authority for live work."""

    gate = _validate_self_digest(
        value,
        expected_keys=_PHASE_B_LIVE_GATE_KEYS,
        digest_key="gate_sha256",
        code="invalid_phase_b_live_gate",
    )
    anchor = gate.get("phase_b_readiness_anchor")
    if (
        gate.get("schema") != PHASE_B_LIVE_GATE_SCHEMA
        or gate.get("ok") is not True
        or gate.get("state") != "phase_b_terminal_ready"
        or gate.get("release_sha") != expected_release_sha
        or gate.get("coordinator_input_sha256")
        != expected_phase_b_apply_receipt.get("coordinator_input_sha256")
        or not isinstance(anchor, Mapping)
        or set(anchor) != _PHASE_B_READINESS_ANCHOR_KEYS
        or gate.get("phase_b_readiness_anchor_sha256")
        != _sha256(_canonical_bytes(anchor))
        or anchor.get("phase_b_release_revision") != expected_release_sha
        or anchor.get("phase_b_plan_sha256")
        != expected_phase_b_apply_receipt.get("phase_b_plan_sha256")
        or anchor.get("phase_b_approval_sha256")
        != expected_phase_b_apply_receipt.get("phase_b_approval_sha256")
        or anchor.get("phase_b_terminal_receipt_sha256")
        != expected_phase_b_apply_receipt.get("phase_b_terminal_receipt_sha256")
        or type(anchor.get("phase_b_readiness_sequence")) is not int
        or anchor["phase_b_readiness_sequence"] < 0
        or type(gate.get("issued_at_unix")) is not int
        or not gate["issued_at_unix"] <= now_unix
        or type(gate.get("expires_at_unix")) is not int
        or not now_unix < gate["expires_at_unix"]
        or gate["expires_at_unix"] - gate["issued_at_unix"] != 300
        or gate["expires_at_unix"] - now_unix > 300
    ):
        raise OwnerLauncherError("invalid_phase_b_live_gate")
    for name in (
        "owner_subject_sha256",
        "approval_source_sha256",
        "phase_b_readiness_anchor_sha256",
    ):
        _require_sha256(gate.get(name), "invalid_phase_b_live_gate")
    for name in _PHASE_B_READINESS_ANCHOR_KEYS - {
        "phase_b_release_revision",
        "phase_b_readiness_sequence",
    }:
        _require_sha256(anchor.get(name), "invalid_phase_b_live_gate")
    _phase_b_secret_free(gate)
    return gate


def validate_current_phase_b_live_gate(
    value: Any,
    *,
    expected_release_sha: str,
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate current terminal Phase-B truth without a historical apply receipt.

    The live gate is a read-only descendant of the pinned Phase-B owner
    authority.  Requiring a launcher-local copy of an earlier apply receipt
    would make durable Cloud truth depend on transient workstation state, so
    the current anchor is validated directly and exactly instead.
    """

    gate = _validate_self_digest(
        value,
        expected_keys=_PHASE_B_LIVE_GATE_KEYS,
        digest_key="gate_sha256",
        code="invalid_phase_b_live_gate",
    )
    anchor = gate.get("phase_b_readiness_anchor")
    if (
        gate.get("schema") != PHASE_B_LIVE_GATE_SCHEMA
        or gate.get("ok") is not True
        or gate.get("state") != "phase_b_terminal_ready"
        or gate.get("release_sha") != expected_release_sha
        or not isinstance(anchor, Mapping)
        or set(anchor) != _PHASE_B_READINESS_ANCHOR_KEYS
        or gate.get("phase_b_readiness_anchor_sha256")
        != _sha256(_canonical_bytes(anchor))
        or anchor.get("phase_b_release_revision") != expected_release_sha
        or anchor.get("phase_b_approval_sha256") is None
        or type(anchor.get("phase_b_readiness_sequence")) is not int
        or anchor["phase_b_readiness_sequence"] < 0
        or gate.get("approval_source_sha256")
        != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
        or type(gate.get("issued_at_unix")) is not int
        or not gate["issued_at_unix"] <= now_unix
        or type(gate.get("expires_at_unix")) is not int
        or not now_unix < gate["expires_at_unix"]
        or gate["expires_at_unix"] - gate["issued_at_unix"] != 300
        or gate["expires_at_unix"] - now_unix > 300
    ):
        raise OwnerLauncherError("invalid_phase_b_live_gate")
    for name in (
        "coordinator_input_sha256",
        "owner_subject_sha256",
        "approval_source_sha256",
        "phase_b_readiness_anchor_sha256",
    ):
        _require_sha256(gate.get(name), "invalid_phase_b_live_gate")
    for name in _PHASE_B_READINESS_ANCHOR_KEYS - {
        "phase_b_release_revision",
        "phase_b_readiness_sequence",
    }:
        _require_sha256(anchor.get(name), "invalid_phase_b_live_gate")
    _phase_b_secret_free(gate)
    return gate


def validate_session_bound_approval_request(
    value: Any,
    *,
    live_gate: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate the in-process approval request emitted by the target run."""

    request = _validate_self_digest(
        value,
        expected_keys=_SESSION_BOUND_APPROVAL_REQUEST_KEYS,
        digest_key="request_sha256",
        code="invalid_owner_approval_request",
    )
    requested = request.get("requested_at_unix")
    cutoff = request.get("owner_input_cutoff_unix")
    deadline = request.get("approval_deadline_unix")
    anchor = live_gate.get("phase_b_readiness_anchor")
    if (
        request.get("schema") != SESSION_BOUND_APPROVAL_REQUEST_SCHEMA
        or request.get("ok") is not True
        or request.get("state") != "awaiting_session_bound_owner_approval"
        or request.get("release_sha") != live_gate.get("release_sha")
        or request.get("coordinator_input_sha256")
        != live_gate.get("coordinator_input_sha256")
        or request.get("owner_subject_sha256")
        != live_gate.get("owner_subject_sha256")
        or request.get("approval_source_sha256")
        != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
        or request.get("approval_source_sha256")
        != live_gate.get("approval_source_sha256")
        or request.get("phase_b_readiness_anchor_sha256")
        != live_gate.get("phase_b_readiness_anchor_sha256")
        or not isinstance(anchor, Mapping)
        or request.get("phase_b_approval_sha256")
        != anchor.get("phase_b_approval_sha256")
        or request.get("staged_plan_path")
        != "/etc/muncho/full-canary/staged/runtime-plan.json"
        or request.get("approval_path") is not None
        or request.get("final_approval_frame_schema")
        != FINAL_APPROVAL_FRAME_SCHEMA
        or type(requested) is not int
        or type(cutoff) is not int
        or type(deadline) is not int
        or not requested <= now_unix <= cutoff < deadline
        or cutoff != deadline - 5
        or not 30 <= deadline - requested <= 900
    ):
        raise OwnerLauncherError("invalid_owner_approval_request")
    for name in (
        "full_canary_plan_sha256",
        "staged_plan_file_sha256",
        "fixture_sha256",
        "phase_b_readiness_anchor_sha256",
        "phase_b_approval_sha256",
        "owner_subject_sha256",
        "approval_source_sha256",
    ):
        _require_sha256(request.get(name), "invalid_owner_approval_request")
    return request


def validate_session_bound_final_owner_approval(
    value: Any,
    *,
    request: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate one final owner decision bound to the just-built plan."""

    if not isinstance(value, Mapping) or set(value) != _FINAL_APPROVAL_KEYS:
        raise OwnerLauncherError("invalid_final_owner_approval")
    approval = copy.deepcopy(dict(value))
    approved = approval.get("approved_at_unix")
    expires = approval.get("expires_at_unix")
    if (
        approval.get("schema") != "muncho-full-canary-owner-approval.v1"
        or approval.get("scope") != "full_canary_runtime_start"
        or approval.get("plan_sha256")
        != request.get("full_canary_plan_sha256")
        or approval.get("authority_kind")
        != "trusted_root_bootstrap_out_of_band_owner"
        or approval.get("cryptographic_owner_proof") is not False
        or approval.get("owner_subject_sha256")
        != request.get("owner_subject_sha256")
        or approval.get("approval_source_sha256")
        != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
        or approval.get("approval_source_sha256")
        != request.get("approval_source_sha256")
        or type(approved) is not int
        or type(expires) is not int
        or not request["requested_at_unix"] <= approved <= now_unix <= expires
        or approved > request["owner_input_cutoff_unix"]
        or not 1 <= expires - approved <= 900
        or expires > request["approval_deadline_unix"]
    ):
        raise OwnerLauncherError("invalid_final_owner_approval")
    _require_sha256(approval.get("nonce_sha256"), "invalid_final_owner_approval")
    return approval


def validate_session_bound_coordinator_receipt(
    value: Any,
    *,
    live_gate: Mapping[str, Any],
    request: Mapping[str, Any],
    approval: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate terminal live truth for the admin-free session-bound run."""

    _reject_secret_echo(
        value,
        active_secrets=(),
        code="invalid_coordinator_receipt",
    )
    receipt = _validate_self_digest(
        value,
        expected_keys=_SESSION_BOUND_COORDINATOR_RECEIPT_KEYS,
        digest_key="receipt_sha256",
        code="invalid_coordinator_receipt",
    )
    result = receipt.get("live_driver_result")
    if (
        receipt.get("schema") != SESSION_BOUND_COORDINATOR_RECEIPT_SCHEMA
        or receipt.get("ok") is not True
        or receipt.get("state")
        != "verified_stopped_and_credentials_retired"
        or receipt.get("release_sha") != live_gate.get("release_sha")
        or receipt.get("coordinator_input_sha256")
        != live_gate.get("coordinator_input_sha256")
        or receipt.get("full_canary_plan_sha256")
        != request.get("full_canary_plan_sha256")
        or receipt.get("owner_approval_sha256")
        != _sha256(_canonical_bytes(approval))
        or receipt.get("phase_b_readiness_anchor_sha256")
        != live_gate.get("phase_b_readiness_anchor_sha256")
        or receipt.get("fixture_sha256") != request.get("fixture_sha256")
        or receipt.get("services_stopped") is not True
        or receipt.get("discord_token_retired") is not True
        or receipt.get("temporary_admin_created") is not False
        or receipt.get("bootstrap_credential_created") is not False
        or not isinstance(result, Mapping)
        or set(result) != _LIVE_DRIVER_RESULT_KEYS
        or result.get("schema") != "muncho-full-canary-live-driver.v1"
        or result.get("ok") is not True
        or result.get("release_sha") != live_gate.get("release_sha")
        or result.get("full_canary_plan_sha256")
        != receipt.get("full_canary_plan_sha256")
        or result.get("discord_ingress_claimed") is not False
        or not isinstance(result.get("offline_invariant_receipt"), Mapping)
        or not isinstance(result.get("lifecycle_verification_receipt"), Mapping)
        or receipt.get("live_driver_receipt_sha256")
        != _sha256(_canonical_bytes(result))
    ):
        raise OwnerLauncherError("invalid_coordinator_receipt")
    for name in (
        "full_canary_plan_sha256",
        "owner_approval_sha256",
        "phase_b_readiness_anchor_sha256",
        "api_session_key_sha256",
        "fixture_sha256",
        "live_driver_receipt_sha256",
    ):
        _require_sha256(receipt.get(name), "invalid_coordinator_receipt")
    evidence_path = result.get("evidence_path")
    expected_evidence_path = (
        f"/var/lib/muncho-full-canary/plans/{live_gate['release_sha']}/"
        f"{receipt['full_canary_plan_sha256']}/live/evidence.json"
    )
    if evidence_path != expected_evidence_path:
        raise OwnerLauncherError("invalid_coordinator_receipt")
    _require_sha256(result.get("evidence_sha256"), "invalid_coordinator_receipt")
    _validate_receipt_time(
        receipt.get("completed_at_unix"),
        now_unix=now_unix,
        code="invalid_coordinator_receipt",
    )
    return receipt


def apply_phase_b_foundation(
    *,
    release_sha: str,
    transport: CoordinatorTransport,
    cloud_sql_client: GoogleRestClient,
    owner_identity: ApprovedOwnerIdentity,
    now: Callable[[], int] = lambda: int(time.time()),
    password_factory: Callable[[], bytearray] | None = None,
    secret_hardener: Callable[[], None] = harden_owner_secret_process,
    provenance_guard: Callable[
        [str], None
    ] = require_owner_runtime_and_launcher_provenance,
) -> Mapping[str, Any]:
    """Drive only the standalone, bounded owner/VM Phase-B protocol."""

    if not _RELEASE_SHA.fullmatch(release_sha):
        raise OwnerLauncherError("invalid_release_sha")
    provenance_guard(release_sha)
    secret_hardener()
    gate = validate_phase_b_apply_gate(
        transport.preflight_phase_b_apply(release_sha),
        expected_release_sha=release_sha,
        now_unix=now(),
    )
    owner_identity.bind_approved_subject(str(gate["owner_subject_sha256"]))
    owner_identity.require_stable()
    session = transport.open_phase_b_apply(release_sha)
    primary: BaseException | None = None
    try:
        current = session.read_gate()
        if current.get("schema") == COORDINATOR_FAILURE_SCHEMA:
            _validated_input_gate(
                session,
                current,
                owner_gate=gate,
            )
            raise AssertionError("unreachable terminal Phase-B failure")
        if current.get("schema") == PHASE_B_APPLY_RECEIPT_SCHEMA:
            receipt = validate_phase_b_apply_receipt(
                current,
                owner_gate=gate,
                now_unix=now(),
            )
            session.mark_validated(receipt)
            owner_identity.require_stable()
            return receipt

        protocol = _PhaseBOwnerProtocol(
            gate=gate,
            cloud_sql_client=cloud_sql_client,
            password_factory=password_factory,
            clock=lambda: float(now()),
            authority_guard=owner_identity.require_stable,
        )
        for _round in range(PHASE_B_MAX_ROUNDS):
            owner_identity.require_stable()
            request = protocol.validate_request(current)
            frame, terminal = protocol.response_frame(request)
            try:
                owner_identity.require_stable()
                current = session.phase_b_exchange(frame, terminal=terminal)
            finally:
                _wipe(frame)
            if current.get("schema") == COORDINATOR_FAILURE_SCHEMA:
                _validated_input_gate(
                    session,
                    current,
                    owner_gate=gate,
                )
                raise AssertionError("unreachable terminal Phase-B failure")
            if terminal:
                receipt = validate_phase_b_apply_receipt(
                    current,
                    owner_gate=gate,
                    now_unix=now(),
                )
                session.mark_validated(receipt)
                owner_identity.require_stable()
                return receipt
            if current.get("schema") == PHASE_B_APPLY_RECEIPT_SCHEMA:
                raise OwnerLauncherError("phase_b_terminal_response_out_of_order")
        raise OwnerLauncherError("phase_b_round_limit_exceeded")
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _close_session_preserving_primary(session, primary)


def _cleanup_schema_reconciliation_admin(
    boundary: CloudSqlTemporaryAdmin,
    *,
    username: str,
) -> Mapping[str, Any]:
    """Prove the deterministic temporary admin absent without blind delete."""

    try:
        boundary.require_current_authority(username)
    except OwnerLauncherError as exc:
        try:
            if boundary.mutation_reconciliation_required():
                boundary.reconcile_ambiguous_mutation_and_confirm_absent(
                    username
                )
            else:
                boundary.delete_and_confirm_absent(username)
        except CleanupBlocked:
            raise
        except OwnerLauncherError as cleanup_error:
            raise CleanupBlocked(cleanup_error.code) from None
    else:
        try:
            boundary.delete_and_confirm_absent(username)
        except CleanupBlocked:
            raise
        except OwnerLauncherError as exc:
            raise CleanupBlocked(exc.code) from None
    try:
        return boundary.reconciliation_receipt()
    except OwnerLauncherError as exc:
        raise CleanupBlocked(exc.code) from None


def reconcile_legacy_canary_schema(
    *,
    release_sha: str,
    transport: IapSchemaReconciliationTransport,
    cloud_sql_client: GoogleRestClient,
    owner_identity: GcloudOwnerAccessToken,
    now: Callable[[], int] = lambda: int(time.time()),
    password_factory: Callable[[], bytearray] = _new_admin_password,
    nonce_factory: Callable[[int], bytes] = secrets.token_bytes,
    signer: _PhaseBOwnerExternalSigner | None = None,
    boundary_factory: Callable[
        [GoogleRestClient], CloudSqlTemporaryAdmin
    ] = CloudSqlSchemaReconciliationAdmin,
    secret_hardener: Callable[[], None] = harden_owner_secret_process,
    provenance_guard: Callable[
        [str], None
    ] = require_owner_runtime_and_launcher_provenance,
) -> Mapping[str, Any]:
    """Run only the stopped G0/A1/P1/A2/I2/C3/T3 repair protocol."""

    if not isinstance(release_sha, str) or _RELEASE_SHA.fullmatch(release_sha) is None:
        raise OwnerLauncherError("invalid_release_sha")
    from gateway import canonical_writer_schema_reconciliation_bootstrap as bootstrap

    signal_fence = _OwnerSignalFence()
    signal_fence.install()
    session: _IapRemoteSession | None = None
    boundary: CloudSqlTemporaryAdmin | None = None
    credential: bytearray | None = None
    admin_frame: bytearray | None = None
    authorization_frame: bytearray | None = None
    cleanup_frame: bytearray | None = None
    cleanup_receipt: Mapping[str, Any] | None = None
    terminal: Mapping[str, Any] | None = None
    expected_username: str | None = None
    primary: BaseException | None = None
    mutation_request_started = False
    mutation_may_exist = False
    database_intermediate_validated = False
    database_capability_terminated = False
    cleanup_complete = False
    try:
        provenance_guard(release_sha)
        secret_hardener()
        account = owner_identity.account_for_read_only_preflight()
        expected_owner_subject = _sha256(account.encode("utf-8"))
        owner_signer = signer or _PhaseBOwnerExternalSigner()
        owner_authority = owner_signer.inspect()
        if (
            owner_authority.public_fingerprint
            != PHASE_B_OWNER_PUBLIC_KEY_FINGERPRINT
            or _sha256(owner_authority.public_fingerprint.encode("ascii"))
            != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
        ):
            raise OwnerLauncherError(
                "schema_reconciliation_owner_authority_invalid"
            )

        session = transport.open_reconciliation(release_sha)
        gate_raw = session.read_gate()
        current = now()
        try:
            gate = bootstrap.validate_gate(
                gate_raw,
                owner_public_key_ed25519_hex=(
                    owner_authority.public_key_ed25519_hex
                ),
                owner_public_fingerprint=owner_authority.public_fingerprint,
                now_unix=current,
            )
        except BaseException as exc:
            if isinstance(exc, OwnerLauncherError):
                raise
            raise OwnerLauncherError(
                "schema_reconciliation_gate_invalid"
            ) from None
        expected_username = ADMIN_USERNAME_PREFIX + gate["plan_sha256"][:16]
        if (
            gate["release_revision"] != release_sha
            or gate["owner_subject_sha256"] != expected_owner_subject
            or gate["owner_public_key_ed25519_hex"]
            != owner_authority.public_key_ed25519_hex
            or gate["owner_key_id"] != owner_authority.key_id
            or gate["owner_public_fingerprint"]
            != owner_authority.public_fingerprint
            or gate["temporary_admin_username"] != expected_username
            or current + SCHEMA_RECONCILIATION_MIN_GATE_REMAINING_SECONDS
            >= gate["expires_at_unix"]
        ):
            raise OwnerLauncherError(
                "schema_reconciliation_gate_binding_invalid"
            )
        owner_identity.bind_approved_subject(expected_owner_subject)
        owner_identity.require_stable()
        session.require_current_authority()
        provenance_guard(release_sha)

        boundary = boundary_factory(cloud_sql_client)
        boundary.begin_mutation_observation(
            expected_owner_subject_sha256=expected_owner_subject,
            expected_mutation_context_sha256=gate["gate_sha256"],
        )
        credential = password_factory()
        if (
            not isinstance(credential, bytearray)
            or len(credential) != SCHEMA_RECONCILIATION_CREDENTIAL_BYTES
        ):
            _wipe(credential if isinstance(credential, bytearray) else None)
            credential = None
            raise OwnerLauncherError(
                "schema_reconciliation_admin_credential_invalid"
            )
        try:
            credential_text = credential.decode("ascii", errors="strict")
        except UnicodeError:
            raise OwnerLauncherError(
                "schema_reconciliation_admin_credential_invalid"
            ) from None
        if re.fullmatch(r"[A-Za-z0-9_-]{64}", credential_text) is None:
            credential_text = ""
            raise OwnerLauncherError(
                "schema_reconciliation_admin_credential_invalid"
            )
        mutation_request_started = True
        try:
            boundary.create_or_rotate_recovery(expected_username, credential_text)
        finally:
            credential_text = ""
        mutation_may_exist = True
        cloud_authority = boundary.temporary_admin_authority_receipt(
            expected_username
        )
        admin_preflight = build_schema_reconciliation_admin_preflight(
            gate=gate,
            cloud_sql_authority_receipt=cloud_authority,
            credential=credential,
            signer=owner_signer,
            owner_authority=owner_authority,
            now_unix=now(),
            nonce_factory=nonce_factory,
        )
        admin_frame = _schema_reconciliation_frame(
            SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_MAGIC,
            admin_preflight,
            credential=credential,
        )

        def guard_admin_first_byte() -> None:
            provenance_guard(release_sha)
            owner_identity.require_stable()
            session.require_current_authority()
            if owner_signer.inspect() != owner_authority:
                raise OwnerLauncherError(
                    "schema_reconciliation_owner_authority_changed"
                )
            if (
                now() + _SECRET_FRAME_TRANSMIT_MARGIN_SECONDS
                >= admin_preflight["expires_at_unix"]
            ):
                raise OwnerLauncherError(
                    "schema_reconciliation_admin_preflight_expired"
                )
            boundary.require_current_authority(expected_username)

        def wipe_admin_material_after_write() -> None:
            nonlocal credential, admin_frame
            _wipe(credential)
            _wipe(admin_frame)
            credential = None
            admin_frame = None

        challenge_raw = session.schema_reconciliation_exchange_before(
            admin_frame,
            write_guard=guard_admin_first_byte,
            on_first_write=lambda: None,
            on_write_complete=wipe_admin_material_after_write,
        )
        _wipe(credential)
        _wipe(admin_frame)
        credential = None
        admin_frame = None
        try:
            challenge = bootstrap.validate_preflight_challenge_for_owner(
                challenge_raw,
                gate=gate,
                admin_preflight=admin_preflight,
                now_unix=now(),
            )
        except BaseException:
            raise OwnerLauncherError(
                "schema_reconciliation_preflight_challenge_invalid"
            ) from None

        owner_identity.require_stable()
        session.require_current_authority()
        provenance_guard(release_sha)
        boundary.require_current_authority(expected_username)
        authorization = build_schema_reconciliation_preflight_authorization(
            gate=gate,
            admin_preflight=admin_preflight,
            challenge=challenge,
            signer=owner_signer,
            owner_authority=owner_authority,
            now_unix=now(),
            nonce_factory=nonce_factory,
        )
        authorization_frame = _schema_reconciliation_frame(
            SCHEMA_RECONCILIATION_PREFLIGHT_AUTHORIZATION_MAGIC,
            authorization,
        )
        intermediate_raw = session.schema_reconciliation_exchange(
            authorization_frame,
            terminal=False,
        )
        _wipe(authorization_frame)
        authorization_frame = None
        try:
            intermediate = bootstrap.validate_database_intermediate_for_owner(
                intermediate_raw,
                gate=gate,
                admin_preflight=admin_preflight,
                challenge=challenge,
                authorization=authorization,
                now_unix=now(),
            )
        except BaseException:
            raise OwnerLauncherError(
                "schema_reconciliation_database_intermediate_invalid"
            ) from None
        database_intermediate_validated = True
        database_capability_terminated = True

        cleanup_receipt = _cleanup_schema_reconciliation_admin(
            boundary,
            username=expected_username,
        )
        cleanup_complete = True
        cleanup = build_schema_reconciliation_admin_cleanup(
            gate=gate,
            admin_preflight=admin_preflight,
            challenge=challenge,
            authorization=authorization,
            intermediate=intermediate,
            cloud_sql_absence_receipt=cleanup_receipt,
            signer=owner_signer,
            owner_authority=owner_authority,
            now_unix=now(),
            nonce_factory=nonce_factory,
        )
        cleanup_frame = _schema_reconciliation_frame(
            SCHEMA_RECONCILIATION_ADMIN_CLEANUP_MAGIC,
            cleanup,
        )
        terminal_raw = session.schema_reconciliation_exchange(
            cleanup_frame,
            terminal=True,
        )
        _wipe(cleanup_frame)
        cleanup_frame = None
        try:
            terminal = bootstrap.validate_terminal_for_owner(
                terminal_raw,
                gate=gate,
                admin_preflight=admin_preflight,
                challenge=challenge,
                authorization=authorization,
                intermediate=intermediate,
                cleanup=cleanup,
                now_unix=now(),
            )
        except BaseException:
            raise OwnerLauncherError(
                "schema_reconciliation_terminal_invalid"
            ) from None
        session.mark_validated(terminal)
        session.close()
        session = None
        owner_identity.require_stable()
        provenance_guard(release_sha)
    except BaseException as exc:
        primary = exc
    finally:
        signal_fence.begin_cleanup()
        _wipe(credential)
        _wipe(admin_frame)
        _wipe(authorization_frame)
        _wipe(cleanup_frame)
        credential = None
        admin_frame = None
        authorization_frame = None
        cleanup_frame = None
        reconciliation_required = False
        if (
            boundary is not None
            and expected_username is not None
            and not cleanup_complete
            and not mutation_request_started
            and not mutation_may_exist
        ):
            try:
                reconciliation_required = (
                    boundary.mutation_reconciliation_required()
                )
            except BaseException as reconciliation_error:
                code = (
                    reconciliation_error.code
                    if isinstance(reconciliation_error, OwnerLauncherError)
                    else "schema_reconciliation_admin_cleanup_state_failed"
                )
                state_error = CleanupBlocked(code)
                if primary is None:
                    primary = state_error
                else:
                    _attach_cleanup_failure(
                        primary,
                        OwnerLauncherError(state_error.cause_code),
                    )
        needs_cleanup = bool(
            boundary is not None
            and expected_username is not None
            and not cleanup_complete
            and (
                mutation_request_started
                or mutation_may_exist
                or reconciliation_required
            )
        )
        if (
            needs_cleanup
            and session is not None
            and not database_intermediate_validated
        ):
            try:
                session.abort_and_prove_terminated()
                database_capability_terminated = True
                session = None
            except BaseException as termination_error:
                code = (
                    termination_error.code
                    if isinstance(termination_error, OwnerLauncherError)
                    else "remote_termination_unconfirmed"
                )
                blocked = CleanupBlocked(code)
                if primary is not None:
                    _attach_cleanup_failure(blocked, primary)
                primary = blocked
        if (
            needs_cleanup
            and boundary is not None
            and expected_username is not None
            and database_capability_terminated
        ):
            try:
                cleanup_receipt = _cleanup_schema_reconciliation_admin(
                    boundary,
                    username=expected_username,
                )
                cleanup_complete = True
            except BaseException as cleanup_error:
                if not isinstance(cleanup_error, CleanupBlocked):
                    code = (
                        cleanup_error.code
                        if isinstance(cleanup_error, OwnerLauncherError)
                        else "schema_reconciliation_admin_cleanup_failed"
                    )
                    cleanup_error = CleanupBlocked(code)
                if primary is not None:
                    _attach_cleanup_failure(cleanup_error, primary)
                primary = cleanup_error
        if session is not None:
            _close_session_preserving_primary(session, primary)
        try:
            signal_fence.restore()
        except BaseException as cleanup_error:
            if primary is None:
                primary = cleanup_error
            else:
                _attach_cleanup_failure(primary, cleanup_error)

    if primary is not None:
        raise primary
    if terminal is None or cleanup_receipt is None or not cleanup_complete:
        raise OwnerLauncherError("schema_reconciliation_terminal_incomplete")
    return terminal


def _validate_stopped_release_service_states(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(_STOPPED_RELEASE_UNITS):
        raise OwnerLauncherError("stopped_release_service_state_invalid")
    result: list[dict[str, Any]] = []
    for raw, unit in zip(value, _STOPPED_RELEASE_UNITS, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != {
            "unit",
            "state",
            "properties",
        }:
            raise OwnerLauncherError("stopped_release_service_state_invalid")
        properties = raw.get("properties")
        if (
            not isinstance(properties, Mapping)
            or set(properties) != set(_STOPPED_RELEASE_SERVICE_PROPERTIES)
            or any(
                not isinstance(properties[name], str)
                or _CONTROL_RE.search(properties[name]) is not None
                for name in _STOPPED_RELEASE_SERVICE_PROPERTIES
            )
            or raw.get("unit") != unit
        ):
            raise OwnerLauncherError("stopped_release_service_state_invalid")
        observed = {
            name: properties[name] for name in _STOPPED_RELEASE_SERVICE_PROPERTIES
        }
        absent = {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "",
            "MainPID": "0",
            "FragmentPath": "",
            "DropInPaths": "",
        }
        disabled = {
            **absent,
            "LoadState": "loaded",
            "UnitFileState": "disabled",
            "FragmentPath": f"/etc/systemd/system/{unit}",
        }
        if observed == absent:
            state = "absent"
        elif observed == disabled:
            state = "disabled_inactive"
        else:
            raise OwnerLauncherError("stopped_release_service_state_invalid")
        if raw.get("state") != state:
            raise OwnerLauncherError("stopped_release_service_state_invalid")
        result.append({
            "unit": unit,
            "state": state,
            "properties": observed,
        })
    return result


def _validate_stopped_release_host(value: Any) -> dict[str, str]:
    fields = {
        "project_id",
        "project_number",
        "zone",
        "instance_name",
        "instance_id",
        "service_account_email",
        "gce_identity_sha256",
        "machine_id_sha256",
        "hostname_sha256",
        "host_identity_sha256",
        "boot_id_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or any(
            not isinstance(item, str)
            or not item
            or _CONTROL_RE.search(item) is not None
            for item in value.values()
        )
    ):
        raise OwnerLauncherError("stopped_release_host_invalid")
    result = {name: str(value[name]) for name in fields}
    expected_gce = {
        "project_id": PROJECT,
        "project_number": "39589465056",
        "zone": ZONE,
        "instance_name": VM_NAME,
        "instance_id": VM_INSTANCE_ID,
        "service_account_email": (
            "muncho-canary-v2-runtime@adventico-ai-platform.iam.gserviceaccount.com"
        ),
    }
    if any(result[name] != item for name, item in expected_gce.items()):
        raise OwnerLauncherError("stopped_release_host_invalid")
    for name in fields:
        if name.endswith("_sha256"):
            _require_sha256(result[name], "stopped_release_host_invalid")
    if result["gce_identity_sha256"] != _sha256(
        _canonical_bytes(expected_gce)
    ) or result["host_identity_sha256"] != _sha256(
        _canonical_bytes({
            "machine_id_sha256": result["machine_id_sha256"],
            "hostname_sha256": result["hostname_sha256"],
        })
    ):
        raise OwnerLauncherError("stopped_release_host_invalid")
    return result


def validate_stopped_release_plan(
    value: Mapping[str, Any],
    *,
    expected_release_sha: str,
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "revision",
        "source",
        "release_root",
        "release_manifest_path",
        "evidence_receipt_path",
        "host_identity_receipt_path",
        "python_version",
        "interpreter",
        "tools",
        "dedicated_host",
        "activation_inventory",
        "service_states",
        "plan_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != STOPPED_RELEASE_PLAN_SCHEMA
        or value.get("revision") != expected_release_sha
        or _RELEASE_SHA.fullmatch(expected_release_sha) is None
    ):
        raise OwnerLauncherError("stopped_release_plan_invalid")
    source_root = f"{STOPPED_RELEASE_SOURCE_BASE}/{expected_release_sha}"
    release_root = f"/opt/muncho-canary-releases/{expected_release_sha}"
    evidence_path = (
        f"{STOPPED_RELEASE_EVIDENCE_BASE}/{expected_release_sha}/"
        "stopped-release-publication.json"
    )
    source = value.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source) != {"repository", "root", "head_sha", "tree_sha"}
        or source.get("repository") != STOPPED_RELEASE_SOURCE_REPOSITORY
        or source.get("root") != source_root
        or source.get("head_sha") != expected_release_sha
        or not isinstance(source.get("tree_sha"), str)
        or _RELEASE_SHA.fullmatch(source["tree_sha"]) is None
    ):
        raise OwnerLauncherError("stopped_release_plan_invalid")
    tools = value.get("tools")
    if tools != {
        "git": "/usr/bin/git",
        "systemctl": "/usr/bin/systemctl",
        "uv": "/usr/local/bin/uv",
        "uv_cache": "/var/cache/muncho-writer-release",
    }:
        raise OwnerLauncherError("stopped_release_plan_invalid")
    expected_inventory = [
        {"path": path, "state": "absent"} for path in _STOPPED_RELEASE_ACTIVATION_PATHS
    ]
    if (
        value.get("release_root") != release_root
        or value.get("release_manifest_path") != f"{release_root}/release-manifest.json"
        or value.get("evidence_receipt_path") != evidence_path
        or value.get("host_identity_receipt_path") != STOPPED_RELEASE_HOST_RECEIPT_PATH
        or value.get("python_version") != STOPPED_RELEASE_PYTHON_VERSION
        or value.get("interpreter") != f"{release_root}/venv/bin/python"
        or value.get("activation_inventory") != expected_inventory
    ):
        raise OwnerLauncherError("stopped_release_plan_invalid")
    _validate_stopped_release_host(value.get("dedicated_host"))
    _validate_stopped_release_service_states(value.get("service_states"))
    plan_sha256 = _require_sha256(
        value.get("plan_sha256"),
        "stopped_release_plan_invalid",
    )
    unsigned = {name: item for name, item in value.items() if name != "plan_sha256"}
    if plan_sha256 != _sha256(_canonical_bytes(unsigned)):
        raise OwnerLauncherError("stopped_release_plan_invalid")
    return copy.deepcopy(dict(value))


def validate_stopped_release_receipt(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "ok",
        "state",
        "release_revision",
        "plan_sha256",
        "source",
        "dedicated_host",
        "activation_inventory",
        "service_state_before",
        "service_state_after",
        "services_stopped_and_disabled",
        "tools",
        "release_root",
        "release_manifest_path",
        "release_manifest_file_sha256",
        "release_artifact_sha256",
        "interpreter",
        "interpreter_sha256",
        "python_version",
        "retained_wheel_path",
        "retained_wheel_sha256",
        "build_constraints_sha256",
        "host_identity_receipt_path",
        "host_identity_receipt_file_sha256",
        "host_identity_receipt_sha256",
        "receipt_path",
        "created_at_unix",
        "receipt_sha256",
    }
    release_sha = plan.get("revision")
    validated_plan = validate_stopped_release_plan(
        plan,
        expected_release_sha=str(release_sha),
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != STOPPED_RELEASE_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "published_services_stopped"
        or value.get("release_revision") != release_sha
        or value.get("plan_sha256") != validated_plan["plan_sha256"]
        or value.get("source") != validated_plan["source"]
        or value.get("dedicated_host") != validated_plan["dedicated_host"]
        or value.get("activation_inventory") != validated_plan["activation_inventory"]
        or value.get("tools") != validated_plan["tools"]
        or value.get("services_stopped_and_disabled") is not True
    ):
        raise OwnerLauncherError("stopped_release_receipt_invalid")
    before = _validate_stopped_release_service_states(value.get("service_state_before"))
    after = _validate_stopped_release_service_states(value.get("service_state_after"))
    if before != validated_plan["service_states"] or after != before:
        raise OwnerLauncherError("stopped_release_receipt_invalid")
    release_root = f"/opt/muncho-canary-releases/{release_sha}"
    receipt_path = (
        f"{STOPPED_RELEASE_EVIDENCE_BASE}/{release_sha}/"
        "stopped-release-publication.json"
    )
    wheel_path = value.get("retained_wheel_path")
    if (
        value.get("release_root") != release_root
        or value.get("release_manifest_path") != f"{release_root}/release-manifest.json"
        or value.get("interpreter") != f"{release_root}/venv/bin/python"
        or value.get("python_version") != STOPPED_RELEASE_PYTHON_VERSION
        or value.get("host_identity_receipt_path") != STOPPED_RELEASE_HOST_RECEIPT_PATH
        or value.get("receipt_path") != receipt_path
        or not isinstance(wheel_path, str)
        or re.fullmatch(
            rf"{re.escape(release_root)}/artifacts/[A-Za-z0-9_.+-]+\.whl",
            wheel_path,
        )
        is None
        or type(value.get("created_at_unix")) is not int
        or value["created_at_unix"] < 0
    ):
        raise OwnerLauncherError("stopped_release_receipt_invalid")
    for name in (
        "release_manifest_file_sha256",
        "release_artifact_sha256",
        "interpreter_sha256",
        "retained_wheel_sha256",
        "build_constraints_sha256",
        "host_identity_receipt_file_sha256",
        "host_identity_receipt_sha256",
    ):
        _require_sha256(value.get(name), "stopped_release_receipt_invalid")
    receipt_sha256 = _require_sha256(
        value.get("receipt_sha256"),
        "stopped_release_receipt_invalid",
    )
    unsigned = {name: item for name, item in value.items() if name != "receipt_sha256"}
    if receipt_sha256 != _sha256(_canonical_bytes(unsigned)):
        raise OwnerLauncherError("stopped_release_receipt_invalid")
    return copy.deepcopy(dict(value))


def _host_receipt_rotation_id(
    *,
    release_sha: str,
    external_iam_policy_sha256: str,
    prior_file_sha256: str,
    prior_receipt_sha256: str,
    prior_boot_id_sha256: str,
    current_boot_id_sha256: str,
) -> str:
    return _sha256(_canonical_bytes({
        "external_iam_policy_sha256": external_iam_policy_sha256,
        "prior_boot_id_sha256": prior_boot_id_sha256,
        "prior_host_identity_receipt_file_sha256": prior_file_sha256,
        "prior_host_identity_receipt_sha256": prior_receipt_sha256,
        "release_revision": release_sha,
        "target_boot_id_sha256": current_boot_id_sha256,
    }))


def validate_host_receipt_rotation_plan(
    value: Mapping[str, Any],
    *,
    expected_release_sha: str,
    expected_external_iam_policy_sha256: str,
    expected_prior_file_sha256: str,
    expected_prior_receipt_sha256: str,
    expected_prior_boot_id_sha256: str,
    expected_current_boot_id_sha256: str,
) -> Mapping[str, Any]:
    """Validate the exact secret-free reboot transition before owner apply."""

    fields = {
        "schema",
        "release_revision",
        "external_iam_policy_sha256",
        "rotation_id",
        "rotation_root",
        "intent_path",
        "prior_archive_path",
        "tombstone_path",
        "completion_path",
        "host_identity_receipt_path",
        "prior_host_identity_receipt_file_sha256",
        "prior_host_identity_receipt_sha256",
        "prior_boot_id_sha256",
        "prior_observed_at_unix",
        "target_host",
        "target_boot_id_sha256",
        "service_states",
        "invariants",
        "plan_sha256",
    }
    expected = {
        "external_iam_policy_sha256": _require_sha256(
            expected_external_iam_policy_sha256,
            "host_receipt_rotation_plan_invalid",
        ),
        "prior_host_identity_receipt_file_sha256": _require_sha256(
            expected_prior_file_sha256,
            "host_receipt_rotation_plan_invalid",
        ),
        "prior_host_identity_receipt_sha256": _require_sha256(
            expected_prior_receipt_sha256,
            "host_receipt_rotation_plan_invalid",
        ),
        "prior_boot_id_sha256": _require_sha256(
            expected_prior_boot_id_sha256,
            "host_receipt_rotation_plan_invalid",
        ),
        "target_boot_id_sha256": _require_sha256(
            expected_current_boot_id_sha256,
            "host_receipt_rotation_plan_invalid",
        ),
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != HOST_RECEIPT_ROTATION_PLAN_SCHEMA
        or value.get("release_revision") != expected_release_sha
        or _RELEASE_SHA.fullmatch(expected_release_sha) is None
        or any(value.get(name) != item for name, item in expected.items())
        or type(value.get("prior_observed_at_unix")) is not int
        or value["prior_observed_at_unix"] < 0
    ):
        raise OwnerLauncherError("host_receipt_rotation_plan_invalid")
    host = _validate_stopped_release_host(value.get("target_host"))
    if host["boot_id_sha256"] != expected["target_boot_id_sha256"]:
        raise OwnerLauncherError("host_receipt_rotation_plan_invalid")
    service_states = _validate_stopped_release_service_states(
        value.get("service_states")
    )
    rotation_id = _host_receipt_rotation_id(
        release_sha=expected_release_sha,
        external_iam_policy_sha256=expected["external_iam_policy_sha256"],
        prior_file_sha256=expected["prior_host_identity_receipt_file_sha256"],
        prior_receipt_sha256=expected["prior_host_identity_receipt_sha256"],
        prior_boot_id_sha256=expected["prior_boot_id_sha256"],
        current_boot_id_sha256=expected["target_boot_id_sha256"],
    )
    root = f"{HOST_RECEIPT_ROTATION_ROOT}/{rotation_id}"
    if (
        value.get("rotation_id") != rotation_id
        or value.get("rotation_root") != root
        or value.get("intent_path") != f"{root}/intent.json"
        or value.get("prior_archive_path")
        != f"{root}/prior-host-identity.json"
        or value.get("tombstone_path") != f"{root}/tombstone.json"
        or value.get("completion_path") != f"{root}/completion.json"
        or value.get("host_identity_receipt_path")
        != STOPPED_RELEASE_HOST_RECEIPT_PATH
        or value.get("invariants")
        != {
            "services_started": False,
            "service_units_changed": False,
            "iam_mutated": False,
            "prior_receipt_archived_no_replace": True,
            "prior_receipt_tombstoned_before_retirement": True,
            "fresh_receipt_collected_on_target_boot": True,
        }
    ):
        raise OwnerLauncherError("host_receipt_rotation_plan_invalid")
    plan_sha256 = _require_sha256(
        value.get("plan_sha256"),
        "host_receipt_rotation_plan_invalid",
    )
    unsigned = {name: item for name, item in value.items() if name != "plan_sha256"}
    if plan_sha256 != _sha256(_canonical_bytes(unsigned)):
        raise OwnerLauncherError("host_receipt_rotation_plan_invalid")
    result = copy.deepcopy(dict(value))
    result["target_host"] = host
    result["service_states"] = service_states
    return result


def validate_host_receipt_rotation_receipt(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "ok",
        "state",
        "release_revision",
        "external_iam_policy_sha256",
        "rotation_id",
        "plan_sha256",
        "prior_archive_path",
        "prior_host_identity_receipt_file_sha256",
        "prior_host_identity_receipt_sha256",
        "prior_boot_id_sha256",
        "tombstone_path",
        "tombstone_receipt_sha256",
        "host_identity_receipt_path",
        "host_identity_receipt_file_sha256",
        "host_identity_receipt_sha256",
        "target_boot_id_sha256",
        "fresh_observed_at_unix",
        "service_states_before",
        "service_states_after",
        "services_started",
        "service_units_changed",
        "iam_mutated",
        "completion_path",
        "receipt_sha256",
    }
    validated_plan = validate_host_receipt_rotation_plan(
        plan,
        expected_release_sha=str(plan.get("release_revision")),
        expected_external_iam_policy_sha256=str(
            plan.get("external_iam_policy_sha256")
        ),
        expected_prior_file_sha256=str(
            plan.get("prior_host_identity_receipt_file_sha256")
        ),
        expected_prior_receipt_sha256=str(
            plan.get("prior_host_identity_receipt_sha256")
        ),
        expected_prior_boot_id_sha256=str(plan.get("prior_boot_id_sha256")),
        expected_current_boot_id_sha256=str(plan.get("target_boot_id_sha256")),
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != HOST_RECEIPT_ROTATION_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("state")
        != "target_boot_receipt_published_services_stopped"
        or value.get("release_revision") != validated_plan["release_revision"]
        or value.get("external_iam_policy_sha256")
        != validated_plan["external_iam_policy_sha256"]
        or value.get("rotation_id") != validated_plan["rotation_id"]
        or value.get("plan_sha256") != validated_plan["plan_sha256"]
        or value.get("prior_archive_path")
        != validated_plan["prior_archive_path"]
        or value.get("prior_host_identity_receipt_file_sha256")
        != validated_plan["prior_host_identity_receipt_file_sha256"]
        or value.get("prior_host_identity_receipt_sha256")
        != validated_plan["prior_host_identity_receipt_sha256"]
        or value.get("prior_boot_id_sha256")
        != validated_plan["prior_boot_id_sha256"]
        or value.get("tombstone_path") != validated_plan["tombstone_path"]
        or value.get("host_identity_receipt_path")
        != STOPPED_RELEASE_HOST_RECEIPT_PATH
        or value.get("target_boot_id_sha256")
        != validated_plan["target_boot_id_sha256"]
        or value.get("completion_path") != validated_plan["completion_path"]
        or value.get("service_states_before") != validated_plan["service_states"]
        or value.get("services_started") is not False
        or value.get("service_units_changed") is not False
        or value.get("iam_mutated") is not False
        or type(value.get("fresh_observed_at_unix")) is not int
        or value["fresh_observed_at_unix"] < 0
    ):
        raise OwnerLauncherError("host_receipt_rotation_receipt_invalid")
    service_states_after = _validate_stopped_release_service_states(
        value.get("service_states_after")
    )
    if service_states_after != validated_plan["service_states"]:
        raise OwnerLauncherError("host_receipt_rotation_receipt_invalid")
    for name in (
        "tombstone_receipt_sha256",
        "host_identity_receipt_file_sha256",
        "host_identity_receipt_sha256",
        "receipt_sha256",
    ):
        _require_sha256(
            value.get(name),
            "host_receipt_rotation_receipt_invalid",
        )
    unsigned = {name: item for name, item in value.items() if name != "receipt_sha256"}
    if value["receipt_sha256"] != _sha256(_canonical_bytes(unsigned)):
        raise OwnerLauncherError("host_receipt_rotation_receipt_invalid")
    return copy.deepcopy(dict(value))


def _validate_fixture_public_key(
    value: Any,
    *,
    expected_path: str,
) -> Mapping[str, Any]:
    fields = {
        "path",
        "file_sha256",
        "public_key_ed25519_hex",
        "key_id",
        "device",
        "inode",
        "uid",
        "gid",
        "mode",
        "size",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("path") != expected_path
        or value.get("uid") != 0
        or type(value.get("gid")) is not int
        or value["gid"] <= 0
        or value.get("mode") != "0440"
        or type(value.get("device")) is not int
        or value["device"] < 0
        or type(value.get("inode")) is not int
        or value["inode"] <= 0
        or type(value.get("size")) is not int
        or not 1 <= value["size"] <= 16 * 1024
        or not isinstance(value.get("public_key_ed25519_hex"), str)
        or re.fullmatch(r"[0-9a-f]{64}", value["public_key_ed25519_hex"])
        is None
    ):
        raise OwnerLauncherError("fixture_publication_plan_invalid")
    _require_sha256(value.get("file_sha256"), "fixture_publication_plan_invalid")
    key_id = _require_sha256(
        value.get("key_id"),
        "fixture_publication_plan_invalid",
    )
    if key_id != _sha256(bytes.fromhex(value["public_key_ed25519_hex"])):
        raise OwnerLauncherError("fixture_publication_plan_invalid")
    return copy.deepcopy(dict(value))


def _validate_fixture_publication_service_states(value: Any) -> Mapping[str, Any]:
    units = (
        "muncho-discord-egress.service",
        "muncho-canonical-writer.service",
        "hermes-cloud-gateway.service",
    )
    properties = {
        "LoadState",
        "ActiveState",
        "SubState",
        "UnitFileState",
        "MainPID",
        "FragmentPath",
        "DropInPaths",
        "Type",
        "NotifyAccess",
        "StatusText",
    }
    if not isinstance(value, Mapping) or set(value) != set(units):
        raise OwnerLauncherError("fixture_publication_plan_invalid")
    result: dict[str, Mapping[str, Any]] = {}
    for unit in units:
        state = value.get(unit)
        if (
            not isinstance(state, Mapping)
            or set(state) != properties
            or type(state.get("MainPID")) is not int
            or state["MainPID"] != 0
            or any(
                not isinstance(item, str)
                or _CONTROL_RE.search(item) is not None
                for name, item in state.items()
                if name != "MainPID"
            )
            or state.get("LoadState") not in {"loaded", "not-found"}
            or state.get("UnitFileState") not in {"disabled", ""}
            or state.get("ActiveState") not in {"inactive", "failed"}
            or state.get("DropInPaths") not in {"", "[]"}
        ):
            raise OwnerLauncherError("fixture_publication_plan_invalid")
        result[unit] = copy.deepcopy(dict(state))
    return result


def _validate_fixture_publication_fixture(
    value: Any,
    *,
    release_sha: str,
    release_artifact_sha256: str,
    canary_run_id: str,
    valid_from_unix_ms: int,
    valid_until_unix_ms: int,
    writer_public_hex: str,
    edge_public_hex: str,
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "canary_run_id",
        "release_sha",
        "release_artifact_sha256",
        "valid_from_unix_ms",
        "valid_until_unix_ms",
        "case_id",
        "owner_discord_user_id",
        "source",
        "model_route",
        "task_policy",
        "public_routeback",
        "discord_public_keys",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or "api_session_key_sha256" in value
        or value.get("schema") != "muncho-full-canary-e2e-fixture.v1"
        or value.get("canary_run_id") != canary_run_id
        or value.get("release_sha") != release_sha
        or value.get("release_artifact_sha256") != release_artifact_sha256
        or value.get("valid_from_unix_ms") != valid_from_unix_ms
        or value.get("valid_until_unix_ms") != valid_until_unix_ms
        or value.get("case_id") != f"case:full-canary:{canary_run_id}"
        or value.get("owner_discord_user_id")
        != FIXTURE_PUBLICATION_OWNER_DISCORD_USER_ID
        or value.get("source")
        != {
            "platform": "api_server",
            "control_protocol": "authenticated_loopback_api_server.v1",
            "host": "127.0.0.1",
            "port": 8642,
            "session_create_endpoint": "/api/sessions",
            "chat_stream_endpoint_template": (
                "/api/sessions/{session_id}/chat/stream"
            ),
        }
        or value.get("model_route")
        != {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "model": "gpt-5.6-sol",
            "initial_effort": "high",
            "elevated_effort": "max",
        }
    ):
        raise OwnerLauncherError("fixture_publication_plan_invalid")
    policy = value.get("task_policy")
    if (
        not isinstance(policy, Mapping)
        or set(policy)
        != {"minimum_completed_steps", "prompt", "prompt_sha256"}
        or policy.get("minimum_completed_steps") != 5
        or not isinstance(policy.get("prompt"), str)
        or not policy["prompt"].strip()
        or len(policy["prompt"]) > 16_000
        or policy.get("prompt_sha256") != FIXTURE_PUBLICATION_PROMPT_SHA256
        or _sha256(policy["prompt"].encode("utf-8"))
        != FIXTURE_PUBLICATION_PROMPT_SHA256
    ):
        raise OwnerLauncherError("fixture_publication_plan_invalid")
    if value.get("public_routeback") != {
        "target": {
            "target_type": "public_guild_channel",
            "guild_id": FIXTURE_PUBLICATION_GUILD_ID,
            "channel_id": FIXTURE_PUBLICATION_CHANNEL_ID,
        },
        "canonical_idempotency_key": f"full-canary-routeback:{canary_run_id}",
    } or value.get("discord_public_keys") != {
        "writer_capability_ed25519_hex": writer_public_hex,
        "edge_receipt_ed25519_hex": edge_public_hex,
    }:
        raise OwnerLauncherError("fixture_publication_plan_invalid")
    return copy.deepcopy(dict(value))


def validate_fixture_publication_plan(
    value: Mapping[str, Any],
    *,
    expected_release_sha: str,
    expected_external_iam_policy_sha256: str,
    expected_canary_run_id: str,
    now_unix_ms: int,
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "release_sha",
        "release_artifact_sha256",
        "activation_plan_sha256",
        "external_iam_policy_sha256",
        "canary_run_id",
        "valid_from_unix_ms",
        "valid_until_unix_ms",
        "owner_discord_user_id",
        "public_target",
        "prompt_sha256",
        "api_session_key_present",
        "fixture",
        "fixture_sha256",
        "fixture_path",
        "fixture_gid",
        "writer_public_key",
        "edge_public_key",
        "config_binding",
        "service_states",
        "publication_receipt_path",
        "invariants",
        "plan_sha256",
    }
    policy_sha256 = _require_sha256(
        expected_external_iam_policy_sha256,
        "fixture_publication_plan_invalid",
    )
    try:
        parsed_run = uuid.UUID(expected_canary_run_id)
    except (TypeError, ValueError, AttributeError) as exc:
        raise OwnerLauncherError("fixture_publication_plan_invalid") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != FIXTURE_PUBLICATION_PLAN_SCHEMA
        or value.get("release_sha") != expected_release_sha
        or _RELEASE_SHA.fullmatch(expected_release_sha) is None
        or value.get("external_iam_policy_sha256") != policy_sha256
        or value.get("canary_run_id") != expected_canary_run_id
        or parsed_run.int == 0
        or str(parsed_run) != expected_canary_run_id
        or value.get("owner_discord_user_id")
        != FIXTURE_PUBLICATION_OWNER_DISCORD_USER_ID
        or value.get("public_target")
        != {
            "target_type": "public_guild_channel",
            "guild_id": FIXTURE_PUBLICATION_GUILD_ID,
            "channel_id": FIXTURE_PUBLICATION_CHANNEL_ID,
        }
        or value.get("prompt_sha256") != FIXTURE_PUBLICATION_PROMPT_SHA256
        or value.get("api_session_key_present") is not False
        or value.get("fixture_path") != FIXTURE_PUBLICATION_PATH
        or type(value.get("fixture_gid")) is not int
        or value["fixture_gid"] <= 0
        or value.get("invariants")
        != {
            "services_started": False,
            "service_units_changed": False,
            "discord_dispatched": False,
            "iam_mutated": False,
            "private_keys_read": False,
            "api_session_secret_present": False,
        }
    ):
        raise OwnerLauncherError("fixture_publication_plan_invalid")
    for name in (
        "release_artifact_sha256",
        "activation_plan_sha256",
        "fixture_sha256",
        "plan_sha256",
    ):
        _require_sha256(value.get(name), "fixture_publication_plan_invalid")
    valid_from = value.get("valid_from_unix_ms")
    valid_until = value.get("valid_until_unix_ms")
    if (
        type(now_unix_ms) is not int
        or type(valid_from) is not int
        or type(valid_until) is not int
        or valid_until <= valid_from
        or valid_until - valid_from > 3_600_000
        or not valid_from <= now_unix_ms < valid_until
        or valid_until - now_unix_ms < 10 * 60 * 1000
    ):
        raise OwnerLauncherError("fixture_publication_plan_invalid")
    writer_key = _validate_fixture_public_key(
        value.get("writer_public_key"),
        expected_path=FIXTURE_WRITER_PUBLIC_KEY_PATH,
    )
    edge_key = _validate_fixture_public_key(
        value.get("edge_public_key"),
        expected_path=FIXTURE_EDGE_PUBLIC_KEY_PATH,
    )
    if writer_key["key_id"] == edge_key["key_id"]:
        raise OwnerLauncherError("fixture_publication_plan_invalid")
    config = value.get("config_binding")
    if (
        not isinstance(config, Mapping)
        or set(config)
        != {
            "writer_config_file_sha256",
            "gateway_config_file_sha256",
            "edge_config_file_sha256",
            "writer_edge_receipt_public_key_file",
            "writer_edge_receipt_public_key_id",
            "edge_writer_capability_public_key_file",
            "edge_writer_capability_public_key_id",
            "edge_receipt_public_key_id",
        }
        or config.get("writer_edge_receipt_public_key_file")
        != FIXTURE_EDGE_PUBLIC_KEY_PATH
        or config.get("writer_edge_receipt_public_key_id") != edge_key["key_id"]
        or config.get("edge_writer_capability_public_key_file")
        != FIXTURE_WRITER_PUBLIC_KEY_PATH
        or config.get("edge_writer_capability_public_key_id")
        != writer_key["key_id"]
        or config.get("edge_receipt_public_key_id") != edge_key["key_id"]
    ):
        raise OwnerLauncherError("fixture_publication_plan_invalid")
    for name in (
        "writer_config_file_sha256",
        "gateway_config_file_sha256",
        "edge_config_file_sha256",
    ):
        _require_sha256(config.get(name), "fixture_publication_plan_invalid")
    fixture = _validate_fixture_publication_fixture(
        value.get("fixture"),
        release_sha=expected_release_sha,
        release_artifact_sha256=value["release_artifact_sha256"],
        canary_run_id=expected_canary_run_id,
        valid_from_unix_ms=valid_from,
        valid_until_unix_ms=valid_until,
        writer_public_hex=writer_key["public_key_ed25519_hex"],
        edge_public_hex=edge_key["public_key_ed25519_hex"],
    )
    if (
        value.get("fixture_sha256") != _sha256(_canonical_bytes(fixture))
        or value.get("publication_receipt_path")
        != f"{FIXTURE_PUBLICATION_ROOT}/{value['fixture_sha256']}.json"
    ):
        raise OwnerLauncherError("fixture_publication_plan_invalid")
    service_states = _validate_fixture_publication_service_states(
        value.get("service_states")
    )
    unsigned = {name: item for name, item in value.items() if name != "plan_sha256"}
    if value["plan_sha256"] != _sha256(_canonical_bytes(unsigned)):
        raise OwnerLauncherError("fixture_publication_plan_invalid")
    result = copy.deepcopy(dict(value))
    result["fixture"] = fixture
    result["writer_public_key"] = writer_key
    result["edge_public_key"] = edge_key
    result["service_states"] = service_states
    return result


def validate_fixture_publication_receipt(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "ok",
        "state",
        "release_sha",
        "release_artifact_sha256",
        "activation_plan_sha256",
        "external_iam_policy_sha256",
        "canary_run_id",
        "owner_discord_user_id",
        "public_target",
        "prompt_sha256",
        "api_session_key_present",
        "fixture_path",
        "fixture_gid",
        "fixture_sha256",
        "fixture_file_sha256",
        "writer_public_key_id",
        "edge_public_key_id",
        "service_states_before",
        "service_states_after",
        "services_started",
        "discord_dispatched",
        "iam_mutated",
        "private_keys_read",
        "api_session_secret_present",
        "approved_plan_sha256",
        "publication_receipt_path",
        "published_at_unix_ms",
        "receipt_sha256",
    }
    if not isinstance(plan, Mapping):
        raise OwnerLauncherError("fixture_publication_receipt_invalid")
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != FIXTURE_PUBLICATION_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "base_fixture_published_services_stopped"
        or value.get("release_sha") != plan.get("release_sha")
        or value.get("release_artifact_sha256")
        != plan.get("release_artifact_sha256")
        or value.get("activation_plan_sha256")
        != plan.get("activation_plan_sha256")
        or value.get("external_iam_policy_sha256")
        != plan.get("external_iam_policy_sha256")
        or value.get("canary_run_id") != plan.get("canary_run_id")
        or value.get("owner_discord_user_id")
        != plan.get("owner_discord_user_id")
        or value.get("public_target") != plan.get("public_target")
        or value.get("prompt_sha256") != plan.get("prompt_sha256")
        or value.get("api_session_key_present") is not False
        or value.get("fixture_path") != plan.get("fixture_path")
        or value.get("fixture_gid") != plan.get("fixture_gid")
        or value.get("fixture_sha256") != plan.get("fixture_sha256")
        or value.get("fixture_file_sha256") != plan.get("fixture_sha256")
        or value.get("writer_public_key_id")
        != plan.get("writer_public_key", {}).get("key_id")
        or value.get("edge_public_key_id")
        != plan.get("edge_public_key", {}).get("key_id")
        or value.get("service_states_before") != plan.get("service_states")
        or value.get("services_started") is not False
        or value.get("discord_dispatched") is not False
        or value.get("iam_mutated") is not False
        or value.get("private_keys_read") is not False
        or value.get("api_session_secret_present") is not False
        or value.get("approved_plan_sha256") != plan.get("plan_sha256")
        or value.get("publication_receipt_path")
        != plan.get("publication_receipt_path")
        or type(value.get("published_at_unix_ms")) is not int
        or not plan.get("valid_from_unix_ms")
        <= value["published_at_unix_ms"]
        < plan.get("valid_until_unix_ms")
    ):
        raise OwnerLauncherError("fixture_publication_receipt_invalid")
    after = _validate_fixture_publication_service_states(
        value.get("service_states_after")
    )
    if after != plan["service_states"]:
        raise OwnerLauncherError("fixture_publication_receipt_invalid")
    receipt_sha = _require_sha256(
        value.get("receipt_sha256"),
        "fixture_publication_receipt_invalid",
    )
    unsigned = {name: item for name, item in value.items() if name != "receipt_sha256"}
    if receipt_sha != _sha256(_canonical_bytes(unsigned)):
        raise OwnerLauncherError("fixture_publication_receipt_invalid")
    return copy.deepcopy(dict(value))


class IapStoppedReleaseTransport(IapCoordinatorTransport):
    """Publish one stopped, revision-addressed release through fixed IAP argv."""

    _MODULE = "scripts.canary.writer_release"
    _REMOTE_PYTHON = "/usr/bin/python3"
    _REMOTE_GIT = "/usr/bin/git"
    _REMOTE_ENV = "/usr/bin/env"
    _REMOTE_FIND = "/usr/bin/find"

    @staticmethod
    def _source_root(release_sha: str) -> str:
        if not _RELEASE_SHA.fullmatch(release_sha):
            raise OwnerLauncherError("invalid_release_sha")
        return f"{STOPPED_RELEASE_SOURCE_BASE}/{release_sha}"

    @staticmethod
    def _fixed_remote_environment(*, chdir: str | None = None) -> tuple[str, ...]:
        values = (
            "HOME=/nonexistent",
            "LANG=C.UTF-8",
            "LC_ALL=C.UTF-8",
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONDONTWRITEBYTECODE=1",
        )
        return (
            IapStoppedReleaseTransport._REMOTE_ENV,
            "-i",
            *((f"--chdir={chdir}",) if chdir is not None else ()),
            *values,
        )

    def _remote_argv(
        self,
        remote_argv: Sequence[str],
        *,
        account: str,
    ) -> tuple[str, ...]:
        if (
            not remote_argv
            or any(
                not isinstance(item, str)
                or not item
                or _CONTROL_RE.search(item) is not None
                for item in remote_argv
            )
            or not isinstance(account, str)
            or GcloudOwnerAccessToken._ACCOUNT.fullmatch(account) is None
        ):
            raise OwnerLauncherError("stopped_release_remote_argv_invalid")
        command_prefix = self._gcloud_executable.trusted_command_prefix()
        if (
            len(command_prefix) != len(_GCLOUD_PYTHON_ISOLATION_ARGS) + 2
            or command_prefix[1:-1] != _GCLOUD_PYTHON_ISOLATION_ARGS
        ):
            raise OwnerLauncherError("invalid_gcloud_command_prefix")
        self._gcloud_configuration.assert_stable()
        known_hosts = self._known_hosts.absolute_path()
        private_key = self._known_hosts.private_key_path()
        self._known_hosts.public_key_line()
        if self._owner_identity.account_for_read_only_preflight() != account:
            raise OwnerLauncherError("stopped_release_owner_identity_changed")
        remote_command = shlex.join((
            "/usr/bin/sudo",
            "--non-interactive",
            "--",
            *remote_argv,
        ))
        return (
            *command_prefix,
            "compute",
            "ssh",
            f"{OS_LOGIN_USERNAME}@{VM_NAME}",
            f"--project={PROJECT}",
            f"--zone={ZONE}",
            f"--account={account}",
            "--plain",
            "--tunnel-through-iap",
            "--quiet",
            f"--command={remote_command}",
            *self._ssh_flags(known_hosts, private_key),
        )

    def _run_remote(
        self,
        remote_argv: Sequence[str],
        *,
        account: str,
        allowed_returncodes: frozenset[int] = frozenset({0}),
        timeout_seconds: float = 300.0,
        maximum_output_bytes: int = _HTTP_RESPONSE_MAX_BYTES,
    ) -> subprocess.CompletedProcess[bytes]:
        if (
            not allowed_returncodes
            or any(
                type(code) is not int or not 0 <= code <= 255
                for code in allowed_returncodes
            )
            or not 0 < timeout_seconds <= 2_400
            or not 0 < maximum_output_bytes <= _HTTP_RESPONSE_MAX_BYTES
        ):
            raise OwnerLauncherError("stopped_release_remote_contract_invalid")
        authorization_before = self._authorization_snapshot(account)
        argv = self._remote_argv(remote_argv, account=account)
        self._validate_dry_run(argv)
        if self._authorization_snapshot(account) != authorization_before:
            raise OwnerLauncherError("iap_ssh_authorization_changed")
        command_prefix = self._gcloud_executable.trusted_command_prefix()
        try:
            completed = self._preflight_runner(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=dict(
                    _owner_gcloud_environment(
                        self._gcloud_configuration,
                        command_prefix[0],
                    )
                ),
                shell=False,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            self._postflight()
            raise OwnerLauncherError("stopped_release_remote_unavailable") from None
        self._postflight()
        if self._authorization_snapshot(account) != authorization_before:
            raise OwnerLauncherError("iap_ssh_authorization_changed")
        if (
            completed.returncode not in allowed_returncodes
            or not isinstance(completed.stdout, bytes)
            or len(completed.stdout) > maximum_output_bytes
        ):
            raise OwnerLauncherError("stopped_release_remote_failed")
        return completed

    def _run_remote_input(
        self,
        remote_argv: Sequence[str],
        *,
        account: str,
        input_bytes: bytes,
        timeout_seconds: float = 300.0,
        maximum_input_bytes: int = _WRITER_AUTHORITY_MAX_FRAME_BYTES + 8,
        maximum_output_bytes: int = _HTTP_RESPONSE_MAX_BYTES,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run one exact remote command with bounded, secret-free framed stdin."""

        if (
            not isinstance(input_bytes, bytes)
            or not input_bytes
            or len(input_bytes) > maximum_input_bytes
            or not 0 < timeout_seconds <= 2_400
            or not 0 < maximum_output_bytes <= _HTTP_RESPONSE_MAX_BYTES
        ):
            raise OwnerLauncherError("stopped_release_remote_input_invalid")
        authorization_before = self._authorization_snapshot(account)
        argv = self._remote_argv(remote_argv, account=account)
        self._validate_dry_run(argv)
        if self._authorization_snapshot(account) != authorization_before:
            raise OwnerLauncherError("iap_ssh_authorization_changed")
        command_prefix = self._gcloud_executable.trusted_command_prefix()
        try:
            completed = self._preflight_runner(
                argv,
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=dict(
                    _owner_gcloud_environment(
                        self._gcloud_configuration,
                        command_prefix[0],
                    )
                ),
                shell=False,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            self._postflight()
            raise OwnerLauncherError("stopped_release_remote_unavailable") from None
        self._postflight()
        if self._authorization_snapshot(account) != authorization_before:
            raise OwnerLauncherError("iap_ssh_authorization_changed")
        if (
            completed.returncode != 0
            or not isinstance(completed.stdout, bytes)
            or len(completed.stdout) > maximum_output_bytes
        ):
            raise OwnerLauncherError("stopped_release_remote_failed")
        return completed

    def _revision_source_exists(self, release_sha: str, *, account: str) -> bool:
        self._source_root(release_sha)
        completed = self._run_remote(
            (
                self._REMOTE_FIND,
                STOPPED_RELEASE_SOURCE_BASE,
                "-mindepth",
                "1",
                "-maxdepth",
                "1",
                "-name",
                release_sha,
                "-printf",
                "%f",
            ),
            account=account,
            maximum_output_bytes=40,
        )
        if completed.stdout == b"":
            return False
        if completed.stdout == release_sha.encode("ascii"):
            return True
        raise OwnerLauncherError("stopped_release_path_probe_invalid")

    def _prepare_source(self, release_sha: str, *, account: str) -> str:
        source_root = self._source_root(release_sha)
        if self._revision_source_exists(release_sha, account=account):
            return source_root
        git_environment = (
            *self._fixed_remote_environment(),
            "GIT_CONFIG_GLOBAL=/dev/null",
            "GIT_CONFIG_NOSYSTEM=1",
        )
        self._run_remote(
            (
                *git_environment,
                self._REMOTE_GIT,
                "clone",
                "--no-checkout",
                "--no-tags",
                STOPPED_RELEASE_SOURCE_REPOSITORY,
                source_root,
            ),
            account=account,
            timeout_seconds=900.0,
        )
        self._run_remote(
            (
                *git_environment,
                self._REMOTE_GIT,
                "-C",
                source_root,
                "checkout",
                "--detach",
                release_sha,
            ),
            account=account,
        )
        return source_root

    def _run_release_command(
        self,
        release_sha: str,
        command: str,
        *,
        account: str,
        approved_plan_sha256: str | None = None,
    ) -> Mapping[str, Any]:
        if command not in {"plan", "apply"}:
            raise OwnerLauncherError("stopped_release_command_invalid")
        if command == "apply":
            approved = _require_sha256(
                approved_plan_sha256,
                "stopped_release_plan_invalid",
            )
        elif approved_plan_sha256 is not None:
            raise OwnerLauncherError("stopped_release_plan_invalid")
        else:
            approved = None
        source_root = self._source_root(release_sha)
        remote = (
            *self._fixed_remote_environment(chdir=source_root),
            self._REMOTE_PYTHON,
            "-B",
            "-E",
            "-s",
            "-m",
            self._MODULE,
            command,
            "--revision",
            release_sha,
            *(() if approved is None else ("--approved-plan-sha256", approved)),
        )
        completed = self._run_remote(
            remote,
            account=account,
            timeout_seconds=2_400.0 if command == "apply" else 300.0,
        )
        if (
            not completed.stdout
            or not completed.stdout.endswith(b"\n")
            or b"\n" in completed.stdout[:-1]
        ):
            raise OwnerLauncherError("stopped_release_output_invalid")
        try:
            return _decode_json_object(
                completed.stdout,
                maximum=_HTTP_RESPONSE_MAX_BYTES,
            )
        except OwnerLauncherError:
            raise OwnerLauncherError("stopped_release_output_invalid") from None

    def publish(self, release_sha: str) -> Mapping[str, Any]:
        if not _RELEASE_SHA.fullmatch(release_sha):
            raise OwnerLauncherError("invalid_release_sha")
        account = self._owner_identity.account_for_read_only_preflight()
        self._prepare_source(release_sha, account=account)
        plan = validate_stopped_release_plan(
            self._run_release_command(
                release_sha,
                "plan",
                account=account,
            ),
            expected_release_sha=release_sha,
        )
        plan_sha256 = str(plan["plan_sha256"])
        return validate_stopped_release_receipt(
            self._run_release_command(
                release_sha,
                "apply",
                account=account,
                approved_plan_sha256=plan_sha256,
            ),
            plan=plan,
        )


class IapHostReceiptRotationTransport(IapStoppedReleaseTransport):
    """Drive one exact stale-boot receipt transition before stopped release."""

    def _run_host_receipt_rotation_command(
        self,
        release_sha: str,
        command: str,
        *,
        account: str,
        external_iam_policy_sha256: str,
        expected_prior_file_sha256: str,
        expected_prior_receipt_sha256: str,
        expected_prior_boot_id_sha256: str,
        expected_current_boot_id_sha256: str,
        approved_plan_sha256: str | None = None,
    ) -> Mapping[str, Any]:
        if command not in {"plan", "apply"}:
            raise OwnerLauncherError("host_receipt_rotation_command_invalid")
        intent = tuple(
            _require_sha256(item, "host_receipt_rotation_plan_invalid")
            for item in (
                external_iam_policy_sha256,
                expected_prior_file_sha256,
                expected_prior_receipt_sha256,
                expected_prior_boot_id_sha256,
                expected_current_boot_id_sha256,
            )
        )
        if command == "apply":
            approved = _require_sha256(
                approved_plan_sha256,
                "host_receipt_rotation_plan_invalid",
            )
        elif approved_plan_sha256 is not None:
            raise OwnerLauncherError("host_receipt_rotation_plan_invalid")
        else:
            approved = None
        source_root = self._source_root(release_sha)
        remote = (
            *self._fixed_remote_environment(chdir=source_root),
            self._REMOTE_PYTHON,
            "-B",
            "-E",
            "-s",
            "-m",
            HOST_RECEIPT_ROTATION_MODULE,
            command,
            "--revision",
            release_sha,
            "--external-iam-policy-sha256",
            intent[0],
            "--expected-prior-file-sha256",
            intent[1],
            "--expected-prior-receipt-sha256",
            intent[2],
            "--expected-prior-boot-id-sha256",
            intent[3],
            "--expected-current-boot-id-sha256",
            intent[4],
            *(() if approved is None else ("--approved-plan-sha256", approved)),
        )
        completed = self._run_remote(
            remote,
            account=account,
            timeout_seconds=300.0,
        )
        if (
            not completed.stdout
            or not completed.stdout.endswith(b"\n")
            or b"\n" in completed.stdout[:-1]
        ):
            raise OwnerLauncherError("host_receipt_rotation_output_invalid")
        try:
            return _decode_json_object(
                completed.stdout,
                maximum=_HTTP_RESPONSE_MAX_BYTES,
            )
        except OwnerLauncherError:
            raise OwnerLauncherError("host_receipt_rotation_output_invalid") from None

    def rotate(
        self,
        release_sha: str,
        *,
        external_iam_policy_sha256: str,
        expected_prior_file_sha256: str,
        expected_prior_receipt_sha256: str,
        expected_prior_boot_id_sha256: str,
        expected_current_boot_id_sha256: str,
    ) -> Mapping[str, Any]:
        if not _RELEASE_SHA.fullmatch(release_sha):
            raise OwnerLauncherError("invalid_release_sha")
        exact = {
            "expected_release_sha": release_sha,
            "expected_external_iam_policy_sha256": external_iam_policy_sha256,
            "expected_prior_file_sha256": expected_prior_file_sha256,
            "expected_prior_receipt_sha256": expected_prior_receipt_sha256,
            "expected_prior_boot_id_sha256": expected_prior_boot_id_sha256,
            "expected_current_boot_id_sha256": expected_current_boot_id_sha256,
        }
        account = self._owner_identity.account_for_read_only_preflight()
        self._prepare_source(release_sha, account=account)
        remote_intent = {
            "external_iam_policy_sha256": external_iam_policy_sha256,
            "expected_prior_file_sha256": expected_prior_file_sha256,
            "expected_prior_receipt_sha256": expected_prior_receipt_sha256,
            "expected_prior_boot_id_sha256": expected_prior_boot_id_sha256,
            "expected_current_boot_id_sha256": expected_current_boot_id_sha256,
        }
        plan = validate_host_receipt_rotation_plan(
            self._run_host_receipt_rotation_command(
                release_sha,
                "plan",
                account=account,
                **remote_intent,
            ),
            **exact,
        )
        receipt = validate_host_receipt_rotation_receipt(
            self._run_host_receipt_rotation_command(
                release_sha,
                "apply",
                account=account,
                approved_plan_sha256=str(plan["plan_sha256"]),
                **remote_intent,
            ),
            plan=plan,
        )
        self._owner_identity.require_stable()
        return receipt


class IapFixturePublicationTransport(IapStoppedReleaseTransport):
    """Publish one fixed public-channel base fixture from the sealed release."""

    def _run_fixture_publication_command(
        self,
        release_sha: str,
        command: str,
        *,
        account: str,
        external_iam_policy_sha256: str,
        canary_run_id: str,
        valid_from_unix_ms: int,
        valid_until_unix_ms: int,
        approved_plan_sha256: str | None = None,
    ) -> Mapping[str, Any]:
        if command not in {"plan", "apply"}:
            raise OwnerLauncherError("fixture_publication_command_invalid")
        external_digest = _require_sha256(
            external_iam_policy_sha256,
            "fixture_publication_plan_invalid",
        )
        try:
            parsed_run_id = uuid.UUID(canary_run_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise OwnerLauncherError("fixture_publication_plan_invalid") from exc
        if parsed_run_id.int == 0 or str(parsed_run_id) != canary_run_id:
            raise OwnerLauncherError("fixture_publication_plan_invalid")
        if (
            type(valid_from_unix_ms) is not int
            or type(valid_until_unix_ms) is not int
            or valid_from_unix_ms < 1
            or valid_until_unix_ms <= valid_from_unix_ms
            or valid_until_unix_ms - valid_from_unix_ms > 3_600_000
        ):
            raise OwnerLauncherError("fixture_publication_plan_invalid")
        if command == "apply":
            approved = _require_sha256(
                approved_plan_sha256,
                "fixture_publication_plan_invalid",
            )
        elif approved_plan_sha256 is not None:
            raise OwnerLauncherError("fixture_publication_plan_invalid")
        else:
            approved = None
        interpreter = f"{self._RELEASE_BASE}/{release_sha}/venv/bin/python"
        remote = (
            *self._fixed_remote_environment(),
            interpreter,
            "-B",
            "-I",
            "-m",
            FIXTURE_PUBLICATION_MODULE,
            command,
            "--release-sha",
            release_sha,
            "--external-iam-policy-sha256",
            external_digest,
            "--canary-run-id",
            canary_run_id,
            "--valid-from-unix-ms",
            str(valid_from_unix_ms),
            "--valid-until-unix-ms",
            str(valid_until_unix_ms),
            "--owner-discord-user-id",
            FIXTURE_PUBLICATION_OWNER_DISCORD_USER_ID,
            "--guild-id",
            FIXTURE_PUBLICATION_GUILD_ID,
            "--channel-id",
            FIXTURE_PUBLICATION_CHANNEL_ID,
            "--expected-prompt-sha256",
            FIXTURE_PUBLICATION_PROMPT_SHA256,
            *(
                ()
                if approved is None
                else ("--approved-plan-sha256", approved)
            ),
        )
        completed = self._run_remote(
            remote,
            account=account,
            timeout_seconds=300.0,
        )
        if (
            not completed.stdout
            or not completed.stdout.endswith(b"\n")
            or b"\n" in completed.stdout[:-1]
        ):
            raise OwnerLauncherError("fixture_publication_output_invalid")
        try:
            return _decode_json_object(
                completed.stdout,
                maximum=_HTTP_RESPONSE_MAX_BYTES,
            )
        except OwnerLauncherError:
            raise OwnerLauncherError("fixture_publication_output_invalid") from None

    def publish(
        self,
        release_sha: str,
        *,
        external_iam_policy_sha256: str,
        clock: Callable[[], float] = time.time,
    ) -> Mapping[str, Any]:
        if not _RELEASE_SHA.fullmatch(release_sha):
            raise OwnerLauncherError("invalid_release_sha")
        external_digest = _require_sha256(
            external_iam_policy_sha256,
            "fixture_publication_plan_invalid",
        )
        first_now = clock()
        if not isinstance(first_now, (int, float)) or not first_now > 0:
            raise OwnerLauncherError("fixture_publication_plan_invalid")
        now_unix_ms = int(first_now * 1000)
        valid_from_unix_ms = now_unix_ms - 30_000
        valid_until_unix_ms = valid_from_unix_ms + 3_600_000
        canary_run_id = str(uuid.uuid4())
        account = self._owner_identity.account_for_read_only_preflight()
        exact = {
            "external_iam_policy_sha256": external_digest,
            "canary_run_id": canary_run_id,
            "valid_from_unix_ms": valid_from_unix_ms,
            "valid_until_unix_ms": valid_until_unix_ms,
        }
        plan_now = int(clock() * 1000)
        plan = validate_fixture_publication_plan(
            self._run_fixture_publication_command(
                release_sha,
                "plan",
                account=account,
                **exact,
            ),
            expected_release_sha=release_sha,
            expected_external_iam_policy_sha256=external_digest,
            expected_canary_run_id=canary_run_id,
            now_unix_ms=plan_now,
        )
        receipt = validate_fixture_publication_receipt(
            self._run_fixture_publication_command(
                release_sha,
                "apply",
                account=account,
                approved_plan_sha256=str(plan["plan_sha256"]),
                **exact,
            ),
            plan=plan,
        )
        self._owner_identity.require_stable()
        return receipt


_WRITER_PREFLIGHT_SYSTEMD_FIELDS = frozenset({
    "LoadState",
    "ActiveState",
    "SubState",
    "MainPID",
    "UnitFileState",
    "FragmentPath",
    "DropInPaths",
    "NeedDaemonReload",
})
_WRITER_PREFLIGHT_SERVICE_PATHS = {
    "muncho-canonical-writer.service": (
        "/etc/systemd/system/muncho-canonical-writer.service"
    ),
    "hermes-cloud-gateway.service": (
        "/etc/systemd/system/hermes-cloud-gateway.service"
    ),
    "muncho-canonical-writer-export.service": None,
    "muncho-discord-egress.service": None,
}
_WRITER_PREFLIGHT_PROVENANCE_FIELDS = frozenset({
    "approved_plan_sha256",
    "release_artifact_sha256",
    "release_manifest_file_sha256",
    "database_ca_sha256",
    "config_collector_receipt_sha256",
    "config_collector_receipt_file_sha256",
    "collector_writer_config_sha256",
    "collector_gateway_config_sha256",
    "native_observation_plan_sha256",
    "native_writer_config_sha256",
    "native_gateway_config_sha256",
    "native_writer_unit_sha256",
    "staged_phase_b_readiness_unit_sha256",
    "native_gateway_unit_sha256",
    "preflight_report_sha256",
    "preflight_report_file_sha256",
    "preflight_time_envelope_sha256",
})


def _validate_writer_preflight_service_state(value: Any) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(_WRITER_PREFLIGHT_SERVICE_PATHS)
    ):
        raise OwnerLauncherError("writer_preflight_plan_invalid")
    canonical: dict[str, dict[str, str]] = {}
    for unit, installed_path in _WRITER_PREFLIGHT_SERVICE_PATHS.items():
        state = value.get(unit)
        if (
            not isinstance(state, Mapping)
            or set(state) != _WRITER_PREFLIGHT_SYSTEMD_FIELDS
            or any(not isinstance(item, str) for item in state.values())
        ):
            raise OwnerLauncherError("writer_preflight_plan_invalid")
        absent = state == {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "MainPID": "0",
            "UnitFileState": "",
            "FragmentPath": "",
            "DropInPaths": "",
            "NeedDaemonReload": "no",
        }
        installed = installed_path is not None and state == {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "MainPID": "0",
            "UnitFileState": "disabled",
            "FragmentPath": installed_path,
            "DropInPaths": "",
            "NeedDaemonReload": "no",
        }
        if not absent and not installed:
            raise OwnerLauncherError("writer_preflight_plan_invalid")
        canonical[unit] = {name: state[name] for name in sorted(state)}
    return canonical


def validate_writer_preflight_plan(
    value: Mapping[str, Any],
    *,
    expected_release_sha: str,
    expected_external_iam_policy_sha256: str,
) -> Mapping[str, Any]:
    """Validate the complete secret-free remote staging envelope."""

    fields = {
        "schema",
        "revision",
        "stopped_release_receipt_path",
        "stopped_release_receipt_file_sha256",
        "stopped_release_receipt_sha256",
        "release_root",
        "release_artifact_sha256",
        "release_manifest_path",
        "release_manifest_file_sha256",
        "host_identity_receipt_path",
        "host_identity_receipt_file_sha256",
        "host_identity_receipt_sha256",
        "host_identity_sha256",
        "boot_id_sha256",
        "database",
        "credential_provenance",
        "owner_discord_user_ids",
        "external_iam_policy_sha256",
        "service_state",
        "fixed_output_paths",
        "invariants",
        "plan_sha256",
    }
    policy_sha256 = _require_sha256(
        expected_external_iam_policy_sha256,
        "writer_preflight_plan_invalid",
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != WRITER_PREFLIGHT_PLAN_SCHEMA
        or value.get("revision") != expected_release_sha
        or _RELEASE_SHA.fullmatch(expected_release_sha) is None
        or value.get("external_iam_policy_sha256") != policy_sha256
        or value.get("owner_discord_user_ids")
        != [WRITER_PREFLIGHT_OWNER_DISCORD_USER_ID]
    ):
        raise OwnerLauncherError("writer_preflight_plan_invalid")
    release_root = f"/opt/muncho-canary-releases/{expected_release_sha}"
    stopped_receipt = (
        f"{STOPPED_RELEASE_EVIDENCE_BASE}/{expected_release_sha}/"
        "stopped-release-publication.json"
    )
    if (
        value.get("release_root") != release_root
        or value.get("release_manifest_path")
        != f"{release_root}/release-manifest.json"
        or value.get("stopped_release_receipt_path") != stopped_receipt
        or value.get("host_identity_receipt_path")
        != STOPPED_RELEASE_HOST_RECEIPT_PATH
    ):
        raise OwnerLauncherError("writer_preflight_plan_invalid")
    for name in (
        "stopped_release_receipt_file_sha256",
        "stopped_release_receipt_sha256",
        "release_artifact_sha256",
        "release_manifest_file_sha256",
        "host_identity_receipt_file_sha256",
        "host_identity_receipt_sha256",
        "host_identity_sha256",
        "boot_id_sha256",
        "external_iam_policy_sha256",
    ):
        _require_sha256(value.get(name), "writer_preflight_plan_invalid")
    database = value.get("database")
    if (
        not isinstance(database, Mapping)
        or set(database)
        != {"host", "port", "database", "user", "tls_server_name", "ca_path", "ca_sha256"}
        or database.get("host") != DATABASE_HOST
        or database.get("port") != DATABASE_PORT
        or database.get("database") != DATABASE_NAME
        or database.get("user") != "muncho_canary_writer_login"
        or database.get("tls_server_name")
        != WRITER_PREFLIGHT_DATABASE_TLS_SERVER_NAME
        or database.get("ca_path")
        != "/etc/muncho/trust/cloudsql-server-ca.pem"
    ):
        raise OwnerLauncherError("writer_preflight_plan_invalid")
    _require_sha256(database.get("ca_sha256"), "writer_preflight_plan_invalid")
    credential = value.get("credential_provenance")
    credential_fields = {
        "path",
        "device",
        "inode",
        "owner_uid",
        "group_gid",
        "mode",
        "link_count",
        "modification_time_ns",
        "change_time_ns",
        "content_or_digest_recorded",
    }
    if (
        not isinstance(credential, Mapping)
        or set(credential) != credential_fields
        or credential.get("path")
        != "/etc/muncho/credentials/canonical-writer-db-password"
        or credential.get("owner_uid") != 999
        or credential.get("group_gid") != 994
        or credential.get("mode") != "0400"
        or credential.get("link_count") != 1
        or credential.get("content_or_digest_recorded") is not False
        or any(
            type(credential.get(name)) is not int or credential[name] < 0
            for name in (
                "device",
                "inode",
                "modification_time_ns",
                "change_time_ns",
            )
        )
    ):
        raise OwnerLauncherError("writer_preflight_plan_invalid")
    outputs = value.get("fixed_output_paths")
    expected_outputs = {
        "writer_config": "/etc/muncho/writer-activation/staged/writer.json",
        "gateway_config": "/etc/muncho/writer-activation/staged/gateway.yaml",
        "writer_unit": (
            "/etc/muncho/writer-activation/staged/"
            "muncho-canonical-writer.service"
        ),
        "phase_b_readiness_unit": (
            "/etc/muncho/writer-activation/staged/"
            "muncho-canonical-writer-phase-b-readiness.service"
        ),
        "gateway_unit": (
            "/etc/muncho/writer-activation/staged/"
            "hermes-cloud-gateway.service"
        ),
        "native_observation_plan": (
            "/etc/muncho/writer-activation/staged/"
            "native-observation-plan.json"
        ),
        "publication_evidence_root": WRITER_PREFLIGHT_EVIDENCE_BASE,
    }
    invariants = {
        "services_started": False,
        "units_installed": False,
        "daemon_reloaded": False,
        "approval_created": False,
        "discord_started": False,
        "credential_content_or_digest_recorded": False,
    }
    if outputs != expected_outputs or value.get("invariants") != invariants:
        raise OwnerLauncherError("writer_preflight_plan_invalid")
    _validate_writer_preflight_service_state(value.get("service_state"))
    plan_sha256 = _require_sha256(
        value.get("plan_sha256"),
        "writer_preflight_plan_invalid",
    )
    unsigned = {name: item for name, item in value.items() if name != "plan_sha256"}
    if plan_sha256 != _sha256(_canonical_bytes(unsigned)):
        raise OwnerLauncherError("writer_preflight_plan_invalid")
    return copy.deepcopy(dict(value))


def validate_writer_preflight_receipt(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "ok",
        "state",
        "revision",
        "approved_plan_sha256",
        "stopped_release_receipt_sha256",
        "release_artifact_sha256",
        "release_manifest_file_sha256",
        "host_identity_receipt_sha256",
        "config_collector_receipt_path",
        "config_collector_receipt_sha256",
        "config_collector_receipt_file_sha256",
        "native_observation_plan_sha256",
        "external_iam_policy_sha256",
        "preflight_report_path",
        "preflight_report_file_sha256",
        "preflight_report_sha256",
        "preflight_observed_at_unix",
        "preflight_collector_hba_observed_at_unix",
        "preflight_collector_collected_at_unix",
        "preflight_collector_hba_expires_at_unix",
        "preflight_time_envelope_sha256",
        "preflight_fresh_at_seal",
        "service_state_before",
        "service_state_after",
        "artifacts",
        "provenance",
        "invariants",
        "sealed_at_unix",
        "receipt_path",
        "receipt_sha256",
    }
    validated_plan = validate_writer_preflight_plan(
        plan,
        expected_release_sha=str(plan.get("revision")),
        expected_external_iam_policy_sha256=str(
            plan.get("external_iam_policy_sha256")
        ),
    )
    revision = str(validated_plan["revision"])
    plan_sha256 = str(validated_plan["plan_sha256"])
    expected_receipt_path = (
        f"{WRITER_PREFLIGHT_EVIDENCE_BASE}/{revision}/{plan_sha256}/"
        "publication.json"
    )
    expected_collector_path = (
        "/var/lib/muncho-writer-canary-evidence/config-collector/"
        f"{revision}/"
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != WRITER_PREFLIGHT_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("state")
        != "staged_preflight_passed_services_stopped"
        or value.get("revision") != revision
        or value.get("approved_plan_sha256") != plan_sha256
        or value.get("stopped_release_receipt_sha256")
        != validated_plan["stopped_release_receipt_sha256"]
        or value.get("release_artifact_sha256")
        != validated_plan["release_artifact_sha256"]
        or value.get("release_manifest_file_sha256")
        != validated_plan["release_manifest_file_sha256"]
        or value.get("host_identity_receipt_sha256")
        != validated_plan["host_identity_receipt_sha256"]
        or value.get("external_iam_policy_sha256")
        != validated_plan["external_iam_policy_sha256"]
        or value.get("invariants") != validated_plan["invariants"]
        or value.get("receipt_path") != expected_receipt_path
        or type(value.get("sealed_at_unix")) is not int
        or value["sealed_at_unix"] < 0
    ):
        raise OwnerLauncherError("writer_preflight_receipt_invalid")
    collector_sha = _require_sha256(
        value.get("config_collector_receipt_sha256"),
        "writer_preflight_receipt_invalid",
    )
    if value.get("config_collector_receipt_path") != (
        expected_collector_path + collector_sha + ".json"
    ):
        raise OwnerLauncherError("writer_preflight_receipt_invalid")
    for name in (
        "config_collector_receipt_file_sha256",
        "native_observation_plan_sha256",
        "preflight_report_file_sha256",
        "preflight_report_sha256",
        "preflight_time_envelope_sha256",
    ):
        _require_sha256(value.get(name), "writer_preflight_receipt_invalid")
    expected_report_path = (
        f"{WRITER_PREFLIGHT_EVIDENCE_BASE}/{revision}/{plan_sha256}/reports/"
        f"{value['preflight_report_sha256']}.json"
    )
    if value.get("preflight_report_path") != expected_report_path:
        raise OwnerLauncherError("writer_preflight_receipt_invalid")
    try:
        service_before = _validate_writer_preflight_service_state(
            value.get("service_state_before")
        )
        service_after = _validate_writer_preflight_service_state(
            value.get("service_state_after")
        )
        planned_service = _validate_writer_preflight_service_state(
            validated_plan.get("service_state")
        )
    except OwnerLauncherError:
        raise OwnerLauncherError("writer_preflight_receipt_invalid") from None
    if service_before != planned_service or service_after != service_before:
        raise OwnerLauncherError("writer_preflight_receipt_invalid")
    artifacts = value.get("artifacts")
    expected_artifact_names = {
        "writer_config",
        "gateway_config",
        "writer_unit",
        "phase_b_readiness_unit",
        "gateway_unit",
        "native_observation_plan",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifact_names:
        raise OwnerLauncherError("writer_preflight_receipt_invalid")
    for name, artifact in artifacts.items():
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {"path", "sha256"}
            or artifact.get("path")
            != validated_plan["fixed_output_paths"][name]
        ):
            raise OwnerLauncherError("writer_preflight_receipt_invalid")
        _require_sha256(
            artifact.get("sha256"),
            "writer_preflight_receipt_invalid",
        )
    if artifacts["native_observation_plan"]["sha256"] != value.get(
        "native_observation_plan_sha256"
    ):
        raise OwnerLauncherError("writer_preflight_receipt_invalid")
    provenance = value.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != _WRITER_PREFLIGHT_PROVENANCE_FIELDS
    ):
        raise OwnerLauncherError("writer_preflight_receipt_invalid")
    for item in provenance.values():
        _require_sha256(item, "writer_preflight_receipt_invalid")
    expected_provenance_bindings = {
        "approved_plan_sha256": validated_plan["plan_sha256"],
        "release_artifact_sha256": validated_plan["release_artifact_sha256"],
        "release_manifest_file_sha256": validated_plan[
            "release_manifest_file_sha256"
        ],
        "database_ca_sha256": validated_plan["database"]["ca_sha256"],
        "config_collector_receipt_sha256": value[
            "config_collector_receipt_sha256"
        ],
        "config_collector_receipt_file_sha256": value[
            "config_collector_receipt_file_sha256"
        ],
        "native_observation_plan_sha256": value[
            "native_observation_plan_sha256"
        ],
        "preflight_report_sha256": value["preflight_report_sha256"],
        "preflight_report_file_sha256": value[
            "preflight_report_file_sha256"
        ],
        "preflight_time_envelope_sha256": value[
            "preflight_time_envelope_sha256"
        ],
    }
    if any(
        provenance.get(name) != expected
        for name, expected in expected_provenance_bindings.items()
    ):
        raise OwnerLauncherError("writer_preflight_receipt_invalid")
    artifact_digest_bindings = {
        "writer_config": (
            "collector_writer_config_sha256",
            "native_writer_config_sha256",
        ),
        "gateway_config": (
            "collector_gateway_config_sha256",
            "native_gateway_config_sha256",
        ),
        "writer_unit": ("native_writer_unit_sha256",),
        "phase_b_readiness_unit": (
            "staged_phase_b_readiness_unit_sha256",
        ),
        "gateway_unit": ("native_gateway_unit_sha256",),
        "native_observation_plan": ("native_observation_plan_sha256",),
    }
    if any(
        artifacts[artifact_name]["sha256"] != provenance[provenance_name]
        for artifact_name, provenance_names in artifact_digest_bindings.items()
        for provenance_name in provenance_names
    ):
        raise OwnerLauncherError("writer_preflight_receipt_invalid")
    hba_observed_at = value.get("preflight_collector_hba_observed_at_unix")
    collector_collected_at = value.get("preflight_collector_collected_at_unix")
    preflight_observed_at = value.get("preflight_observed_at_unix")
    hba_expires_at = value.get("preflight_collector_hba_expires_at_unix")
    sealed_at = value["sealed_at_unix"]
    time_envelope = {
        "config_collector_receipt_sha256": value[
            "config_collector_receipt_sha256"
        ],
        "native_observation_plan_sha256": value[
            "native_observation_plan_sha256"
        ],
        "preflight_report_sha256": value["preflight_report_sha256"],
        "collector_hba_observed_at_unix": hba_observed_at,
        "collector_collected_at_unix": collector_collected_at,
        "observed_at_unix": preflight_observed_at,
        "collector_hba_expires_at_unix": hba_expires_at,
    }
    if (
        any(
            type(item) is not int or item < 0
            for item in (
                hba_observed_at,
                collector_collected_at,
                preflight_observed_at,
                hba_expires_at,
            )
        )
        or hba_expires_at - hba_observed_at != 300
        or not hba_observed_at
        <= collector_collected_at
        <= preflight_observed_at
        <= hba_expires_at
        or sealed_at < preflight_observed_at
        or type(value.get("preflight_fresh_at_seal")) is not bool
        or value["preflight_fresh_at_seal"] != (sealed_at <= hba_expires_at)
        or value.get("preflight_time_envelope_sha256")
        != _sha256(_canonical_bytes(time_envelope))
    ):
        raise OwnerLauncherError("writer_preflight_receipt_invalid")
    receipt_sha256 = _require_sha256(
        value.get("receipt_sha256"),
        "writer_preflight_receipt_invalid",
    )
    unsigned = {name: item for name, item in value.items() if name != "receipt_sha256"}
    if receipt_sha256 != _sha256(_canonical_bytes(unsigned)):
        raise OwnerLauncherError("writer_preflight_receipt_invalid")
    return copy.deepcopy(dict(value))


def _writer_owner_approval_path(
    *,
    scope: str,
    plan_sha256: str,
    receipt_sha256: str,
) -> str:
    if scope not in {"native_observation", "activation"}:
        raise OwnerLauncherError("writer_owner_approval_invalid")
    plan = _require_sha256(plan_sha256, "writer_owner_approval_invalid")
    receipt = _require_sha256(receipt_sha256, "writer_owner_approval_invalid")
    return f"{WRITER_OWNER_APPROVAL_ROOT}/{scope}/{plan}/{receipt}.json"


def build_writer_owner_approval(
    *,
    scope: str,
    plan_sha256: str,
    owner_identity: GcloudOwnerAccessToken,
    now_unix: int,
    nonce_factory: Callable[[int], bytes] = secrets.token_bytes,
) -> Mapping[str, Any]:
    """Author the existing honest out-of-band approval for one exact plan."""

    if scope not in {"native_observation", "activation"}:
        raise OwnerLauncherError("writer_owner_approval_invalid")
    plan = _require_sha256(plan_sha256, "writer_owner_approval_invalid")
    if type(now_unix) is not int or now_unix < 0:
        raise OwnerLauncherError("writer_owner_approval_invalid")
    account = owner_identity.account_for_read_only_preflight()
    owner_identity.require_stable()
    owner_subject = _sha256(account.encode("utf-8"))
    if owner_identity.owner_subject_sha256 != owner_subject:
        raise OwnerLauncherError("writer_owner_identity_invalid")
    authority = _PhaseBOwnerExternalSigner().inspect()
    if (
        authority.public_fingerprint != PHASE_B_OWNER_PUBLIC_KEY_FINGERPRINT
        or _sha256(authority.public_fingerprint.encode("ascii"))
        != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
    ):
        raise OwnerLauncherError("writer_owner_approval_source_invalid")
    try:
        nonce = nonce_factory(32)
    except BaseException:
        raise OwnerLauncherError("writer_owner_nonce_failed") from None
    if not isinstance(nonce, bytes) or len(nonce) != 32:
        raise OwnerLauncherError("writer_owner_nonce_failed")
    value = {
        "schema": "muncho-writer-owner-approval.v1",
        "scope": scope,
        "plan_sha256": plan,
        "authority_kind": "trusted_root_bootstrap_out_of_band_owner",
        "cryptographic_owner_proof": False,
        "owner_subject_sha256": owner_subject,
        "approval_source_sha256": PHASE_B_PINNED_APPROVAL_SOURCE_SHA256,
        "nonce_sha256": _sha256(nonce),
        "approved_at_unix": now_unix,
        "expires_at_unix": now_unix + 900,
    }
    try:
        from gateway.canonical_writer_host_authority import OwnerApprovalReceipt

        approval = OwnerApprovalReceipt.from_mapping(value)
        approval.require(scope=scope, plan_sha256=plan, now_unix=now_unix)
    except (ImportError, PermissionError, TypeError, ValueError):
        raise OwnerLauncherError("writer_owner_approval_invalid") from None
    owner_identity.require_stable()
    return approval.to_mapping()


def collect_fresh_writer_external_iam(
    *,
    owner_identity: GcloudOwnerAccessToken,
    source_approval_sha256: str,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Project fresh evidence through the exact injected read-only inventory."""

    source = _require_sha256(
        source_approval_sha256,
        "writer_external_iam_invalid",
    )
    if now_unix is not None and (type(now_unix) is not int or now_unix < 0):
        raise OwnerLauncherError("writer_external_iam_invalid")
    owner_identity.require_stable()
    try:
        from gateway.canonical_writer_host_authority import (
            build_external_iam_receipt,
        )
        from scripts.canary.foundation_preflight import (
            collect as collect_foundation,
            evaluate as evaluate_foundation,
        )
        from scripts.canary.host_preflight import (
            collect as collect_host,
            evaluate as evaluate_host,
        )

        runner = owner_identity.run_canary_iam_read_only_json
        foundation_report = evaluate_foundation(
            collect_foundation(run_json=runner)
        )
        host_report = evaluate_host(collect_host(run_json=runner))
        current = int(time.time()) if now_unix is None else now_unix
        receipt = build_external_iam_receipt(
            foundation_report,
            host_report,
            source_approval_sha256=source,
            now_unix=current,
        )
        receipt.require_fresh(current, minimum_remaining_seconds=720)
    except OwnerLauncherError:
        raise
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError):
        raise OwnerLauncherError("writer_external_iam_invalid") from None
    owner_identity.require_stable()
    return receipt.to_mapping()


def build_writer_authority_frame(
    *,
    action: str,
    revision: str,
    plan_sha256: str,
    owner_approval: Mapping[str, Any],
    external_iam_receipt: Mapping[str, Any],
    previous_owner_approval_sha256: str | None,
    previous_external_iam_receipt_sha256: str | None,
    now_unix: int,
) -> bytes:
    if action == "stage-native-authority":
        scope = "native_observation"
        if (
            previous_owner_approval_sha256 is not None
            or previous_external_iam_receipt_sha256 is not None
        ):
            raise OwnerLauncherError("writer_authority_frame_invalid")
    elif action == "replace-final-authority":
        scope = "activation"
        _require_sha256(
            previous_owner_approval_sha256,
            "writer_authority_frame_invalid",
        )
        _require_sha256(
            previous_external_iam_receipt_sha256,
            "writer_authority_frame_invalid",
        )
    else:
        raise OwnerLauncherError("writer_authority_frame_invalid")
    if _RELEASE_SHA.fullmatch(revision) is None or type(now_unix) is not int:
        raise OwnerLauncherError("writer_authority_frame_invalid")
    try:
        from gateway.canonical_writer_host_authority import (
            ExternalIAMReceipt,
            OwnerApprovalReceipt,
        )

        approval = OwnerApprovalReceipt.from_mapping(owner_approval)
        approval.require(
            scope=scope,
            plan_sha256=plan_sha256,
            now_unix=now_unix,
        )
        iam = ExternalIAMReceipt.from_mapping(external_iam_receipt)
        iam.require_fresh(now_unix, minimum_remaining_seconds=720)
    except (ImportError, PermissionError, TypeError, ValueError):
        raise OwnerLauncherError("writer_authority_frame_invalid") from None
    if (
        approval.value.get("approval_source_sha256")
        != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
        or iam.value.get("source_approval_sha256") != approval.sha256
    ):
        raise OwnerLauncherError("writer_authority_frame_invalid")
    unsigned = {
        "schema": WRITER_AUTHORITY_FRAME_SCHEMA,
        "action": action,
        "scope": scope,
        "revision": revision,
        "plan_sha256": _require_sha256(
            plan_sha256,
            "writer_authority_frame_invalid",
        ),
        "owner_subject_sha256": approval.value["owner_subject_sha256"],
        "approval_source_sha256": approval.value["approval_source_sha256"],
        "owner_approval": approval.to_mapping(),
        "external_iam_receipt": iam.to_mapping(),
        "previous_owner_approval_sha256": previous_owner_approval_sha256,
        "previous_external_iam_receipt_sha256": (
            previous_external_iam_receipt_sha256
        ),
        "framed_at_unix": now_unix,
    }
    payload = _canonical_bytes({
        **unsigned,
        "frame_sha256": _sha256(_canonical_bytes(unsigned)),
    })
    if len(payload) > _WRITER_AUTHORITY_MAX_FRAME_BYTES:
        raise OwnerLauncherError("writer_authority_frame_invalid")
    return WRITER_AUTHORITY_FRAME_MAGIC + struct.pack(">I", len(payload)) + payload


def _writer_authority_frame_sha256(frame: bytes) -> str:
    if (
        not isinstance(frame, bytes)
        or len(frame) < 9
        or frame[:4] != WRITER_AUTHORITY_FRAME_MAGIC
        or struct.unpack(">I", frame[4:8])[0] != len(frame) - 8
    ):
        raise OwnerLauncherError("writer_authority_frame_invalid")
    try:
        value = _decode_json_object(
            frame[8:],
            maximum=_WRITER_AUTHORITY_MAX_FRAME_BYTES,
        )
    except OwnerLauncherError:
        raise OwnerLauncherError("writer_authority_frame_invalid") from None
    return _require_sha256(
        value.get("frame_sha256"),
        "writer_authority_frame_invalid",
    )


class IapWriterPreflightTransport(IapStoppedReleaseTransport):
    """Run the sealed, no-service-start writer staging publisher over IAP."""

    _FAILURE_TYPES = {
        "RuntimeError": "runtime_failed",
        "ValueError": "validation_failed",
        "PermissionError": "approval_failed",
        "OSError": "io_failed",
    }

    @classmethod
    def _raise_structured_failure(
        cls,
        command: str,
        value: Mapping[str, Any],
    ) -> None:
        if (
            set(value) != {"schema", "ok", "error_code", "error_type"}
            or value.get("schema") != WRITER_PREFLIGHT_FAILURE_SCHEMA
            or value.get("ok") is not False
            or value.get("error_code") != "writer_preflight_publication_failed"
            or not isinstance(value.get("error_type"), str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value["error_type"])
            is None
        ):
            raise OwnerLauncherError("writer_preflight_output_invalid")
        category = cls._FAILURE_TYPES.get(
            value["error_type"],
            "remote_failed",
        )
        raise OwnerLauncherError(f"writer_preflight_{command}_{category}")

    def _run_writer_preflight_command(
        self,
        release_sha: str,
        command: str,
        *,
        account: str,
        external_iam_policy_sha256: str,
        approved_plan_sha256: str | None = None,
    ) -> Mapping[str, Any]:
        if command not in {"plan", "apply"}:
            raise OwnerLauncherError("writer_preflight_command_invalid")
        external_digest = _require_sha256(
            external_iam_policy_sha256,
            "writer_preflight_plan_invalid",
        )
        if command == "apply":
            approved = _require_sha256(
                approved_plan_sha256,
                "writer_preflight_plan_invalid",
            )
        elif approved_plan_sha256 is not None:
            raise OwnerLauncherError("writer_preflight_plan_invalid")
        else:
            approved = None
        interpreter = f"/opt/muncho-canary-releases/{release_sha}/venv/bin/python"
        remote = (
            *self._fixed_remote_environment(chdir="/"),
            interpreter,
            "-B",
            "-I",
            "-m",
            WRITER_PREFLIGHT_MODULE,
            command,
            "--revision",
            release_sha,
            "--external-iam-policy-sha256",
            external_digest,
            *(() if approved is None else ("--approved-plan-sha256", approved)),
        )
        completed = self._run_remote(
            remote,
            account=account,
            allowed_returncodes=frozenset({0, 2}),
            timeout_seconds=900.0 if command == "apply" else 300.0,
        )
        if (
            not completed.stdout
            or not completed.stdout.endswith(b"\n")
            or b"\n" in completed.stdout[:-1]
        ):
            raise OwnerLauncherError("writer_preflight_output_invalid")
        try:
            value = _decode_json_object(
                completed.stdout,
                maximum=_HTTP_RESPONSE_MAX_BYTES,
            )
        except OwnerLauncherError:
            raise OwnerLauncherError("writer_preflight_output_invalid") from None
        if completed.returncode == 2:
            self._raise_structured_failure(command, value)
        if value.get("schema") == WRITER_PREFLIGHT_FAILURE_SCHEMA:
            raise OwnerLauncherError("writer_preflight_output_invalid")
        return value

    def publish(
        self,
        release_sha: str,
        *,
        external_iam_policy_sha256: str,
    ) -> Mapping[str, Any]:
        if _RELEASE_SHA.fullmatch(release_sha) is None:
            raise OwnerLauncherError("invalid_release_sha")
        account = self._owner_identity.account_for_read_only_preflight()
        plan = validate_writer_preflight_plan(
            self._run_writer_preflight_command(
                release_sha,
                "plan",
                account=account,
                external_iam_policy_sha256=external_iam_policy_sha256,
            ),
            expected_release_sha=release_sha,
            expected_external_iam_policy_sha256=external_iam_policy_sha256,
        )
        return validate_writer_preflight_receipt(
            self._run_writer_preflight_command(
                release_sha,
                "apply",
                account=account,
                external_iam_policy_sha256=external_iam_policy_sha256,
                approved_plan_sha256=str(plan["plan_sha256"]),
            ),
            plan=plan,
        )


class IapWriterActivationBridgeTransport(IapStoppedReleaseTransport):
    """Drive the fixed packaged native/final stopped-only activation sequence."""

    _MODULES = frozenset({
        WRITER_ACTIVATION_BRIDGE_MODULE,
        WRITER_ACTIVATION_MODULE,
        WRITER_PLANNER_MODULE,
    })

    def _run_packaged_json(
        self,
        release_sha: str,
        *,
        module: str,
        arguments: Sequence[str],
        account: str,
        stdin_frame: bytes | None = None,
        timeout_seconds: float = 900.0,
        allowed_returncodes: frozenset[int] = frozenset({0}),
    ) -> Mapping[str, Any]:
        if (
            _RELEASE_SHA.fullmatch(release_sha) is None
            or module not in self._MODULES
            or not arguments
            or any(
                not isinstance(item, str)
                or not item
                or _CONTROL_RE.search(item) is not None
                for item in arguments
            )
        ):
            raise OwnerLauncherError("writer_activation_command_invalid")
        interpreter = f"/opt/muncho-canary-releases/{release_sha}/venv/bin/python"
        remote = (
            *self._fixed_remote_environment(chdir="/"),
            interpreter,
            "-B",
            "-I",
            "-m",
            module,
            *arguments,
        )
        if stdin_frame is None:
            completed = self._run_remote(
                remote,
                account=account,
                allowed_returncodes=allowed_returncodes,
                timeout_seconds=timeout_seconds,
            )
        else:
            if allowed_returncodes != frozenset({0}):
                raise OwnerLauncherError("writer_activation_command_invalid")
            completed = self._run_remote_input(
                remote,
                account=account,
                input_bytes=stdin_frame,
                timeout_seconds=timeout_seconds,
            )
        if (
            not completed.stdout
            or not completed.stdout.endswith(b"\n")
            or b"\n" in completed.stdout[:-1]
        ):
            raise OwnerLauncherError("writer_activation_output_invalid")
        try:
            decoded = _decode_json_object(
                completed.stdout,
                maximum=_HTTP_RESPONSE_MAX_BYTES,
            )
        except OwnerLauncherError:
            raise OwnerLauncherError("writer_activation_output_invalid") from None
        if completed.stdout[:-1] != _canonical_bytes(decoded):
            raise OwnerLauncherError("writer_activation_output_invalid")
        return decoded

    @staticmethod
    def _approval_receipt(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
        try:
            from gateway.canonical_writer_host_authority import OwnerApprovalReceipt

            receipt = OwnerApprovalReceipt.from_mapping(value)
        except (ImportError, TypeError, ValueError):
            raise OwnerLauncherError("writer_owner_approval_invalid") from None
        return receipt.to_mapping(), receipt.sha256

    @staticmethod
    def _external_iam_receipt(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], str, str]:
        try:
            from gateway.canonical_writer_host_authority import ExternalIAMReceipt

            receipt = ExternalIAMReceipt.from_mapping(value)
        except (ImportError, TypeError, ValueError):
            raise OwnerLauncherError("writer_external_iam_invalid") from None
        return receipt.to_mapping(), receipt.sha256, receipt.policy_sha256

    @staticmethod
    def _validate_install_plan(
        value: Mapping[str, Any],
        *,
        final: bool,
        release_sha: str,
    ) -> str:
        if final:
            expected_keys = {
                "ok",
                "schema",
                "revision",
                "activation_plan_sha256",
                "installed_path",
            }
            digest_key = "activation_plan_sha256"
            schema = "muncho-writer-only-activation-plan.v4"
            path = WRITER_FINAL_PLAN_PATH
        else:
            expected_keys = {
                "ok",
                "schema",
                "revision",
                "native_observation_plan_sha256",
                "installed_path",
            }
            digest_key = "native_observation_plan_sha256"
            schema = "muncho-writer-native-observation-plan.v2"
            path = WRITER_NATIVE_PLAN_PATH
        if (
            not isinstance(value, Mapping)
            or set(value) != expected_keys
            or value.get("ok") is not True
            or value.get("schema") != schema
            or value.get("revision") != release_sha
            or value.get("installed_path") != path
        ):
            raise OwnerLauncherError("writer_activation_plan_install_invalid")
        return _require_sha256(
            value.get(digest_key),
            "writer_activation_plan_install_invalid",
        )

    @staticmethod
    def _validate_authority_stage(
        value: Mapping[str, Any],
        *,
        action: str,
        release_sha: str,
        plan_sha256: str,
        owner_subject_sha256: str,
        owner_approval_sha256: str,
        external_iam_receipt_sha256: str,
        external_iam_policy_sha256: str,
        frame_sha256: str,
        previous_owner_approval_sha256: str | None,
        previous_external_iam_receipt_sha256: str | None,
    ) -> Mapping[str, Any]:
        expected_keys = {
            "schema",
            "ok",
            "state",
            "action",
            "scope",
            "revision",
            "plan_sha256",
            "frame_sha256",
            "owner_subject_sha256",
            "approval_source_sha256",
            "owner_approval_sha256",
            "external_iam_receipt_sha256",
            "external_iam_policy_sha256",
            "previous_owner_approval_sha256",
            "previous_external_iam_receipt_sha256",
            "archive",
            "owner_staged_present",
            "external_iam_staged_present",
            "services_started",
            "services_stopped",
            "intent_path",
            "intent_sha256",
            "receipt_path",
            "completed_at_unix",
            "receipt_sha256",
        }
        scope = (
            "native_observation"
            if action == "stage-native-authority"
            else "activation"
        )
        if (
            not isinstance(value, Mapping)
            or set(value) != expected_keys
            or value.get("schema") != WRITER_AUTHORITY_STAGE_RECEIPT_SCHEMA
            or value.get("ok") is not True
            or value.get("state") != "authority_staged_services_stopped"
            or value.get("action") != action
            or value.get("scope") != scope
            or value.get("revision") != release_sha
            or value.get("plan_sha256") != plan_sha256
            or value.get("frame_sha256") != frame_sha256
            or value.get("owner_subject_sha256") != owner_subject_sha256
            or value.get("approval_source_sha256")
            != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
            or value.get("owner_approval_sha256") != owner_approval_sha256
            or value.get("external_iam_receipt_sha256")
            != external_iam_receipt_sha256
            or value.get("external_iam_policy_sha256")
            != external_iam_policy_sha256
            or value.get("previous_owner_approval_sha256")
            != previous_owner_approval_sha256
            or value.get("previous_external_iam_receipt_sha256")
            != previous_external_iam_receipt_sha256
            or value.get("owner_staged_present") is not True
            or value.get("external_iam_staged_present") is not True
            or value.get("services_started") is not False
            or value.get("services_stopped") is not True
            or type(value.get("completed_at_unix")) is not int
        ):
            raise OwnerLauncherError("writer_authority_stage_invalid")
        for name in ("frame_sha256", "intent_sha256", "receipt_sha256"):
            _require_sha256(value.get(name), "writer_authority_stage_invalid")
        unsigned = {
            name: copy.deepcopy(item)
            for name, item in value.items()
            if name != "receipt_sha256"
        }
        if value["receipt_sha256"] != _sha256(_canonical_bytes(unsigned)):
            raise OwnerLauncherError("writer_authority_stage_invalid")
        expected_root = (
            "/var/lib/muncho-writer-canary-evidence/authority-bridge/"
            f"{release_sha}/{plan_sha256}/{frame_sha256}"
        )
        if (
            value.get("intent_path") != f"{expected_root}/intent.json"
            or value.get("receipt_path") != f"{expected_root}/receipt.json"
            or (
                action == "stage-native-authority"
                and value.get("archive") is not None
            )
            or (
                action == "replace-final-authority"
                and not isinstance(value.get("archive"), Mapping)
            )
        ):
            raise OwnerLauncherError("writer_authority_stage_invalid")
        if action == "replace-final-authority":
            archive = value["archive"]
            expected_archive_root = (
                "/var/lib/muncho-writer-canary-evidence/authority-bridge/retired/"
                f"{previous_owner_approval_sha256}/"
                f"{previous_external_iam_receipt_sha256}"
            )
            if archive != {
                "owner_approval_path": f"{expected_archive_root}/owner-approval.json",
                "owner_approval_file_sha256": previous_owner_approval_sha256,
                "external_iam_path": (
                    f"{expected_archive_root}/external-iam-receipt.json"
                ),
                "external_iam_file_sha256": (
                    previous_external_iam_receipt_sha256
                ),
            }:
                raise OwnerLauncherError("writer_authority_stage_invalid")
        return copy.deepcopy(dict(value))

    @staticmethod
    def _validate_install_approval(
        value: Mapping[str, Any],
        *,
        scope: str,
        plan_sha256: str,
        approval_sha256: str,
    ) -> str:
        path = _writer_owner_approval_path(
            scope=scope,
            plan_sha256=plan_sha256,
            receipt_sha256=approval_sha256,
        )
        if value != {
            "ok": True,
            "schema": "muncho-writer-owner-approval.v1",
            "scope": scope,
            "plan_sha256": plan_sha256,
            "owner_approval_receipt_sha256": approval_sha256,
            "installed_path": path,
        }:
            raise OwnerLauncherError("writer_approval_install_invalid")
        return path

    @staticmethod
    def _validate_install_iam(
        value: Mapping[str, Any],
        *,
        scope: str,
        plan_sha256: str,
        approval_sha256: str,
        iam_sha256: str,
        policy_sha256: str,
    ) -> Mapping[str, Any]:
        expected_keys = {
            "ok",
            "schema",
            "scope",
            "plan_sha256",
            "owner_approval_receipt_sha256",
            "external_iam_receipt_sha256",
            "external_iam_policy_sha256",
            "archive",
            "live_path",
            "live_replaced",
        }
        archive = value.get("archive")
        if (
            not isinstance(value, Mapping)
            or set(value) != expected_keys
            or value.get("ok") is not True
            or value.get("schema") != "muncho-writer-external-iam-evidence.v1"
            or value.get("scope") != scope
            or value.get("plan_sha256") != plan_sha256
            or value.get("owner_approval_receipt_sha256") != approval_sha256
            or value.get("external_iam_receipt_sha256") != iam_sha256
            or value.get("external_iam_policy_sha256") != policy_sha256
            or value.get("live_path") != WRITER_EXTERNAL_IAM_LIVE_PATH
            or type(value.get("live_replaced")) is not bool
            or not isinstance(archive, Mapping)
            or set(archive)
            != {"path", "sha256", "policy_sha256", "mode", "owner_uid", "group_gid"}
            or archive.get("path")
            != (
                "/etc/muncho/writer-activation/external-iam/"
                f"{policy_sha256}/{iam_sha256}.json"
            )
            or archive.get("sha256") != iam_sha256
            or archive.get("policy_sha256") != policy_sha256
            or archive.get("mode") != "0400"
            or archive.get("owner_uid") != 0
            or archive.get("group_gid") != 0
        ):
            raise OwnerLauncherError("writer_iam_install_invalid")
        return copy.deepcopy(dict(value))

    @staticmethod
    def _validate_native_observation(
        value: Mapping[str, Any],
        *,
        release_sha: str,
        plan_sha256: str,
        iam_sha256: str,
    ) -> str:
        expected_keys = {
            "ok",
            "schema",
            "revision",
            "native_observation_plan_sha256",
            "native_observation_receipt_sha256",
            "idempotent",
            "host_preparation_receipt_path",
            "host_preparation_receipt_sha256",
            "external_iam_evidence",
        }
        evidence = value.get("external_iam_evidence")
        if (
            not isinstance(value, Mapping)
            or set(value) != expected_keys
            or value.get("ok") is not True
            or value.get("schema") != "muncho-writer-native-observation.v1"
            or value.get("revision") != release_sha
            or value.get("native_observation_plan_sha256") != plan_sha256
            or type(value.get("idempotent")) is not bool
            or not isinstance(evidence, Mapping)
            or evidence.get("sha256") != iam_sha256
        ):
            raise OwnerLauncherError("writer_native_observation_invalid")
        _require_sha256(
            value.get("host_preparation_receipt_sha256"),
            "writer_native_observation_invalid",
        )
        return _require_sha256(
            value.get("native_observation_receipt_sha256"),
            "writer_native_observation_invalid",
        )

    @staticmethod
    def _validate_final_plan_build(
        value: Mapping[str, Any],
        *,
        native_plan_sha256: str,
        native_receipt_sha256: str,
    ) -> str:
        expected_keys = {
            "activation_plan_sha256",
            "artifact_sha256",
            "config_collector_receipt_sha256",
            "native_observation_plan_sha256",
            "native_observation_receipt_sha256",
            "release_manifest_file_sha256",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected_keys
            or value.get("native_observation_plan_sha256")
            != native_plan_sha256
            or value.get("native_observation_receipt_sha256")
            != native_receipt_sha256
        ):
            raise OwnerLauncherError("writer_final_plan_invalid")
        for name in expected_keys:
            _require_sha256(value.get(name), "writer_final_plan_invalid")
        return str(value["activation_plan_sha256"])

    @staticmethod
    def _validate_read_only_preflight(
        value: Mapping[str, Any],
        *,
        release_sha: str,
        plan_sha256: str,
    ) -> Mapping[str, Any]:
        expected_keys = {
            "schema",
            "ok",
            "revision",
            "activation_plan_sha256",
            "checks",
            "failed_checks",
            "checked_at_unix",
            "report_sha256",
            "evidence",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected_keys
            or value.get("schema")
            != "muncho-writer-activation-read-only-preflight.v1"
            or value.get("ok") is not True
            or value.get("revision") != release_sha
            or value.get("activation_plan_sha256") != plan_sha256
            or value.get("failed_checks") != []
            or not isinstance(value.get("checks"), list)
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"name", "passed"}
                or item.get("passed") is not True
                for item in value["checks"]
            )
        ):
            raise OwnerLauncherError("writer_activation_preflight_invalid")
        report_sha = _require_sha256(
            value.get("report_sha256"),
            "writer_activation_preflight_invalid",
        )
        unsigned = {
            name: copy.deepcopy(item)
            for name, item in value.items()
            if name not in {"report_sha256", "evidence"}
        }
        evidence = value.get("evidence")
        if (
            report_sha != _sha256(_canonical_bytes(unsigned))
            or not isinstance(evidence, Mapping)
            or evidence.get("report_sha256") != report_sha
        ):
            raise OwnerLauncherError("writer_activation_preflight_invalid")
        return copy.deepcopy(dict(value))

    @staticmethod
    def _validate_terminal_activation(
        value: Mapping[str, Any],
        *,
        release_sha: str,
        plan_sha256: str,
        native_plan_sha256: str,
        native_receipt_sha256: str,
        approval: Mapping[str, Any],
        approval_sha256: str,
        iam_sha256: str,
    ) -> Mapping[str, Any]:
        expected_keys = {
            "schema",
            "revision",
            "activation_plan_sha256",
            "approved_plan_sha256",
            "native_observation_plan_sha256",
            "native_observation_receipt_sha256",
            "owner_approval_receipt_sha256",
            "owner_approval_receipt",
            "external_iam_evidence",
            "read_only_preflight",
            "projection_export",
            "live_preflight",
            "services_stopped",
            "discord_started",
            "completed_at_unix",
            "activation_receipt_path",
            "receipt_sha256",
        }
        evidence = value.get("external_iam_evidence")
        if (
            not isinstance(value, Mapping)
            or set(value) != expected_keys
            or value.get("schema") != "muncho-writer-only-activation-receipt.v1"
            or value.get("revision") != release_sha
            or value.get("activation_plan_sha256") != plan_sha256
            or value.get("approved_plan_sha256") != plan_sha256
            or value.get("native_observation_plan_sha256")
            != native_plan_sha256
            or value.get("native_observation_receipt_sha256")
            != native_receipt_sha256
            or value.get("owner_approval_receipt_sha256") != approval_sha256
            or value.get("owner_approval_receipt") != approval
            or not isinstance(evidence, Mapping)
            or evidence.get("sha256") != iam_sha256
            or not isinstance(value.get("read_only_preflight"), Mapping)
            or not isinstance(value.get("projection_export"), Mapping)
            or not isinstance(value.get("live_preflight"), Mapping)
            or value.get("services_stopped") is not True
            or value.get("discord_started") is not False
            or type(value.get("completed_at_unix")) is not int
            or value.get("activation_receipt_path")
            != (
                "/var/lib/muncho-writer-activation/plans/"
                f"{release_sha}/{plan_sha256}/success/activation.json"
            )
        ):
            raise OwnerLauncherError("writer_activation_terminal_invalid")
        receipt_sha = _require_sha256(
            value.get("receipt_sha256"),
            "writer_activation_terminal_invalid",
        )
        unsigned = {
            name: copy.deepcopy(item)
            for name, item in value.items()
            if name != "receipt_sha256"
        }
        if receipt_sha != _sha256(_canonical_bytes(unsigned)):
            raise OwnerLauncherError("writer_activation_terminal_invalid")
        return copy.deepcopy(dict(value))

    def activate(
        self,
        release_sha: str,
        *,
        external_iam_policy_sha256: str,
        now: Callable[[], int] = lambda: int(time.time()),
    ) -> Mapping[str, Any]:
        """Complete native observation and final activation, ending stopped."""

        if _RELEASE_SHA.fullmatch(release_sha) is None:
            raise OwnerLauncherError("invalid_release_sha")
        policy_sha = _require_sha256(
            external_iam_policy_sha256,
            "writer_activation_policy_invalid",
        )
        account = self._owner_identity.account_for_read_only_preflight()
        self._owner_identity.require_stable()

        native_install = self._run_packaged_json(
            release_sha,
            module=WRITER_ACTIVATION_MODULE,
            arguments=(
                "install-native-plan",
                "--plan",
                WRITER_STAGED_NATIVE_PLAN_PATH,
            ),
            account=account,
        )
        native_plan_sha = self._validate_install_plan(
            native_install,
            final=False,
            release_sha=release_sha,
        )
        native_approval = build_writer_owner_approval(
            scope="native_observation",
            plan_sha256=native_plan_sha,
            owner_identity=self._owner_identity,
            now_unix=now(),
        )
        native_approval, native_approval_sha = self._approval_receipt(
            native_approval
        )
        native_iam = collect_fresh_writer_external_iam(
            owner_identity=self._owner_identity,
            source_approval_sha256=native_approval_sha,
        )
        native_iam, native_iam_sha, native_policy = self._external_iam_receipt(
            native_iam
        )
        if native_policy != policy_sha:
            raise OwnerLauncherError("writer_activation_policy_mismatch")
        frame_now = now()
        native_frame = build_writer_authority_frame(
            action="stage-native-authority",
            revision=release_sha,
            plan_sha256=native_plan_sha,
            owner_approval=native_approval,
            external_iam_receipt=native_iam,
            previous_owner_approval_sha256=None,
            previous_external_iam_receipt_sha256=None,
            now_unix=frame_now,
        )
        native_stage = self._validate_authority_stage(
            self._run_packaged_json(
                release_sha,
                module=WRITER_ACTIVATION_BRIDGE_MODULE,
                arguments=("stage-native-authority",),
                account=account,
                stdin_frame=native_frame,
            ),
            action="stage-native-authority",
            release_sha=release_sha,
            plan_sha256=native_plan_sha,
            owner_subject_sha256=str(
                native_approval["owner_subject_sha256"]
            ),
            owner_approval_sha256=native_approval_sha,
            external_iam_receipt_sha256=native_iam_sha,
            external_iam_policy_sha256=policy_sha,
            frame_sha256=_writer_authority_frame_sha256(native_frame),
            previous_owner_approval_sha256=None,
            previous_external_iam_receipt_sha256=None,
        )
        native_approval_path = self._validate_install_approval(
            self._run_packaged_json(
                release_sha,
                module=WRITER_ACTIVATION_MODULE,
                arguments=(
                    "install-approval",
                    "--staged-receipt",
                    WRITER_STAGED_OWNER_APPROVAL_PATH,
                ),
                account=account,
            ),
            scope="native_observation",
            plan_sha256=native_plan_sha,
            approval_sha256=native_approval_sha,
        )
        native_iam_install = self._validate_install_iam(
            self._run_packaged_json(
                release_sha,
                module=WRITER_ACTIVATION_MODULE,
                arguments=(
                    "install-external-iam",
                    "--staged-receipt",
                    WRITER_STAGED_EXTERNAL_IAM_PATH,
                    "--external-iam-policy-sha256",
                    policy_sha,
                    "--plan",
                    WRITER_NATIVE_PLAN_PATH,
                    "--approved-plan-sha256",
                    native_plan_sha,
                    "--owner-approval-receipt",
                    native_approval_path,
                ),
                account=account,
            ),
            scope="native_observation",
            plan_sha256=native_plan_sha,
            approval_sha256=native_approval_sha,
            iam_sha256=native_iam_sha,
            policy_sha256=policy_sha,
        )
        native_receipt_sha = self._validate_native_observation(
            self._run_packaged_json(
                release_sha,
                module=WRITER_ACTIVATION_MODULE,
                arguments=(
                    "observe-native",
                    "--plan",
                    WRITER_NATIVE_PLAN_PATH,
                    "--approved-plan-sha256",
                    native_plan_sha,
                    "--owner-approval-receipt",
                    native_approval_path,
                    "--external-iam-receipt",
                    WRITER_EXTERNAL_IAM_LIVE_PATH,
                ),
                account=account,
                timeout_seconds=2_400.0,
            ),
            release_sha=release_sha,
            plan_sha256=native_plan_sha,
            iam_sha256=native_iam_sha,
        )
        final_plan_sha = self._validate_final_plan_build(
            self._run_packaged_json(
                release_sha,
                module=WRITER_PLANNER_MODULE,
                arguments=(
                    "build-final-plan",
                    "--native-observation-receipt-sha256",
                    native_receipt_sha,
                ),
                account=account,
            ),
            native_plan_sha256=native_plan_sha,
            native_receipt_sha256=native_receipt_sha,
        )
        installed_final_sha = self._validate_install_plan(
            self._run_packaged_json(
                release_sha,
                module=WRITER_ACTIVATION_MODULE,
                arguments=(
                    "install-plan",
                    "--plan",
                    WRITER_STAGED_FINAL_PLAN_PATH,
                ),
                account=account,
            ),
            final=True,
            release_sha=release_sha,
        )
        if installed_final_sha != final_plan_sha:
            raise OwnerLauncherError("writer_final_plan_install_invalid")
        final_approval = build_writer_owner_approval(
            scope="activation",
            plan_sha256=final_plan_sha,
            owner_identity=self._owner_identity,
            now_unix=now(),
        )
        final_approval, final_approval_sha = self._approval_receipt(final_approval)
        final_iam = collect_fresh_writer_external_iam(
            owner_identity=self._owner_identity,
            source_approval_sha256=final_approval_sha,
        )
        final_iam, final_iam_sha, final_policy = self._external_iam_receipt(final_iam)
        if final_policy != policy_sha:
            raise OwnerLauncherError("writer_activation_policy_mismatch")
        final_frame = build_writer_authority_frame(
            action="replace-final-authority",
            revision=release_sha,
            plan_sha256=final_plan_sha,
            owner_approval=final_approval,
            external_iam_receipt=final_iam,
            previous_owner_approval_sha256=native_approval_sha,
            previous_external_iam_receipt_sha256=native_iam_sha,
            now_unix=now(),
        )
        final_stage = self._validate_authority_stage(
            self._run_packaged_json(
                release_sha,
                module=WRITER_ACTIVATION_BRIDGE_MODULE,
                arguments=("replace-final-authority",),
                account=account,
                stdin_frame=final_frame,
            ),
            action="replace-final-authority",
            release_sha=release_sha,
            plan_sha256=final_plan_sha,
            owner_subject_sha256=str(final_approval["owner_subject_sha256"]),
            owner_approval_sha256=final_approval_sha,
            external_iam_receipt_sha256=final_iam_sha,
            external_iam_policy_sha256=policy_sha,
            frame_sha256=_writer_authority_frame_sha256(final_frame),
            previous_owner_approval_sha256=native_approval_sha,
            previous_external_iam_receipt_sha256=native_iam_sha,
        )
        final_approval_path = self._validate_install_approval(
            self._run_packaged_json(
                release_sha,
                module=WRITER_ACTIVATION_MODULE,
                arguments=(
                    "install-approval",
                    "--staged-receipt",
                    WRITER_STAGED_OWNER_APPROVAL_PATH,
                ),
                account=account,
            ),
            scope="activation",
            plan_sha256=final_plan_sha,
            approval_sha256=final_approval_sha,
        )
        final_iam_install = self._validate_install_iam(
            self._run_packaged_json(
                release_sha,
                module=WRITER_ACTIVATION_MODULE,
                arguments=(
                    "install-external-iam",
                    "--staged-receipt",
                    WRITER_STAGED_EXTERNAL_IAM_PATH,
                    "--external-iam-policy-sha256",
                    policy_sha,
                    "--plan",
                    WRITER_FINAL_PLAN_PATH,
                    "--approved-plan-sha256",
                    final_plan_sha,
                    "--owner-approval-receipt",
                    final_approval_path,
                ),
                account=account,
            ),
            scope="activation",
            plan_sha256=final_plan_sha,
            approval_sha256=final_approval_sha,
            iam_sha256=final_iam_sha,
            policy_sha256=policy_sha,
        )
        preflight = self._validate_read_only_preflight(
            self._run_packaged_json(
                release_sha,
                module=WRITER_ACTIVATION_MODULE,
                arguments=(
                    "validate-plan",
                    "--plan",
                    WRITER_FINAL_PLAN_PATH,
                    "--approved-plan-sha256",
                    final_plan_sha,
                    "--owner-approval-receipt",
                    final_approval_path,
                ),
                account=account,
            ),
            release_sha=release_sha,
            plan_sha256=final_plan_sha,
        )
        terminal = self._validate_terminal_activation(
            self._run_packaged_json(
                release_sha,
                module=WRITER_ACTIVATION_MODULE,
                arguments=(
                    "apply",
                    "--plan",
                    WRITER_FINAL_PLAN_PATH,
                    "--approved-plan-sha256",
                    final_plan_sha,
                    "--owner-approval-receipt",
                    final_approval_path,
                ),
                account=account,
                timeout_seconds=2_400.0,
            ),
            release_sha=release_sha,
            plan_sha256=final_plan_sha,
            native_plan_sha256=native_plan_sha,
            native_receipt_sha256=native_receipt_sha,
            approval=final_approval,
            approval_sha256=final_approval_sha,
            iam_sha256=final_iam_sha,
        )
        self._owner_identity.require_stable()
        unsigned = {
            "schema": WRITER_ACTIVATION_OWNER_RECEIPT_SCHEMA,
            "ok": True,
            "state": "writer_activation_verified_stopped",
            "release_sha": release_sha,
            "owner_subject_sha256": final_approval["owner_subject_sha256"],
            "approval_source_sha256": PHASE_B_PINNED_APPROVAL_SOURCE_SHA256,
            "external_iam_policy_sha256": policy_sha,
            "native_observation_plan_sha256": native_plan_sha,
            "native_owner_approval_sha256": native_approval_sha,
            "native_external_iam_receipt_sha256": native_iam_sha,
            "native_authority_stage_receipt_sha256": native_stage[
                "receipt_sha256"
            ],
            "native_external_iam_install_receipt_sha256": _sha256(
                _canonical_bytes(native_iam_install)
            ),
            "native_observation_receipt_sha256": native_receipt_sha,
            "activation_plan_sha256": final_plan_sha,
            "activation_owner_approval_sha256": final_approval_sha,
            "activation_external_iam_receipt_sha256": final_iam_sha,
            "activation_authority_stage_receipt_sha256": final_stage[
                "receipt_sha256"
            ],
            "activation_external_iam_install_receipt_sha256": _sha256(
                _canonical_bytes(final_iam_install)
            ),
            "activation_preflight_report_sha256": preflight["report_sha256"],
            "activation_terminal_receipt_sha256": terminal["receipt_sha256"],
            "services_stopped": True,
            "discord_started": False,
            "completed_at_unix": int(terminal["completed_at_unix"]),
        }
        return {
            **unsigned,
            "receipt_sha256": _sha256(_canonical_bytes(unsigned)),
        }


def _retire_discord_token_only(
    *,
    release_sha: str,
    transport: CoordinatorTransport,
    owner_gate: Mapping[str, Any],
    install_receipt: Mapping[str, Any] | None,
    owner_identity: ApprovedOwnerIdentity,
    now: Callable[[], int],
) -> Mapping[str, Any]:
    session = transport.open_discord_retirement(release_sha)
    closed = False
    primary: BaseException | None = None
    try:
        raw_gate = _validated_input_gate(
            session,
            session.read_gate(),
            owner_gate=owner_gate,
        )
        gate = validate_discord_retirement_gate(
            raw_gate,
            owner_gate=owner_gate,
            install_receipt=install_receipt,
            now_unix=now(),
        )
        owner_identity.require_stable()
        ack = build_discord_retirement_ack(gate, now_unix=now())
        raw_receipt = _validated_input_gate(
            session,
            session.finish(build_discord_retirement_ack_frame(ack)),
            owner_gate=owner_gate,
        )
        receipt = validate_discord_retirement_receipt(
            raw_receipt,
            owner_gate=owner_gate,
            install_receipt=install_receipt,
            retirement_gate=gate,
            now_unix=now(),
        )
        session.mark_validated(receipt)
        owner_identity.require_stable()
        session.close()
        closed = True
        return receipt
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if not closed:
            _close_session_preserving_primary(session, primary)


def launch_session_bound_full_canary(
    *,
    release_sha: str,
    transport: CoordinatorTransport,
    token_source: DiscordTokenSource,
    owner_identity: ApprovedOwnerIdentity,
    final_approval_source: FinalApprovalSource,
    approval_request_sink: Callable[[Mapping[str, Any]], None],
    now: Callable[[], int] = lambda: int(time.time()),
    secret_hardener: Callable[[], None] = harden_owner_secret_process,
    provenance_guard: Callable[
        [str], None
    ] = require_owner_runtime_and_launcher_provenance,
) -> Mapping[str, Any]:
    """Run the target canary without Cloud SQL admin or bootstrap authority.

    The same sealed coordinator process builds the final plan, emits its
    approval request, receives MFA1, executes the live driver, proves stopped
    terminal truth, and retires the Discord credential.  The launcher never
    creates a database principal and never receives the API session key.
    """

    if not isinstance(release_sha, str) or _RELEASE_SHA.fullmatch(release_sha) is None:
        raise OwnerLauncherError("invalid_release_sha")

    signal_fence = _OwnerSignalFence()
    signal_fence.install()
    live_gate: Mapping[str, Any] | None = None
    discord_install_receipt: Mapping[str, Any] | None = None
    request: Mapping[str, Any] | None = None
    approval: Mapping[str, Any] | None = None
    coordinator_receipt: Mapping[str, Any] | None = None
    discord_session: RemoteSecretSession | None = None
    run_session: RemoteSecretSession | None = None
    token: bytearray | None = None
    discord_frame: bytearray | None = None
    approval_frame: bytes | None = None
    primary: BaseException | None = None
    token_installed = False
    token_retired = False
    try:
        provenance_guard(release_sha)
        secret_hardener()
        live_gate = validate_current_phase_b_live_gate(
            transport.preflight_phase_b_live_run(release_sha),
            expected_release_sha=release_sha,
            now_unix=now(),
        )
        owner_identity.bind_approved_subject(str(live_gate["owner_subject_sha256"]))
        owner_identity.require_stable()

        discord_session = transport.open_discord_install(release_sha)
        discord_gate = validate_discord_install_gate(
            discord_session.read_gate(),
            owner_gate=live_gate,
            now_unix=now(),
        )
        owner_identity.require_stable()
        token = token_source.read_discord_token()
        discord_frame = build_discord_frame(token)

        def guard_discord_write() -> None:
            owner_identity.require_stable()
            if (
                now() + _SECRET_FRAME_TRANSMIT_MARGIN_SECONDS
                >= discord_gate["expires_at_unix"]
            ):
                raise OwnerLauncherError("discord_install_gate_expired")

        raw_install_receipt = discord_session.finish_before(
            discord_frame,
            write_guard=guard_discord_write,
            on_first_write=lambda: None,
        )
        discord_install_receipt = validate_discord_install_receipt(
            raw_install_receipt,
            gate=discord_gate,
            token=token,
            now_unix=now(),
        )
        token_installed = True
        discord_session.mark_validated(discord_install_receipt)
        owner_identity.require_stable()
        discord_session.close()
        discord_session = None
        _wipe(token)
        token = None
        _wipe(discord_frame)
        discord_frame = None

        run_session = transport.open_run(release_sha)
        request = validate_session_bound_approval_request(
            run_session.read_gate(),
            live_gate=live_gate,
            now_unix=now(),
        )
        approval_request_sink(request)
        owner_identity.require_stable()
        if now() > request["owner_input_cutoff_unix"]:
            raise OwnerLauncherError("final_approval_delivery_window_exhausted")
        approval = validate_session_bound_final_owner_approval(
            final_approval_source.read_final_approval(request),
            request=request,
            now_unix=now(),
        )
        owner_identity.require_stable()
        approval_frame = build_final_approval_frame(approval)

        def guard_final_approval_write() -> None:
            owner_identity.require_stable()
            if now() > request["owner_input_cutoff_unix"]:
                raise OwnerLauncherError("final_approval_delivery_window_exhausted")

        raw_coordinator_receipt = run_session.finish_before(
            approval_frame,
            write_guard=guard_final_approval_write,
            on_first_write=lambda: None,
        )
        coordinator_receipt = validate_session_bound_coordinator_receipt(
            raw_coordinator_receipt,
            live_gate=live_gate,
            request=request,
            approval=approval,
            now_unix=now(),
        )
        token_retired = True
        run_session.mark_validated(coordinator_receipt)
        owner_identity.require_stable()
        run_session.close()
        run_session = None
        provenance_guard(release_sha)
    except BaseException as exc:
        primary = exc
    finally:
        signal_fence.begin_cleanup()
        _wipe(token)
        _wipe(discord_frame)
        token = None
        discord_frame = None
        approval_frame = None
        for session in (run_session, discord_session):
            if session is not None:
                _close_session_preserving_primary(session, primary)
        if token_installed and not token_retired and live_gate is not None:
            try:
                _retire_discord_token_only(
                    release_sha=release_sha,
                    transport=transport,
                    owner_gate=live_gate,
                    install_receipt=discord_install_receipt,
                    owner_identity=owner_identity,
                    now=now,
                )
                token_retired = True
            except BaseException as cleanup_error:
                if primary is None:
                    primary = cleanup_error
                else:
                    _attach_cleanup_failure(primary, cleanup_error)
        try:
            signal_fence.restore()
        except BaseException as cleanup_error:
            if primary is None:
                primary = cleanup_error
            else:
                _attach_cleanup_failure(primary, cleanup_error)

    if primary is not None:
        raise primary
    if (
        live_gate is None
        or discord_install_receipt is None
        or request is None
        or approval is None
        or coordinator_receipt is None
        or not token_retired
    ):
        raise OwnerLauncherError("owner_launcher_terminal_truth_incomplete")
    unsigned = {
        "schema": SESSION_BOUND_OWNER_RECEIPT_SCHEMA,
        "ok": True,
        "state": "verified_stopped_and_credentials_retired",
        "release_sha": release_sha,
        "coordinator_input_sha256": live_gate["coordinator_input_sha256"],
        "full_canary_plan_sha256": request["full_canary_plan_sha256"],
        "owner_approval_sha256": _sha256(_canonical_bytes(approval)),
        "phase_b_readiness_anchor_sha256": live_gate[
            "phase_b_readiness_anchor_sha256"
        ],
        "api_session_key_sha256": coordinator_receipt["api_session_key_sha256"],
        "fixture_sha256": coordinator_receipt["fixture_sha256"],
        "discord_token_install_receipt_sha256": discord_install_receipt[
            "receipt_sha256"
        ],
        "coordinator_receipt_sha256": coordinator_receipt["receipt_sha256"],
        "live_driver_receipt_sha256": coordinator_receipt[
            "live_driver_receipt_sha256"
        ],
        "services_stopped": True,
        "discord_token_retired": True,
        "temporary_admin_created": False,
        "bootstrap_credential_created": False,
        "completed_at_unix": now(),
    }
    return {**unsigned, "receipt_sha256": _sha256(_canonical_bytes(unsigned))}


# The session-bound implementation is the only live owner target.
launch_full_canary = launch_session_bound_full_canary


class _OwnerStoreOnce(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        del parser
        if getattr(namespace, self.dest, None) is not None:
            raise argparse.ArgumentError(
                self,
                f"{option_string or self.dest} was repeated",
            )
        setattr(namespace, self.dest, values)


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Owner-side isolated full-canary credential launcher",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--release-sha",
        required=True,
        help="exact sealed 40-character fork release SHA",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--bootstrap-trusted-runtime",
        action="store_true",
        help="publish the reviewed owner-only gcloud SDK snapshot and receipt",
    )
    actions.add_argument(
        "--publish-stopped-release",
        action="store_true",
        help="publish the exact fork revision while every canary service is stopped",
    )
    actions.add_argument(
        "--rotate-host-identity-receipt",
        action="store_true",
        help=(
            "archive one exact stale boot receipt and publish the exact current "
            "VM boot receipt while services stay stopped"
        ),
    )
    actions.add_argument(
        "--publish-full-canary-fixture",
        action="store_true",
        help=(
            "publish the fixed public-channel, secret-free base fixture while "
            "all canary services remain stopped"
        ),
    )
    actions.add_argument(
        "--reconcile-legacy-canary-db",
        action="store_true",
        help=(
            "repair or adopt only the sealed legacy canary schema while all "
            "canary services remain stopped"
        ),
    )
    actions.add_argument(
        "--publish-writer-preflight",
        action="store_true",
        help="stage and attest writer-only inputs without starting services",
    )
    actions.add_argument(
        "--activate-writer-stopped",
        action="store_true",
        help=(
            "run the packaged native/final writer activation and require its "
            "terminal stopped receipt"
        ),
    )
    actions.add_argument(
        "--apply-phase-b-foundation",
        action="store_true",
        help=(
            "apply the reviewed writer foundation through the stopped-only "
            "owner protocol"
        ),
    )
    actions.add_argument(
        "--publish-coordinator-input",
        action="store_true",
        help=(
            "publish the fixed coordinator input after terminal writer-only "
            "activation"
        ),
    )
    parser.add_argument(
        "--external-iam-policy-sha256",
        help="exact external IAM policy digest bound into writer staging",
        action=_OwnerStoreOnce,
    )
    parser.add_argument(
        "--expected-prior-host-receipt-file-sha256",
        help="exact immutable file digest of the stale host receipt",
        action=_OwnerStoreOnce,
    )
    parser.add_argument(
        "--expected-prior-host-receipt-sha256",
        help="exact self-digest inside the stale host receipt",
        action=_OwnerStoreOnce,
    )
    parser.add_argument(
        "--expected-prior-boot-id-sha256",
        help="exact stale boot-id digest being retired",
        action=_OwnerStoreOnce,
    )
    parser.add_argument(
        "--expected-current-boot-id-sha256",
        help="exact current dedicated-VM boot-id digest being published",
        action=_OwnerStoreOnce,
    )
    return parser


def _emit_canonical_line(value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value)
    if len(payload) > _MAX_JSON_LINE_BYTES:
        raise OwnerLauncherError("owner_output_oversized")
    sys.stdout.buffer.write(payload + b"\n")
    sys.stdout.buffer.flush()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _cli_parser().parse_args(argv)
    release_sha = arguments.release_sha
    if not isinstance(release_sha, str) or not _RELEASE_SHA.fullmatch(release_sha):
        failure = {
            "schema": "muncho-full-canary-owner-launcher-cli-failure.v1",
            "ok": False,
            "error_code": "invalid_release_sha",
        }
        _emit_canonical_line({
            **failure,
            "receipt_sha256": _sha256(_canonical_bytes(failure)),
        })
        return 2
    try:
        host_rotation_bindings = (
            arguments.expected_prior_host_receipt_file_sha256,
            arguments.expected_prior_host_receipt_sha256,
            arguments.expected_prior_boot_id_sha256,
            arguments.expected_current_boot_id_sha256,
        )
        if (
            not arguments.rotate_host_identity_receipt
            and any(item is not None for item in host_rotation_bindings)
        ):
            raise OwnerLauncherError("host_receipt_rotation_plan_invalid")
        if (
            (
                arguments.reconcile_legacy_canary_db
                or arguments.apply_phase_b_foundation
                or arguments.publish_coordinator_input
            )
            and arguments.external_iam_policy_sha256 is not None
        ):
            code = (
                "schema_reconciliation_owner_cli_invalid"
                if arguments.reconcile_legacy_canary_db
                else (
                    "phase_b_owner_cli_invalid"
                    if arguments.apply_phase_b_foundation
                    else "coordinator_input_owner_cli_invalid"
                )
            )
            raise OwnerLauncherError(code)
        if arguments.bootstrap_trusted_runtime:
            if arguments.external_iam_policy_sha256 is not None:
                raise OwnerLauncherError("host_receipt_rotation_plan_invalid")
            require_trusted_bootstrap_interpreter()
            launcher_sha256 = require_local_launcher_provenance(release_sha)
            receipt = bootstrap_trusted_gcloud_runtime(
                release_sha,
                launcher_sha256=launcher_sha256,
            )
            require_trusted_bootstrap_interpreter()
            require_local_launcher_provenance(release_sha)
            _emit_canonical_line(receipt)
            return 0
        # The fixed isolated interpreter and its release-bound runtime receipt
        # precede git, auth, IAP, and every secret-bearing source.
        gcloud_executable = require_trusted_owner_runtime(release_sha)
        require_local_launcher_provenance(release_sha)
        gcloud_configuration = PinnedGcloudConfiguration()
        owner_identity = GcloudOwnerAccessToken(
            gcloud_executable=gcloud_executable,
            gcloud_configuration=gcloud_configuration,
        )

        def runtime_and_provenance_guard(exact_release: str) -> None:
            command = gcloud_executable.trusted_command_prefix()
            _validate_owner_interpreter_invocation(command[0])
            require_local_launcher_provenance(exact_release)

        if arguments.rotate_host_identity_receipt:
            external_iam_policy_sha256 = _require_sha256(
                arguments.external_iam_policy_sha256,
                "host_receipt_rotation_plan_invalid",
            )
            prior_file_sha256, prior_receipt_sha256, prior_boot_sha256, current_boot_sha256 = (
                _require_sha256(
                    item,
                    "host_receipt_rotation_plan_invalid",
                )
                for item in host_rotation_bindings
            )
            rotation_transport = IapHostReceiptRotationTransport(
                owner_identity,
                gcloud_executable=gcloud_executable,
                gcloud_configuration=gcloud_configuration,
            )
            receipt = rotation_transport.rotate(
                release_sha,
                external_iam_policy_sha256=external_iam_policy_sha256,
                expected_prior_file_sha256=prior_file_sha256,
                expected_prior_receipt_sha256=prior_receipt_sha256,
                expected_prior_boot_id_sha256=prior_boot_sha256,
                expected_current_boot_id_sha256=current_boot_sha256,
            )
            runtime_and_provenance_guard(release_sha)
            _emit_canonical_line(receipt)
            return 0
        if arguments.publish_stopped_release:
            if arguments.external_iam_policy_sha256 is not None:
                raise OwnerLauncherError("writer_preflight_plan_invalid")
            release_transport = IapStoppedReleaseTransport(
                owner_identity,
                gcloud_executable=gcloud_executable,
                gcloud_configuration=gcloud_configuration,
            )
            receipt = release_transport.publish(release_sha)
            runtime_and_provenance_guard(release_sha)
            _emit_canonical_line(receipt)
            return 0
        if arguments.publish_full_canary_fixture:
            external_iam_policy_sha256 = _require_sha256(
                arguments.external_iam_policy_sha256,
                "fixture_publication_plan_invalid",
            )
            fixture_transport = IapFixturePublicationTransport(
                owner_identity,
                gcloud_executable=gcloud_executable,
                gcloud_configuration=gcloud_configuration,
            )
            receipt = fixture_transport.publish(
                release_sha,
                external_iam_policy_sha256=external_iam_policy_sha256,
            )
            runtime_and_provenance_guard(release_sha)
            _emit_canonical_line(receipt)
            return 0
        if arguments.reconcile_legacy_canary_db:
            transport = IapSchemaReconciliationTransport(
                owner_identity,
                gcloud_executable=gcloud_executable,
                gcloud_configuration=gcloud_configuration,
            )
            receipt = reconcile_legacy_canary_schema(
                release_sha=release_sha,
                transport=transport,
                cloud_sql_client=GoogleRestClient(owner_identity),
                owner_identity=owner_identity,
                provenance_guard=runtime_and_provenance_guard,
            )
            runtime_and_provenance_guard(release_sha)
            _emit_canonical_line(receipt)
            return 0
        if arguments.publish_writer_preflight:
            external_iam_policy_sha256 = _require_sha256(
                arguments.external_iam_policy_sha256,
                "writer_preflight_plan_invalid",
            )
            writer_transport = IapWriterPreflightTransport(
                owner_identity,
                gcloud_executable=gcloud_executable,
                gcloud_configuration=gcloud_configuration,
            )
            receipt = writer_transport.publish(
                release_sha,
                external_iam_policy_sha256=external_iam_policy_sha256,
            )
            runtime_and_provenance_guard(release_sha)
            _emit_canonical_line(receipt)
            return 0
        if arguments.activate_writer_stopped:
            external_iam_policy_sha256 = _require_sha256(
                arguments.external_iam_policy_sha256,
                "writer_activation_policy_invalid",
            )
            transport = IapWriterActivationBridgeTransport(
                owner_identity,
                gcloud_executable=gcloud_executable,
                gcloud_configuration=gcloud_configuration,
            )
            receipt = transport.activate(
                release_sha,
                external_iam_policy_sha256=external_iam_policy_sha256,
            )
            runtime_and_provenance_guard(release_sha)
            _emit_canonical_line(receipt)
            return 0
        if arguments.apply_phase_b_foundation:
            transport = IapCoordinatorTransport(
                owner_identity,
                gcloud_executable=gcloud_executable,
                gcloud_configuration=gcloud_configuration,
            )
            receipt = apply_phase_b_foundation(
                release_sha=release_sha,
                transport=transport,
                cloud_sql_client=GoogleRestClient(owner_identity),
                owner_identity=owner_identity,
                provenance_guard=runtime_and_provenance_guard,
            )
            runtime_and_provenance_guard(release_sha)
            _emit_canonical_line(receipt)
            return 0
        if arguments.publish_coordinator_input:
            transport = IapCoordinatorTransport(
                owner_identity,
                gcloud_executable=gcloud_executable,
                gcloud_configuration=gcloud_configuration,
            )
            receipt = transport.publish_coordinator_input(release_sha)
            runtime_and_provenance_guard(release_sha)
            _emit_canonical_line(receipt)
            return 0
        if arguments.external_iam_policy_sha256 is not None:
            raise OwnerLauncherError("writer_preflight_plan_invalid")
        transport = IapCoordinatorTransport(
            owner_identity,
            gcloud_executable=gcloud_executable,
            gcloud_configuration=gcloud_configuration,
        )
        try:
            receipt = launch_full_canary(
                release_sha=release_sha,
                transport=transport,
                token_source=OwnerDiscordTokenReader(),
                owner_identity=owner_identity,
                final_approval_source=FixedLocalFinalApprovalFile(),
                approval_request_sink=_emit_canonical_line,
                provenance_guard=runtime_and_provenance_guard,
            )
        except BaseException as exc:
            code = (
                exc.code
                if isinstance(exc, OwnerLauncherError)
                else "owner_launcher_failed"
            )
            failure = {
                "schema": "muncho-full-canary-owner-launcher-cli-failure.v1",
                "ok": False,
                "error_code": code,
            }
            receipt = {
                **failure,
                "receipt_sha256": _sha256(_canonical_bytes(failure)),
            }
        # Terminal provenance is mandatory after both success and failure.
        # Any drift takes precedence over the earlier outcome before emission.
        try:
            runtime_and_provenance_guard(release_sha)
        except BaseException as exc:
            code = (
                exc.code
                if isinstance(exc, OwnerLauncherError)
                else "owner_launcher_failed"
            )
            failure = {
                "schema": "muncho-full-canary-owner-launcher-cli-failure.v1",
                "ok": False,
                "error_code": code,
            }
            receipt = {
                **failure,
                "receipt_sha256": _sha256(_canonical_bytes(failure)),
            }
    except BaseException as exc:
        code = (
            exc.code if isinstance(exc, OwnerLauncherError) else "owner_launcher_failed"
        )
        failure = {
            "schema": "muncho-full-canary-owner-launcher-cli-failure.v1",
            "ok": False,
            "error_code": code,
        }
        receipt = {
            **failure,
            "receipt_sha256": _sha256(_canonical_bytes(failure)),
        }
    _emit_canonical_line(receipt)
    return 0 if receipt.get("ok") is True else 2


__all__ = [
    "ADMIN_FRAME_MAGIC",
    "ADMIN_USERNAME_PREFIX",
    "CANARY_BOOTSTRAP_ABSENCE_EVIDENCE_SCHEMA",
    "CANARY_BOOTSTRAP_AUTHORITY_RECEIPT_SCHEMA",
    "CANARY_BOOTSTRAP_DATABASE_ROLE",
    "CANARY_BOOTSTRAP_LOGIN",
    "CleanupBlocked",
    "CloudSqlCanaryBootstrapLogin",
    "CloudSqlCreateNotCommitted",
    "CloudSqlSchemaReconciliationAdmin",
    "CloudSqlTemporaryAdmin",
    "COORDINATOR_FAILURE_SCHEMA",
    "COORDINATOR_INPUT_PUBLICATION_SCHEMA",
    "COORDINATOR_RECEIPT_SCHEMA",
    "COORDINATOR_SECRET_GATE_SCHEMA",
    "CoordinatorTransport",
    "DATABASE_HOST",
    "DATABASE_NAME",
    "DATABASE_PORT",
    "DISCORD_INSTALL_GATE_SCHEMA",
    "DISCORD_INSTALL_RECEIPT_SCHEMA",
    "DISCORD_RETIREMENT_ACK_FRAME_MAGIC",
    "DISCORD_RETIREMENT_ACK_FRAME_SCHEMA",
    "DISCORD_RETIREMENT_ACK_SCHEMA",
    "DISCORD_RETIREMENT_GATE_SCHEMA",
    "DISCORD_RETIREMENT_RECEIPT_SCHEMA",
    "FINAL_APPROVAL_CANCEL_RECEIPT_SCHEMA",
    "FINAL_APPROVAL_FRAME_MAGIC",
    "FINAL_APPROVAL_FRAME_SCHEMA",
    "FINAL_APPROVAL_INSTALL_RECEIPT_SCHEMA",
    "FINAL_APPROVAL_MAX_WAIT_SECONDS",
    "FINAL_APPROVAL_PATH",
    "FIXTURE_PUBLICATION_MODULE",
    "FIXTURE_PUBLICATION_PLAN_SCHEMA",
    "FIXTURE_PUBLICATION_RECEIPT_SCHEMA",
    "FixedLocalFinalApprovalFile",
    "GoogleRestClient",
    "GcloudOwnerAccessToken",
    "HOST_RECEIPT_ROTATION_MODULE",
    "HOST_RECEIPT_ROTATION_PLAN_SCHEMA",
    "HOST_RECEIPT_ROTATION_RECEIPT_SCHEMA",
    "HOST_RECEIPT_ROTATION_ROOT",
    "IapCoordinatorTransport",
    "IapFixturePublicationTransport",
    "IapHostReceiptRotationTransport",
    "IapSchemaReconciliationTransport",
    "IapStoppedReleaseTransport",
    "IapWriterActivationBridgeTransport",
    "IapWriterPreflightTransport",
    "HttpResponse",
    "LocalLauncherProvenance",
    "OWNER_DISCORD_INPUT_MAGIC",
    "OWNER_RECEIPT_SCHEMA",
    "OwnerDiscordTokenReader",
    "OwnerLauncherError",
    "OwnerStdinDiscordTokenReader",
    "PinnedGoogleComputeKnownHosts",
    "PinnedGcloudConfiguration",
    "PROJECT",
    "RECOVERY_ACK_FRAME_MAGIC",
    "RECOVERY_ACK_FRAME_SCHEMA",
    "RECOVERY_ACK_SCHEMA",
    "RECOVERY_ADMIN_FRAME_MAGIC",
    "RECOVERY_ADMIN_FRAME_SCHEMA",
    "RECOVERY_CONCURRENT_LOSER_RECEIPT_SCHEMA",
    "RECOVERY_FINALIZE_PENDING_RECEIPT_SCHEMA",
    "RECOVERY_GATE_SCHEMA",
    "RECOVERY_RECEIPT_SCHEMA",
    "RECOVERY_SECRET_GATE_SCHEMA",
    "RECOVERY_WORKER_COMPLETION_SCHEMA",
    "RECOVERY_WORKER_LEASE_SCHEMA",
    "SCHEMA_RECONCILIATION_ADMIN_CLEANUP_MAGIC",
    "SCHEMA_RECONCILIATION_ADMIN_CLEANUP_SSHSIG_NAMESPACE",
    "SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_MAGIC",
    "SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_SSHSIG_NAMESPACE",
    "SCHEMA_RECONCILIATION_CREDENTIAL_BYTES",
    "SCHEMA_RECONCILIATION_MIN_GATE_REMAINING_SECONDS",
    "SCHEMA_RECONCILIATION_PREFLIGHT_AUTHORIZATION_MAGIC",
    "SCHEMA_RECONCILIATION_PREFLIGHT_AUTHORIZATION_SSHSIG_NAMESPACE",
    "SCHEMA_RECONCILIATION_DATABASE_ROLES",
    "SESSION_BOUND_APPROVAL_REQUEST_SCHEMA",
    "SESSION_BOUND_COORDINATOR_RECEIPT_SCHEMA",
    "SESSION_BOUND_OWNER_RECEIPT_SCHEMA",
    "SQL_INSTANCE",
    "TEMPORARY_ADMIN_AUTHORITY_RECEIPT_SCHEMA",
    "STOPPED_RELEASE_EVIDENCE_BASE",
    "STOPPED_RELEASE_HOST_RECEIPT_PATH",
    "STOPPED_RELEASE_PLAN_SCHEMA",
    "STOPPED_RELEASE_PYTHON_VERSION",
    "STOPPED_RELEASE_RECEIPT_SCHEMA",
    "WRITER_PREFLIGHT_PLAN_SCHEMA",
    "WRITER_PREFLIGHT_RECEIPT_SCHEMA",
    "WRITER_PREFLIGHT_FAILURE_SCHEMA",
    "STOPPED_RELEASE_SOURCE_BASE",
    "STOPPED_RELEASE_SOURCE_REPOSITORY",
    "TRUSTED_RUNTIME_BOOTSTRAP_RECEIPT_SCHEMA",
    "TRUSTED_SDK_PUBLICATION_INTENT_SCHEMA",
    "TrustedGcloudExecutable",
    "VM_INSTANCE_ID",
    "VM_NAME",
    "OS_LOGIN_PROFILE_ID",
    "OS_LOGIN_USERNAME",
    "ZONE",
    "build_discord_frame",
    "build_discord_retirement_ack",
    "build_discord_retirement_ack_frame",
    "build_final_approval_frame",
    "build_schema_reconciliation_admin_cleanup",
    "build_schema_reconciliation_admin_preflight",
    "build_schema_reconciliation_preflight_authorization",
    "build_writer_authority_frame",
    "build_writer_owner_approval",
    "bootstrap_trusted_gcloud_runtime",
    "harden_owner_secret_process",
    "launch_full_canary",
    "launch_session_bound_full_canary",
    "apply_phase_b_foundation",
    "collect_fresh_writer_external_iam",
    "main",
    "reconcile_legacy_canary_schema",
    "require_local_launcher_provenance",
    "validate_current_phase_b_live_gate",
    "validate_coordinator_input_publication_receipt",
    "validate_discord_install_gate",
    "validate_discord_install_receipt",
    "validate_discord_retirement_gate",
    "validate_discord_retirement_receipt",
    "validate_fixture_publication_plan",
    "validate_fixture_publication_receipt",
    "validate_host_receipt_rotation_plan",
    "validate_host_receipt_rotation_receipt",
    "validate_phase_b_apply_gate",
    "validate_phase_b_apply_receipt",
    "validate_session_bound_approval_request",
    "validate_session_bound_coordinator_receipt",
    "validate_session_bound_final_owner_approval",
    "validate_stopped_release_plan",
    "validate_stopped_release_receipt",
    "validate_writer_preflight_plan",
    "validate_writer_preflight_receipt",
    "validate_terminal_first_failure",
]


if __name__ == "__main__":
    raise SystemExit(main())
