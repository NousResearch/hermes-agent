#!/usr/bin/env python3
"""Owner-side credential edge for the isolated full Cloud Muncho canary.

This module deliberately contains no model, routing, or task semantics.  Its
secret-free stopped-release action uses only fixed IAP argv to publish one
approved fork revision while every canary service remains inactive.  Its live
action does four mechanical things after an exact, fresh coordinator
authorization:

* reads the canary-only Discord credential from the owner's inherited stdin;
* asks the sealed remote coordinator to install that opaque credential;
* creates one approval-derived temporary Cloud SQL BUILT_IN administrator;
* streams that administrator over IAP/SSH stdin and deletes it in ``finally``.

Secret values are never accepted on argv, placed in the environment, written
to a file, logged, or included in a receipt.  The temporary administrator is
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
STOPPED_RELEASE_HOST_RECEIPT_PATH = "/etc/muncho/full-canary/host-identity.json"
STOPPED_RELEASE_PYTHON_VERSION = "3.11.15"
WRITER_PREFLIGHT_PLAN_SCHEMA = "muncho-writer-preflight-publication-plan.v2"
WRITER_PREFLIGHT_RECEIPT_SCHEMA = "muncho-writer-preflight-publication.v3"
WRITER_PREFLIGHT_MODULE = "gateway.canonical_writer_preflight_publisher"
WRITER_PREFLIGHT_EVIDENCE_BASE = (
    "/var/lib/muncho-writer-canary-evidence/staged-publication"
)
WRITER_PREFLIGHT_OWNER_DISCORD_USER_ID = "1279454038731264061"
WRITER_PREFLIGHT_DATABASE_TLS_SERVER_NAME = (
    "14-0d81ef63-2cac-4a64-84ad-c4f58c0cfd56.europe-west3.sql.goog"
)
SQL_INSTANCE = "muncho-canary-pg18-v2"
DATABASE_HOST = "10.91.0.3"
DATABASE_PORT = 5432
DATABASE_NAME = "muncho_canary_brain"
OWNER_GATE_SCHEMA = "muncho-full-canary-owner-launch-gate.v1"
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
COORDINATOR_FAILURE_SCHEMA = "muncho-full-canary-coordinator-failure.v1"
OWNER_APPROVAL_REQUEST_SCHEMA = "muncho-full-canary-owner-approval-request.v1"
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
_HTTP_TIMEOUT_SECONDS = 30.0
_OPERATION_TIMEOUT_SECONDS = 180.0
_GATE_MAX_FUTURE_SECONDS = 900
FINAL_APPROVAL_MAX_WAIT_SECONDS = 240
_FINAL_APPROVAL_DELIVERY_RESERVE_SECONDS = 30
_FINAL_APPROVAL_TERMINAL_CLEANUP_GRACE_SECONDS = 300
_HBA_EXPIRY_SAFETY_MARGIN_SECONDS = 30
_SECRET_FRAME_TRANSMIT_MARGIN_SECONDS = 30
_MAX_JSON_LINE_BYTES = 256 * 1024
_STOPPED_RELEASE_ACTIVATION_PATHS = (
    "/etc/muncho/writer-activation/staged/writer.json",
    "/etc/muncho/writer-activation/staged/gateway.yaml",
    "/etc/muncho/writer-activation/staged/native-observation-plan.json",
    "/etc/muncho/writer-activation/staged/activation-plan.json",
    "/etc/muncho/writer-activation/staged/owner-approval.json",
    "/etc/muncho/writer-activation/staged/external-iam-receipt.json",
    "/etc/muncho/writer-activation/staged/muncho-canonical-writer.service",
    "/etc/muncho/writer-activation/staged/hermes-cloud-gateway.service",
    "/etc/muncho/writer-activation/native-observation-plan.json",
    "/etc/muncho/writer-activation/activation-plan.json",
    "/etc/muncho/writer-activation/deployment-manifest.json",
    "/etc/systemd/system/muncho-canonical-writer.service",
    "/etc/systemd/system/hermes-cloud-gateway.service",
    "/etc/systemd/system/muncho-canonical-writer-export.service",
    "/etc/tmpfiles.d/muncho-canonical-writer.conf",
    "/etc/muncho-canonical-writer/writer.json",
    "/etc/hermes/config.yaml",
)
_STOPPED_RELEASE_UNITS = (
    "muncho-discord-egress.service",
    "muncho-canonical-writer.service",
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
_OWNER_GATE_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "release_sha",
    "database_host",
    "database_port",
    "database_name",
    "admin_username",
    "expires_at_unix",
    "gate_sha256",
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
_RECOVERY_GATE_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "predecessor_kind",
    "predecessor_schema",
    "predecessor_journal_sha256",
    "predecessor_generation",
    "original_run_process_lease_sha256",
    "causal_recovery_state_sha256",
    "target_pid",
    "target_process_start_time_ticks",
    "target_boot_id_sha256",
    "target_boot_time_ns",
    "target_uid",
    "target_gid",
    "target_module_origin",
    "target_module_sha256",
    "target_process_exe_sha256",
    "target_process_cmdline_sha256",
    "target_process_identity_state",
    "discord_token_state",
    "discord_token_install_receipt_sha256",
    "discord_retirement_receipt_sha256",
    "token_device",
    "token_inode",
    "db_secret_accepted",
    "frame_schema",
    "observed_at_unix",
    "expires_at_unix",
    "gate_sha256",
})
_RECOVERY_ACK_KEYS = frozenset({
    "schema",
    "scope",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "predecessor_kind",
    "predecessor_schema",
    "predecessor_journal_sha256",
    "predecessor_generation",
    "original_run_process_lease_sha256",
    "causal_recovery_state_sha256",
    "target_pid",
    "target_process_start_time_ticks",
    "target_boot_id_sha256",
    "target_boot_time_ns",
    "target_uid",
    "target_gid",
    "target_module_origin",
    "target_module_sha256",
    "target_process_exe_sha256",
    "target_process_cmdline_sha256",
    "target_process_identity_state",
    "discord_token_state",
    "discord_token_install_receipt_sha256",
    "discord_retirement_receipt_sha256",
    "token_device",
    "token_inode",
    "recovery_takeover_gate_sha256",
    "nonce_sha256",
    "approved_at_unix",
    "expires_at_unix",
    "ack_sha256",
})
_RECOVERY_SECRET_GATE_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "recovery_takeover_gate_sha256",
    "owner_recovery_takeover_ack_sha256",
    "predecessor_kind",
    "predecessor_journal_sha256",
    "predecessor_generation",
    "original_run_process_lease_sha256",
    "causal_recovery_state_sha256",
    "recovery_worker_lease_sha256",
    "recovery_worker_state",
    "recovery_worker_transition_seq",
    "recovery_worker_pid",
    "recovery_worker_process_start_time_ticks",
    "recovery_worker_boot_id_sha256",
    "recovery_worker_boot_time_ns",
    "recovery_worker_uid",
    "recovery_worker_gid",
    "recovery_worker_module_origin",
    "recovery_worker_module_sha256",
    "recovery_worker_process_exe_sha256",
    "recovery_worker_process_cmdline_sha256",
    "database_host",
    "database_port",
    "database_name",
    "tls_server_name",
    "tls_peer_certificate_sha256",
    "gate_nonce_sha256",
    "admin_frame_schema",
    "expires_at_unix",
    "gate_sha256",
})
_RECOVERY_WORKER_COMPLETION_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "original_run_process_lease",
    "original_run_process_lease_sha256",
    "causal_recovery_state_sha256",
    "predecessor_kind",
    "predecessor_journal_sha256",
    "predecessor_generation",
    "recovery_generation",
    "recovery_takeover_gate_sha256",
    "owner_recovery_takeover_ack_sha256",
    "recovery_worker_lease_sha256",
    "recovery_worker_pid",
    "recovery_worker_process_start_time_ticks",
    "recovery_worker_boot_id_sha256",
    "recovery_worker_boot_time_ns",
    "recovery_worker_uid",
    "recovery_worker_gid",
    "recovery_worker_module_origin",
    "recovery_worker_module_sha256",
    "recovery_worker_process_exe_sha256",
    "recovery_worker_process_cmdline_sha256",
    "predecessor_termination_proven",
    "predecessor_process_lock_acquired",
    "predecessor_journal_replaced",
    "canonical_stop_receipt_sha256",
    "preplan_stopped_report_sha256",
    "preclaim_reconciliation_receipt_sha256",
    "preclaim_reconciliation_state",
    "admin_frame_zeroized",
    "admin_session_closed",
    "migration_owner_membership_removed",
    "bootstrap_login_password_disabled",
    "bootstrap_credential_removed",
    "discord_token_removed",
    "discord_install_receipt_removed",
    "discord_retirement_receipt_sha256",
    "services_stopped_proven",
    "services_enabled",
    "recovery_worker_exit_proven",
    "safe_to_delete_temporary_admin",
    "cleanup_completed_at_unix",
    "completion_sha256",
})
_RECOVERY_RECEIPT_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "original_run_process_lease",
    "original_run_process_lease_sha256",
    "causal_recovery_state_sha256",
    "predecessor_kind",
    "predecessor_journal_sha256",
    "predecessor_generation",
    "recovery_generation",
    "recovery_takeover_gate_sha256",
    "owner_recovery_takeover_ack_sha256",
    "recovery_worker_lease_sha256",
    "recovery_worker_completion_sha256",
    "recovery_worker_pid",
    "recovery_worker_process_start_time_ticks",
    "recovery_worker_boot_id_sha256",
    "recovery_worker_boot_time_ns",
    "recovery_worker_uid",
    "recovery_worker_gid",
    "recovery_worker_module_origin",
    "recovery_worker_module_sha256",
    "recovery_worker_process_exe_sha256",
    "recovery_worker_process_cmdline_sha256",
    "predecessor_termination_proven",
    "predecessor_process_lock_acquired",
    "predecessor_journal_replaced",
    "canonical_stop_receipt_sha256",
    "preplan_stopped_report_sha256",
    "preclaim_reconciliation_receipt_sha256",
    "preclaim_reconciliation_state",
    "admin_frame_zeroized",
    "admin_session_closed",
    "migration_owner_membership_removed",
    "bootstrap_login_password_disabled",
    "bootstrap_credential_removed",
    "discord_token_removed",
    "discord_install_receipt_removed",
    "discord_retirement_receipt_sha256",
    "services_stopped_proven",
    "services_enabled",
    "recovery_worker_lock_acquired",
    "recovery_worker_exit_proven",
    "safe_to_delete_temporary_admin",
    "cleanup_completed_at_unix",
    "finalized_at_unix",
    "receipt_sha256",
})
_RECOVERY_CONCURRENT_LOSER_RECEIPT_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "recovery_takeover_gate_sha256",
    "owner_recovery_takeover_ack_sha256",
    "predecessor_kind",
    "predecessor_schema",
    "predecessor_journal_sha256",
    "predecessor_generation",
    "original_run_process_lease_sha256",
    "target_pid",
    "target_process_start_time_ticks",
    "target_boot_id_sha256",
    "target_boot_time_ns",
    "target_uid",
    "target_gid",
    "target_module_origin",
    "target_module_sha256",
    "target_process_exe_sha256",
    "target_process_cmdline_sha256",
    "target_signal_attempted_by_loser",
    "target_termination_proven_by_loser",
    "process_lock_acquired_by_loser",
    "journal_cas_attempted_by_loser",
    "journal_cas_succeeded_by_loser",
    "observed_successor_schema",
    "observed_successor_state",
    "observed_successor_journal_sha256",
    "observed_successor_generation",
    "observed_successor_worker_pid",
    "observed_successor_worker_process_start_time_ticks",
    "observed_successor_worker_boot_id_sha256",
    "observed_successor_worker_module_sha256",
    "observed_successor_worker_process_exe_sha256",
    "observed_successor_worker_process_cmdline_sha256",
    "secret_gate_emitted_by_loser",
    "admin_frame_bytes_received_by_loser",
    "admin_session_opened_by_loser",
    "admin_credential_mutation_performed_by_loser",
    "worker_lease_published_by_loser",
    "retryable",
    "completed_at_unix",
    "receipt_sha256",
})
_RECOVERY_FINALIZE_PENDING_RECEIPT_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "recovery_worker_completion_sha256",
    "recovery_worker_pid",
    "recovery_worker_process_start_time_ticks",
    "recovery_worker_boot_id_sha256",
    "recovery_worker_boot_time_ns",
    "recovery_worker_uid",
    "recovery_worker_gid",
    "recovery_worker_module_origin",
    "recovery_worker_module_sha256",
    "recovery_worker_process_exe_sha256",
    "recovery_worker_process_cmdline_sha256",
    "completion_admin_authority_may_have_been_used",
    "completion_admin_frame_zeroized",
    "completion_admin_session_closed",
    "worker_identity_state",
    "target_signal_attempted_by_finalizer",
    "target_termination_proven_by_finalizer",
    "process_lock_acquired_by_finalizer",
    "completion_cas_attempted_by_finalizer",
    "completion_cas_succeeded_by_finalizer",
    "observed_journal_sha256",
    "secret_gate_emitted_by_finalizer",
    "admin_frame_bytes_received_by_finalizer",
    "admin_session_opened_by_finalizer",
    "admin_credential_mutation_performed_by_finalizer",
    "retryable",
    "completed_at_unix",
    "receipt_sha256",
})
_ORIGINAL_RUN_PROCESS_LEASE_KEYS = frozenset({
    "schema",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "pid",
    "process_start_time_ticks",
    "boot_id_sha256",
    "boot_time_ns",
    "module_origin",
    "module_sha256",
    "process_exe_sha256",
    "process_cmdline_sha256",
    "created_at_unix",
    "lease_sha256",
})
_COORDINATOR_SECRET_GATE_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "release_sha",
    "admin_username",
    "database_host",
    "database_port",
    "database_name",
    "tls_server_name",
    "tls_peer_certificate_sha256",
    "coordinator_process_lease_sha256",
    "coordinator_pid",
    "coordinator_start_time_ticks",
    "coordinator_boot_id_sha256",
    "expires_at_unix",
    "frame_schema",
    "gate_sha256",
})
_COORDINATOR_RECEIPT_KEYS = frozenset({
    "schema",
    "ok",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "temporary_admin_delete_required",
    "admin_session_closed",
    "full_canary_plan_sha256",
    "owner_approval_sha256",
    "live_driver_result",
    "live_driver_receipt_sha256",
    "bootstrap_login_password_disabled",
    "bootstrap_credential_removed",
    "discord_token_removed",
    "coordinator_process_lease_retired",
    "services_enabled",
    "completed_at_unix",
    "receipt_sha256",
})
_COORDINATOR_FAILURE_KEYS = frozenset({
    "schema",
    "ok",
    "phase",
    "error_code",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "full_canary_plan_sha256",
    "cleanup_status",
    "recovery_material_preserved",
    "admin_session_closed",
    "bootstrap_login_password_disabled",
    "bootstrap_credential_removed",
    "discord_token_removed",
    "coordinator_process_lease_retired",
    "services_enabled",
    "receipt_sha256",
})
_OWNER_APPROVAL_REQUEST_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "full_canary_plan_sha256",
    "staged_plan_path",
    "staged_plan_file_sha256",
    "approval_path",
    "hba_receipt_sha256",
    "hba_expires_at_unix",
    "fixture_expires_at_unix",
    "credential_approval_expires_at_unix",
    "approval_deadline_unix",
    "owner_input_cutoff_unix",
    "final_approval_transmit_margin_seconds",
    "max_wait_seconds",
    "requested_at_unix",
    "approval_source_sha256",
    "approval_request_path",
    "final_approval_frame_schema",
    "prior_approval_file_sha256",
    "request_sha256",
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
_FINAL_APPROVAL_INSTALL_RECEIPT_KEYS = frozenset({
    "schema",
    "ok",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "full_canary_plan_sha256",
    "approval_request_sha256",
    "owner_approval_sha256",
    "approval_path",
    "installed_at_unix",
    "receipt_sha256",
})
_FINAL_APPROVAL_CANCEL_RECEIPT_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "reason",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "full_canary_plan_sha256",
    "approval_request_sha256",
    "approval_request_path",
    "expected_approval_request_file_sha256",
    "observed_approval_request_file_sha256",
    "approval_request_artifact_state",
    "approval_request_present",
    "approval_request_remains_active",
    "staged_plan_path",
    "expected_staged_plan_file_sha256",
    "observed_staged_plan_file_sha256",
    "staged_plan_artifact_state",
    "staged_plan_present",
    "approval_path",
    "prior_approval_file_sha256",
    "observed_approval_file_sha256",
    "owner_approval_artifact_state",
    "approval_path_matches_prior",
    "new_owner_approval_installed",
    "frame_bytes_received",
    "owner_approval_mutation_performed_by_this_helper",
    "cancelled_at_unix",
    "receipt_sha256",
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
_INVARIANT_RECEIPT_KEYS = frozenset({
    "schema",
    "ok",
    "fixture_sha256",
    "evidence_sha256",
    "full_canary_start_receipt_sha256",
    "release_sha",
    "canary_run_id",
    "invariants",
    "invariant_receipt_sha256",
})
_LIFECYCLE_RECEIPT_KEYS = frozenset({
    "operation",
    "full_canary_start_receipt_sha256",
    "full_canary_start_receipt_internal_sha256",
    "evidence_path",
    "evidence_sha256",
    "live_report_sha256",
    "verifier_result",
    "stop_order",
    "preclaim_reconciliation",
    "stopped_report_sha256",
    "units_enabled",
    "verified",
    "error_type",
    "error_sha256",
    "completed_at_unix",
    "schema",
    "stage",
    "revision",
    "full_canary_plan_sha256",
    "receipt_path",
    "receipt_sha256",
})
_PRECLAIM_RECEIPT_KEYS = frozenset({
    "version",
    "observed_at_unix",
    "source_config_path",
    "source_config_sha256",
    "database_identity",
    "database_identity_sha256",
    "result",
    "receipt_sha256",
})
_PRECLAIM_RESULT_KEYS = frozenset({
    "success",
    "outcome",
    "grant_id",
    "case_id",
    "release_sha256",
    "fixture_sha256",
    "run_id",
    "session_key_sha256",
    "expires_at",
    "approved_by",
    "approval_source_sha256",
    "provisioning_receipt_sha256",
    "preapproval_event_id",
    "bootstrap_consumption_event_id",
    "claim_event_id",
    "retirement_event_id",
    "revocation_event_id",
    "claimed_at",
    "retired_at",
    "reason",
    "scope_retired",
    "authority_active",
    "inserted",
    "deduped",
})
_CANARY_INVARIANTS = (
    "live_provenance_bound",
    "canonical_writer_ready",
    "owner_preapproved_one_shot_scope_claimed_and_durably_revoked",
    "gpt56_model_authored_high_to_xhigh",
    "canonical_plan_event_verification_truth_complete",
    "public_discord_routeback_signed_and_readback_verified",
    "discord_dm_private_target_denied_without_dispatch",
    "sustained_multistep_task_completed_nonpartial",
)
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
    "bootstrap_login_password_disabled",
    "bootstrap_credential_removed",
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
class _RecoveryAttemptState:
    """Outer-visible truth for a privileged recovery-admin attempt."""

    gate: Mapping[str, Any] | None = None
    takeover_ack: Mapping[str, Any] | None = None
    secret_gate: Mapping[str, Any] | None = None
    worker_completion: Mapping[str, Any] | None = None
    final_receipt: Mapping[str, Any] | None = None
    concurrent_loser_receipt: Mapping[str, Any] | None = None
    finalize_pending_receipt: Mapping[str, Any] | None = None
    username: str | None = None
    password: bytearray | None = None
    password_wiped: bool | None = None
    admin_mutation_attempted: bool = False
    admin_mutation_confirmed: bool = False
    admin_mutation_ambiguous: bool = False
    admin_mutation_explicitly_not_committed: bool = False
    admin_mutation_ambiguity_observed: bool = False
    admin_reconciliation_performed: bool = False
    admin_reconciliation_evidence_sha256: str | None = None
    admin_reconciliation_quiet_window_seconds: float | None = None
    admin_response_known_candidate_observed: bool | None = None
    admin_post_baseline_authority_operation_count: int | None = None
    admin_delete_pending: bool = False
    frame_write_attempted: bool = False
    frame_disclosed_or_ambiguous: bool = False
    remote_termination_proven: bool = False
    terminal_receipt_validated: bool = False
    admin_session_closed: bool | None = None
    recovery_required: bool = False
    cleanup_failure_codes: set[str] = field(default_factory=set)

    def wipe_password(self) -> None:
        if self.password is None:
            return
        _wipe(self.password)
        self.password_wiped = True

    def absorb(self, later: "_RecoveryAttemptState") -> None:
        if (
            self.username is not None
            and later.username is not None
            and self.username != later.username
        ):
            self.cleanup_failure_codes.add("recovery_admin_username_drifted")
            self.recovery_required = True
            return
        self.gate = later.gate or self.gate
        self.takeover_ack = later.takeover_ack or self.takeover_ack
        self.secret_gate = later.secret_gate or self.secret_gate
        self.worker_completion = later.worker_completion or self.worker_completion
        self.final_receipt = later.final_receipt or self.final_receipt
        self.concurrent_loser_receipt = (
            later.concurrent_loser_receipt or self.concurrent_loser_receipt
        )
        self.finalize_pending_receipt = (
            later.finalize_pending_receipt or self.finalize_pending_receipt
        )
        self.username = later.username or self.username
        self.admin_mutation_attempted |= later.admin_mutation_attempted
        self.admin_mutation_confirmed |= later.admin_mutation_confirmed
        self.admin_mutation_ambiguous |= later.admin_mutation_ambiguous
        self.admin_mutation_explicitly_not_committed |= (
            later.admin_mutation_explicitly_not_committed
        )
        self.admin_mutation_ambiguity_observed |= (
            later.admin_mutation_ambiguity_observed
        )
        self.admin_reconciliation_performed |= later.admin_reconciliation_performed
        if later.admin_reconciliation_evidence_sha256 is not None:
            if (
                self.admin_reconciliation_evidence_sha256 is not None
                and self.admin_reconciliation_evidence_sha256
                != later.admin_reconciliation_evidence_sha256
            ):
                self.cleanup_failure_codes.add(
                    "cloud_sql_reconciliation_evidence_drifted"
                )
                self.recovery_required = True
            else:
                self.admin_reconciliation_evidence_sha256 = (
                    later.admin_reconciliation_evidence_sha256
                )
                self.admin_reconciliation_quiet_window_seconds = (
                    later.admin_reconciliation_quiet_window_seconds
                )
                self.admin_response_known_candidate_observed = (
                    later.admin_response_known_candidate_observed
                )
                self.admin_post_baseline_authority_operation_count = (
                    later.admin_post_baseline_authority_operation_count
                )
        self.admin_delete_pending |= later.admin_delete_pending
        self.frame_write_attempted |= later.frame_write_attempted
        self.frame_disclosed_or_ambiguous |= later.frame_disclosed_or_ambiguous
        self.remote_termination_proven |= later.remote_termination_proven
        self.terminal_receipt_validated |= later.terminal_receipt_validated
        if later.admin_session_closed is not None:
            self.admin_session_closed = later.admin_session_closed
        if later.final_receipt is not None:
            self.recovery_required = False
        else:
            self.recovery_required |= later.recovery_required
        if later.password_wiped is not None:
            self.password_wiped = later.password_wiped
        self.cleanup_failure_codes.update(later.cleanup_failure_codes)


def _validated_sql_reconciliation_evidence(
    sql_admin: Any,
) -> Mapping[str, Any]:
    value = sql_admin.reconciliation_evidence()
    if not isinstance(value, Mapping) or set(value) != {
        "mutation_ambiguity_observed",
        "reconciliation_proven",
        "reconciliation_evidence_sha256",
        "quiet_window_seconds",
        "response_known_candidate_observed",
        "post_baseline_authority_operation_count",
    }:
        raise OwnerLauncherError("cloud_sql_reconciliation_evidence_invalid")
    ambiguity = value.get("mutation_ambiguity_observed")
    proven = value.get("reconciliation_proven")
    digest = value.get("reconciliation_evidence_sha256")
    quiet = value.get("quiet_window_seconds")
    candidate_observed = value.get("response_known_candidate_observed")
    operation_count = value.get("post_baseline_authority_operation_count")
    if (
        type(ambiguity) is not bool
        or type(proven) is not bool
        or (
            proven
            and (
                not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
                or not isinstance(quiet, (int, float))
                or isinstance(quiet, bool)
                or quiet <= 0
                or type(candidate_observed) is not bool
                or type(operation_count) is not int
                or operation_count < 0
                or candidate_observed
                and operation_count < 1
            )
        )
        or (
            not proven
            and (
                digest is not None
                or quiet is not None
                or candidate_observed is not None
                or operation_count is not None
            )
        )
    ):
        raise OwnerLauncherError("cloud_sql_reconciliation_evidence_invalid")
    return value


def _adopt_sql_reconciliation_evidence(
    state: _RecoveryAttemptState,
    evidence: Mapping[str, Any],
) -> None:
    state.admin_mutation_ambiguity_observed |= bool(
        evidence["mutation_ambiguity_observed"]
    )
    if evidence["reconciliation_proven"] is True:
        state.admin_reconciliation_performed = True
        state.admin_reconciliation_evidence_sha256 = str(
            evidence["reconciliation_evidence_sha256"]
        )
        state.admin_reconciliation_quiet_window_seconds = float(
            evidence["quiet_window_seconds"]
        )
        state.admin_response_known_candidate_observed = bool(
            evidence["response_known_candidate_observed"]
        )
        state.admin_post_baseline_authority_operation_count = int(
            evidence["post_baseline_authority_operation_count"]
        )


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


def _require_sha256(value: object, code: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise OwnerLauncherError(code)
    return value


def validate_owner_launch_gate(
    gate: Mapping[str, Any],
    *,
    expected_release_sha: str,
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate the exact read-only approval gate before any mutation."""

    if not _RELEASE_SHA.fullmatch(expected_release_sha):
        raise OwnerLauncherError("invalid_release_sha")
    if type(now_unix) is not int or now_unix < 0:
        raise OwnerLauncherError("invalid_clock")
    if not isinstance(gate, Mapping) or set(gate) != _OWNER_GATE_KEYS:
        raise OwnerLauncherError("invalid_owner_gate")
    if (
        gate.get("schema") != OWNER_GATE_SCHEMA
        or gate.get("ok") is not True
        or gate.get("state") != "credential_prepare_authorized"
        or gate.get("release_sha") != expected_release_sha
        or gate.get("database_host") != DATABASE_HOST
        or gate.get("database_port") != DATABASE_PORT
        or gate.get("database_name") != DATABASE_NAME
    ):
        raise OwnerLauncherError("invalid_owner_gate")
    _require_sha256(gate.get("coordinator_input_sha256"), "invalid_owner_gate")
    _require_sha256(gate.get("owner_subject_sha256"), "invalid_owner_gate")
    approval_sha = _require_sha256(
        gate.get("credential_prepare_approval_sha256"), "invalid_owner_gate"
    )
    expected_username = f"{ADMIN_USERNAME_PREFIX}{approval_sha[:16]}"
    if gate.get("admin_username") != expected_username:
        raise OwnerLauncherError("invalid_owner_gate")
    expires = gate.get("expires_at_unix")
    if (
        type(expires) is not int
        or expires <= now_unix
        or expires - now_unix > _GATE_MAX_FUTURE_SECONDS
    ):
        raise OwnerLauncherError("stale_owner_gate")
    gate_sha = _require_sha256(gate.get("gate_sha256"), "invalid_owner_gate")
    unsigned = dict(gate)
    del unsigned["gate_sha256"]
    if gate_sha != _sha256(_canonical_bytes(unsigned)):
        raise OwnerLauncherError("invalid_owner_gate")
    return dict(gate)


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


def validate_recovery_gate(
    gate: Mapping[str, Any],
    *,
    expected_release_sha: str,
    owner_gate: Mapping[str, Any] | None,
    now_unix: int,
) -> Mapping[str, Any]:
    value = _validate_self_digest(
        gate,
        expected_keys=_RECOVERY_GATE_KEYS,
        digest_key="gate_sha256",
        code="invalid_recovery_gate",
    )
    approval_sha = value.get("credential_prepare_approval_sha256")
    expires = value.get("expires_at_unix")
    module_origin = value.get("target_module_origin")
    discord_token_state = value.get("discord_token_state")
    discord_retirement_sha = value.get("discord_retirement_receipt_sha256")
    token_identity = (value.get("token_device"), value.get("token_inode"))
    token_identity_materialized = (
        type(token_identity[0]) is int
        and token_identity[0] > 0
        and type(token_identity[1]) is int
        and token_identity[1] > 0
    )
    token_identity_retired = token_identity == (None, None) or (
        token_identity_materialized
    )
    module_pattern = re.compile(
        rf"^/opt/muncho-canary-releases/{re.escape(expected_release_sha)}/"
        r"venv/lib/python3\.[0-9]+/site-packages/gateway/"
        r"canonical_full_canary_coordinator\.py$"
    )
    if (
        value.get("schema") != RECOVERY_GATE_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "awaiting_owner_recovery_takeover_ack"
        or value.get("release_sha") != expected_release_sha
        or not isinstance(approval_sha, str)
        or _SHA256.fullmatch(approval_sha) is None
        or value.get("ephemeral_admin_username")
        != f"{ADMIN_USERNAME_PREFIX}{approval_sha[:16]}"
        or value.get("predecessor_kind")
        not in {"run_process_lease", "recovery_worker_lease"}
        or value.get("predecessor_schema")
        not in {
            "muncho-full-canary-coordinator-process-lease.v1",
            "muncho-full-canary-recovery-worker-lease.v1",
        }
        or type(value.get("predecessor_generation")) is not int
        or value["predecessor_generation"] < 0
        or value.get("predecessor_kind") == "run_process_lease"
        and (
            value.get("predecessor_schema")
            != "muncho-full-canary-coordinator-process-lease.v1"
            or value["predecessor_generation"] != 0
        )
        or value.get("predecessor_kind") == "recovery_worker_lease"
        and (
            value.get("predecessor_schema") != RECOVERY_WORKER_LEASE_SCHEMA
            or value["predecessor_generation"] < 1
        )
        or type(value.get("target_pid")) is not int
        or value["target_pid"] <= 1
        or type(value.get("target_process_start_time_ticks")) is not int
        or value["target_process_start_time_ticks"] <= 0
        or type(value.get("target_boot_time_ns")) is not int
        or value["target_boot_time_ns"] < 0
        or value.get("target_uid") != 0
        or value.get("target_gid") != 0
        or value.get("target_process_identity_state")
        not in {"exact_alive", "not_alive"}
        or discord_token_state not in {"installed", "retirement_prepared", "retired"}
        or discord_token_state == "installed"
        and (discord_retirement_sha is not None or not token_identity_materialized)
        or discord_token_state == "retirement_prepared"
        and (
            not token_identity_retired
            or not isinstance(discord_retirement_sha, str)
            or _SHA256.fullmatch(discord_retirement_sha) is None
        )
        or discord_token_state == "retired"
        and (
            not token_identity_retired
            or not isinstance(discord_retirement_sha, str)
            or _SHA256.fullmatch(discord_retirement_sha) is None
        )
        or value.get("db_secret_accepted") is not False
        or value.get("frame_schema") != RECOVERY_ACK_FRAME_SCHEMA
        or not isinstance(module_origin, str)
        or module_pattern.fullmatch(module_origin) is None
        or type(value.get("observed_at_unix")) is not int
        or type(expires) is not int
        or not value["observed_at_unix"] <= now_unix < expires
        or expires - value["observed_at_unix"] > 300
    ):
        raise OwnerLauncherError("invalid_recovery_gate")
    for name in (
        "coordinator_input_sha256",
        "owner_subject_sha256",
        "predecessor_journal_sha256",
        "original_run_process_lease_sha256",
        "causal_recovery_state_sha256",
        "discord_token_install_receipt_sha256",
        "target_boot_id_sha256",
        "target_module_sha256",
        "target_process_exe_sha256",
        "target_process_cmdline_sha256",
    ):
        _require_sha256(value.get(name), "invalid_recovery_gate")
    if owner_gate is not None and (
        value.get("coordinator_input_sha256")
        != owner_gate.get("coordinator_input_sha256")
        or approval_sha != owner_gate.get("credential_prepare_approval_sha256")
        or value.get("owner_subject_sha256") != owner_gate.get("owner_subject_sha256")
        or value.get("ephemeral_admin_username") != owner_gate.get("admin_username")
    ):
        raise OwnerLauncherError("invalid_recovery_gate")
    return value


def build_recovery_ack(
    gate: Mapping[str, Any],
    *,
    now_unix: int,
    nonce: bytes | None = None,
) -> Mapping[str, Any]:
    nonce_value = secrets.token_bytes(32) if nonce is None else nonce
    if not isinstance(nonce_value, bytes) or len(nonce_value) < 16:
        raise OwnerLauncherError("invalid_recovery_ack")
    observed_at = gate.get("observed_at_unix")
    if type(observed_at) is not int or now_unix < observed_at:
        raise OwnerLauncherError("invalid_recovery_ack")
    expires = min(int(gate["expires_at_unix"]), now_unix + 300)
    if expires <= now_unix:
        raise OwnerLauncherError("invalid_recovery_ack")
    unsigned = {
        "schema": RECOVERY_ACK_SCHEMA,
        "scope": "terminate_exact_recovery_predecessor_and_claim_worker",
        "release_sha": gate["release_sha"],
        "coordinator_input_sha256": gate["coordinator_input_sha256"],
        "credential_prepare_approval_sha256": gate[
            "credential_prepare_approval_sha256"
        ],
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "ephemeral_admin_username": gate["ephemeral_admin_username"],
        "predecessor_kind": gate["predecessor_kind"],
        "predecessor_schema": gate["predecessor_schema"],
        "predecessor_journal_sha256": gate["predecessor_journal_sha256"],
        "predecessor_generation": gate["predecessor_generation"],
        "original_run_process_lease_sha256": gate["original_run_process_lease_sha256"],
        "causal_recovery_state_sha256": gate["causal_recovery_state_sha256"],
        "target_pid": gate["target_pid"],
        "target_process_start_time_ticks": gate["target_process_start_time_ticks"],
        "target_boot_id_sha256": gate["target_boot_id_sha256"],
        "target_boot_time_ns": gate["target_boot_time_ns"],
        "target_uid": gate["target_uid"],
        "target_gid": gate["target_gid"],
        "target_module_origin": gate["target_module_origin"],
        "target_module_sha256": gate["target_module_sha256"],
        "target_process_exe_sha256": gate["target_process_exe_sha256"],
        "target_process_cmdline_sha256": gate["target_process_cmdline_sha256"],
        "target_process_identity_state": gate["target_process_identity_state"],
        "discord_token_state": gate["discord_token_state"],
        "discord_token_install_receipt_sha256": gate[
            "discord_token_install_receipt_sha256"
        ],
        "discord_retirement_receipt_sha256": gate["discord_retirement_receipt_sha256"],
        "token_device": gate["token_device"],
        "token_inode": gate["token_inode"],
        "recovery_takeover_gate_sha256": gate["gate_sha256"],
        "nonce_sha256": _sha256(nonce_value),
        "approved_at_unix": now_unix,
        "expires_at_unix": expires,
    }
    return {**unsigned, "ack_sha256": _sha256(_canonical_bytes(unsigned))}


def build_recovery_ack_frame(
    gate: Mapping[str, Any],
    ack: Mapping[str, Any],
    *,
    password: bytearray | None = None,
) -> bytearray:
    if not isinstance(ack, Mapping) or set(ack) != _RECOVERY_ACK_KEYS:
        raise OwnerLauncherError("invalid_recovery_ack")
    expected = _require_sha256(ack.get("ack_sha256"), "invalid_recovery_ack")
    unsigned = dict(ack)
    del unsigned["ack_sha256"]
    if expected != _sha256(_canonical_bytes(unsigned)):
        raise OwnerLauncherError("invalid_recovery_ack")
    payload = _canonical_bytes(ack)
    frame = bytearray(RECOVERY_ACK_FRAME_MAGIC)
    frame.extend(struct.pack(">I", len(payload)))
    frame.extend(payload)
    if password is not None:
        raise OwnerLauncherError("unexpected_recovery_admin_credential")
    if len(frame) > 128 * 1024 + 8:
        raise OwnerLauncherError("invalid_recovery_ack")
    return frame


def validate_recovery_secret_gate(
    secret_gate: Mapping[str, Any],
    *,
    takeover_gate: Mapping[str, Any],
    takeover_ack: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    value = _validate_self_digest(
        secret_gate,
        expected_keys=_RECOVERY_SECRET_GATE_KEYS,
        digest_key="gate_sha256",
        code="invalid_recovery_secret_gate",
    )
    module_origin = value.get("recovery_worker_module_origin")
    expires = value.get("expires_at_unix")
    if (
        value.get("schema") != RECOVERY_SECRET_GATE_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "awaiting_recovery_admin_credential"
        or value.get("release_sha") != takeover_gate.get("release_sha")
        or value.get("coordinator_input_sha256")
        != takeover_gate.get("coordinator_input_sha256")
        or value.get("credential_prepare_approval_sha256")
        != takeover_gate.get("credential_prepare_approval_sha256")
        or value.get("owner_subject_sha256")
        != takeover_gate.get("owner_subject_sha256")
        or value.get("ephemeral_admin_username")
        != takeover_gate.get("ephemeral_admin_username")
        or value.get("recovery_takeover_gate_sha256")
        != takeover_gate.get("gate_sha256")
        or value.get("owner_recovery_takeover_ack_sha256")
        != takeover_ack.get("ack_sha256")
        or value.get("predecessor_kind") != takeover_gate.get("predecessor_kind")
        or value.get("predecessor_journal_sha256")
        != takeover_gate.get("predecessor_journal_sha256")
        or value.get("predecessor_generation")
        != takeover_gate.get("predecessor_generation")
        or value.get("original_run_process_lease_sha256")
        != takeover_gate.get("original_run_process_lease_sha256")
        or value.get("causal_recovery_state_sha256")
        != takeover_gate.get("causal_recovery_state_sha256")
        or value.get("recovery_worker_state") != "admin_authority_may_be_in_use"
        or value.get("recovery_worker_transition_seq") != 2
        or type(value.get("recovery_worker_pid")) is not int
        or value["recovery_worker_pid"] <= 1
        or type(value.get("recovery_worker_process_start_time_ticks")) is not int
        or value["recovery_worker_process_start_time_ticks"] <= 0
        or type(value.get("recovery_worker_boot_time_ns")) is not int
        or value["recovery_worker_boot_time_ns"] < 0
        or value.get("recovery_worker_uid") != 0
        or value.get("recovery_worker_gid") != 0
        or not isinstance(module_origin, str)
        or re.fullmatch(
            rf"/opt/muncho-canary-releases/{re.escape(str(value['release_sha']))}/"
            r"venv/lib/python3\.[0-9]+/site-packages/gateway/"
            r"canonical_full_canary_coordinator\.py",
            module_origin,
        )
        is None
        or value.get("database_host") != DATABASE_HOST
        or value.get("database_port") != DATABASE_PORT
        or value.get("database_name") != DATABASE_NAME
        or not isinstance(value.get("tls_server_name"), str)
        or _TLS_SERVER_NAME.fullmatch(value["tls_server_name"]) is None
        or value.get("admin_frame_schema") != RECOVERY_ADMIN_FRAME_SCHEMA
        or type(expires) is not int
        or not now_unix < expires
        or expires > takeover_gate.get("expires_at_unix")
        or expires > takeover_ack.get("expires_at_unix")
    ):
        raise OwnerLauncherError("invalid_recovery_secret_gate")
    for name in (
        "recovery_worker_lease_sha256",
        "recovery_worker_boot_id_sha256",
        "recovery_worker_module_sha256",
        "recovery_worker_process_exe_sha256",
        "recovery_worker_process_cmdline_sha256",
        "tls_peer_certificate_sha256",
        "gate_nonce_sha256",
    ):
        _require_sha256(value.get(name), "invalid_recovery_secret_gate")
    return value


def validate_recovery_concurrent_loser_receipt(
    receipt: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    ack: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    value = _validate_self_digest(
        receipt,
        expected_keys=_RECOVERY_CONCURRENT_LOSER_RECEIPT_KEYS,
        digest_key="receipt_sha256",
        code="invalid_recovery_concurrent_loser_receipt",
    )
    if (
        value.get("schema") != RECOVERY_CONCURRENT_LOSER_RECEIPT_SCHEMA
        or value.get("ok") is not False
        or value.get("state") != "recovery_worker_claim_lost_no_secret"
        or value.get("release_sha") != gate.get("release_sha")
        or value.get("coordinator_input_sha256") != gate.get("coordinator_input_sha256")
        or value.get("credential_prepare_approval_sha256")
        != gate.get("credential_prepare_approval_sha256")
        or value.get("owner_subject_sha256") != gate.get("owner_subject_sha256")
        or value.get("ephemeral_admin_username") != gate.get("ephemeral_admin_username")
        or value.get("recovery_takeover_gate_sha256") != gate.get("gate_sha256")
        or value.get("owner_recovery_takeover_ack_sha256") != ack.get("ack_sha256")
        or any(
            value.get(receipt_name) != gate.get(gate_name)
            for receipt_name, gate_name in (
                ("predecessor_kind", "predecessor_kind"),
                ("predecessor_schema", "predecessor_schema"),
                ("predecessor_journal_sha256", "predecessor_journal_sha256"),
                ("predecessor_generation", "predecessor_generation"),
                (
                    "original_run_process_lease_sha256",
                    "original_run_process_lease_sha256",
                ),
                ("target_pid", "target_pid"),
                (
                    "target_process_start_time_ticks",
                    "target_process_start_time_ticks",
                ),
                ("target_boot_id_sha256", "target_boot_id_sha256"),
                ("target_boot_time_ns", "target_boot_time_ns"),
                ("target_uid", "target_uid"),
                ("target_gid", "target_gid"),
                ("target_module_origin", "target_module_origin"),
                ("target_module_sha256", "target_module_sha256"),
                ("target_process_exe_sha256", "target_process_exe_sha256"),
                (
                    "target_process_cmdline_sha256",
                    "target_process_cmdline_sha256",
                ),
            )
        )
        or type(value.get("target_signal_attempted_by_loser")) is not bool
        or type(value.get("target_termination_proven_by_loser")) is not bool
        or type(value.get("process_lock_acquired_by_loser")) is not bool
        or type(value.get("journal_cas_attempted_by_loser")) is not bool
        or value.get("journal_cas_succeeded_by_loser") is not False
        or value.get("observed_successor_journal_sha256")
        == gate.get("predecessor_journal_sha256")
        or value.get("secret_gate_emitted_by_loser") is not False
        or value.get("admin_frame_bytes_received_by_loser") != 0
        or value.get("admin_session_opened_by_loser") is not False
        or value.get("admin_credential_mutation_performed_by_loser") is not False
        or value.get("worker_lease_published_by_loser") is not False
        or value.get("retryable") is not True
        or type(value.get("completed_at_unix")) is not int
        or not 0 <= value["completed_at_unix"] <= now_unix + 30
    ):
        raise OwnerLauncherError("invalid_recovery_concurrent_loser_receipt")
    for name in (
        "observed_successor_journal_sha256",
        "observed_successor_worker_boot_id_sha256",
        "observed_successor_worker_module_sha256",
        "observed_successor_worker_process_exe_sha256",
        "observed_successor_worker_process_cmdline_sha256",
    ):
        _require_sha256(
            value.get(name),
            "invalid_recovery_concurrent_loser_receipt",
        )
    if (
        not isinstance(value.get("observed_successor_schema"), str)
        or not isinstance(value.get("observed_successor_state"), str)
        or type(value.get("observed_successor_generation")) is not int
        or value["observed_successor_generation"] < 1
        or type(value.get("observed_successor_worker_pid")) is not int
        or value["observed_successor_worker_pid"] <= 1
        or type(value.get("observed_successor_worker_process_start_time_ticks"))
        is not int
        or value["observed_successor_worker_process_start_time_ticks"] <= 0
    ):
        raise OwnerLauncherError("invalid_recovery_concurrent_loser_receipt")
    return value


def validate_recovery_finalize_pending_receipt(
    receipt: Mapping[str, Any],
    *,
    completion: Mapping[str, Any] | None = None,
    expected_release_sha: str | None = None,
    owner_gate: Mapping[str, Any] | None = None,
    now_unix: int,
) -> Mapping[str, Any]:
    value = _validate_self_digest(
        receipt,
        expected_keys=_RECOVERY_FINALIZE_PENDING_RECEIPT_KEYS,
        digest_key="receipt_sha256",
        code="invalid_recovery_finalize_pending_receipt",
    )
    expected_release = (
        completion.get("release_sha")
        if completion is not None
        else expected_release_sha
    )
    if (
        not isinstance(expected_release, str)
        or _RELEASE_SHA.fullmatch(expected_release) is None
    ):
        raise OwnerLauncherError("invalid_recovery_finalize_pending_receipt")
    completion_bound = completion is not None
    worker_identity_names = (
        "recovery_worker_pid",
        "recovery_worker_process_start_time_ticks",
        "recovery_worker_boot_id_sha256",
        "recovery_worker_boot_time_ns",
        "recovery_worker_uid",
        "recovery_worker_gid",
        "recovery_worker_module_origin",
        "recovery_worker_module_sha256",
        "recovery_worker_process_exe_sha256",
        "recovery_worker_process_cmdline_sha256",
    )
    module_origin = value.get("recovery_worker_module_origin")
    approval_sha = value.get("credential_prepare_approval_sha256")
    if (
        value.get("schema") != RECOVERY_FINALIZE_PENDING_RECEIPT_SCHEMA
        or value.get("ok") is not False
        or value.get("state") != "recovery_finalization_pending_no_secret"
        or value.get("release_sha") != expected_release
        or not isinstance(approval_sha, str)
        or _SHA256.fullmatch(approval_sha) is None
        or value.get("ephemeral_admin_username")
        != f"{ADMIN_USERNAME_PREFIX}{approval_sha[:16]}"
        or completion_bound
        and (
            value.get("coordinator_input_sha256")
            != completion.get("coordinator_input_sha256")
            or approval_sha != completion.get("credential_prepare_approval_sha256")
            or value.get("owner_subject_sha256")
            != completion.get("owner_subject_sha256")
            or value.get("ephemeral_admin_username")
            != completion.get("ephemeral_admin_username")
            or value.get("recovery_worker_completion_sha256")
            != completion.get("completion_sha256")
            or any(
                value.get(name) != completion.get(name)
                for name in worker_identity_names
            )
        )
        or owner_gate is not None
        and (
            value.get("coordinator_input_sha256")
            != owner_gate.get("coordinator_input_sha256")
            or approval_sha != owner_gate.get("credential_prepare_approval_sha256")
            or value.get("owner_subject_sha256")
            != owner_gate.get("owner_subject_sha256")
            or value.get("ephemeral_admin_username")
            != owner_gate.get(
                "admin_username",
                owner_gate.get("ephemeral_admin_username"),
            )
        )
        or type(value.get("recovery_worker_pid")) is not int
        or value["recovery_worker_pid"] <= 1
        or type(value.get("recovery_worker_process_start_time_ticks")) is not int
        or value["recovery_worker_process_start_time_ticks"] <= 0
        or type(value.get("recovery_worker_boot_time_ns")) is not int
        or value["recovery_worker_boot_time_ns"] < 0
        or value.get("recovery_worker_uid") != 0
        or value.get("recovery_worker_gid") != 0
        or not isinstance(module_origin, str)
        or re.fullmatch(
            rf"/opt/muncho-canary-releases/{re.escape(expected_release)}/"
            r"venv/lib/python3\.[0-9]+/site-packages/gateway/"
            r"canonical_full_canary_coordinator\.py",
            module_origin,
        )
        is None
        or value.get("completion_admin_authority_may_have_been_used") is not True
        or value.get("completion_admin_frame_zeroized") is not True
        or value.get("completion_admin_session_closed") is not True
        or value.get("worker_identity_state") not in {"exact_alive", "not_alive"}
        or type(value.get("target_signal_attempted_by_finalizer")) is not bool
        or type(value.get("target_termination_proven_by_finalizer")) is not bool
        or type(value.get("process_lock_acquired_by_finalizer")) is not bool
        or type(value.get("completion_cas_attempted_by_finalizer")) is not bool
        or value.get("completion_cas_succeeded_by_finalizer") is not False
        or value.get("secret_gate_emitted_by_finalizer") is not False
        or value.get("admin_frame_bytes_received_by_finalizer") != 0
        or value.get("admin_session_opened_by_finalizer") is not False
        or value.get("admin_credential_mutation_performed_by_finalizer") is not False
        or value.get("retryable") is not True
        or type(value.get("completed_at_unix")) is not int
        or not 0 <= value["completed_at_unix"] <= now_unix + 30
    ):
        raise OwnerLauncherError("invalid_recovery_finalize_pending_receipt")
    for name in (
        "coordinator_input_sha256",
        "credential_prepare_approval_sha256",
        "owner_subject_sha256",
        "recovery_worker_completion_sha256",
        "recovery_worker_boot_id_sha256",
        "recovery_worker_module_sha256",
        "recovery_worker_process_exe_sha256",
        "recovery_worker_process_cmdline_sha256",
        "observed_journal_sha256",
    ):
        _require_sha256(
            value.get(name),
            "invalid_recovery_finalize_pending_receipt",
        )
    return value


def _validate_original_run_process_lease(
    lease: Any,
    *,
    value: Mapping[str, Any],
    expected_release_sha: str,
    code: str,
) -> Mapping[str, Any]:
    original = _validate_self_digest(
        lease,
        expected_keys=_ORIGINAL_RUN_PROCESS_LEASE_KEYS,
        digest_key="lease_sha256",
        code=code,
    )
    module_origin = original.get("module_origin")
    approval_sha = original.get("credential_prepare_approval_sha256")
    if (
        original.get("schema") != "muncho-full-canary-coordinator-process-lease.v1"
        or original.get("release_sha") != expected_release_sha
        or original.get("coordinator_input_sha256")
        != value.get("coordinator_input_sha256")
        or approval_sha != value.get("credential_prepare_approval_sha256")
        or original.get("owner_subject_sha256") != value.get("owner_subject_sha256")
        or original.get("ephemeral_admin_username")
        != value.get("ephemeral_admin_username")
        or original.get("lease_sha256")
        != value.get("original_run_process_lease_sha256")
        or not isinstance(approval_sha, str)
        or _SHA256.fullmatch(approval_sha) is None
        or type(original.get("pid")) is not int
        or original["pid"] <= 1
        or type(original.get("process_start_time_ticks")) is not int
        or original["process_start_time_ticks"] <= 0
        or type(original.get("boot_time_ns")) is not int
        or original["boot_time_ns"] < 0
        or type(original.get("created_at_unix")) is not int
        or original["created_at_unix"] < 0
        or not isinstance(module_origin, str)
        or re.fullmatch(
            rf"/opt/muncho-canary-releases/{re.escape(expected_release_sha)}/"
            r"venv/lib/python3\.[0-9]+/site-packages/gateway/"
            r"canonical_full_canary_coordinator\.py",
            module_origin,
        )
        is None
    ):
        raise OwnerLauncherError(code)
    for name in (
        "coordinator_input_sha256",
        "credential_prepare_approval_sha256",
        "owner_subject_sha256",
        "boot_id_sha256",
        "module_sha256",
        "process_exe_sha256",
        "process_cmdline_sha256",
    ):
        _require_sha256(original.get(name), code)
    return original


def _validate_recovery_cleanup_matrix(
    value: Mapping[str, Any],
    *,
    expected_release_sha: str,
    now_unix: int,
    final: bool,
    require_fresh: bool,
    code: str,
) -> None:
    canonical_stop = value.get("canonical_stop_receipt_sha256")
    preplan_stop = value.get("preplan_stopped_report_sha256")
    preclaim_digest = value.get("preclaim_reconciliation_receipt_sha256")
    preclaim_state = value.get("preclaim_reconciliation_state")
    approval_sha = value.get("credential_prepare_approval_sha256")
    _validate_original_run_process_lease(
        value.get("original_run_process_lease"),
        value=value,
        expected_release_sha=expected_release_sha,
        code=code,
    )
    if (
        value.get("release_sha") != expected_release_sha
        or not isinstance(approval_sha, str)
        or _SHA256.fullmatch(approval_sha) is None
        or value.get("ephemeral_admin_username")
        != f"{ADMIN_USERNAME_PREFIX}{approval_sha[:16]}"
        or value.get("predecessor_kind")
        not in {"run_process_lease", "recovery_worker_lease"}
        or type(value.get("predecessor_generation")) is not int
        or type(value.get("recovery_generation")) is not int
        or value["predecessor_generation"] < 0
        or value.get("predecessor_kind") == "run_process_lease"
        and value["predecessor_generation"] != 0
        or value.get("predecessor_kind") == "recovery_worker_lease"
        and value["predecessor_generation"] < 1
        or value["recovery_generation"] != value["predecessor_generation"] + 1
        or type(value.get("recovery_worker_pid")) is not int
        or value["recovery_worker_pid"] <= 1
        or type(value.get("recovery_worker_process_start_time_ticks")) is not int
        or value["recovery_worker_process_start_time_ticks"] <= 0
        or type(value.get("recovery_worker_boot_time_ns")) is not int
        or value["recovery_worker_boot_time_ns"] < 0
        or value.get("recovery_worker_uid") != 0
        or value.get("recovery_worker_gid") != 0
        or not isinstance(value.get("recovery_worker_module_origin"), str)
        or re.fullmatch(
            rf"/opt/muncho-canary-releases/{re.escape(expected_release_sha)}/"
            r"venv/lib/python3\.[0-9]+/site-packages/gateway/"
            r"canonical_full_canary_coordinator\.py",
            value["recovery_worker_module_origin"],
        )
        is None
        or (canonical_stop is None) == (preplan_stop is None)
        or any(
            value.get(name) is not True
            for name in (
                "predecessor_termination_proven",
                "predecessor_process_lock_acquired",
                "predecessor_journal_replaced",
                "admin_frame_zeroized",
                "admin_session_closed",
                "migration_owner_membership_removed",
                "bootstrap_login_password_disabled",
                "bootstrap_credential_removed",
                "discord_token_removed",
                "discord_install_receipt_removed",
                "services_stopped_proven",
            )
        )
        or value.get("services_enabled") is not False
        or value.get("recovery_worker_exit_proven") is not final
        or value.get("safe_to_delete_temporary_admin") is not final
    ):
        raise OwnerLauncherError(code)
    for name in (
        "coordinator_input_sha256",
        "owner_subject_sha256",
        "original_run_process_lease_sha256",
        "causal_recovery_state_sha256",
        "predecessor_journal_sha256",
        "recovery_takeover_gate_sha256",
        "owner_recovery_takeover_ack_sha256",
        "recovery_worker_lease_sha256",
        "recovery_worker_boot_id_sha256",
        "recovery_worker_module_sha256",
        "recovery_worker_process_exe_sha256",
        "recovery_worker_process_cmdline_sha256",
        "discord_retirement_receipt_sha256",
    ):
        _require_sha256(value.get(name), code)
    for digest in (canonical_stop, preplan_stop):
        if digest is not None:
            _require_sha256(digest, code)
    if preplan_stop is not None:
        if preclaim_digest is not None or preclaim_state is not None:
            raise OwnerLauncherError(code)
    else:
        _require_sha256(preclaim_digest, code)
        if preclaim_state not in {"retired", "claimed", "not_preapproved"}:
            raise OwnerLauncherError(code)
    completed = value.get("cleanup_completed_at_unix")
    if (
        type(completed) is not int
        or completed < 0
        or completed > now_unix + 30
        or require_fresh
        and now_unix - completed > _GATE_MAX_FUTURE_SECONDS
    ):
        raise OwnerLauncherError(code)


def validate_recovery_worker_completion(
    completion: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    ack: Mapping[str, Any],
    secret_gate: Mapping[str, Any],
    password: bytearray,
    now_unix: int,
) -> Mapping[str, Any]:
    _reject_secret_echo(
        completion,
        active_secrets=(password,),
        code="invalid_recovery_worker_completion",
    )
    value = _validate_self_digest(
        completion,
        expected_keys=_RECOVERY_WORKER_COMPLETION_KEYS,
        digest_key="completion_sha256",
        code="invalid_recovery_worker_completion",
    )
    if (
        value.get("schema") != RECOVERY_WORKER_COMPLETION_SCHEMA
        or value.get("ok") is not False
        or value.get("state") != "cleanup_complete_awaiting_worker_exit"
        or value.get("coordinator_input_sha256") != gate.get("coordinator_input_sha256")
        or value.get("credential_prepare_approval_sha256")
        != gate.get("credential_prepare_approval_sha256")
        or value.get("owner_subject_sha256") != gate.get("owner_subject_sha256")
        or value.get("ephemeral_admin_username") != gate.get("ephemeral_admin_username")
        or value.get("original_run_process_lease_sha256")
        != gate.get("original_run_process_lease_sha256")
        or value.get("causal_recovery_state_sha256")
        != gate.get("causal_recovery_state_sha256")
        or value.get("predecessor_kind") != gate.get("predecessor_kind")
        or value.get("predecessor_journal_sha256")
        != gate.get("predecessor_journal_sha256")
        or value.get("predecessor_generation") != gate.get("predecessor_generation")
        or value.get("recovery_takeover_gate_sha256") != gate.get("gate_sha256")
        or value.get("owner_recovery_takeover_ack_sha256") != ack.get("ack_sha256")
        or value.get("recovery_worker_lease_sha256")
        != secret_gate.get("recovery_worker_lease_sha256")
        or any(
            value.get(name) != secret_gate.get(name)
            for name in (
                "recovery_worker_pid",
                "recovery_worker_process_start_time_ticks",
                "recovery_worker_boot_id_sha256",
                "recovery_worker_boot_time_ns",
                "recovery_worker_uid",
                "recovery_worker_gid",
                "recovery_worker_module_origin",
                "recovery_worker_module_sha256",
                "recovery_worker_process_exe_sha256",
                "recovery_worker_process_cmdline_sha256",
            )
        )
    ):
        raise OwnerLauncherError("invalid_recovery_worker_completion")
    _validate_recovery_cleanup_matrix(
        value,
        expected_release_sha=str(gate.get("release_sha")),
        now_unix=now_unix,
        final=False,
        require_fresh=True,
        code="invalid_recovery_worker_completion",
    )
    return value


def validate_persisted_recovery_worker_completion(
    completion: Mapping[str, Any],
    *,
    expected_release_sha: str,
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate durable cleanup truth that still awaits worker-exit proof."""

    value = _validate_self_digest(
        completion,
        expected_keys=_RECOVERY_WORKER_COMPLETION_KEYS,
        digest_key="completion_sha256",
        code="invalid_recovery_worker_completion",
    )
    if (
        value.get("schema") != RECOVERY_WORKER_COMPLETION_SCHEMA
        or value.get("ok") is not False
        or value.get("state") != "cleanup_complete_awaiting_worker_exit"
    ):
        raise OwnerLauncherError("invalid_recovery_worker_completion")
    _validate_recovery_cleanup_matrix(
        value,
        expected_release_sha=expected_release_sha,
        now_unix=now_unix,
        final=False,
        require_fresh=False,
        code="invalid_recovery_worker_completion",
    )
    return value


def _reconstruct_worker_completion_from_final_receipt(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    reconstructed = {
        name: value[name]
        for name in _RECOVERY_WORKER_COMPLETION_KEYS
        if name
        not in {
            "schema",
            "ok",
            "state",
            "completion_sha256",
            "recovery_worker_exit_proven",
            "safe_to_delete_temporary_admin",
        }
    }
    reconstructed.update({
        "schema": RECOVERY_WORKER_COMPLETION_SCHEMA,
        "ok": False,
        "state": "cleanup_complete_awaiting_worker_exit",
        "recovery_worker_exit_proven": False,
        "safe_to_delete_temporary_admin": False,
    })
    reconstructed["completion_sha256"] = _sha256(_canonical_bytes(reconstructed))
    return reconstructed


def validate_recovery_receipt(
    receipt: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    ack: Mapping[str, Any],
    secret_gate: Mapping[str, Any],
    completion: Mapping[str, Any],
    password: bytearray | None,
    now_unix: int,
) -> Mapping[str, Any]:
    active = () if password is None else (password,)
    _reject_secret_echo(
        receipt,
        active_secrets=active,
        code="invalid_recovery_receipt",
    )
    value = _validate_self_digest(
        receipt,
        expected_keys=_RECOVERY_RECEIPT_KEYS,
        digest_key="receipt_sha256",
        code="invalid_recovery_receipt",
    )
    reconstructed_completion = _reconstruct_worker_completion_from_final_receipt(value)
    if (
        value.get("schema") != RECOVERY_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "recovered"
        or value.get("coordinator_input_sha256") != gate.get("coordinator_input_sha256")
        or value.get("credential_prepare_approval_sha256")
        != gate.get("credential_prepare_approval_sha256")
        or value.get("owner_subject_sha256") != gate.get("owner_subject_sha256")
        or value.get("ephemeral_admin_username") != gate.get("ephemeral_admin_username")
        or value.get("recovery_takeover_gate_sha256") != gate.get("gate_sha256")
        or value.get("owner_recovery_takeover_ack_sha256") != ack.get("ack_sha256")
        or value.get("recovery_worker_lease_sha256")
        != secret_gate.get("recovery_worker_lease_sha256")
        or value.get("recovery_worker_completion_sha256")
        != completion.get("completion_sha256")
        or value.get("recovery_worker_completion_sha256")
        != reconstructed_completion.get("completion_sha256")
        or reconstructed_completion != completion
        or ack.get("discord_token_install_receipt_sha256")
        != gate.get("discord_token_install_receipt_sha256")
        or ack.get("discord_token_state") != gate.get("discord_token_state")
        or ack.get("discord_retirement_receipt_sha256")
        != gate.get("discord_retirement_receipt_sha256")
        or ack.get("token_device") != gate.get("token_device")
        or ack.get("token_inode") != gate.get("token_inode")
    ):
        raise OwnerLauncherError("invalid_recovery_receipt")
    _validate_recovery_cleanup_matrix(
        value,
        expected_release_sha=str(gate.get("release_sha")),
        now_unix=now_unix,
        final=True,
        require_fresh=True,
        code="invalid_recovery_receipt",
    )
    finalized = value.get("finalized_at_unix")
    if (
        value.get("recovery_worker_lock_acquired") is not True
        or type(finalized) is not int
        or not value["cleanup_completed_at_unix"] <= finalized <= now_unix + 30
    ):
        raise OwnerLauncherError("invalid_recovery_receipt")
    return value


def validate_persisted_recovery_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_release_sha: str,
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate durable terminal recovery truth after owner-side interruption."""

    _reject_secret_echo(
        receipt,
        active_secrets=(),
        code="invalid_recovery_receipt",
    )
    value = _validate_self_digest(
        receipt,
        expected_keys=_RECOVERY_RECEIPT_KEYS,
        digest_key="receipt_sha256",
        code="invalid_recovery_receipt",
    )
    reconstructed_completion = _reconstruct_worker_completion_from_final_receipt(value)
    finalized = value.get("finalized_at_unix")
    if (
        value.get("schema") != RECOVERY_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "recovered"
        or value.get("recovery_worker_lock_acquired") is not True
        or value.get("recovery_worker_completion_sha256")
        != reconstructed_completion.get("completion_sha256")
        or type(finalized) is not int
        or finalized < value.get("cleanup_completed_at_unix", -1)
    ):
        raise OwnerLauncherError("invalid_recovery_receipt")
    _validate_recovery_cleanup_matrix(
        value,
        expected_release_sha=expected_release_sha,
        now_unix=now_unix,
        final=True,
        require_fresh=False,
        code="invalid_recovery_receipt",
    )
    return value


def validate_coordinator_secret_gate(
    gate: Mapping[str, Any],
    *,
    owner_gate: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    value = _validate_self_digest(
        gate,
        expected_keys=_COORDINATOR_SECRET_GATE_KEYS,
        digest_key="gate_sha256",
        code="invalid_coordinator_secret_gate",
    )
    expires = value.get("expires_at_unix")
    if (
        value.get("schema") != COORDINATOR_SECRET_GATE_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "awaiting_admin_credential"
        or value.get("coordinator_input_sha256")
        != owner_gate.get("coordinator_input_sha256")
        or value.get("credential_prepare_approval_sha256")
        != owner_gate.get("credential_prepare_approval_sha256")
        or value.get("owner_subject_sha256") != owner_gate.get("owner_subject_sha256")
        or value.get("release_sha") != owner_gate.get("release_sha")
        or value.get("admin_username") != owner_gate.get("admin_username")
        or value.get("database_host") != DATABASE_HOST
        or value.get("database_port") != DATABASE_PORT
        or value.get("database_name") != DATABASE_NAME
        or not isinstance(value.get("tls_server_name"), str)
        or _TLS_SERVER_NAME.fullmatch(value["tls_server_name"]) is None
        or value.get("frame_schema") != ADMIN_FRAME_SCHEMA
        or type(value.get("coordinator_pid")) is not int
        or value["coordinator_pid"] <= 1
        or type(value.get("coordinator_start_time_ticks")) is not int
        or value["coordinator_start_time_ticks"] <= 0
        or type(expires) is not int
        or expires <= now_unix
        or expires > owner_gate.get("expires_at_unix")
    ):
        raise OwnerLauncherError("invalid_coordinator_secret_gate")
    _require_sha256(
        value.get("tls_peer_certificate_sha256"),
        "invalid_coordinator_secret_gate",
    )
    _require_sha256(
        value.get("coordinator_process_lease_sha256"),
        "invalid_coordinator_secret_gate",
    )
    _require_sha256(
        value.get("coordinator_boot_id_sha256"),
        "invalid_coordinator_secret_gate",
    )
    return value


def _validate_invariant_receipt(
    value: Any,
    *,
    release_sha: str,
    evidence_sha256: str,
) -> Mapping[str, Any]:
    receipt = _validate_self_digest(
        value,
        expected_keys=_INVARIANT_RECEIPT_KEYS,
        digest_key="invariant_receipt_sha256",
        code="invalid_coordinator_receipt",
    )
    if (
        receipt.get("schema") != "muncho-full-canary-e2e-verification.v1"
        or receipt.get("ok") is not True
        or receipt.get("release_sha") != release_sha
        or receipt.get("evidence_sha256") != evidence_sha256
        or receipt.get("invariants") != list(_CANARY_INVARIANTS)
        or not isinstance(receipt.get("canary_run_id"), str)
        or not receipt["canary_run_id"]
    ):
        raise OwnerLauncherError("invalid_coordinator_receipt")
    for name in ("fixture_sha256", "full_canary_start_receipt_sha256"):
        _require_sha256(receipt.get(name), "invalid_coordinator_receipt")
    return receipt


def _validate_preclaim_receipt(value: Any) -> Mapping[str, Any]:
    receipt = _validate_self_digest(
        value,
        expected_keys=_PRECLAIM_RECEIPT_KEYS,
        digest_key="receipt_sha256",
        code="invalid_coordinator_receipt",
    )
    result = receipt.get("result")
    database = receipt.get("database_identity")
    if (
        receipt.get("version") != "canonical-canary-preclaim-reconciliation-v1"
        or type(receipt.get("observed_at_unix")) is not int
        or receipt["observed_at_unix"] < 0
        or receipt.get("source_config_path")
        != "/etc/muncho/full-canary/staged/writer.json"
        or not isinstance(database, Mapping)
        or set(database) != {"host", "tls_server_name", "port", "database", "user"}
        or database.get("host") != DATABASE_HOST
        or database.get("port") != DATABASE_PORT
        or database.get("database") != DATABASE_NAME
        or not isinstance(database.get("tls_server_name"), str)
        or _TLS_SERVER_NAME.fullmatch(database["tls_server_name"]) is None
        or not isinstance(database.get("user"), str)
        or not database["user"]
        or not isinstance(result, Mapping)
        or set(result) != _PRECLAIM_RESULT_KEYS
        or result.get("success") is not True
        or result.get("outcome") != "claimed"
        or result.get("reason") != "claim_already_committed_session_retired"
        or result.get("authority_active") is not False
        or result.get("scope_retired") is not False
        or type(result.get("inserted")) is not bool
        or type(result.get("deduped")) is not bool
        or result.get("inserted") == result.get("deduped")
    ):
        raise OwnerLauncherError("invalid_coordinator_receipt")
    for name in (
        "source_config_sha256",
        "database_identity_sha256",
    ):
        _require_sha256(receipt.get(name), "invalid_coordinator_receipt")
    if receipt["database_identity_sha256"] != _sha256(_canonical_bytes(database)):
        raise OwnerLauncherError("invalid_coordinator_receipt")
    for name in (
        "release_sha256",
        "fixture_sha256",
        "session_key_sha256",
        "approval_source_sha256",
        "provisioning_receipt_sha256",
    ):
        _require_sha256(result.get(name), "invalid_coordinator_receipt")
    return receipt


def _validate_lifecycle_receipt(
    value: Any,
    *,
    release_sha: str,
    full_canary_plan_sha256: str,
    evidence_path: str,
    evidence_sha256: str,
    invariant_receipt: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    receipt = _validate_self_digest(
        value,
        expected_keys=_LIFECYCLE_RECEIPT_KEYS,
        digest_key="receipt_sha256",
        code="invalid_coordinator_receipt",
    )
    if (
        receipt.get("schema") != "muncho-full-canary-runtime-receipt.v1"
        or receipt.get("stage") != "verified_stopped"
        or receipt.get("operation") != "verify_and_stop"
        or receipt.get("revision") != release_sha
        or receipt.get("full_canary_plan_sha256") != full_canary_plan_sha256
        or receipt.get("evidence_path") != evidence_path
        or receipt.get("evidence_sha256") != evidence_sha256
        or receipt.get("verifier_result") != invariant_receipt
        or receipt.get("stop_order")
        != [
            "hermes-cloud-gateway.service",
            "muncho-canonical-writer.service",
            "muncho-discord-egress.service",
        ]
        or receipt.get("units_enabled") is not False
        or receipt.get("verified") is not True
        or receipt.get("error_type") is not None
        or receipt.get("error_sha256") is not None
    ):
        raise OwnerLauncherError("invalid_coordinator_receipt")
    receipt_path_raw = receipt.get("receipt_path")
    if not isinstance(receipt_path_raw, str):
        raise OwnerLauncherError("invalid_coordinator_receipt")
    receipt_path = PurePosixPath(receipt_path_raw)
    expected_parent = (
        PurePosixPath("/var/lib/muncho-full-canary/plans")
        / release_sha
        / full_canary_plan_sha256
        / "verified_stopped"
    )
    if (
        receipt_path.parent != expected_parent
        or re.fullmatch(r"[0-9]+-[1-9][0-9]*-[0-9a-f]{32}\.json", receipt_path.name)
        is None
    ):
        raise OwnerLauncherError("invalid_coordinator_receipt")
    for name in (
        "full_canary_start_receipt_sha256",
        "full_canary_start_receipt_internal_sha256",
        "live_report_sha256",
        "stopped_report_sha256",
    ):
        _require_sha256(receipt.get(name), "invalid_coordinator_receipt")
    _validate_receipt_time(
        receipt.get("completed_at_unix"),
        now_unix=now_unix,
        code="invalid_coordinator_receipt",
    )
    _validate_preclaim_receipt(receipt.get("preclaim_reconciliation"))
    return receipt


def validate_owner_approval_request(
    request: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    value = _validate_self_digest(
        request,
        expected_keys=_OWNER_APPROVAL_REQUEST_KEYS,
        digest_key="request_sha256",
        code="invalid_owner_approval_request",
    )
    if (
        value.get("schema") != OWNER_APPROVAL_REQUEST_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "awaiting_final_owner_approval"
        or value.get("release_sha") != gate.get("release_sha")
        or value.get("coordinator_input_sha256") != gate.get("coordinator_input_sha256")
        or value.get("credential_prepare_approval_sha256")
        != gate.get("credential_prepare_approval_sha256")
        or value.get("owner_subject_sha256") != gate.get("owner_subject_sha256")
        or value.get("ephemeral_admin_username") != gate.get("admin_username")
        or value.get("staged_plan_path")
        != "/etc/muncho/full-canary/staged/runtime-plan.json"
        or value.get("approval_path") != FINAL_APPROVAL_PATH
        or value.get("approval_request_path") != APPROVAL_REQUEST_PATH
        or value.get("final_approval_frame_schema") != FINAL_APPROVAL_FRAME_SCHEMA
        or (
            value.get("prior_approval_file_sha256") is not None
            and (
                not isinstance(value.get("prior_approval_file_sha256"), str)
                or _SHA256.fullmatch(value["prior_approval_file_sha256"]) is None
            )
        )
    ):
        raise OwnerLauncherError("invalid_owner_approval_request")
    for name in (
        "full_canary_plan_sha256",
        "staged_plan_file_sha256",
        "hba_receipt_sha256",
        "approval_source_sha256",
    ):
        _require_sha256(value.get(name), "invalid_owner_approval_request")
    hba_expires = value.get("hba_expires_at_unix")
    fixture_expires = value.get("fixture_expires_at_unix")
    credential_expires = value.get("credential_approval_expires_at_unix")
    deadline = value.get("approval_deadline_unix")
    input_cutoff = value.get("owner_input_cutoff_unix")
    transmit_margin = value.get("final_approval_transmit_margin_seconds")
    maximum_wait = value.get("max_wait_seconds")
    requested_at = value.get("requested_at_unix")
    if (
        type(hba_expires) is not int
        or type(fixture_expires) is not int
        or type(credential_expires) is not int
        or type(deadline) is not int
        or type(input_cutoff) is not int
        or type(transmit_margin) is not int
        or type(maximum_wait) is not int
        or type(requested_at) is not int
        or transmit_margin != _FINAL_APPROVAL_DELIVERY_RESERVE_SECONDS
        or not transmit_margin + 1 <= maximum_wait <= FINAL_APPROVAL_MAX_WAIT_SECONDS
        or deadline != requested_at + maximum_wait
        or input_cutoff != deadline - transmit_margin
        or not requested_at <= now_unix < input_cutoff < deadline
        or deadline > hba_expires - _HBA_EXPIRY_SAFETY_MARGIN_SECONDS
        or deadline > fixture_expires
        or type(gate.get("expires_at_unix")) is not int
        or credential_expires != gate["expires_at_unix"]
        or deadline > credential_expires - _FINAL_APPROVAL_DELIVERY_RESERVE_SECONDS
        or deadline - now_unix > maximum_wait
    ):
        raise OwnerLauncherError("invalid_owner_approval_request")
    return value


def validate_final_owner_approval(
    approval: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    if not isinstance(approval, Mapping) or set(approval) != _FINAL_APPROVAL_KEYS:
        raise OwnerLauncherError("invalid_final_owner_approval")
    value = dict(approval)
    if (
        value.get("schema") != "muncho-full-canary-owner-approval.v1"
        or value.get("scope") != "full_canary_runtime_start"
        or value.get("plan_sha256") != request.get("full_canary_plan_sha256")
        or value.get("authority_kind") != "trusted_root_bootstrap_out_of_band_owner"
        or value.get("cryptographic_owner_proof") is not False
        or value.get("owner_subject_sha256") != request.get("owner_subject_sha256")
        or value.get("approval_source_sha256") != request.get("approval_source_sha256")
    ):
        raise OwnerLauncherError("invalid_final_owner_approval")
    _require_sha256(value.get("nonce_sha256"), "invalid_final_owner_approval")
    approved = value.get("approved_at_unix")
    expires = value.get("expires_at_unix")
    if (
        type(approved) is not int
        or type(expires) is not int
        or not request["requested_at_unix"] <= approved <= now_unix <= expires
        or approved > request["owner_input_cutoff_unix"]
        or not 1 <= expires - approved <= 900
        or expires
        > min(
            request["hba_expires_at_unix"],
            request["fixture_expires_at_unix"],
            request["credential_approval_expires_at_unix"],
        )
    ):
        raise OwnerLauncherError("invalid_final_owner_approval")
    return value


def build_final_approval_frame(approval: Mapping[str, Any]) -> bytes:
    payload = _canonical_bytes(approval)
    if not payload or len(payload) > 128 * 1024:
        raise OwnerLauncherError("invalid_final_owner_approval")
    return FINAL_APPROVAL_FRAME_MAGIC + struct.pack(">I", len(payload)) + payload


def validate_final_approval_install_receipt(
    receipt: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    approval: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    value = _validate_self_digest(
        receipt,
        expected_keys=_FINAL_APPROVAL_INSTALL_RECEIPT_KEYS,
        digest_key="receipt_sha256",
        code="invalid_final_approval_install_receipt",
    )
    if (
        value.get("schema") != FINAL_APPROVAL_INSTALL_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("release_sha") != request.get("release_sha")
        or value.get("coordinator_input_sha256")
        != request.get("coordinator_input_sha256")
        or value.get("credential_prepare_approval_sha256")
        != request.get("credential_prepare_approval_sha256")
        or value.get("owner_subject_sha256") != request.get("owner_subject_sha256")
        or value.get("full_canary_plan_sha256")
        != request.get("full_canary_plan_sha256")
        or value.get("approval_request_sha256") != request.get("request_sha256")
        or value.get("owner_approval_sha256") != _sha256(_canonical_bytes(approval))
        or value.get("approval_path") != FINAL_APPROVAL_PATH
    ):
        raise OwnerLauncherError("invalid_final_approval_install_receipt")
    _validate_receipt_time(
        value.get("installed_at_unix"),
        now_unix=now_unix,
        code="invalid_final_approval_install_receipt",
    )
    return value


def validate_final_approval_cancel_receipt(
    receipt: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    value = _validate_self_digest(
        receipt,
        expected_keys=_FINAL_APPROVAL_CANCEL_RECEIPT_KEYS,
        digest_key="receipt_sha256",
        code="invalid_final_approval_cancel_receipt",
    )
    prior = request.get("prior_approval_file_sha256")
    request_state = value.get("approval_request_artifact_state")
    staged_state = value.get("staged_plan_artifact_state")
    owner_state = value.get("owner_approval_artifact_state")
    expected_request_sha = _sha256(_canonical_bytes(request))
    expected_staged_sha = request.get("staged_plan_file_sha256")
    observed_request_sha = value.get("observed_approval_request_file_sha256")
    observed_staged_sha = value.get("observed_staged_plan_file_sha256")
    observed_owner_sha = value.get("observed_approval_file_sha256")
    clean_request_states = {
        "matching_active",
        "matching_expired",
        "retired_absent",
    }
    clean_staged_states = {"matching_present", "retired_absent"}
    clean_owner_states = {"matching_prior"}
    request_conflict_states = {"superseded", "drifted"}
    staged_conflict_states = {"superseded", "drifted"}
    owner_conflict_states = {"drifted"}
    if (
        value.get("schema") != FINAL_APPROVAL_CANCEL_RECEIPT_SCHEMA
        or value.get("ok") is not False
        or value.get("state")
        not in {"cancelled_no_secret", "cancelled_no_secret_state_conflict"}
        or value.get("reason") != "eof_before_mfa1"
        or value.get("release_sha") != request.get("release_sha")
        or value.get("coordinator_input_sha256")
        != request.get("coordinator_input_sha256")
        or value.get("credential_prepare_approval_sha256")
        != request.get("credential_prepare_approval_sha256")
        or value.get("owner_subject_sha256") != request.get("owner_subject_sha256")
        or value.get("full_canary_plan_sha256")
        != request.get("full_canary_plan_sha256")
        or value.get("approval_request_sha256") != request.get("request_sha256")
        or value.get("approval_request_path")
        != request.get("approval_request_path", APPROVAL_REQUEST_PATH)
        or value.get("expected_approval_request_file_sha256") != expected_request_sha
        or value.get("staged_plan_path") != request.get("staged_plan_path")
        or value.get("expected_staged_plan_file_sha256") != expected_staged_sha
        or value.get("approval_path") != FINAL_APPROVAL_PATH
        or value.get("prior_approval_file_sha256") != prior
        or value.get("frame_bytes_received") != 0
        or value.get("owner_approval_mutation_performed_by_this_helper") is not False
        or type(value.get("approval_request_present")) is not bool
        or type(value.get("approval_request_remains_active")) is not bool
        or type(value.get("staged_plan_present")) is not bool
        or value.get("approval_path_matches_prior") is not None
        and type(value.get("approval_path_matches_prior")) is not bool
        or value.get("new_owner_approval_installed") is not None
        and type(value.get("new_owner_approval_installed")) is not bool
        or request_state not in clean_request_states | request_conflict_states
        or staged_state not in clean_staged_states | staged_conflict_states
        or owner_state not in clean_owner_states | owner_conflict_states
    ):
        raise OwnerLauncherError("invalid_final_approval_cancel_receipt")
    for observed in (observed_request_sha, observed_staged_sha, observed_owner_sha):
        if observed is not None:
            _require_sha256(observed, "invalid_final_approval_cancel_receipt")
    cancelled = value.get("cancelled_at_unix")
    if (
        type(cancelled) is not int
        or not request["requested_at_unix"] <= cancelled <= now_unix + 30
    ):
        raise OwnerLauncherError("invalid_final_approval_cancel_receipt")
    request_clean = (
        request_state == "matching_active"
        and value.get("approval_request_present") is True
        and value.get("approval_request_remains_active") is True
        and observed_request_sha == expected_request_sha
        and cancelled <= request["approval_deadline_unix"]
        or request_state == "matching_expired"
        and value.get("approval_request_present") is True
        and value.get("approval_request_remains_active") is False
        and observed_request_sha == expected_request_sha
        and cancelled > request["approval_deadline_unix"]
        or request_state == "retired_absent"
        and value.get("approval_request_present") is False
        and value.get("approval_request_remains_active") is False
        and observed_request_sha is None
    )
    staged_clean = (
        staged_state == "matching_present"
        and value.get("staged_plan_present") is True
        and observed_staged_sha == expected_staged_sha
        or staged_state == "retired_absent"
        and value.get("staged_plan_present") is False
        and observed_staged_sha is None
    )
    owner_clean = (
        owner_state == "matching_prior"
        and observed_owner_sha == prior
        and value.get("approval_path_matches_prior") is True
        and value.get("new_owner_approval_installed") is False
    )
    request_shape_valid = (
        (
            request_state == "retired_absent"
            and value.get("approval_request_present") is False
            and observed_request_sha is None
            or request_state != "retired_absent"
            and value.get("approval_request_present") is True
            and observed_request_sha is not None
        )
        and value.get("approval_request_remains_active")
        is (request_state == "matching_active")
        and not (
            request_state == "superseded"
            and observed_request_sha == expected_request_sha
        )
        and not (
            request_state == "drifted" and observed_request_sha != expected_request_sha
        )
    )
    staged_shape_valid = (
        (
            staged_state == "retired_absent"
            and value.get("staged_plan_present") is False
            and observed_staged_sha is None
            or staged_state != "retired_absent"
            and value.get("staged_plan_present") is True
            and observed_staged_sha is not None
        )
        and not (
            staged_state == "superseded" and observed_staged_sha == expected_staged_sha
        )
        and not (
            staged_state == "drifted" and observed_staged_sha != expected_staged_sha
        )
    )
    owner_shape_valid = owner_clean or (
        owner_state == "drifted"
        and value.get("approval_path_matches_prior") is False
        and value.get("new_owner_approval_installed") is None
        and (prior is not None or observed_owner_sha is not None)
    )
    artifact_pair_clean = (
        request_state in {"matching_active", "matching_expired"}
        and staged_state == "matching_present"
        or request_state == "retired_absent"
        and staged_state == "retired_absent"
    )
    all_clean = request_clean and staged_clean and owner_clean and artifact_pair_clean
    has_conflict = (
        request_state in request_conflict_states
        or staged_state in staged_conflict_states
        or owner_state in owner_conflict_states
        or not artifact_pair_clean
    )
    if (
        not request_shape_valid
        or not staged_shape_valid
        or not owner_shape_valid
        or value.get("state") == "cancelled_no_secret"
        and not all_clean
        or value.get("state") == "cancelled_no_secret_state_conflict"
        and (all_clean or not has_conflict)
    ):
        raise OwnerLauncherError("invalid_final_approval_cancel_receipt")
    return value


def validate_coordinator_receipt(
    receipt: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    password: bytearray,
    now_unix: int,
) -> Mapping[str, Any]:
    _reject_secret_echo(
        receipt,
        active_secrets=(password,),
        code="invalid_coordinator_receipt",
    )
    schema = receipt.get("schema") if isinstance(receipt, Mapping) else None
    if schema == COORDINATOR_FAILURE_SCHEMA:
        value = _validate_self_digest(
            receipt,
            expected_keys=_COORDINATOR_FAILURE_KEYS,
            digest_key="receipt_sha256",
            code="invalid_coordinator_receipt",
        )
        phase = value.get("phase")
        cleanup_status = value.get("cleanup_status")
        plan_sha = value.get("full_canary_plan_sha256")
        recovery = value.get("recovery_material_preserved")
        session_closed = value.get("admin_session_closed")
        password_disabled = value.get("bootstrap_login_password_disabled")
        credential_removed = value.get("bootstrap_credential_removed")
        discord_removed = value.get("discord_token_removed")
        services_enabled = value.get("services_enabled")
        process_lease_retired = value.get("coordinator_process_lease_retired")
        allowed_phases = {
            "process_hardening",
            "coordinator_input",
            "secret_gate",
            "coordinator_secret_gate",
            "admin_credential",
            "credential_read",
            "admin_connect",
            "database_operation",
            "bootstrap_credential",
            "bootstrap_credential_cleanup",
            "plan_and_final_approval",
            "owner_approval_request",
            "owner_approval_request_cleanup",
            "root_publication_cleanup",
            "root_snapshot_cleanup",
            "secret_cleanup",
            "discord_token_retirement",
            "staged_writer_cleanup",
            "signal",
            "live_driver",
            "terminal_cleanup",
            "terminal",
        }
        null_plan_phases = {
            "process_hardening",
            "coordinator_input",
            "secret_gate",
            "coordinator_secret_gate",
            "admin_credential",
            "credential_read",
            "admin_connect",
            "database_operation",
            "bootstrap_credential",
            "bootstrap_credential_cleanup",
            "plan_and_final_approval",
            "root_publication_cleanup",
            "root_snapshot_cleanup",
            "secret_cleanup",
            "discord_token_retirement",
            "staged_writer_cleanup",
            "signal",
        }
        if (
            value.get("ok") is not False
            or value.get("release_sha") != gate.get("release_sha")
            or value.get("coordinator_input_sha256")
            != gate.get("coordinator_input_sha256")
            or value.get("credential_prepare_approval_sha256")
            != gate.get("credential_prepare_approval_sha256")
            or value.get("owner_subject_sha256") != gate.get("owner_subject_sha256")
            or value.get("ephemeral_admin_username") != gate.get("admin_username")
            or phase not in allowed_phases
            or (plan_sha is None and phase not in null_plan_phases)
            or (
                plan_sha is not None
                and (
                    not isinstance(plan_sha, str) or _SHA256.fullmatch(plan_sha) is None
                )
            )
            or not isinstance(value.get("error_code"), str)
            or _STABLE_CODE.fullmatch(value["error_code"]) is None
            or cleanup_status not in {"complete", "cleanup_blocked"}
            or type(recovery) is not bool
            or type(session_closed) is not bool
            or type(password_disabled) is not bool
            or type(credential_removed) is not bool
            or type(discord_removed) is not bool
            or type(process_lease_retired) is not bool
            or credential_removed is True
            and password_disabled is not True
            or cleanup_status == "complete"
            and (
                recovery is not False
                or session_closed is not True
                or password_disabled != credential_removed
                or discord_removed is not True
                or process_lease_retired is not True
            )
            or recovery is True
            and cleanup_status != "cleanup_blocked"
            or discord_removed is False
            and (cleanup_status != "cleanup_blocked" or recovery is not True)
            or process_lease_retired is False
            and (cleanup_status != "cleanup_blocked" or recovery is not True)
            or session_closed is False
            and (cleanup_status != "cleanup_blocked" or recovery is not True)
            or services_enabled not in {False, None}
            or services_enabled is None
            and (cleanup_status != "cleanup_blocked" or recovery is not True)
        ):
            raise OwnerLauncherError("invalid_coordinator_receipt")
        return value

    value = _validate_self_digest(
        receipt,
        expected_keys=_COORDINATOR_RECEIPT_KEYS,
        digest_key="receipt_sha256",
        code="invalid_coordinator_receipt",
    )
    result = value.get("live_driver_result")
    if (
        value.get("schema") != COORDINATOR_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("release_sha") != gate.get("release_sha")
        or value.get("coordinator_input_sha256") != gate.get("coordinator_input_sha256")
        or value.get("credential_prepare_approval_sha256")
        != gate.get("credential_prepare_approval_sha256")
        or value.get("owner_subject_sha256") != gate.get("owner_subject_sha256")
        or value.get("ephemeral_admin_username") != gate.get("admin_username")
        or value.get("temporary_admin_delete_required") is not True
        or value.get("admin_session_closed") is not True
        or value.get("bootstrap_login_password_disabled") is not True
        or value.get("bootstrap_credential_removed") is not True
        or value.get("discord_token_removed") is not True
        or value.get("coordinator_process_lease_retired") is not True
        or value.get("services_enabled") is not False
        or not isinstance(result, Mapping)
        or set(result) != _LIVE_DRIVER_RESULT_KEYS
        or result.get("schema") != "muncho-full-canary-live-driver.v1"
        or result.get("ok") is not True
        or result.get("release_sha") != gate.get("release_sha")
        or result.get("full_canary_plan_sha256") != value.get("full_canary_plan_sha256")
        or result.get("discord_ingress_claimed") is not False
        or not isinstance(result.get("offline_invariant_receipt"), Mapping)
        or not isinstance(result.get("lifecycle_verification_receipt"), Mapping)
    ):
        raise OwnerLauncherError("invalid_coordinator_receipt")
    for name in (
        "full_canary_plan_sha256",
        "owner_approval_sha256",
        "live_driver_receipt_sha256",
    ):
        _require_sha256(value.get(name), "invalid_coordinator_receipt")
    if _sha256(_canonical_bytes(result)) != value["live_driver_receipt_sha256"]:
        raise OwnerLauncherError("invalid_coordinator_receipt")
    evidence_sha256 = _require_sha256(
        result.get("evidence_sha256"), "invalid_coordinator_receipt"
    )
    evidence_path = result.get("evidence_path")
    expected_evidence_path = (
        f"/var/lib/muncho-full-canary/plans/{gate['release_sha']}/"
        f"{value['full_canary_plan_sha256']}/live/evidence.json"
    )
    if evidence_path != expected_evidence_path:
        raise OwnerLauncherError("invalid_coordinator_receipt")
    invariant = _validate_invariant_receipt(
        result["offline_invariant_receipt"],
        release_sha=str(gate["release_sha"]),
        evidence_sha256=evidence_sha256,
    )
    _validate_lifecycle_receipt(
        result["lifecycle_verification_receipt"],
        release_sha=str(gate["release_sha"]),
        full_canary_plan_sha256=str(value["full_canary_plan_sha256"]),
        evidence_path=evidence_path,
        evidence_sha256=evidence_sha256,
        invariant_receipt=invariant,
        now_unix=now_unix,
    )
    _validate_receipt_time(
        value.get("completed_at_unix"),
        now_unix=now_unix,
        code="invalid_coordinator_receipt",
    )
    return value


def validate_terminal_first_failure(
    receipt: Mapping[str, Any],
    *,
    owner_gate: Mapping[str, Any] | None,
    token_lease_expected: bool,
    process_lease_expected: bool,
    expected_release_sha: str | None = None,
    admin_frame_disclosed: bool = False,
    active_secrets: Sequence[bytes | bytearray | memoryview] = (),
) -> Mapping[str, Any]:
    """Validate an exact coordinator failure emitted before an input gate."""

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
    error_code = value.get("error_code")
    cleanup_status = value.get("cleanup_status")
    recovery = value.get("recovery_material_preserved")
    discord_removed = value.get("discord_token_removed")
    services_enabled = value.get("services_enabled")
    process_lease_retired = value.get("coordinator_process_lease_retired")
    session_closed = value.get("admin_session_closed")
    plan_sha = value.get("full_canary_plan_sha256")
    password_disabled = value.get("bootstrap_login_password_disabled")
    credential_removed = value.get("bootstrap_credential_removed")
    binding_names = (
        "release_sha",
        "coordinator_input_sha256",
        "credential_prepare_approval_sha256",
        "owner_subject_sha256",
        "ephemeral_admin_username",
    )
    bindings = tuple(value.get(name) for name in binding_names)
    null_bindings = (None, None, None, None, None)
    expected_bindings = None
    if owner_gate is not None:
        expected_bindings = (
            owner_gate.get("release_sha"),
            owner_gate.get("coordinator_input_sha256"),
            owner_gate.get("credential_prepare_approval_sha256"),
            owner_gate.get("owner_subject_sha256"),
            owner_gate.get("admin_username"),
        )
    elif bindings != null_bindings:
        approval_sha = bindings[2]
        if (
            expected_release_sha is None
            or bindings[0] != expected_release_sha
            or not isinstance(bindings[1], str)
            or _SHA256.fullmatch(bindings[1]) is None
            or not isinstance(approval_sha, str)
            or _SHA256.fullmatch(approval_sha) is None
            or not isinstance(bindings[3], str)
            or _SHA256.fullmatch(bindings[3]) is None
        ):
            raise OwnerLauncherError("invalid_terminal_first_failure")
        expected_bindings = (
            expected_release_sha,
            bindings[1],
            approval_sha,
            bindings[3],
            f"{ADMIN_USERNAME_PREFIX}{approval_sha[:16]}",
        )
    if (
        value.get("schema") != COORDINATOR_FAILURE_SCHEMA
        or value.get("ok") is not False
        or not isinstance(phase, str)
        or _STABLE_CODE.fullmatch(phase) is None
        or not isinstance(error_code, str)
        or _STABLE_CODE.fullmatch(error_code) is None
        or bindings != null_bindings
        and (expected_bindings is None or bindings != expected_bindings)
        or (
            plan_sha is not None
            and (
                not admin_frame_disclosed
                or not isinstance(plan_sha, str)
                or _SHA256.fullmatch(plan_sha) is None
            )
        )
        or cleanup_status not in {"complete", "cleanup_blocked"}
        or type(recovery) is not bool
        or type(session_closed) is not bool
        or type(password_disabled) is not bool
        or type(credential_removed) is not bool
        or admin_frame_disclosed
        and credential_removed is True
        and password_disabled is not True
        or not admin_frame_disclosed
        and password_disabled is not False
        or type(discord_removed) is not bool
        or type(process_lease_retired) is not bool
        or services_enabled not in {False, None}
        or cleanup_status == "complete"
        and recovery is not False
        or cleanup_status == "complete"
        and session_closed is not True
        or cleanup_status == "complete"
        and admin_frame_disclosed
        and password_disabled != credential_removed
        or cleanup_status == "cleanup_blocked"
        and recovery is not True
        or services_enabled is None
        and cleanup_status != "cleanup_blocked"
        or session_closed is False
        and (cleanup_status != "cleanup_blocked" or recovery is not True)
        or token_lease_expected
        and discord_removed is False
        and (cleanup_status != "cleanup_blocked" or recovery is not True)
        or process_lease_expected
        and process_lease_retired is False
        and (cleanup_status != "cleanup_blocked" or recovery is not True)
    ):
        raise OwnerLauncherError("invalid_terminal_first_failure")
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
            not self._approved
            or self._pinned_account is None
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
        self._mutation_ambiguous = False
        self._mutation_ambiguity_observed = False
        self._reconciliation_proven = False
        self._reconciliation_evidence_sha256: str | None = None
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
    ) -> None:
        """Capture the complete operation namespace before a user mutation."""

        operations = self._stable_instance_operations()
        relevant = self._relevant_user_operations(operations)
        if expected_owner_subject_sha256 is not None:
            _require_sha256(
                expected_owner_subject_sha256,
                "invalid_owner_subject_sha256",
            )
        if any(value[1] != "DONE" for value in relevant.values()):
            raise OwnerLauncherError("cloud_sql_user_operations_not_quiescent")
        self._expected_owner_subject_sha256 = expected_owner_subject_sha256
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
            if not _ADMIN_USERNAME.fullmatch(username):
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

    def _user_names(self, *, exact_admin_username: str | None = None) -> frozenset[str]:
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
                if not isinstance(name, str):
                    raise OwnerLauncherError("invalid_cloud_sql_users")
                if name == exact_admin_username and (
                    item.get("kind") != "sql#user"
                    or item.get("project") != PROJECT
                    or item.get("instance") != SQL_INSTANCE
                    or item.get("type") != "BUILT_IN"
                    or name in names
                ):
                    raise OwnerLauncherError("invalid_cloud_sql_users")
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
                return frozenset(names)
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

    def require_absent(self, username: str) -> None:
        if not _ADMIN_USERNAME.fullmatch(username):
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
        if not _ADMIN_USERNAME.fullmatch(username):
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
                body={
                    "instance": SQL_INSTANCE,
                    "name": username,
                    "password": password,
                    "project": PROJECT,
                    "type": "BUILT_IN",
                },
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

        if not _ADMIN_USERNAME.fullmatch(username):
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
        query = urllib.parse.urlencode({"host": "", "name": username})
        self._mutation_ambiguous = True
        self._mutation_ambiguity_observed = True
        try:
            operation = self._client.request_json(
                "PUT",
                f"{self._users_url}?{query}",
                body={
                    "name": username,
                    "password": password,
                    "type": "BUILT_IN",
                },
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

        if not _ADMIN_USERNAME.fullmatch(username):
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
        evidence = {
            "schema": "muncho-cloud-sql-admin-absence-evidence.v1",
            "project": PROJECT,
            "instance": SQL_INSTANCE,
            "username_sha256": _sha256(username.encode("utf-8")),
            "baseline_operation_names": sorted(baseline),
            "known_operation_names": sorted(self._mutation_known_operations),
            "response_known_authority_operation_names": sorted(
                self._mutation_authority_known_operations
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
            "temporary_admin_absent": True,
        }
        self._reconciliation_evidence_sha256 = _sha256(_canonical_bytes(evidence))
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

        if not _ADMIN_USERNAME.fullmatch(username):
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
            not _ADMIN_USERNAME.fullmatch(username)
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
        if (
            type(deadline) is not int
            or type(delivery_deadline) is not int
            or margin != _FINAL_APPROVAL_DELIVERY_RESERVE_SECONDS
            or delivery_deadline != deadline - margin
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

    def close(self) -> None: ...


class CoordinatorTransport(Protocol):
    def preflight_recovery(self, release_sha: str) -> Mapping[str, Any]: ...

    def preflight_owner_launch(self, release_sha: str) -> Mapping[str, Any]: ...

    def open_discord_install(self, release_sha: str) -> RemoteSecretSession: ...

    def open_run(self, release_sha: str) -> RemoteSecretSession: ...

    def open_final_approval_install(self, release_sha: str) -> RemoteSecretSession: ...

    def open_discord_retirement(self, release_sha: str) -> RemoteSecretSession: ...

    def open_recovery(self, release_sha: str) -> RemoteSecretSession: ...

    def open_recovery_finalizer(self, release_sha: str) -> RemoteSecretSession: ...


def _validated_input_gate(
    session: RemoteSecretSession,
    raw: Mapping[str, Any],
    *,
    owner_gate: Mapping[str, Any] | None,
    token_lease_expected: bool,
    process_lease_expected: bool = False,
    admin_frame_disclosed: bool = False,
    active_secrets: Sequence[bytes | bytearray | memoryview] = (),
) -> Mapping[str, Any]:
    if raw.get("schema") != COORDINATOR_FAILURE_SCHEMA:
        return raw
    terminal = validate_terminal_first_failure(
        raw,
        owner_gate=owner_gate,
        token_lease_expected=token_lease_expected,
        process_lease_expected=process_lease_expected,
        admin_frame_disclosed=admin_frame_disclosed,
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
        postflight_guard: Callable[[], None] | None = None,
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
        self._postflight_guard = postflight_guard or (lambda: None)
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
                    value = _decode_json_object(raw, maximum=_MAX_JSON_LINE_BYTES)
                    if raw != _canonical_bytes(value):
                        raise OwnerLauncherError("remote_ndjson_noncanonical")
                    self._messages_read += 1
                    self._last_mapping = dict(value)
                    return dict(value)
                if len(self._buffer) > _MAX_JSON_LINE_BYTES:
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
    ) -> Mapping[str, Any]:
        if self._stdin_closed:
            raise OwnerLauncherError("remote_frame_state_invalid")
        if (
            not isinstance(frame, (bytes, bytearray, memoryview))
            or not frame
            or len(frame) > 128 * 1024 + 8
        ):
            raise OwnerLauncherError("remote_frame_invalid")
        descriptor = self._process.stdin.fileno()
        view = memoryview(frame)
        offset = 0
        first_write = True
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
            while offset < len(frame):
                if write_guard is not None:
                    write_guard()
                    if selector is None or not selector.select(0.05):
                        if self._process.poll() is not None:
                            raise OwnerLauncherError("iap_ssh_secret_write_failed")
                        continue
                    write_guard()
                if first_write and on_first_write is not None:
                    on_first_write()
                try:
                    written = os.write(descriptor, view[offset:])
                except BlockingIOError:
                    continue
                if written <= 0:
                    raise OSError("short write")
                first_write = False
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
        self._frames_written += 1
        return self._read_line(self._post_frame_timeout)

    def exchange(self, frame: bytes | bytearray | memoryview) -> Mapping[str, Any]:
        if self._messages_read != 1 or self._frames_written != 0:
            raise OwnerLauncherError("remote_frame_state_invalid")
        return self._write_frame(frame, close_stdin=False)

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
        if command not in {
            "preflight-owner-launch",
            "install-discord-token",
            "run",
            "install-final-approval",
            "stop-and-retire-discord-token",
            "preflight-recovery",
            "recover",
            "finalize-recovery",
        }:
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
            postflight_guard=postflight,
        )

    def _postflight(self) -> None:
        self._gcloud_executable.trusted_command_prefix()
        self._gcloud_configuration.assert_stable()
        self._known_hosts.absolute_path()
        self._known_hosts.private_key_path()
        self._known_hosts.public_key_line()

    def preflight_owner_launch(self, release_sha: str) -> Mapping[str, Any]:
        session = self._open(
            release_sha,
            "preflight-owner-launch",
            approved=False,
        )
        primary: BaseException | None = None
        try:
            value = session.read_gate()
            if value.get("schema") == COORDINATOR_FAILURE_SCHEMA:
                terminal = validate_terminal_first_failure(
                    value,
                    owner_gate=None,
                    token_lease_expected=False,
                    process_lease_expected=False,
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

    def preflight_recovery(self, release_sha: str) -> Mapping[str, Any]:
        session = self._open(release_sha, "preflight-recovery", approved=False)
        primary: BaseException | None = None
        try:
            value = session.read_gate()
            if value.get("schema") == COORDINATOR_FAILURE_SCHEMA:
                terminal = validate_terminal_first_failure(
                    value,
                    owner_gate=None,
                    token_lease_expected=False,
                    process_lease_expected=False,
                    expected_release_sha=release_sha,
                )
                session.mark_validated(terminal)
                return terminal
            session.complete_read_only()
            return value
        except BaseException as exc:
            primary = exc
            raise
        finally:
            _close_session_preserving_primary(session, primary)

    def open_discord_install(self, release_sha: str) -> RemoteSecretSession:
        return self._open(release_sha, "install-discord-token", approved=True)

    def open_run(self, release_sha: str) -> RemoteSecretSession:
        return self._open(release_sha, "run", approved=True)

    def open_final_approval_install(self, release_sha: str) -> RemoteSecretSession:
        return self._open(
            release_sha,
            "install-final-approval",
            approved=True,
            gate_timeout_seconds=30.0,
            post_frame_timeout_seconds=30.0,
        )

    def open_discord_retirement(self, release_sha: str) -> RemoteSecretSession:
        return self._open(
            release_sha,
            "stop-and-retire-discord-token",
            approved=True,
            post_frame_timeout_seconds=2_400.0,
        )

    def open_recovery(self, release_sha: str) -> RemoteSecretSession:
        return self._open(
            release_sha,
            "recover",
            approved=True,
            post_frame_timeout_seconds=2_400.0,
        )

    def open_recovery_finalizer(self, release_sha: str) -> RemoteSecretSession:
        return self._open(
            release_sha,
            "finalize-recovery",
            approved=True,
            post_frame_timeout_seconds=2_400.0,
        )


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


class IapWriterPreflightTransport(IapStoppedReleaseTransport):
    """Run the sealed, no-service-start writer staging publisher over IAP."""

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
            timeout_seconds=900.0 if command == "apply" else 300.0,
        )
        if (
            not completed.stdout
            or not completed.stdout.endswith(b"\n")
            or b"\n" in completed.stdout[:-1]
        ):
            raise OwnerLauncherError("writer_preflight_output_invalid")
        try:
            return _decode_json_object(
                completed.stdout,
                maximum=_HTTP_RESPONSE_MAX_BYTES,
            )
        except OwnerLauncherError:
            raise OwnerLauncherError("writer_preflight_output_invalid") from None

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


def _validate_admin_credential(username: str, password: bytearray) -> bytes:
    if not _ADMIN_USERNAME.fullmatch(username):
        raise OwnerLauncherError("invalid_admin_username")
    if not isinstance(password, bytearray):
        raise OwnerLauncherError("invalid_admin_password")
    username_bytes = username.encode("ascii")
    if (
        not _ADMIN_PASSWORD_MIN_UTF8 <= len(password) <= _ADMIN_PASSWORD_MAX_UTF8
        or b"\x00" in password
        or password != password.strip()
        or any(value < 0x20 or value == 0x7F for value in password)
    ):
        raise OwnerLauncherError("invalid_admin_password")
    try:
        password.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OwnerLauncherError("invalid_admin_password") from exc
    return username_bytes


def build_admin_frame(username: str, password: bytearray) -> bytearray:
    """Build the exact MCA2 frame consumed only through remote stdin."""

    username_bytes = _validate_admin_credential(username, password)
    frame = bytearray(ADMIN_FRAME_MAGIC)
    frame.extend(struct.pack(">H", len(username_bytes)))
    frame.extend(struct.pack(">I", len(password)))
    frame.extend(username_bytes)
    frame.extend(password)
    return frame


def build_recovery_admin_frame(
    secret_gate: Mapping[str, Any],
    username: str,
    password: bytearray,
) -> bytearray:
    """Build MRC2 bound to the one validated stage-two worker gate."""

    username_bytes = _validate_admin_credential(username, password)
    if (
        not isinstance(secret_gate, Mapping)
        or secret_gate.get("schema") != RECOVERY_SECRET_GATE_SCHEMA
        or secret_gate.get("admin_frame_schema") != RECOVERY_ADMIN_FRAME_SCHEMA
        or secret_gate.get("ephemeral_admin_username") != username
    ):
        raise OwnerLauncherError("invalid_recovery_secret_gate")
    gate_sha256 = _require_sha256(
        secret_gate.get("gate_sha256"),
        "invalid_recovery_secret_gate",
    )
    gate_nonce_sha256 = _require_sha256(
        secret_gate.get("gate_nonce_sha256"),
        "invalid_recovery_secret_gate",
    )
    try:
        gate_digest = bytes.fromhex(gate_sha256)
        nonce_digest = bytes.fromhex(gate_nonce_sha256)
    except ValueError:
        raise OwnerLauncherError("invalid_recovery_secret_gate") from None
    frame = bytearray(
        struct.pack(
            "!4s32s32sHI",
            RECOVERY_ADMIN_FRAME_MAGIC,
            gate_digest,
            nonce_digest,
            len(username_bytes),
            len(password),
        )
    )
    frame.extend(username_bytes)
    frame.extend(password)
    return frame


def build_discord_frame(token: bytearray) -> bytearray:
    if (
        not isinstance(token, bytearray)
        or not token
        or len(token) > _DISCORD_TOKEN_MAX_BYTES
    ):
        raise OwnerLauncherError("invalid_discord_token")
    if b"\x00" in token or any(value < 0x20 or value == 0x7F for value in token):
        raise OwnerLauncherError("invalid_discord_token")
    frame = bytearray(DISCORD_FRAME_MAGIC)
    frame.extend(struct.pack(">I", len(token)))
    frame.extend(token)
    return frame


def _wipe(value: bytearray | None) -> None:
    if value is not None:
        value[:] = b"\x00" * len(value)


def _new_admin_password() -> bytearray:
    # URL-safe base64 has no controls or surrounding whitespace and is accepted
    # by Cloud SQL.  Keep the mutable representation so the launcher can wipe it.
    return bytearray(
        base64.urlsafe_b64encode(secrets.token_bytes(_ADMIN_PASSWORD_BYTES))
    )


def _owner_binding_view(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "release_sha": value.get("release_sha"),
        "coordinator_input_sha256": value.get("coordinator_input_sha256"),
        "credential_prepare_approval_sha256": value.get(
            "credential_prepare_approval_sha256"
        ),
        "owner_subject_sha256": value.get("owner_subject_sha256"),
        "admin_username": value.get(
            "admin_username", value.get("ephemeral_admin_username")
        ),
    }


def _adopt_persisted_recovery_completion(
    state: _RecoveryAttemptState,
    completion: Mapping[str, Any],
) -> None:
    state.gate = {
        "release_sha": completion["release_sha"],
        "coordinator_input_sha256": completion["coordinator_input_sha256"],
        "credential_prepare_approval_sha256": completion[
            "credential_prepare_approval_sha256"
        ],
        "owner_subject_sha256": completion["owner_subject_sha256"],
        "ephemeral_admin_username": completion["ephemeral_admin_username"],
        "original_run_process_lease_sha256": completion[
            "original_run_process_lease_sha256"
        ],
        "causal_recovery_state_sha256": completion["causal_recovery_state_sha256"],
        "gate_sha256": completion["recovery_takeover_gate_sha256"],
    }
    state.worker_completion = completion
    state.username = str(completion["ephemeral_admin_username"])
    state.admin_mutation_attempted = True
    state.admin_mutation_confirmed = True
    state.admin_delete_pending = True
    state.frame_write_attempted = True
    state.frame_disclosed_or_ambiguous = True
    state.admin_session_closed = True
    state.recovery_required = True


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
            token_lease_expected=True,
            process_lease_expected=False,
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
            token_lease_expected=True,
            process_lease_expected=False,
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


def _finalize_recovery_worker(
    *,
    release_sha: str,
    transport: CoordinatorTransport,
    owner_identity: ApprovedOwnerIdentity,
    now: Callable[[], int],
    state: _RecoveryAttemptState,
) -> Mapping[str, Any]:
    """Finish one durable worker completion without sending any secret."""

    if state.gate is None:
        raise OwnerLauncherError("recovery_finalization_context_missing")
    owner_gate = _owner_binding_view(state.gate)
    session = transport.open_recovery_finalizer(release_sha)
    closed = False
    primary: BaseException | None = None
    try:
        raw = _validated_input_gate(
            session,
            session.read_gate(),
            owner_gate=owner_gate,
            token_lease_expected=False,
            process_lease_expected=True,
            admin_frame_disclosed=state.frame_disclosed_or_ambiguous,
            active_secrets=(
                (state.password,)
                if state.password is not None and state.password_wiped is not True
                else ()
            ),
        )
        if raw.get("schema") == RECOVERY_FINALIZE_PENDING_RECEIPT_SCHEMA:
            pending = validate_recovery_finalize_pending_receipt(
                raw,
                completion=state.worker_completion,
                expected_release_sha=release_sha,
                owner_gate=owner_gate,
                now_unix=now(),
            )
            state.finalize_pending_receipt = pending
            state.recovery_required = True
            session.mark_validated(pending)
            state.remote_termination_proven = bool(
                state.remote_termination_proven or session.termination_proven
            )
            state.terminal_receipt_validated = bool(
                state.terminal_receipt_validated or session.terminal_receipt_validated
            )
            session.close()
            closed = True
            raise OwnerLauncherError("recovery_finalization_pending_no_secret")
        if (
            state.takeover_ack is not None
            and state.secret_gate is not None
            and state.worker_completion is not None
            and state.password is not None
            and state.password_wiped is not True
        ):
            receipt = validate_recovery_receipt(
                raw,
                gate=state.gate,
                ack=state.takeover_ack,
                secret_gate=state.secret_gate,
                completion=state.worker_completion,
                password=state.password,
                now_unix=now(),
            )
        else:
            receipt = validate_persisted_recovery_receipt(
                raw,
                expected_release_sha=release_sha,
                now_unix=now(),
            )
            if state.worker_completion is not None:
                reconstructed_completion = (
                    _reconstruct_worker_completion_from_final_receipt(receipt)
                )
                if (
                    receipt.get("recovery_worker_completion_sha256")
                    != state.worker_completion.get("completion_sha256")
                    or reconstructed_completion != state.worker_completion
                ):
                    raise OwnerLauncherError(
                        "recovery_final_receipt_completion_drifted"
                    )
            if any(
                receipt.get(receipt_name) != state.gate.get(gate_name)
                for receipt_name, gate_name in (
                    ("coordinator_input_sha256", "coordinator_input_sha256"),
                    (
                        "credential_prepare_approval_sha256",
                        "credential_prepare_approval_sha256",
                    ),
                    ("owner_subject_sha256", "owner_subject_sha256"),
                    ("ephemeral_admin_username", "ephemeral_admin_username"),
                    (
                        "original_run_process_lease_sha256",
                        "original_run_process_lease_sha256",
                    ),
                    (
                        "causal_recovery_state_sha256",
                        "causal_recovery_state_sha256",
                    ),
                )
            ):
                raise OwnerLauncherError("recovery_final_receipt_drifted")
        state.final_receipt = receipt
        state.admin_session_closed = True
        state.recovery_required = False
        session.mark_validated(receipt)
        owner_identity.require_stable()
        state.remote_termination_proven = bool(
            state.remote_termination_proven or session.termination_proven
        )
        state.terminal_receipt_validated = bool(
            state.terminal_receipt_validated or session.terminal_receipt_validated
        )
        session.close()
        closed = True
        return receipt
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if not closed:
            _close_session_preserving_primary(session, primary)
        if primary is not None:
            state.cleanup_failure_codes.update(_attached_cleanup_failure_codes(primary))


def _recover_stale_process(
    *,
    release_sha: str,
    preflight_gate: Mapping[str, Any],
    transport: CoordinatorTransport,
    sql_admin: CloudSqlTemporaryAdmin,
    owner_identity: ApprovedOwnerIdentity,
    now: Callable[[], int],
    password_factory: Callable[[], bytearray],
    state: _RecoveryAttemptState,
) -> tuple[Mapping[str, Any], bytearray]:
    owner_gate = _owner_binding_view(preflight_gate)
    gate = validate_recovery_gate(
        preflight_gate,
        expected_release_sha=release_sha,
        owner_gate=owner_gate,
        now_unix=now(),
    )
    state.gate = gate
    state.username = str(gate["ephemeral_admin_username"])
    state.recovery_required = True
    session: RemoteSecretSession | None = None
    closed = False
    primary: BaseException | None = None
    try:
        session = transport.open_recovery(release_sha)
        raw_gate = _validated_input_gate(
            session,
            session.read_gate(),
            owner_gate=owner_gate,
            token_lease_expected=True,
            process_lease_expected=True,
        )
        gate = validate_recovery_gate(
            raw_gate,
            expected_release_sha=release_sha,
            owner_gate=owner_gate,
            now_unix=now(),
        )
        state.gate = gate
        state.username = str(gate["ephemeral_admin_username"])
        stable_names = (
            "release_sha",
            "coordinator_input_sha256",
            "credential_prepare_approval_sha256",
            "owner_subject_sha256",
            "ephemeral_admin_username",
            "predecessor_kind",
            "predecessor_schema",
            "predecessor_journal_sha256",
            "predecessor_generation",
            "original_run_process_lease_sha256",
            "causal_recovery_state_sha256",
            "target_pid",
            "target_process_start_time_ticks",
            "target_boot_id_sha256",
            "target_boot_time_ns",
            "target_uid",
            "target_gid",
            "target_module_origin",
            "target_module_sha256",
            "target_process_exe_sha256",
            "target_process_cmdline_sha256",
            "target_process_identity_state",
            "discord_token_state",
            "discord_token_install_receipt_sha256",
            "discord_retirement_receipt_sha256",
            "token_device",
            "token_inode",
            "db_secret_accepted",
            "frame_schema",
        )
        if any(gate.get(name) != preflight_gate.get(name) for name in stable_names):
            raise OwnerLauncherError("recovery_gate_drifted")
        owner_identity.require_stable()
        ack = build_recovery_ack(gate, now_unix=now())
        state.takeover_ack = ack
        ack_frame = build_recovery_ack_frame(gate, ack)
        try:
            raw_secret_gate = session.exchange(ack_frame)
        finally:
            _wipe(ack_frame)
        if raw_secret_gate.get("schema") == RECOVERY_CONCURRENT_LOSER_RECEIPT_SCHEMA:
            concurrent = validate_recovery_concurrent_loser_receipt(
                raw_secret_gate,
                gate=gate,
                ack=ack,
                now_unix=now(),
            )
            state.concurrent_loser_receipt = concurrent
            state.recovery_required = True
            session.mark_validated(concurrent)
            state.remote_termination_proven = session.termination_proven
            state.terminal_receipt_validated = session.terminal_receipt_validated
            session.close()
            closed = True
            raise OwnerLauncherError("recovery_worker_claim_lost_no_secret")
        raw_secret_gate = _validated_input_gate(
            session,
            raw_secret_gate,
            owner_gate=owner_gate,
            token_lease_expected=True,
            process_lease_expected=True,
        )
        secret_gate = validate_recovery_secret_gate(
            raw_secret_gate,
            takeover_gate=gate,
            takeover_ack=ack,
            now_unix=now(),
        )
        state.secret_gate = secret_gate
        owner_identity.require_stable()
        password = password_factory()
        state.password = password
        if not isinstance(password, bytearray):
            raise OwnerLauncherError("invalid_admin_password")
        _validate_admin_credential(state.username, password)
        sql_admin.begin_mutation_observation(
            expected_owner_subject_sha256=str(gate["owner_subject_sha256"]),
        )
        state.admin_delete_pending = True
        state.admin_mutation_attempted = True
        state.admin_mutation_ambiguous = True
        try:
            sql_admin.create_or_rotate_recovery(
                state.username,
                password.decode("utf-8"),
            )
        except CloudSqlCreateNotCommitted:
            state.admin_mutation_ambiguous = False
            state.admin_mutation_explicitly_not_committed = True
            state.admin_delete_pending = False
            state.recovery_required = True
            raise
        except BaseException:
            state.admin_mutation_ambiguous = bool(
                state.admin_mutation_ambiguous
                or sql_admin.mutation_reconciliation_required()
            )
            state.admin_mutation_ambiguity_observed = True
            raise
        state.admin_mutation_confirmed = True
        state.admin_mutation_ambiguous = bool(
            sql_admin.mutation_reconciliation_required()
        )
        state.admin_mutation_ambiguity_observed = bool(
            _validated_sql_reconciliation_evidence(sql_admin)[
                "mutation_ambiguity_observed"
            ]
        )
        owner_identity.require_stable()
        if state.admin_mutation_ambiguous:
            raise OwnerLauncherError("cloud_sql_mutation_ambiguous")
        if (
            now() + _SECRET_FRAME_TRANSMIT_MARGIN_SECONDS
            >= secret_gate["expires_at_unix"]
        ):
            raise OwnerLauncherError("recovery_secret_gate_expired")
        frame = build_recovery_admin_frame(secret_gate, state.username, password)

        def guard_recovery_admin_write() -> None:
            owner_identity.require_stable()
            if (
                state.admin_mutation_ambiguous
                or sql_admin.mutation_reconciliation_required()
            ):
                raise OwnerLauncherError("cloud_sql_mutation_ambiguous")
            sql_admin.require_current_authority(state.username)
            owner_identity.require_stable()
            if (
                now() + _SECRET_FRAME_TRANSMIT_MARGIN_SECONDS
                >= secret_gate["expires_at_unix"]
            ):
                raise OwnerLauncherError("recovery_secret_gate_expired")

        def mark_recovery_admin_write() -> None:
            state.frame_write_attempted = True
            state.frame_disclosed_or_ambiguous = True

        try:
            try:
                raw_completion = session.finish_before(
                    frame,
                    write_guard=guard_recovery_admin_write,
                    on_first_write=mark_recovery_admin_write,
                )
            finally:
                _wipe(frame)
        except BaseException:
            state.admin_mutation_ambiguous = bool(
                state.admin_mutation_ambiguous
                or sql_admin.mutation_reconciliation_required()
            )
            state.admin_mutation_ambiguity_observed = bool(
                state.admin_mutation_ambiguity_observed
                or sql_admin.mutation_reconciliation_required()
            )
            raise
        raw_completion = _validated_input_gate(
            session,
            raw_completion,
            owner_gate=owner_gate,
            token_lease_expected=True,
            process_lease_expected=True,
            admin_frame_disclosed=True,
            active_secrets=(password,),
        )
        completion = validate_recovery_worker_completion(
            raw_completion,
            gate=gate,
            ack=ack,
            secret_gate=secret_gate,
            password=password,
            now_unix=now(),
        )
        state.worker_completion = completion
        state.admin_session_closed = True
        state.recovery_required = True
        session.mark_validated(completion)
        state.remote_termination_proven = session.termination_proven
        state.terminal_receipt_validated = session.terminal_receipt_validated
        owner_identity.require_stable()
        session.close()
        closed = True
        receipt = _finalize_recovery_worker(
            release_sha=release_sha,
            transport=transport,
            owner_identity=owner_identity,
            now=now,
            state=state,
        )
        return receipt, password
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if session is not None and not closed:
            _close_session_preserving_primary(session, primary)
        if primary is not None:
            state.cleanup_failure_codes.update(_attached_cleanup_failure_codes(primary))
        if session is not None:
            state.remote_termination_proven = bool(
                state.remote_termination_proven or session.termination_proven
            )
            state.terminal_receipt_validated = bool(
                state.terminal_receipt_validated or session.terminal_receipt_validated
            )


def _resume_recovery_attempt(
    *,
    release_sha: str,
    transport: CoordinatorTransport,
    sql_admin: CloudSqlTemporaryAdmin,
    owner_identity: ApprovedOwnerIdentity,
    now: Callable[[], int],
    password_factory: Callable[[], bytearray],
    state: _RecoveryAttemptState,
) -> Mapping[str, Any]:
    """Use only durable recovery truth to reach the final v2 receipt."""

    if state.final_receipt is not None:
        return state.final_receipt
    if state.gate is None:
        raise OwnerLauncherError("recovery_resume_context_missing")
    if state.worker_completion is not None:
        return _finalize_recovery_worker(
            release_sha=release_sha,
            transport=transport,
            owner_identity=owner_identity,
            now=now,
            state=state,
        )
    probe = transport.preflight_recovery(release_sha)
    if probe.get("schema") == RECOVERY_RECEIPT_SCHEMA:
        receipt = validate_persisted_recovery_receipt(
            probe,
            expected_release_sha=release_sha,
            now_unix=now(),
        )
        if any(
            receipt.get(receipt_name) != state.gate.get(gate_name)
            for receipt_name, gate_name in (
                ("coordinator_input_sha256", "coordinator_input_sha256"),
                (
                    "credential_prepare_approval_sha256",
                    "credential_prepare_approval_sha256",
                ),
                ("owner_subject_sha256", "owner_subject_sha256"),
                ("ephemeral_admin_username", "ephemeral_admin_username"),
                (
                    "original_run_process_lease_sha256",
                    "original_run_process_lease_sha256",
                ),
                (
                    "causal_recovery_state_sha256",
                    "causal_recovery_state_sha256",
                ),
            )
        ):
            raise OwnerLauncherError("persisted_recovery_receipt_drifted")
        state.final_receipt = receipt
        state.admin_session_closed = True
        state.remote_termination_proven = True
        state.terminal_receipt_validated = True
        state.recovery_required = False
        return receipt
    if probe.get("schema") == RECOVERY_WORKER_COMPLETION_SCHEMA:
        completion = validate_persisted_recovery_worker_completion(
            probe,
            expected_release_sha=release_sha,
            now_unix=now(),
        )
        if any(
            completion.get(completion_name) != state.gate.get(gate_name)
            for completion_name, gate_name in (
                ("coordinator_input_sha256", "coordinator_input_sha256"),
                (
                    "credential_prepare_approval_sha256",
                    "credential_prepare_approval_sha256",
                ),
                ("owner_subject_sha256", "owner_subject_sha256"),
                ("ephemeral_admin_username", "ephemeral_admin_username"),
                (
                    "original_run_process_lease_sha256",
                    "original_run_process_lease_sha256",
                ),
                (
                    "causal_recovery_state_sha256",
                    "causal_recovery_state_sha256",
                ),
            )
        ):
            raise OwnerLauncherError("persisted_recovery_completion_drifted")
        _adopt_persisted_recovery_completion(state, completion)
        owner_identity.require_stable()
        return _finalize_recovery_worker(
            release_sha=release_sha,
            transport=transport,
            owner_identity=owner_identity,
            now=now,
            state=state,
        )
    if probe.get("schema") == COORDINATOR_FAILURE_SCHEMA:
        terminal = validate_terminal_first_failure(
            probe,
            owner_gate=_owner_binding_view(state.gate),
            token_lease_expected=True,
            process_lease_expected=True,
            expected_release_sha=release_sha,
            admin_frame_disclosed=state.frame_disclosed_or_ambiguous,
        )
        if terminal.get("error_code") == "recovery_completion_requires_finalization":
            return _finalize_recovery_worker(
                release_sha=release_sha,
                transport=transport,
                owner_identity=owner_identity,
                now=now,
                state=state,
            )
        raise RemoteCommandFailed(terminal)
    takeover_gate = validate_recovery_gate(
        probe,
        expected_release_sha=release_sha,
        owner_gate=_owner_binding_view(state.gate),
        now_unix=now(),
    )
    owner_identity.require_stable()
    later = _RecoveryAttemptState()
    try:
        receipt, _ = _recover_stale_process(
            release_sha=release_sha,
            preflight_gate=takeover_gate,
            transport=transport,
            sql_admin=sql_admin,
            owner_identity=owner_identity,
            now=now,
            password_factory=password_factory,
            state=later,
        )
        return receipt
    finally:
        later.wipe_password()
        state.absorb(later)


def _receipt(
    *,
    ok: bool,
    state: str,
    release_sha: str,
    gate: Mapping[str, Any] | None,
    discord_install_receipt: Mapping[str, Any] | None,
    final_approval_install_receipt: Mapping[str, Any] | None,
    final_approval_cancel_receipt: Mapping[str, Any] | None,
    coordinator_receipt: Mapping[str, Any] | None,
    remote_failure_receipt: Mapping[str, Any] | None,
    recovery_preflight_receipt: Mapping[str, Any] | None,
    recovery_receipt: Mapping[str, Any] | None,
    discord_retirement_receipt: Mapping[str, Any] | None,
    discord_token_retired: bool | None,
    failure_code: str | None,
    primary_failure_code: str | None,
    cleanup_failure_codes: Sequence[str],
    recovery_attempt: _RecoveryAttemptState | None,
    admin_mutation_ambiguous: bool,
    admin_mutation_ambiguity_observed: bool,
    admin_reconciliation_performed: bool,
    admin_reconciliation_evidence_sha256: str | None,
    admin_reconciliation_quiet_window_seconds: float | None,
    admin_response_known_candidate_observed: bool | None,
    admin_post_baseline_authority_operation_count: int | None,
    temporary_admin_absent: bool | None,
    remote_coordinator_terminated: bool | None,
    terminal_receipt_validated: bool | None,
    admin_frame_disclosed: bool,
    discord_frame_disclosed: bool,
    final_approval_frame_disclosed: bool,
    coordinator_admin_session_closed: bool | None,
) -> Mapping[str, Any]:
    normalized_cleanup_codes = sorted(set(cleanup_failure_codes))
    if (
        any(_STABLE_CODE.fullmatch(item) is None for item in normalized_cleanup_codes)
        or primary_failure_code is not None
        and _STABLE_CODE.fullmatch(primary_failure_code) is None
        or admin_reconciliation_performed
        and (
            admin_reconciliation_evidence_sha256 is None
            or _SHA256.fullmatch(admin_reconciliation_evidence_sha256) is None
            or not isinstance(admin_reconciliation_quiet_window_seconds, (int, float))
            or isinstance(admin_reconciliation_quiet_window_seconds, bool)
            or admin_reconciliation_quiet_window_seconds <= 0
            or type(admin_response_known_candidate_observed) is not bool
            or type(admin_post_baseline_authority_operation_count) is not int
            or admin_post_baseline_authority_operation_count < 0
            or admin_response_known_candidate_observed
            and admin_post_baseline_authority_operation_count < 1
        )
        or not admin_reconciliation_performed
        and (
            admin_reconciliation_evidence_sha256 is not None
            or admin_reconciliation_quiet_window_seconds is not None
            or admin_response_known_candidate_observed is not None
            or admin_post_baseline_authority_operation_count is not None
        )
        or recovery_attempt is not None
        and recovery_attempt.admin_reconciliation_performed
        and (
            recovery_attempt.admin_reconciliation_evidence_sha256 is None
            or _SHA256.fullmatch(recovery_attempt.admin_reconciliation_evidence_sha256)
            is None
            or not isinstance(
                recovery_attempt.admin_reconciliation_quiet_window_seconds,
                (int, float),
            )
            or isinstance(
                recovery_attempt.admin_reconciliation_quiet_window_seconds,
                bool,
            )
            or recovery_attempt.admin_reconciliation_quiet_window_seconds <= 0
            or type(recovery_attempt.admin_response_known_candidate_observed)
            is not bool
            or type(recovery_attempt.admin_post_baseline_authority_operation_count)
            is not int
            or recovery_attempt.admin_post_baseline_authority_operation_count < 0
            or recovery_attempt.admin_response_known_candidate_observed
            and recovery_attempt.admin_post_baseline_authority_operation_count < 1
        )
        or recovery_attempt is not None
        and not recovery_attempt.admin_reconciliation_performed
        and (
            recovery_attempt.admin_reconciliation_evidence_sha256 is not None
            or recovery_attempt.admin_reconciliation_quiet_window_seconds is not None
            or recovery_attempt.admin_response_known_candidate_observed is not None
            or recovery_attempt.admin_post_baseline_authority_operation_count
            is not None
        )
        or temporary_admin_absent is True
        and not admin_reconciliation_performed
        or admin_mutation_ambiguity_observed
        and temporary_admin_absent is None
    ):
        raise OwnerLauncherError("owner_receipt_failure_truth_invalid")
    value: dict[str, Any] = {
        "schema": OWNER_RECEIPT_SCHEMA,
        "ok": ok,
        "state": state,
        "release_sha": release_sha,
        "coordinator_input_sha256": None
        if gate is None
        else gate.get("coordinator_input_sha256"),
        "credential_prepare_approval_sha256": None
        if gate is None
        else gate.get("credential_prepare_approval_sha256"),
        "owner_gate_sha256": None if gate is None else gate.get("gate_sha256"),
        "admin_username": None if gate is None else gate.get("admin_username"),
        "discord_install_receipt_sha256": None
        if discord_install_receipt is None
        else discord_install_receipt.get("receipt_sha256"),
        "final_approval_install_receipt_sha256": None
        if final_approval_install_receipt is None
        else final_approval_install_receipt.get("receipt_sha256"),
        "final_approval_cancel_receipt_sha256": None
        if final_approval_cancel_receipt is None
        else final_approval_cancel_receipt.get("receipt_sha256"),
        "coordinator_receipt_sha256": None
        if coordinator_receipt is None
        else coordinator_receipt.get("receipt_sha256"),
        "remote_failure_receipt_sha256": None
        if remote_failure_receipt is None
        else remote_failure_receipt.get("receipt_sha256"),
        "recovery_preflight_receipt_sha256": None
        if recovery_preflight_receipt is None
        else recovery_preflight_receipt.get("receipt_sha256"),
        "recovery_receipt_sha256": None
        if recovery_receipt is None
        else recovery_receipt.get("receipt_sha256"),
        "discord_retirement_receipt_sha256": None
        if discord_retirement_receipt is None
        else discord_retirement_receipt.get("receipt_sha256"),
        "discord_token_retired": discord_token_retired,
        "temporary_admin_absent": temporary_admin_absent,
        "admin_mutation_ambiguous": admin_mutation_ambiguous,
        "admin_mutation_ambiguity_observed": (admin_mutation_ambiguity_observed),
        "admin_reconciliation_performed": admin_reconciliation_performed,
        "admin_reconciliation_evidence_sha256": (admin_reconciliation_evidence_sha256),
        "admin_reconciliation_quiet_window_seconds": (
            admin_reconciliation_quiet_window_seconds
        ),
        "admin_response_known_candidate_observed": (
            admin_response_known_candidate_observed
        ),
        "admin_post_baseline_authority_operation_count": (
            admin_post_baseline_authority_operation_count
        ),
        "temporary_admin_preserved_for_recovery": temporary_admin_absent is False,
        "admin_cleanup_status": (
            "not_required"
            if temporary_admin_absent is None
            else (
                "absent_proven"
                if temporary_admin_absent is True
                else (
                    "preserved_remote_unconfirmed"
                    if remote_coordinator_terminated is False
                    else "delete_unconfirmed"
                )
            )
        ),
        "remote_coordinator_terminated": remote_coordinator_terminated,
        "terminal_receipt_validated": terminal_receipt_validated,
        "admin_frame_disclosed": admin_frame_disclosed,
        "discord_frame_disclosed": discord_frame_disclosed,
        "final_approval_frame_disclosed": final_approval_frame_disclosed,
        "coordinator_admin_session_closed": coordinator_admin_session_closed,
        "remote_cleanup_status": (
            "not_started"
            if remote_coordinator_terminated is None
            else (
                "terminated_proven"
                if remote_coordinator_terminated is True
                else "termination_unconfirmed"
            )
        ),
        "failure_code": failure_code,
        "primary_failure_code": primary_failure_code,
        "cleanup_failure_codes": normalized_cleanup_codes,
        "recovery_gate_sha256": (
            None
            if recovery_attempt is None or recovery_attempt.gate is None
            else recovery_attempt.gate.get("gate_sha256")
        ),
        "recovery_secret_gate_sha256": (
            None
            if recovery_attempt is None or recovery_attempt.secret_gate is None
            else recovery_attempt.secret_gate.get("gate_sha256")
        ),
        "recovery_takeover_ack_sha256": (
            None
            if recovery_attempt is None or recovery_attempt.takeover_ack is None
            else recovery_attempt.takeover_ack.get("ack_sha256")
        ),
        "recovery_worker_completion_sha256": (
            None
            if recovery_attempt is None or recovery_attempt.worker_completion is None
            else recovery_attempt.worker_completion.get("completion_sha256")
        ),
        "recovery_final_receipt_sha256": (
            None
            if recovery_attempt is None or recovery_attempt.final_receipt is None
            else recovery_attempt.final_receipt.get("receipt_sha256")
        ),
        "recovery_concurrent_loser_receipt_sha256": (
            None
            if recovery_attempt is None
            or recovery_attempt.concurrent_loser_receipt is None
            else recovery_attempt.concurrent_loser_receipt.get("receipt_sha256")
        ),
        "recovery_finalize_pending_receipt_sha256": (
            None
            if recovery_attempt is None
            or recovery_attempt.finalize_pending_receipt is None
            else recovery_attempt.finalize_pending_receipt.get("receipt_sha256")
        ),
        "recovery_admin_username": (
            None if recovery_attempt is None else recovery_attempt.username
        ),
        "recovery_admin_mutation_attempted": bool(
            recovery_attempt is not None and recovery_attempt.admin_mutation_attempted
        ),
        "recovery_admin_mutation_confirmed": bool(
            recovery_attempt is not None and recovery_attempt.admin_mutation_confirmed
        ),
        "recovery_admin_mutation_ambiguous": bool(
            recovery_attempt is not None and recovery_attempt.admin_mutation_ambiguous
        ),
        "recovery_admin_mutation_explicitly_not_committed": bool(
            recovery_attempt is not None
            and recovery_attempt.admin_mutation_explicitly_not_committed
        ),
        "recovery_admin_mutation_ambiguity_observed": bool(
            recovery_attempt is not None
            and recovery_attempt.admin_mutation_ambiguity_observed
        ),
        "recovery_admin_reconciliation_performed": bool(
            recovery_attempt is not None
            and recovery_attempt.admin_reconciliation_performed
        ),
        "recovery_admin_reconciliation_evidence_sha256": (
            None
            if recovery_attempt is None
            else recovery_attempt.admin_reconciliation_evidence_sha256
        ),
        "recovery_admin_reconciliation_quiet_window_seconds": (
            None
            if recovery_attempt is None
            else recovery_attempt.admin_reconciliation_quiet_window_seconds
        ),
        "recovery_admin_response_known_candidate_observed": (
            None
            if recovery_attempt is None
            else recovery_attempt.admin_response_known_candidate_observed
        ),
        "recovery_admin_post_baseline_authority_operation_count": (
            None
            if recovery_attempt is None
            else recovery_attempt.admin_post_baseline_authority_operation_count
        ),
        "recovery_admin_delete_pending": bool(
            recovery_attempt is not None and recovery_attempt.admin_delete_pending
        ),
        "recovery_frame_write_attempted": bool(
            recovery_attempt is not None and recovery_attempt.frame_write_attempted
        ),
        "recovery_frame_disclosed_or_ambiguous": bool(
            recovery_attempt is not None
            and recovery_attempt.frame_disclosed_or_ambiguous
        ),
        "recovery_remote_termination_proven": bool(
            recovery_attempt is not None and recovery_attempt.remote_termination_proven
        ),
        "recovery_terminal_receipt_validated": bool(
            recovery_attempt is not None and recovery_attempt.terminal_receipt_validated
        ),
        "recovery_admin_session_closed": (
            None if recovery_attempt is None else recovery_attempt.admin_session_closed
        ),
        "recovery_password_wiped": (
            None if recovery_attempt is None else recovery_attempt.password_wiped
        ),
        "recovery_required": bool(
            recovery_attempt is not None and recovery_attempt.recovery_required
        ),
    }
    value["receipt_sha256"] = _sha256(_canonical_bytes(value))
    return value


def launch_full_canary(
    *,
    release_sha: str,
    transport: CoordinatorTransport,
    sql_admin: CloudSqlTemporaryAdmin,
    token_source: DiscordTokenSource,
    owner_identity: ApprovedOwnerIdentity,
    final_approval_source: FinalApprovalSource,
    approval_request_sink: Callable[[Mapping[str, Any]], None],
    now: Callable[[], int] = lambda: int(time.time()),
    password_factory: Callable[[], bytearray] = _new_admin_password,
    secret_hardener: Callable[[], None] = harden_owner_secret_process,
    provenance_guard: Callable[
        [str], None
    ] = require_owner_runtime_and_launcher_provenance,
) -> Mapping[str, Any]:
    """Run the exact approved owner edge and always clean the temporary admin."""

    if not _RELEASE_SHA.fullmatch(release_sha):
        raise OwnerLauncherError("invalid_release_sha")
    signal_fence = _OwnerSignalFence()
    signal_fence.install()
    gate: Mapping[str, Any] | None = None
    discord_install_receipt: Mapping[str, Any] | None = None
    final_approval_install_receipt: Mapping[str, Any] | None = None
    final_approval_cancel_receipt: Mapping[str, Any] | None = None
    coordinator_receipt: Mapping[str, Any] | None = None
    remote_failure_receipt: Mapping[str, Any] | None = None
    recovery_preflight_receipt: Mapping[str, Any] | None = None
    recovery_receipt: Mapping[str, Any] | None = None
    discord_retirement_receipt: Mapping[str, Any] | None = None
    secret_gate: Mapping[str, Any] | None = None
    approval_request: Mapping[str, Any] | None = None
    recovery_attempt: _RecoveryAttemptState | None = None
    discord_session: RemoteSecretSession | None = None
    approval_session: RemoteSecretSession | None = None
    run_session: RemoteSecretSession | None = None
    token: bytearray | None = None
    password: bytearray | None = None
    admin_create_attempted = False
    admin_mutation_ambiguous = False
    admin_mutation_ambiguity_observed = False
    admin_reconciliation_performed = False
    admin_reconciliation_evidence_sha256: str | None = None
    admin_reconciliation_quiet_window_seconds: float | None = None
    admin_response_known_candidate_observed: bool | None = None
    admin_post_baseline_authority_operation_count: int | None = None
    admin_frame_disclosed = False
    discord_frame_disclosed = False
    final_approval_frame_disclosed = False
    temporary_admin_absent: bool | None = None
    remote_coordinator_terminated: bool | None = None
    terminal_receipt_validated: bool | None = None
    coordinator_admin_session_closed: bool | None = None
    remote_cleanup_blocked = False
    admin_cleanup_blocked = False
    admin_preserved_remote = False
    token_cleanup_blocked = False
    cleanup_failure_codes: set[str] = set()
    current_discord_token_retired: bool | None = None
    failure: OwnerLauncherError | None = None
    secrets_hardened = False
    recovery_safe_to_delete = False

    def capture_sql_reconciliation_evidence(
        state: _RecoveryAttemptState | None = None,
    ) -> None:
        nonlocal admin_mutation_ambiguity_observed
        nonlocal admin_reconciliation_performed
        nonlocal admin_reconciliation_evidence_sha256
        nonlocal admin_reconciliation_quiet_window_seconds
        nonlocal admin_response_known_candidate_observed
        nonlocal admin_post_baseline_authority_operation_count
        evidence = _validated_sql_reconciliation_evidence(sql_admin)
        admin_mutation_ambiguity_observed = bool(
            admin_mutation_ambiguity_observed or evidence["mutation_ambiguity_observed"]
        )
        if evidence["reconciliation_proven"] is True:
            admin_reconciliation_performed = True
            admin_reconciliation_evidence_sha256 = str(
                evidence["reconciliation_evidence_sha256"]
            )
            admin_reconciliation_quiet_window_seconds = float(
                evidence["quiet_window_seconds"]
            )
            admin_response_known_candidate_observed = bool(
                evidence["response_known_candidate_observed"]
            )
            admin_post_baseline_authority_operation_count = int(
                evidence["post_baseline_authority_operation_count"]
            )
        if state is not None:
            _adopt_sql_reconciliation_evidence(state, evidence)

    def recovery_cleanup_state(username: str) -> _RecoveryAttemptState:
        nonlocal recovery_attempt
        if recovery_attempt is None:
            recovery_attempt = _RecoveryAttemptState(username=username)
        elif recovery_attempt.username is None:
            recovery_attempt.username = username
        elif recovery_attempt.username != username:
            raise OwnerLauncherError("recovery_admin_username_drifted")
        return recovery_attempt

    try:
        provenance_guard(release_sha)
        # Read-only gcloud/IAP and Cloud SQL calls still acquire bearer
        # credentials. Make the owner process non-dumpable before any such
        # authentication, not merely before reading the Discord/admin values.
        secret_hardener()
        secrets_hardened = True
        recovery_probe = transport.preflight_recovery(release_sha)
        if recovery_probe.get("schema") == RECOVERY_RECEIPT_SCHEMA:
            recovery_receipt = validate_persisted_recovery_receipt(
                recovery_probe,
                expected_release_sha=release_sha,
                now_unix=now(),
            )
            current_discord_token_retired = True
            owner_identity.bind_approved_subject(
                str(recovery_receipt["owner_subject_sha256"])
            )
            owner_identity.require_stable()
            recovery_safe_to_delete = True
            persisted_recovery_state = recovery_cleanup_state(
                str(recovery_receipt["ephemeral_admin_username"])
            )
            try:
                sql_admin.delete_and_confirm_absent(
                    str(recovery_receipt["ephemeral_admin_username"])
                )
                capture_sql_reconciliation_evidence(persisted_recovery_state)
                temporary_admin_absent = True
            except BaseException:
                temporary_admin_absent = False
                admin_cleanup_blocked = True
                raise
            owner_identity.require_stable()
        elif recovery_probe.get("schema") == RECOVERY_WORKER_COMPLETION_SCHEMA:
            persisted_completion = validate_persisted_recovery_worker_completion(
                recovery_probe,
                expected_release_sha=release_sha,
                now_unix=now(),
            )
            recovery_attempt = _RecoveryAttemptState()
            _adopt_persisted_recovery_completion(
                recovery_attempt,
                persisted_completion,
            )
            owner_identity.bind_approved_subject(
                str(persisted_completion["owner_subject_sha256"])
            )
            owner_identity.require_stable()
            recovery_receipt = _finalize_recovery_worker(
                release_sha=release_sha,
                transport=transport,
                owner_identity=owner_identity,
                now=now,
                state=recovery_attempt,
            )
            current_discord_token_retired = True
            recovery_safe_to_delete = True
            try:
                sql_admin.delete_and_confirm_absent(
                    str(persisted_completion["ephemeral_admin_username"])
                )
                capture_sql_reconciliation_evidence(recovery_attempt)
                temporary_admin_absent = True
                recovery_attempt.admin_delete_pending = False
                recovery_attempt.recovery_required = False
            except BaseException:
                temporary_admin_absent = False
                admin_cleanup_blocked = True
                raise
            owner_identity.require_stable()
        elif recovery_probe.get("schema") == COORDINATOR_FAILURE_SCHEMA:
            recovery_preflight_receipt = validate_terminal_first_failure(
                recovery_probe,
                owner_gate=None,
                token_lease_expected=False,
                process_lease_expected=False,
                expected_release_sha=release_sha,
            )
            no_recovery_required = (
                recovery_preflight_receipt.get("phase") == "recovery_preflight"
                and recovery_preflight_receipt.get("error_code")
                == "coordinator_process_recovery_not_required"
                and recovery_preflight_receipt.get("cleanup_status") == "complete"
                and recovery_preflight_receipt.get("recovery_material_preserved")
                is False
                and recovery_preflight_receipt.get("admin_session_closed") is True
                and recovery_preflight_receipt.get("bootstrap_login_password_disabled")
                is False
                and recovery_preflight_receipt.get("bootstrap_credential_removed")
                is False
                and recovery_preflight_receipt.get("discord_token_removed") is True
                and recovery_preflight_receipt.get("coordinator_process_lease_retired")
                is True
                and recovery_preflight_receipt.get("services_enabled") is False
            )
            token_only_required = (
                recovery_preflight_receipt.get("cleanup_status") == "cleanup_blocked"
                and recovery_preflight_receipt.get("recovery_material_preserved")
                is True
                and recovery_preflight_receipt.get("discord_token_removed") is False
                and recovery_preflight_receipt.get("coordinator_process_lease_retired")
                is True
            )
            if token_only_required:
                recovery_owner_gate = _owner_binding_view(recovery_preflight_receipt)
                owner_identity.bind_approved_subject(
                    str(recovery_preflight_receipt["owner_subject_sha256"])
                )
                owner_identity.require_stable()
                discord_retirement_receipt = _retire_discord_token_only(
                    release_sha=release_sha,
                    transport=transport,
                    owner_gate=recovery_owner_gate,
                    install_receipt=None,
                    owner_identity=owner_identity,
                    now=now,
                )
                current_discord_token_retired = True
                token_only_state = recovery_cleanup_state(
                    str(recovery_preflight_receipt["ephemeral_admin_username"])
                )
                # The run command durably publishes its process lease before
                # accepting MCA2. Lease absence therefore proves no remote
                # admin session received this credential. After exact token
                # retirement it is safe to reconcile the approval-derived
                # local Cloud SQL user in this same invocation.
                try:
                    sql_admin.delete_and_confirm_absent(
                        str(recovery_preflight_receipt["ephemeral_admin_username"])
                    )
                    capture_sql_reconciliation_evidence(token_only_state)
                    temporary_admin_absent = True
                except BaseException:
                    temporary_admin_absent = False
                    admin_cleanup_blocked = True
                    raise
                owner_identity.require_stable()
            elif no_recovery_required:
                current_discord_token_retired = True
                owner_identity.bind_approved_subject(
                    str(recovery_preflight_receipt["owner_subject_sha256"])
                )
                owner_identity.require_stable()
                no_recovery_state = recovery_cleanup_state(
                    str(recovery_preflight_receipt["ephemeral_admin_username"])
                )
                try:
                    sql_admin.delete_and_confirm_absent(
                        str(recovery_preflight_receipt["ephemeral_admin_username"])
                    )
                    capture_sql_reconciliation_evidence(no_recovery_state)
                    temporary_admin_absent = True
                except BaseException:
                    temporary_admin_absent = False
                    admin_cleanup_blocked = True
                    raise
                owner_identity.require_stable()
            elif not no_recovery_required:
                raise RemoteCommandFailed(recovery_preflight_receipt)
        else:
            recovery_gate = validate_recovery_gate(
                recovery_probe,
                expected_release_sha=release_sha,
                owner_gate=None,
                now_unix=now(),
            )
            owner_identity.bind_approved_subject(
                str(recovery_gate["owner_subject_sha256"])
            )
            owner_identity.require_stable()
            recovery_attempt = _RecoveryAttemptState()
            try:
                recovery_receipt, _ = _recover_stale_process(
                    release_sha=release_sha,
                    preflight_gate=recovery_gate,
                    transport=transport,
                    sql_admin=sql_admin,
                    owner_identity=owner_identity,
                    now=now,
                    password_factory=password_factory,
                    state=recovery_attempt,
                )
                current_discord_token_retired = True
                recovery_safe_to_delete = True
                try:
                    sql_admin.delete_and_confirm_absent(
                        str(recovery_gate["ephemeral_admin_username"])
                    )
                    capture_sql_reconciliation_evidence(recovery_attempt)
                    temporary_admin_absent = True
                    recovery_attempt.admin_delete_pending = False
                except BaseException:
                    temporary_admin_absent = False
                    admin_cleanup_blocked = True
                    raise
                owner_identity.require_stable()
            finally:
                recovery_attempt.wipe_password()

        # Any stale user authorized by the recovery receipt was handled above.
        # A fresh temporary admin created below needs its own termination proof.
        recovery_safe_to_delete = False
        temporary_admin_absent = None
        admin_mutation_ambiguous = False
        admin_mutation_ambiguity_observed = False
        admin_reconciliation_performed = False
        admin_reconciliation_evidence_sha256 = None
        admin_reconciliation_quiet_window_seconds = None
        admin_response_known_candidate_observed = None
        admin_post_baseline_authority_operation_count = None

        # After bounded stale-state reconciliation, this fresh read-only proof
        # must still precede every mutation belonging to the new canary run.
        gate = validate_owner_launch_gate(
            transport.preflight_owner_launch(release_sha),
            expected_release_sha=release_sha,
            now_unix=now(),
        )
        username = str(gate["admin_username"])
        owner_identity.bind_approved_subject(str(gate["owner_subject_sha256"]))
        owner_identity.require_stable()
        if not secrets_hardened:
            secret_hardener()
            secrets_hardened = True

        discord_session = transport.open_discord_install(release_sha)
        raw_discord_gate = _validated_input_gate(
            discord_session,
            discord_session.read_gate(),
            owner_gate=gate,
            token_lease_expected=False,
        )
        discord_gate = validate_discord_install_gate(
            raw_discord_gate,
            owner_gate=gate,
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

        def mark_discord_write() -> None:
            nonlocal discord_frame_disclosed
            discord_frame_disclosed = True

        try:
            raw_discord_receipt = discord_session.finish_before(
                discord_frame,
                write_guard=guard_discord_write,
                on_first_write=mark_discord_write,
            )
        finally:
            _wipe(discord_frame)
        raw_install_receipt = _validated_input_gate(
            discord_session,
            raw_discord_receipt,
            owner_gate=gate,
            token_lease_expected=True,
            active_secrets=(token,),
        )
        discord_install_receipt = validate_discord_install_receipt(
            raw_install_receipt,
            gate=discord_gate,
            token=token,
            now_unix=now(),
        )
        current_discord_token_retired = False
        discord_session.mark_validated(discord_install_receipt)
        owner_identity.require_stable()
        discord_session.close()
        discord_session = None
        _wipe(token)
        token = None

        owner_identity.require_stable()
        sql_admin.require_absent(username)
        owner_identity.require_stable()
        password = password_factory()
        if not isinstance(password, bytearray):
            raise OwnerLauncherError("invalid_admin_password")
        # Validate before serializing the process-memory HTTPS request.
        validation_frame = build_admin_frame(username, password)
        _wipe(validation_frame)
        # A complete paginated operations snapshot is the causal baseline
        # for any later lost-response reconciliation.
        sql_admin.begin_mutation_observation(
            expected_owner_subject_sha256=str(gate["owner_subject_sha256"]),
        )
        # From this point onward a transport failure may be ambiguous: cleanup
        # is mandatory even when the create response never reaches us.
        admin_create_attempted = True
        admin_mutation_ambiguous = True
        temporary_admin_absent = None
        owner_identity.require_stable()
        try:
            sql_admin.create(username, password.decode("utf-8"))
        except CloudSqlCreateNotCommitted:
            admin_mutation_ambiguous = False
            raise
        except BaseException:
            admin_mutation_ambiguous = bool(
                admin_mutation_ambiguous or sql_admin.mutation_reconciliation_required()
            )
            admin_mutation_ambiguity_observed = True
            raise
        admin_mutation_ambiguous = bool(sql_admin.mutation_reconciliation_required())
        admin_mutation_ambiguity_observed = bool(
            _validated_sql_reconciliation_evidence(sql_admin)[
                "mutation_ambiguity_observed"
            ]
        )
        owner_identity.require_stable()
        if admin_mutation_ambiguous:
            raise OwnerLauncherError("cloud_sql_mutation_ambiguous")

        remote_coordinator_terminated = False
        terminal_receipt_validated = False
        run_session = transport.open_run(release_sha)
        try:
            raw_secret_gate = _validated_input_gate(
                run_session,
                run_session.read_gate(),
                owner_gate=gate,
                token_lease_expected=True,
                process_lease_expected=True,
                active_secrets=(password,),
            )
        except RemoteCommandFailed as exc:
            coordinator_receipt = exc.receipt
            current_discord_token_retired = bool(
                exc.receipt.get("discord_token_removed")
            )
            remote_failure_receipt = exc.receipt
            coordinator_admin_session_closed = bool(
                exc.receipt.get("admin_session_closed")
            )
            remote_coordinator_terminated = run_session.termination_proven
            terminal_receipt_validated = run_session.terminal_receipt_validated
            raise
        except BaseException:
            admin_mutation_ambiguous = bool(
                admin_mutation_ambiguous or sql_admin.mutation_reconciliation_required()
            )
            admin_mutation_ambiguity_observed = bool(
                admin_mutation_ambiguity_observed
                or sql_admin.mutation_reconciliation_required()
            )
            raise
        secret_gate = validate_coordinator_secret_gate(
            raw_secret_gate,
            owner_gate=gate,
            now_unix=now(),
        )
        owner_identity.require_stable()
        if (
            now() + _SECRET_FRAME_TRANSMIT_MARGIN_SECONDS
            >= secret_gate["expires_at_unix"]
        ):
            raise OwnerLauncherError("coordinator_secret_gate_expired")
        admin_frame = build_admin_frame(username, password)

        def guard_admin_write() -> None:
            owner_identity.require_stable()
            if admin_mutation_ambiguous or sql_admin.mutation_reconciliation_required():
                raise OwnerLauncherError("cloud_sql_mutation_ambiguous")
            sql_admin.require_current_authority(username)
            owner_identity.require_stable()
            if (
                now() + _SECRET_FRAME_TRANSMIT_MARGIN_SECONDS
                >= secret_gate["expires_at_unix"]
            ):
                raise OwnerLauncherError("coordinator_secret_gate_expired")

        def mark_admin_write() -> None:
            nonlocal admin_frame_disclosed
            admin_frame_disclosed = True

        try:
            try:
                raw_admin_result = run_session.finish_before(
                    admin_frame,
                    write_guard=guard_admin_write,
                    on_first_write=mark_admin_write,
                )
            finally:
                _wipe(admin_frame)
            raw_approval_request = _validated_input_gate(
                run_session,
                raw_admin_result,
                owner_gate=gate,
                token_lease_expected=True,
                process_lease_expected=True,
                admin_frame_disclosed=True,
                active_secrets=(password,),
            )
        except RemoteCommandFailed as exc:
            coordinator_receipt = exc.receipt
            current_discord_token_retired = bool(
                exc.receipt.get("discord_token_removed")
            )
            remote_failure_receipt = exc.receipt
            coordinator_admin_session_closed = bool(
                exc.receipt.get("admin_session_closed")
            )
            remote_coordinator_terminated = run_session.termination_proven
            terminal_receipt_validated = run_session.terminal_receipt_validated
            raise
        except BaseException:
            # require_current_authority() marks its SQL observation ambiguous
            # before failing. Preserve that original baseline through cleanup
            # instead of silently starting a generic delete baseline.
            admin_mutation_ambiguous = bool(
                admin_mutation_ambiguous or sql_admin.mutation_reconciliation_required()
            )
            admin_mutation_ambiguity_observed = bool(
                admin_mutation_ambiguity_observed
                or sql_admin.mutation_reconciliation_required()
            )
            raise
        approval_request = validate_owner_approval_request(
            raw_approval_request,
            gate=secret_gate,
            now_unix=now(),
        )
        owner_identity.require_stable()
        approval_deadline = approval_request.get("approval_deadline_unix")
        owner_input_cutoff = approval_request.get("owner_input_cutoff_unix")
        if (
            type(approval_deadline) is not int
            or type(owner_input_cutoff) is not int
            or now() > owner_input_cutoff
        ):
            raise OwnerLauncherError("final_approval_delivery_window_exhausted")
        # Publish the coordinator-authored window immediately.  The remote
        # install transport is then opened while the owner may decide, but the
        # local approval is not read and no MFA1 byte can be disclosed until
        # that independent session returns the exact matching read-only gate.
        approval_request_sink(approval_request)
        approval_session = transport.open_final_approval_install(release_sha)
        try:
            install_gate = _validated_input_gate(
                approval_session,
                approval_session.read_gate(),
                owner_gate=gate,
                token_lease_expected=False,
                active_secrets=(password,),
            )
        except RemoteCommandFailed as exc:
            remote_failure_receipt = exc.receipt
            raise
        if install_gate != approval_request:
            raise OwnerLauncherError("invalid_final_approval_install_gate")
        if now() > owner_input_cutoff:
            raise OwnerLauncherError("final_approval_delivery_window_exhausted")
        owner_identity.require_stable()
        if now() > owner_input_cutoff:
            raise OwnerLauncherError("final_approval_delivery_window_exhausted")
        final_approval = validate_final_owner_approval(
            final_approval_source.read_final_approval(approval_request),
            request=approval_request,
            now_unix=now(),
        )
        owner_identity.require_stable()
        if now() > owner_input_cutoff:
            raise OwnerLauncherError("final_approval_delivery_window_exhausted")
        final_approval_frame = build_final_approval_frame(final_approval)
        if now() > owner_input_cutoff:
            raise OwnerLauncherError("final_approval_delivery_window_exhausted")

        def guard_final_approval_write() -> None:
            owner_identity.require_stable()
            if now() > owner_input_cutoff:
                raise OwnerLauncherError("final_approval_delivery_window_exhausted")

        def mark_final_approval_write() -> None:
            nonlocal final_approval_frame_disclosed
            final_approval_frame_disclosed = True

        raw_approval_install_receipt = _validated_input_gate(
            approval_session,
            approval_session.finish_before(
                final_approval_frame,
                write_guard=guard_final_approval_write,
                on_first_write=mark_final_approval_write,
            ),
            owner_gate=gate,
            token_lease_expected=False,
            active_secrets=(password,),
        )
        final_approval_install_receipt = validate_final_approval_install_receipt(
            raw_approval_install_receipt,
            request=approval_request,
            approval=final_approval,
            now_unix=now(),
        )
        approval_session.mark_validated(final_approval_install_receipt)
        approval_session.close()
        approval_session = None
        owner_identity.require_stable()

        raw_coordinator_receipt = run_session.read_next()
        coordinator_receipt = validate_coordinator_receipt(
            raw_coordinator_receipt,
            gate=secret_gate,
            password=password,
            now_unix=now(),
        )
        current_discord_token_retired = bool(
            coordinator_receipt.get("discord_token_removed")
        )
        coordinator_admin_session_closed = bool(
            coordinator_receipt.get("admin_session_closed")
        )
        run_session.mark_validated(coordinator_receipt)
        remote_coordinator_terminated = run_session.termination_proven
        terminal_receipt_validated = run_session.terminal_receipt_validated
        owner_identity.require_stable()
        run_session.close()
        run_session = None
        if coordinator_receipt.get("ok") is not True:
            raise OwnerLauncherError("coordinator_failed")
    except RemoteCommandFailed as exc:
        remote_failure_receipt = exc.receipt
        cleanup_failure_codes.update(_attached_cleanup_failure_codes(exc))
        if "remote_termination_unconfirmed" in cleanup_failure_codes:
            remote_cleanup_blocked = True
        failure = exc
    except CloudSqlCreateNotCommitted as exc:
        # An explicit create rejection proves this launcher never acquired
        # authority over any same-named account that may have raced into view.
        # Do not delete that account in owner cleanup.
        admin_create_attempted = False
        temporary_admin_absent = None
        failure = exc
    except OwnerLauncherError as exc:
        cleanup_failure_codes.update(_attached_cleanup_failure_codes(exc))
        cleanup_cause = _cleanup_blocked_cause(exc)
        if cleanup_cause not in {None, "cleanup_blocked"}:
            cleanup_failure_codes.add(cleanup_cause)
        if "remote_termination_unconfirmed" in cleanup_failure_codes:
            remote_cleanup_blocked = True
        failure = exc
    except BaseException as exc:
        cleanup_failure_codes.update(_attached_cleanup_failure_codes(exc))
        if "remote_termination_unconfirmed" in cleanup_failure_codes:
            remote_cleanup_blocked = True
        failure = OwnerLauncherError("owner_launcher_interrupted")
    finally:
        signal_fence.begin_cleanup()
        if (
            approval_session is not None
            and approval_request is not None
            and not final_approval_frame_disclosed
            and not approval_session.terminal_receipt_validated
        ):
            try:
                raw_cancel = approval_session.cancel_no_secret()
                final_approval_cancel_receipt = validate_final_approval_cancel_receipt(
                    raw_cancel,
                    request=approval_request,
                    now_unix=now(),
                )
                approval_session.mark_validated(final_approval_cancel_receipt)
                approval_session.close()
                approval_session = None
                if (
                    final_approval_cancel_receipt.get("state")
                    == "cancelled_no_secret_state_conflict"
                ):
                    cleanup_failure_codes.add("final_approval_cancel_state_conflict")
            except BaseException as exc:
                cleanup_failure_codes.update(_attached_cleanup_failure_codes(exc))
                cleanup_failure_codes.add(
                    exc.code
                    if isinstance(exc, OwnerLauncherError)
                    else "final_approval_cancel_unconfirmed"
                )
                if not approval_session.termination_proven:
                    remote_cleanup_blocked = True
        for session in (approval_session,):
            if session is not None:
                try:
                    session.close()
                except BaseException:
                    remote_cleanup_blocked = True
        if (
            run_session is not None
            and approval_request is not None
            and secret_gate is not None
            and not run_session.terminal_receipt_validated
            and password is not None
        ):
            try:
                deadline = approval_request.get("approval_deadline_unix")
                margin = approval_request.get("final_approval_transmit_margin_seconds")
                if type(deadline) is not int or margin != (
                    _FINAL_APPROVAL_DELIVERY_RESERVE_SECONDS
                ):
                    raise OwnerLauncherError("invalid_owner_approval_request")
                cleanup_wait = max(
                    1.0,
                    min(
                        float(
                            FINAL_APPROVAL_MAX_WAIT_SECONDS
                            + _FINAL_APPROVAL_TERMINAL_CLEANUP_GRACE_SECONDS
                        ),
                        float(
                            deadline
                            - now()
                            + _FINAL_APPROVAL_TERMINAL_CLEANUP_GRACE_SECONDS
                        ),
                    ),
                )
                drained = run_session.read_next_bounded(cleanup_wait)
                coordinator_receipt = validate_coordinator_receipt(
                    drained,
                    gate=secret_gate,
                    password=password,
                    now_unix=now(),
                )
                current_discord_token_retired = bool(
                    coordinator_receipt.get("discord_token_removed")
                )
                coordinator_admin_session_closed = bool(
                    coordinator_receipt.get("admin_session_closed")
                )
                run_session.mark_validated(coordinator_receipt)
                remote_coordinator_terminated = run_session.termination_proven
                terminal_receipt_validated = run_session.terminal_receipt_validated
            except BaseException:
                pass
        for session in (run_session, discord_session):
            if session is not None:
                try:
                    session.close()
                except BaseException:
                    remote_cleanup_blocked = True
                    if session is run_session:
                        remote_coordinator_terminated = False
                        terminal_receipt_validated = False
        if run_session is not None and not remote_cleanup_blocked:
            remote_coordinator_terminated = run_session.termination_proven
            terminal_receipt_validated = run_session.terminal_receipt_validated
        if (
            (discord_install_receipt is not None or discord_frame_disclosed)
            and discord_retirement_receipt is None
            and gate is not None
            and not (
                coordinator_receipt is not None
                and coordinator_receipt.get("discord_token_removed") is True
                and coordinator_receipt.get("coordinator_process_lease_retired") is True
            )
        ):
            cleanup_recovery_attempt = _RecoveryAttemptState()
            try:
                owner_identity.require_stable()
                cleanup_probe = transport.preflight_recovery(release_sha)
                if cleanup_probe.get("schema") == RECOVERY_RECEIPT_SCHEMA:
                    persisted_cleanup_receipt = validate_persisted_recovery_receipt(
                        cleanup_probe,
                        expected_release_sha=release_sha,
                        now_unix=now(),
                    )
                    if recovery_receipt is None or persisted_cleanup_receipt.get(
                        "receipt_sha256"
                    ) != recovery_receipt.get("receipt_sha256"):
                        raise OwnerLauncherError("persisted_recovery_receipt_drifted")
                    discord_retirement_receipt = _retire_discord_token_only(
                        release_sha=release_sha,
                        transport=transport,
                        owner_gate=gate,
                        install_receipt=discord_install_receipt,
                        owner_identity=owner_identity,
                        now=now,
                    )
                    current_discord_token_retired = True
                    if not admin_frame_disclosed:
                        recovery_safe_to_delete = True
                elif cleanup_probe.get("schema") == COORDINATOR_FAILURE_SCHEMA:
                    recovery_preflight_receipt = validate_terminal_first_failure(
                        cleanup_probe,
                        owner_gate=gate,
                        token_lease_expected=False,
                        process_lease_expected=False,
                        expected_release_sha=release_sha,
                        active_secrets=(password,) if password is not None else (),
                    )
                    if (
                        recovery_preflight_receipt.get(
                            "coordinator_process_lease_retired"
                        )
                        is not True
                    ):
                        raise OwnerLauncherError("recovery_process_lease_state_invalid")
                    discord_retirement_receipt = _retire_discord_token_only(
                        release_sha=release_sha,
                        transport=transport,
                        owner_gate=gate,
                        install_receipt=discord_install_receipt,
                        owner_identity=owner_identity,
                        now=now,
                    )
                    current_discord_token_retired = True
                    if not admin_frame_disclosed:
                        recovery_safe_to_delete = True
                else:
                    cleanup_recovery_gate = validate_recovery_gate(
                        cleanup_probe,
                        expected_release_sha=release_sha,
                        owner_gate=gate,
                        now_unix=now(),
                    )
                    recovery_receipt, _ = _recover_stale_process(
                        release_sha=release_sha,
                        preflight_gate=cleanup_recovery_gate,
                        transport=transport,
                        sql_admin=sql_admin,
                        owner_identity=owner_identity,
                        now=now,
                        password_factory=password_factory,
                        state=cleanup_recovery_attempt,
                    )
                    current_discord_token_retired = True
                    recovery_safe_to_delete = True
                    coordinator_admin_session_closed = True
                    remote_coordinator_terminated = True
                    terminal_receipt_validated = True
            except BaseException as exc:
                token_cleanup_blocked = True
                cleanup_failure_codes.update(_attached_cleanup_failure_codes(exc))
                cleanup_cause = _cleanup_blocked_cause(exc)
                if cleanup_cause not in {None, "cleanup_blocked"}:
                    cleanup_failure_codes.add(cleanup_cause)
                if failure is None:
                    failure = (
                        exc
                        if isinstance(exc, OwnerLauncherError)
                        else OwnerLauncherError("discord_token_cleanup_blocked")
                    )
            finally:
                cleanup_recovery_attempt.wipe_password()
                if (
                    cleanup_recovery_attempt.gate is not None
                    or cleanup_recovery_attempt.concurrent_loser_receipt is not None
                ):
                    if recovery_attempt is None:
                        recovery_attempt = cleanup_recovery_attempt
                    else:
                        recovery_attempt.absorb(cleanup_recovery_attempt)
        if recovery_attempt is not None:
            cleanup_failure_codes.update(recovery_attempt.cleanup_failure_codes)
            if recovery_attempt.admin_delete_pending:
                recovery_delete_safe = recovery_attempt.final_receipt is not None
                if (
                    not recovery_delete_safe
                    and not recovery_attempt.frame_write_attempted
                ):
                    # The rotated credential was never offered to any remote
                    # process.  The exact approval-derived user is therefore
                    # safe to remove even if the SQL response itself failed.
                    recovery_delete_safe = True
                elif not recovery_delete_safe:
                    try:
                        recovery_receipt = _resume_recovery_attempt(
                            release_sha=release_sha,
                            transport=transport,
                            sql_admin=sql_admin,
                            owner_identity=owner_identity,
                            now=now,
                            password_factory=password_factory,
                            state=recovery_attempt,
                        )
                        recovery_delete_safe = True
                        recovery_safe_to_delete = True
                        current_discord_token_retired = True
                        recovery_attempt.cleanup_failure_codes.discard(
                            "remote_termination_unconfirmed"
                        )
                        cleanup_failure_codes.discard("remote_termination_unconfirmed")
                        remote_cleanup_blocked = False
                        remote_coordinator_terminated = True
                        terminal_receipt_validated = True
                    except BaseException as exc:
                        recovery_attempt.recovery_required = True
                        cleanup_failure_codes.update(
                            _attached_cleanup_failure_codes(exc)
                        )
                        cleanup_failure_codes.add(
                            exc.code
                            if isinstance(exc, OwnerLauncherError)
                            else "recovery_reconciliation_blocked"
                        )
                if recovery_delete_safe:
                    try:
                        owner_identity.require_stable()
                        if recovery_attempt.admin_mutation_ambiguous:
                            sql_admin.reconcile_ambiguous_mutation_and_confirm_absent(
                                str(recovery_attempt.username)
                            )
                        else:
                            sql_admin.delete_and_confirm_absent(
                                str(recovery_attempt.username)
                            )
                        capture_sql_reconciliation_evidence(recovery_attempt)
                        owner_identity.require_stable()
                    except BaseException as exc:
                        temporary_admin_absent = False
                        admin_cleanup_blocked = True
                        recovery_attempt.recovery_required = True
                        cleanup_failure_codes.update(
                            _attached_cleanup_failure_codes(exc)
                        )
                        cleanup_cause = _cleanup_blocked_cause(exc)
                        if cleanup_cause not in {None, "cleanup_blocked"}:
                            cleanup_failure_codes.add(cleanup_cause)
                    else:
                        recovery_attempt.admin_delete_pending = False
                        recovery_attempt.admin_mutation_ambiguous = False
                        recovery_attempt.recovery_required = (
                            recovery_attempt.final_receipt is None
                        )
                        temporary_admin_absent = True
                        if (
                            gate is not None
                            and gate.get("admin_username") == recovery_attempt.username
                        ):
                            admin_create_attempted = False
                else:
                    temporary_admin_absent = False
                    admin_preserved_remote = True
            recovery_attempt.wipe_password()
            cleanup_failure_codes.update(recovery_attempt.cleanup_failure_codes)
        _wipe(token)
        _wipe(password)
        if admin_create_attempted and gate is not None:
            safe_to_delete = (
                recovery_safe_to_delete
                or remote_coordinator_terminated is None
                or (
                    remote_coordinator_terminated is True
                    and terminal_receipt_validated is True
                    and (
                        not admin_frame_disclosed
                        or coordinator_admin_session_closed is True
                    )
                )
            )
            if not safe_to_delete:
                temporary_admin_absent = False
                admin_preserved_remote = True
            else:
                try:
                    if admin_mutation_ambiguous:
                        sql_admin.reconcile_ambiguous_mutation_and_confirm_absent(
                            str(gate["admin_username"])
                        )
                    else:
                        sql_admin.delete_and_confirm_absent(str(gate["admin_username"]))
                    # This is the fresh run's admin, not the earlier recovery
                    # admin. Keep the recovery evidence bound to its own
                    # phase instead of overwriting it with this digest.
                    capture_sql_reconciliation_evidence()
                    temporary_admin_absent = True
                    admin_mutation_ambiguous = False
                    if (
                        recovery_attempt is not None
                        and recovery_attempt.username == gate.get("admin_username")
                    ):
                        recovery_attempt.admin_delete_pending = False
                        recovery_attempt.recovery_required = False
                except BaseException as exc:
                    temporary_admin_absent = False
                    admin_cleanup_blocked = True
                    cleanup_cause = _cleanup_blocked_cause(exc)
                    if cleanup_cause not in {None, "cleanup_blocked"}:
                        cleanup_failure_codes.add(cleanup_cause)
                else:
                    try:
                        owner_identity.require_stable()
                    except BaseException:
                        if failure is None:
                            failure = OwnerLauncherError(
                                "approved_owner_identity_changed"
                            )
    if admin_create_attempted:
        try:
            # The SQL object exposes the most recent fresh phase. Recovery
            # evidence was adopted at its own cleanup boundary and is immutable.
            capture_sql_reconciliation_evidence()
        except BaseException:
            admin_cleanup_blocked = True
            cleanup_failure_codes.add("cloud_sql_reconciliation_evidence_invalid")
    if signal_fence.received and failure is None:
        failure = OwnerLauncherError("owner_launcher_interrupted")
    try:
        signal_fence.restore()
    except OwnerLauncherError as exc:
        if failure is None:
            failure = exc

    cleanup_blocked = (
        admin_cleanup_blocked
        or admin_preserved_remote
        or remote_cleanup_blocked
        or token_cleanup_blocked
        or bool(cleanup_failure_codes)
    )
    if admin_cleanup_blocked:
        cleanup_failure_codes.add("admin_cleanup_blocked")
    if admin_preserved_remote:
        cleanup_failure_codes.add("temporary_admin_preserved_remote")
    if remote_cleanup_blocked:
        cleanup_failure_codes.add("remote_termination_unconfirmed")
    if token_cleanup_blocked:
        cleanup_failure_codes.add("discord_token_cleanup_blocked")
    primary_failure_code = None if failure is None else failure.code
    cleanup_failure_code = (
        next(iter(cleanup_failure_codes))
        if len(cleanup_failure_codes) == 1
        else "multiple_cleanup_blocked"
    )
    if cleanup_blocked:
        return _receipt(
            ok=False,
            state="cleanup_blocked",
            release_sha=release_sha,
            gate=gate,
            discord_install_receipt=discord_install_receipt,
            final_approval_install_receipt=final_approval_install_receipt,
            final_approval_cancel_receipt=final_approval_cancel_receipt,
            coordinator_receipt=coordinator_receipt,
            remote_failure_receipt=remote_failure_receipt,
            recovery_preflight_receipt=recovery_preflight_receipt,
            recovery_receipt=recovery_receipt,
            discord_retirement_receipt=discord_retirement_receipt,
            discord_token_retired=current_discord_token_retired,
            failure_code=cleanup_failure_code,
            primary_failure_code=primary_failure_code,
            cleanup_failure_codes=cleanup_failure_codes,
            recovery_attempt=recovery_attempt,
            admin_mutation_ambiguous=admin_mutation_ambiguous,
            admin_mutation_ambiguity_observed=(admin_mutation_ambiguity_observed),
            admin_reconciliation_performed=admin_reconciliation_performed,
            admin_reconciliation_evidence_sha256=(admin_reconciliation_evidence_sha256),
            admin_reconciliation_quiet_window_seconds=(
                admin_reconciliation_quiet_window_seconds
            ),
            admin_response_known_candidate_observed=(
                admin_response_known_candidate_observed
            ),
            admin_post_baseline_authority_operation_count=(
                admin_post_baseline_authority_operation_count
            ),
            temporary_admin_absent=temporary_admin_absent,
            remote_coordinator_terminated=remote_coordinator_terminated,
            terminal_receipt_validated=terminal_receipt_validated,
            admin_frame_disclosed=admin_frame_disclosed,
            discord_frame_disclosed=discord_frame_disclosed,
            final_approval_frame_disclosed=final_approval_frame_disclosed,
            coordinator_admin_session_closed=coordinator_admin_session_closed,
        )

    if failure is not None:
        return _receipt(
            ok=False,
            state="blocked",
            release_sha=release_sha,
            gate=gate,
            discord_install_receipt=discord_install_receipt,
            final_approval_install_receipt=final_approval_install_receipt,
            final_approval_cancel_receipt=final_approval_cancel_receipt,
            coordinator_receipt=coordinator_receipt,
            remote_failure_receipt=remote_failure_receipt,
            recovery_preflight_receipt=recovery_preflight_receipt,
            recovery_receipt=recovery_receipt,
            discord_retirement_receipt=discord_retirement_receipt,
            discord_token_retired=current_discord_token_retired,
            failure_code=failure.code,
            primary_failure_code=primary_failure_code,
            cleanup_failure_codes=cleanup_failure_codes,
            recovery_attempt=recovery_attempt,
            admin_mutation_ambiguous=admin_mutation_ambiguous,
            admin_mutation_ambiguity_observed=(admin_mutation_ambiguity_observed),
            admin_reconciliation_performed=admin_reconciliation_performed,
            admin_reconciliation_evidence_sha256=(admin_reconciliation_evidence_sha256),
            admin_reconciliation_quiet_window_seconds=(
                admin_reconciliation_quiet_window_seconds
            ),
            admin_response_known_candidate_observed=(
                admin_response_known_candidate_observed
            ),
            admin_post_baseline_authority_operation_count=(
                admin_post_baseline_authority_operation_count
            ),
            temporary_admin_absent=temporary_admin_absent,
            remote_coordinator_terminated=remote_coordinator_terminated,
            terminal_receipt_validated=terminal_receipt_validated,
            admin_frame_disclosed=admin_frame_disclosed,
            discord_frame_disclosed=discord_frame_disclosed,
            final_approval_frame_disclosed=final_approval_frame_disclosed,
            coordinator_admin_session_closed=coordinator_admin_session_closed,
        )
    return _receipt(
        ok=True,
        state="completed",
        release_sha=release_sha,
        gate=gate,
        discord_install_receipt=discord_install_receipt,
        final_approval_install_receipt=final_approval_install_receipt,
        final_approval_cancel_receipt=final_approval_cancel_receipt,
        coordinator_receipt=coordinator_receipt,
        remote_failure_receipt=remote_failure_receipt,
        recovery_preflight_receipt=recovery_preflight_receipt,
        recovery_receipt=recovery_receipt,
        discord_retirement_receipt=discord_retirement_receipt,
        discord_token_retired=current_discord_token_retired,
        failure_code=None,
        primary_failure_code=None,
        cleanup_failure_codes=cleanup_failure_codes,
        recovery_attempt=recovery_attempt,
        admin_mutation_ambiguous=admin_mutation_ambiguous,
        admin_mutation_ambiguity_observed=admin_mutation_ambiguity_observed,
        admin_reconciliation_performed=admin_reconciliation_performed,
        admin_reconciliation_evidence_sha256=(admin_reconciliation_evidence_sha256),
        admin_reconciliation_quiet_window_seconds=(
            admin_reconciliation_quiet_window_seconds
        ),
        admin_response_known_candidate_observed=(
            admin_response_known_candidate_observed
        ),
        admin_post_baseline_authority_operation_count=(
            admin_post_baseline_authority_operation_count
        ),
        temporary_admin_absent=temporary_admin_absent,
        remote_coordinator_terminated=remote_coordinator_terminated,
        terminal_receipt_validated=terminal_receipt_validated,
        admin_frame_disclosed=admin_frame_disclosed,
        discord_frame_disclosed=discord_frame_disclosed,
        final_approval_frame_disclosed=final_approval_frame_disclosed,
        coordinator_admin_session_closed=coordinator_admin_session_closed,
    )


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
        "--publish-writer-preflight",
        action="store_true",
        help="stage and attest writer-only inputs without starting services",
    )
    parser.add_argument(
        "--external-iam-policy-sha256",
        help="exact external IAM policy digest bound into writer staging",
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
        if arguments.bootstrap_trusted_runtime:
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
        if arguments.external_iam_policy_sha256 is not None:
            raise OwnerLauncherError("writer_preflight_plan_invalid")
        transport = IapCoordinatorTransport(
            owner_identity,
            gcloud_executable=gcloud_executable,
            gcloud_configuration=gcloud_configuration,
        )
        sql_admin = CloudSqlTemporaryAdmin(GoogleRestClient(owner_identity))

        try:
            receipt = launch_full_canary(
                release_sha=release_sha,
                transport=transport,
                sql_admin=sql_admin,
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
    "CleanupBlocked",
    "CloudSqlCreateNotCommitted",
    "CloudSqlTemporaryAdmin",
    "COORDINATOR_FAILURE_SCHEMA",
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
    "FixedLocalFinalApprovalFile",
    "GoogleRestClient",
    "GcloudOwnerAccessToken",
    "IapCoordinatorTransport",
    "IapStoppedReleaseTransport",
    "IapWriterPreflightTransport",
    "HttpResponse",
    "LocalLauncherProvenance",
    "OWNER_GATE_SCHEMA",
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
    "SQL_INSTANCE",
    "STOPPED_RELEASE_EVIDENCE_BASE",
    "STOPPED_RELEASE_HOST_RECEIPT_PATH",
    "STOPPED_RELEASE_PLAN_SCHEMA",
    "STOPPED_RELEASE_PYTHON_VERSION",
    "STOPPED_RELEASE_RECEIPT_SCHEMA",
    "WRITER_PREFLIGHT_PLAN_SCHEMA",
    "WRITER_PREFLIGHT_RECEIPT_SCHEMA",
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
    "build_admin_frame",
    "build_discord_frame",
    "build_discord_retirement_ack",
    "build_discord_retirement_ack_frame",
    "build_final_approval_frame",
    "build_recovery_ack",
    "build_recovery_ack_frame",
    "build_recovery_admin_frame",
    "bootstrap_trusted_gcloud_runtime",
    "harden_owner_secret_process",
    "launch_full_canary",
    "main",
    "require_local_launcher_provenance",
    "validate_coordinator_receipt",
    "validate_coordinator_secret_gate",
    "validate_discord_install_gate",
    "validate_discord_install_receipt",
    "validate_discord_retirement_gate",
    "validate_discord_retirement_receipt",
    "validate_final_approval_install_receipt",
    "validate_final_approval_cancel_receipt",
    "validate_final_owner_approval",
    "validate_owner_launch_gate",
    "validate_owner_approval_request",
    "validate_persisted_recovery_receipt",
    "validate_persisted_recovery_worker_completion",
    "validate_recovery_gate",
    "validate_recovery_secret_gate",
    "validate_recovery_worker_completion",
    "validate_recovery_receipt",
    "validate_stopped_release_plan",
    "validate_stopped_release_receipt",
    "validate_writer_preflight_plan",
    "validate_writer_preflight_receipt",
    "validate_terminal_first_failure",
]


if __name__ == "__main__":
    raise SystemExit(main())
