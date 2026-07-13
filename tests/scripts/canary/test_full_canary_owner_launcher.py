from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import signal
import struct
import subprocess
import sys
import tarfile
from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import scripts.canary.full_canary_owner_launcher as launcher

from scripts.canary.full_canary_owner_launcher import (
    ADMIN_FRAME_MAGIC,
    COORDINATOR_FAILURE_SCHEMA,
    COORDINATOR_RECEIPT_SCHEMA,
    COORDINATOR_SECRET_GATE_SCHEMA,
    CleanupBlocked,
    CloudSqlTemporaryAdmin,
    DISCORD_FRAME_MAGIC,
    DISCORD_INSTALL_GATE_SCHEMA,
    DISCORD_INSTALL_RECEIPT_SCHEMA,
    DISCORD_RETIREMENT_RECEIPT_SCHEMA,
    DISCORD_TOKEN_PATH,
    GoogleRestClient,
    GcloudOwnerAccessToken,
    HttpResponse,
    IapCoordinatorTransport,
    IapStoppedReleaseTransport,
    OWNER_GATE_SCHEMA,
    OWNER_DISCORD_INPUT_MAGIC,
    OwnerLauncherError,
    OwnerDiscordTokenReader,
    OwnerStdinDiscordTokenReader,
    PROJECT,
    SQL_INSTANCE,
    build_admin_frame,
    build_discord_frame,
    launch_full_canary,
    validate_owner_launch_gate,
    validate_coordinator_receipt,
    validate_discord_retirement_receipt,
)


RELEASE_SHA = "a" * 40
APPROVAL_SHA = "1234567890abcdef" + "b" * 48
OWNER_SUBJECT_SHA = hashlib.sha256(b"owner@example.com").hexdigest()
USERNAME = "muncho_canary_admin_1234567890abcdef"
PASSWORD = b"owner-only-random-password-1234567890"
DISCORD_TOKEN = b"discord-canary-only-token"
GCLOUD_COMMAND_PREFIX = (
    "/trusted/bin/python3.13",
    "-I",
    "-S",
    "-B",
    "-X",
    "pycache_prefix=/var/empty/muncho-canary",
    "/trusted/sdk/lib/gcloud.py",
)


@pytest.fixture(autouse=True)
def _clear_ambient_custom_ca_bundle(monkeypatch):
    """Keep gateway import side effects from weakening owner-boundary tests."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)


class _StableExecutable:
    def __init__(
        self,
        path: str = "/trusted/bin/gcloud",
        python_path: str = "/trusted/bin/python3.13",
    ) -> None:
        self.path = path
        self.python_path = python_path
        self.calls = 0

    def absolute_path(self) -> str:
        self.calls += 1
        return self.path

    def trusted_command_prefix(self) -> tuple[str, ...]:
        self.calls += 1
        return (
            self.python_path,
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/var/empty/muncho-canary",
            "/trusted/sdk/lib/gcloud.py",
        )

    def private_key_path(self) -> str:
        return "/trusted/google_compute_engine"

    def public_key_line(self) -> str:
        return "ssh-ed25519 fixed-public-key owner@example.com"


class _StableGcloudConfiguration:
    account = "owner@example.com"

    def __init__(self) -> None:
        self.calls = 0

    def assert_stable(self) -> None:
        self.calls += 1

    def environment_values(self) -> Mapping[str, str]:
        self.assert_stable()
        return {
            "HOME": "/trusted/home",
            "CLOUDSDK_CONFIG": "/trusted/home/.config/gcloud",
        }


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _gate(*, now: int = 1_000, **changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": OWNER_GATE_SCHEMA,
        "ok": True,
        "state": "credential_prepare_authorized",
        "coordinator_input_sha256": "c" * 64,
        "credential_prepare_approval_sha256": APPROVAL_SHA,
        "owner_subject_sha256": OWNER_SUBJECT_SHA,
        "release_sha": RELEASE_SHA,
        "database_host": "10.91.0.3",
        "database_port": 5432,
        "database_name": "muncho_canary_brain",
        "admin_username": USERNAME,
        "expires_at_unix": now + 400,
    }
    value.update(changes)
    value["gate_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def _self_digest(value: dict[str, object], key: str) -> dict[str, object]:
    value[key] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def _discord_gate() -> dict[str, object]:
    return _self_digest(
        {
            "schema": DISCORD_INSTALL_GATE_SCHEMA,
            "ok": True,
            "state": "token_install_authorized",
            "coordinator_input_sha256": "c" * 64,
            "discord_token_install_approval_sha256": "d" * 64,
            "owner_subject_sha256": OWNER_SUBJECT_SHA,
            "release_sha": RELEASE_SHA,
            "token_path": DISCORD_TOKEN_PATH,
            "edge_uid": 1001,
            "edge_gid": 1002,
            "expires_at_unix": 1_400,
            "frame_schema": "DCT1-u32be-opaque-eof.v1",
        },
        "gate_sha256",
    )


def _discord_receipt() -> dict[str, object]:
    return _self_digest(
        {
            "schema": DISCORD_INSTALL_RECEIPT_SCHEMA,
            "ok": True,
            "release_sha": RELEASE_SHA,
            "coordinator_input_sha256": "c" * 64,
            "discord_token_install_approval_sha256": "d" * 64,
            "owner_subject_sha256": OWNER_SUBJECT_SHA,
            "token_path": DISCORD_TOKEN_PATH,
            "device": 10,
            "inode": 11,
            "owner_uid": 1001,
            "group_gid": 1002,
            "mode": "0400",
            "size": len(DISCORD_TOKEN),
            "link_count": 1,
            "content_or_digest_recorded": False,
            "installed_at_unix": 1_000,
        },
        "receipt_sha256",
    )


def _discord_retirement_receipt(**changes: object) -> dict[str, object]:
    install = _discord_receipt()
    value = {
        "schema": DISCORD_RETIREMENT_RECEIPT_SCHEMA,
        "ok": True,
        "state": "retired",
        "release_sha": RELEASE_SHA,
        "coordinator_input_sha256": "c" * 64,
        "discord_token_install_receipt_sha256": install["receipt_sha256"],
        "token_path": DISCORD_TOKEN_PATH,
        "token_device": 10,
        "token_inode": 11,
        "services_stopped_proven": True,
        "services_enabled": False,
        "token_removed": True,
        "install_receipt_removed": True,
        "prepared_at_unix": 999,
        "retired_at_unix": 1_000,
    }
    value.update(changes)
    return _self_digest(value, "receipt_sha256")


def _discord_retirement_gate(**changes: object) -> dict[str, object]:
    install = _discord_receipt()
    value = {
        "schema": launcher.DISCORD_RETIREMENT_GATE_SCHEMA,
        "ok": True,
        "state": "awaiting_owner_discord_retirement_ack",
        "release_sha": RELEASE_SHA,
        "coordinator_input_sha256": "c" * 64,
        "owner_subject_sha256": OWNER_SUBJECT_SHA,
        "discord_token_install_receipt_sha256": install["receipt_sha256"],
        "token_device": 10,
        "token_inode": 11,
        "process_lease_absent": True,
        "services_stopped_proven": True,
        "frame_schema": launcher.DISCORD_RETIREMENT_ACK_FRAME_SCHEMA,
        "expires_at_unix": 1_300,
    }
    value.update(changes)
    return _self_digest(value, "gate_sha256")


def _coordinator_gate() -> dict[str, object]:
    return _self_digest(
        {
            "schema": COORDINATOR_SECRET_GATE_SCHEMA,
            "ok": True,
            "state": "awaiting_admin_credential",
            "coordinator_input_sha256": "c" * 64,
            "credential_prepare_approval_sha256": APPROVAL_SHA,
            "owner_subject_sha256": OWNER_SUBJECT_SHA,
            "release_sha": RELEASE_SHA,
            "admin_username": USERNAME,
            "database_host": "10.91.0.3",
            "database_port": 5432,
            "database_name": "muncho_canary_brain",
            "tls_server_name": "db.europe-west3.sql.goog",
            "tls_peer_certificate_sha256": "e" * 64,
            "coordinator_process_lease_sha256": "1" * 64,
            "coordinator_pid": 1234,
            "coordinator_start_time_ticks": 5678,
            "coordinator_boot_id_sha256": "2" * 64,
            "expires_at_unix": 1_400,
            "frame_schema": "MCA2-u16be-u32be-utf8-eof.v1",
        },
        "gate_sha256",
    )


def _approval_request() -> dict[str, object]:
    return _self_digest(
        {
            "schema": "muncho-full-canary-owner-approval-request.v1",
            "ok": True,
            "state": "awaiting_final_owner_approval",
            "release_sha": RELEASE_SHA,
            "coordinator_input_sha256": "c" * 64,
            "credential_prepare_approval_sha256": APPROVAL_SHA,
            "owner_subject_sha256": OWNER_SUBJECT_SHA,
            "ephemeral_admin_username": USERNAME,
            "full_canary_plan_sha256": "8" * 64,
            "staged_plan_path": "/etc/muncho/full-canary/staged/runtime-plan.json",
            "staged_plan_file_sha256": "7" * 64,
            "approval_path": "/etc/muncho/full-canary/owner-approval.json",
            "hba_receipt_sha256": "6" * 64,
            "hba_expires_at_unix": 1_400,
            "fixture_expires_at_unix": 1_400,
            "credential_approval_expires_at_unix": 1_400,
            "approval_deadline_unix": 1_240,
            "owner_input_cutoff_unix": 1_210,
            "final_approval_transmit_margin_seconds": 30,
            "max_wait_seconds": 240,
            "requested_at_unix": 1_000,
            "approval_source_sha256": "5" * 64,
            "approval_request_path": "/etc/muncho/full-canary/approval-request.json",
            "final_approval_frame_schema": "MFA1-u32be-canonical-json-eof.v1",
            "prior_approval_file_sha256": None,
        },
        "request_sha256",
    )


def _final_approval() -> dict[str, object]:
    return {
        "schema": "muncho-full-canary-owner-approval.v1",
        "scope": "full_canary_runtime_start",
        "plan_sha256": "8" * 64,
        "authority_kind": "trusted_root_bootstrap_out_of_band_owner",
        "cryptographic_owner_proof": False,
        "owner_subject_sha256": OWNER_SUBJECT_SHA,
        "approval_source_sha256": "5" * 64,
        "nonce_sha256": "4" * 64,
        "approved_at_unix": 1_000,
        "expires_at_unix": 1_300,
    }


def _approval_install_receipt() -> dict[str, object]:
    request = _approval_request()
    approval = _final_approval()
    return _self_digest(
        {
            "schema": "muncho-full-canary-final-approval-install-receipt.v1",
            "ok": True,
            "release_sha": RELEASE_SHA,
            "coordinator_input_sha256": "c" * 64,
            "credential_prepare_approval_sha256": APPROVAL_SHA,
            "owner_subject_sha256": OWNER_SUBJECT_SHA,
            "full_canary_plan_sha256": "8" * 64,
            "approval_request_sha256": request["request_sha256"],
            "owner_approval_sha256": hashlib.sha256(_canonical(approval)).hexdigest(),
            "approval_path": "/etc/muncho/full-canary/owner-approval.json",
            "installed_at_unix": 1_000,
        },
        "receipt_sha256",
    )


def _approval_cancel_receipt(**changes: object) -> dict[str, object]:
    request = _approval_request()
    value = {
        "schema": launcher.FINAL_APPROVAL_CANCEL_RECEIPT_SCHEMA,
        "ok": False,
        "state": "cancelled_no_secret",
        "reason": "eof_before_mfa1",
        "release_sha": RELEASE_SHA,
        "coordinator_input_sha256": "c" * 64,
        "credential_prepare_approval_sha256": APPROVAL_SHA,
        "owner_subject_sha256": OWNER_SUBJECT_SHA,
        "full_canary_plan_sha256": "8" * 64,
        "approval_request_sha256": request["request_sha256"],
        "approval_request_path": launcher.APPROVAL_REQUEST_PATH,
        "expected_approval_request_file_sha256": hashlib.sha256(
            _canonical(request)
        ).hexdigest(),
        "observed_approval_request_file_sha256": hashlib.sha256(
            _canonical(request)
        ).hexdigest(),
        "approval_request_artifact_state": "matching_active",
        "approval_request_present": True,
        "approval_request_remains_active": True,
        "staged_plan_path": request["staged_plan_path"],
        "expected_staged_plan_file_sha256": request["staged_plan_file_sha256"],
        "observed_staged_plan_file_sha256": request["staged_plan_file_sha256"],
        "staged_plan_artifact_state": "matching_present",
        "staged_plan_present": True,
        "approval_path": launcher.FINAL_APPROVAL_PATH,
        "prior_approval_file_sha256": None,
        "observed_approval_file_sha256": None,
        "owner_approval_artifact_state": "matching_prior",
        "approval_path_matches_prior": True,
        "new_owner_approval_installed": False,
        "frame_bytes_received": 0,
        "owner_approval_mutation_performed_by_this_helper": False,
        "cancelled_at_unix": 1_000,
    }
    value.update(changes)
    return _self_digest(value, "receipt_sha256")


def _invariant_receipt() -> dict[str, object]:
    return _self_digest(
        {
            "schema": "muncho-full-canary-e2e-verification.v1",
            "ok": True,
            "fixture_sha256": "f" * 64,
            "evidence_sha256": "1" * 64,
            "full_canary_start_receipt_sha256": "2" * 64,
            "release_sha": RELEASE_SHA,
            "canary_run_id": "run-1",
            "invariants": [
                "live_provenance_bound",
                "canonical_writer_ready",
                "owner_preapproved_one_shot_scope_claimed_and_durably_revoked",
                "gpt56_model_authored_high_to_xhigh",
                "canonical_plan_event_verification_truth_complete",
                "public_discord_routeback_signed_and_readback_verified",
                "discord_dm_private_target_denied_without_dispatch",
                "sustained_multistep_task_completed_nonpartial",
            ],
        },
        "invariant_receipt_sha256",
    )


def _preclaim_receipt() -> dict[str, object]:
    database = {
        "host": "10.91.0.3",
        "tls_server_name": "db.europe-west3.sql.goog",
        "port": 5432,
        "database": "muncho_canary_brain",
        "user": "canonical_writer",
    }
    result = {
        "success": True,
        "outcome": "claimed",
        "grant_id": "grant-1",
        "case_id": "case:1",
        "release_sha256": "3" * 64,
        "fixture_sha256": "f" * 64,
        "run_id": "run-1",
        "session_key_sha256": "4" * 64,
        "expires_at": "2026-07-13T12:00:00+00:00",
        "approved_by": "owner",
        "approval_source_sha256": "5" * 64,
        "provisioning_receipt_sha256": "6" * 64,
        "preapproval_event_id": "00000000-0000-4000-8000-000000000001",
        "bootstrap_consumption_event_id": "00000000-0000-4000-8000-000000000002",
        "claim_event_id": "00000000-0000-4000-8000-000000000003",
        "retirement_event_id": None,
        "revocation_event_id": "00000000-0000-4000-8000-000000000004",
        "claimed_at": "2026-07-13T11:00:00+00:00",
        "retired_at": None,
        "reason": "claim_already_committed_session_retired",
        "scope_retired": False,
        "authority_active": False,
        "inserted": True,
        "deduped": False,
    }
    return _self_digest(
        {
            "version": "canonical-canary-preclaim-reconciliation-v1",
            "observed_at_unix": 1_000,
            "source_config_path": "/etc/muncho/full-canary/staged/writer.json",
            "source_config_sha256": "7" * 64,
            "database_identity": database,
            "database_identity_sha256": hashlib.sha256(
                _canonical(database)
            ).hexdigest(),
            "result": result,
        },
        "receipt_sha256",
    )


def _coordinator_receipt() -> dict[str, object]:
    plan_sha = "8" * 64
    evidence_path = (
        f"/var/lib/muncho-full-canary/plans/{RELEASE_SHA}/{plan_sha}/live/evidence.json"
    )
    invariant = _invariant_receipt()
    lifecycle = _self_digest(
        {
            "operation": "verify_and_stop",
            "full_canary_start_receipt_sha256": "2" * 64,
            "full_canary_start_receipt_internal_sha256": "9" * 64,
            "evidence_path": evidence_path,
            "evidence_sha256": "1" * 64,
            "live_report_sha256": "a" * 64,
            "verifier_result": invariant,
            "stop_order": [
                "hermes-cloud-gateway.service",
                "muncho-canonical-writer.service",
                "muncho-discord-egress.service",
            ],
            "preclaim_reconciliation": _preclaim_receipt(),
            "stopped_report_sha256": "b" * 64,
            "units_enabled": False,
            "verified": True,
            "error_type": None,
            "error_sha256": None,
            "completed_at_unix": 1_000,
            "schema": "muncho-full-canary-runtime-receipt.v1",
            "stage": "verified_stopped",
            "revision": RELEASE_SHA,
            "full_canary_plan_sha256": plan_sha,
            "receipt_path": (
                f"/var/lib/muncho-full-canary/plans/{RELEASE_SHA}/{plan_sha}/"
                "verified_stopped/1000000000-123-0123456789abcdef0123456789abcdef.json"
            ),
        },
        "receipt_sha256",
    )
    live_result = {
        "schema": "muncho-full-canary-live-driver.v1",
        "ok": True,
        "release_sha": RELEASE_SHA,
        "full_canary_plan_sha256": plan_sha,
        "canary_run_id": "run-1",
        "evidence_path": evidence_path,
        "evidence_sha256": "1" * 64,
        "offline_invariant_receipt": invariant,
        "lifecycle_verification_receipt": lifecycle,
        "discord_ingress_claimed": False,
    }
    return _self_digest(
        {
            "schema": COORDINATOR_RECEIPT_SCHEMA,
            "ok": True,
            "release_sha": RELEASE_SHA,
            "coordinator_input_sha256": "c" * 64,
            "credential_prepare_approval_sha256": APPROVAL_SHA,
            "owner_subject_sha256": OWNER_SUBJECT_SHA,
            "ephemeral_admin_username": USERNAME,
            "temporary_admin_delete_required": True,
            "admin_session_closed": True,
            "full_canary_plan_sha256": plan_sha,
            "owner_approval_sha256": "d" * 64,
            "live_driver_result": live_result,
            "live_driver_receipt_sha256": hashlib.sha256(
                _canonical(live_result)
            ).hexdigest(),
            "bootstrap_login_password_disabled": True,
            "bootstrap_credential_removed": True,
            "discord_token_removed": True,
            "coordinator_process_lease_retired": True,
            "services_enabled": False,
            "completed_at_unix": 1_000,
        },
        "receipt_sha256",
    )


def _coordinator_failure() -> dict[str, object]:
    return _self_digest(
        {
            "schema": COORDINATOR_FAILURE_SCHEMA,
            "ok": False,
            "phase": "plan_and_final_approval",
            "error_code": "final_owner_approval_timeout",
            "release_sha": RELEASE_SHA,
            "coordinator_input_sha256": "c" * 64,
            "credential_prepare_approval_sha256": APPROVAL_SHA,
            "owner_subject_sha256": OWNER_SUBJECT_SHA,
            "ephemeral_admin_username": USERNAME,
            "full_canary_plan_sha256": "8" * 64,
            "cleanup_status": "complete",
            "recovery_material_preserved": False,
            "admin_session_closed": True,
            "bootstrap_login_password_disabled": True,
            "bootstrap_credential_removed": True,
            "discord_token_removed": True,
            "coordinator_process_lease_retired": True,
            "services_enabled": False,
        },
        "receipt_sha256",
    )


def _early_terminal_failure(
    *,
    bound: bool,
    admin_frame_disclosed: bool = False,
    discord_removed: bool = True,
) -> dict[str, object]:
    blocked = not discord_removed
    return _self_digest(
        {
            "schema": COORDINATOR_FAILURE_SCHEMA,
            "ok": False,
            "phase": "coordinator_input",
            "error_code": "coordinator_input_invalid",
            "release_sha": RELEASE_SHA if bound else None,
            "coordinator_input_sha256": "c" * 64 if bound else None,
            "credential_prepare_approval_sha256": APPROVAL_SHA if bound else None,
            "owner_subject_sha256": OWNER_SUBJECT_SHA if bound else None,
            "ephemeral_admin_username": USERNAME if bound else None,
            "full_canary_plan_sha256": "8" * 64 if admin_frame_disclosed else None,
            "cleanup_status": "cleanup_blocked" if blocked else "complete",
            "recovery_material_preserved": blocked,
            "admin_session_closed": True,
            "bootstrap_login_password_disabled": admin_frame_disclosed,
            "bootstrap_credential_removed": admin_frame_disclosed,
            "discord_token_removed": discord_removed,
            "coordinator_process_lease_retired": True,
            "services_enabled": False,
        },
        "receipt_sha256",
    )


def _no_recovery_required() -> dict[str, object]:
    value = _early_terminal_failure(bound=True)
    value.update({
        "phase": "recovery_preflight",
        "error_code": "coordinator_process_recovery_not_required",
        "bootstrap_login_password_disabled": False,
        "bootstrap_credential_removed": False,
        "discord_token_removed": True,
        "coordinator_process_lease_retired": True,
        "cleanup_status": "complete",
        "recovery_material_preserved": False,
    })
    value["receipt_sha256"] = hashlib.sha256(
        _canonical({
            key: item for key, item in value.items() if key != "receipt_sha256"
        })
    ).hexdigest()
    return value


def _token_only_recovery_required() -> dict[str, object]:
    value = _no_recovery_required()
    value.update({
        "cleanup_status": "cleanup_blocked",
        "recovery_material_preserved": True,
        "discord_token_removed": False,
        "coordinator_process_lease_retired": True,
    })
    value["receipt_sha256"] = hashlib.sha256(
        _canonical({
            key: item for key, item in value.items() if key != "receipt_sha256"
        })
    ).hexdigest()
    return value


def _original_run_process_lease() -> dict[str, object]:
    return _self_digest(
        {
            "schema": "muncho-full-canary-coordinator-process-lease.v1",
            "release_sha": RELEASE_SHA,
            "coordinator_input_sha256": "c" * 64,
            "credential_prepare_approval_sha256": APPROVAL_SHA,
            "owner_subject_sha256": OWNER_SUBJECT_SHA,
            "ephemeral_admin_username": USERNAME,
            "pid": 4321,
            "process_start_time_ticks": 9876,
            "boot_id_sha256": "4" * 64,
            "boot_time_ns": 123456,
            "module_origin": (
                f"/opt/muncho-canary-releases/{RELEASE_SHA}/venv/lib/"
                "python3.11/site-packages/gateway/"
                "canonical_full_canary_coordinator.py"
            ),
            "module_sha256": "5" * 64,
            "process_exe_sha256": "6" * 64,
            "process_cmdline_sha256": "7" * 64,
            "created_at_unix": 900,
        },
        "lease_sha256",
    )


def _recovery_gate(**changes: object) -> dict[str, object]:
    original_lease = _original_run_process_lease()
    value = {
        "schema": launcher.RECOVERY_GATE_SCHEMA,
        "ok": True,
        "state": "awaiting_owner_recovery_takeover_ack",
        "release_sha": RELEASE_SHA,
        "coordinator_input_sha256": "c" * 64,
        "credential_prepare_approval_sha256": APPROVAL_SHA,
        "owner_subject_sha256": OWNER_SUBJECT_SHA,
        "ephemeral_admin_username": USERNAME,
        "predecessor_kind": "run_process_lease",
        "predecessor_schema": "muncho-full-canary-coordinator-process-lease.v1",
        "predecessor_journal_sha256": "3" * 64,
        "predecessor_generation": 0,
        "original_run_process_lease_sha256": original_lease["lease_sha256"],
        "causal_recovery_state_sha256": "b" * 64,
        "target_pid": 4321,
        "target_process_start_time_ticks": 9876,
        "target_boot_id_sha256": "4" * 64,
        "target_boot_time_ns": 123456,
        "target_uid": 0,
        "target_gid": 0,
        "target_module_origin": (
            f"/opt/muncho-canary-releases/{RELEASE_SHA}/venv/lib/"
            "python3.11/site-packages/gateway/"
            "canonical_full_canary_coordinator.py"
        ),
        "target_module_sha256": "5" * 64,
        "target_process_exe_sha256": "6" * 64,
        "target_process_cmdline_sha256": "7" * 64,
        "target_process_identity_state": "exact_alive",
        "discord_token_state": "installed",
        "discord_token_install_receipt_sha256": _discord_receipt()["receipt_sha256"],
        "discord_retirement_receipt_sha256": None,
        "token_device": 10,
        "token_inode": 11,
        "db_secret_accepted": False,
        "frame_schema": launcher.RECOVERY_ACK_FRAME_SCHEMA,
        "observed_at_unix": 1_000,
        "expires_at_unix": 1_300,
    }
    value.update(changes)
    return _self_digest(value, "gate_sha256")


def _recovery_secret_gate(gate, ack, **changes: object) -> dict[str, object]:
    value = {
        "schema": launcher.RECOVERY_SECRET_GATE_SCHEMA,
        "ok": True,
        "state": "awaiting_recovery_admin_credential",
        "release_sha": RELEASE_SHA,
        "coordinator_input_sha256": "c" * 64,
        "credential_prepare_approval_sha256": APPROVAL_SHA,
        "owner_subject_sha256": OWNER_SUBJECT_SHA,
        "ephemeral_admin_username": USERNAME,
        "recovery_takeover_gate_sha256": gate["gate_sha256"],
        "owner_recovery_takeover_ack_sha256": ack["ack_sha256"],
        "predecessor_kind": gate["predecessor_kind"],
        "predecessor_journal_sha256": gate["predecessor_journal_sha256"],
        "predecessor_generation": gate["predecessor_generation"],
        "original_run_process_lease_sha256": gate["original_run_process_lease_sha256"],
        "causal_recovery_state_sha256": gate["causal_recovery_state_sha256"],
        "recovery_worker_lease_sha256": "1" * 64,
        "recovery_worker_state": "admin_authority_may_be_in_use",
        "recovery_worker_transition_seq": 2,
        "recovery_worker_pid": 5432,
        "recovery_worker_process_start_time_ticks": 8765,
        "recovery_worker_boot_id_sha256": "2" * 64,
        "recovery_worker_boot_time_ns": 123456,
        "recovery_worker_uid": 0,
        "recovery_worker_gid": 0,
        "recovery_worker_module_origin": (
            f"/opt/muncho-canary-releases/{RELEASE_SHA}/venv/lib/"
            "python3.11/site-packages/gateway/"
            "canonical_full_canary_coordinator.py"
        ),
        "recovery_worker_module_sha256": "d" * 64,
        "recovery_worker_process_exe_sha256": "e" * 64,
        "recovery_worker_process_cmdline_sha256": "f" * 64,
        "database_host": launcher.DATABASE_HOST,
        "database_port": launcher.DATABASE_PORT,
        "database_name": launcher.DATABASE_NAME,
        "tls_server_name": "db.europe-west3.sql.goog",
        "tls_peer_certificate_sha256": "0" * 64,
        "gate_nonce_sha256": "9" * 64,
        "admin_frame_schema": launcher.RECOVERY_ADMIN_FRAME_SCHEMA,
        "expires_at_unix": 1_300,
    }
    value.update(changes)
    return _self_digest(value, "gate_sha256")


def _recovery_completion(gate, ack, secret_gate, **changes: object):
    original_lease = _original_run_process_lease()
    value = {
        "schema": launcher.RECOVERY_WORKER_COMPLETION_SCHEMA,
        "ok": False,
        "state": "cleanup_complete_awaiting_worker_exit",
        "release_sha": RELEASE_SHA,
        "coordinator_input_sha256": "c" * 64,
        "credential_prepare_approval_sha256": APPROVAL_SHA,
        "owner_subject_sha256": OWNER_SUBJECT_SHA,
        "ephemeral_admin_username": USERNAME,
        "original_run_process_lease": original_lease,
        "original_run_process_lease_sha256": gate["original_run_process_lease_sha256"],
        "causal_recovery_state_sha256": gate["causal_recovery_state_sha256"],
        "predecessor_kind": gate["predecessor_kind"],
        "predecessor_journal_sha256": gate["predecessor_journal_sha256"],
        "predecessor_generation": gate["predecessor_generation"],
        "recovery_generation": gate["predecessor_generation"] + 1,
        "recovery_takeover_gate_sha256": gate["gate_sha256"],
        "owner_recovery_takeover_ack_sha256": ack["ack_sha256"],
        "recovery_worker_lease_sha256": secret_gate["recovery_worker_lease_sha256"],
        **{
            name: secret_gate[name]
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
        },
        "predecessor_termination_proven": True,
        "predecessor_process_lock_acquired": True,
        "predecessor_journal_replaced": True,
        "canonical_stop_receipt_sha256": None,
        "preplan_stopped_report_sha256": "8" * 64,
        "preclaim_reconciliation_receipt_sha256": None,
        "preclaim_reconciliation_state": None,
        "admin_frame_zeroized": True,
        "admin_session_closed": True,
        "migration_owner_membership_removed": True,
        "bootstrap_login_password_disabled": True,
        "bootstrap_credential_removed": True,
        "discord_token_removed": True,
        "discord_install_receipt_removed": True,
        "discord_retirement_receipt_sha256": "9" * 64,
        "services_stopped_proven": True,
        "services_enabled": False,
        "recovery_worker_exit_proven": False,
        "safe_to_delete_temporary_admin": False,
        "cleanup_completed_at_unix": 1_000,
    }
    value.update(changes)
    return _self_digest(value, "completion_sha256")


def _recovery_receipt(
    gate,
    ack,
    *,
    secret_gate=None,
    completion=None,
    **changes: object,
) -> dict[str, object]:
    secret_gate = secret_gate or _recovery_secret_gate(gate, ack)
    completion = completion or _recovery_completion(gate, ack, secret_gate)
    value = {
        key: item
        for key, item in completion.items()
        if key
        not in {
            "schema",
            "ok",
            "state",
            "completion_sha256",
            "recovery_worker_exit_proven",
            "safe_to_delete_temporary_admin",
        }
    }
    value.update({
        "schema": launcher.RECOVERY_RECEIPT_SCHEMA,
        "ok": True,
        "state": "recovered",
        "recovery_worker_completion_sha256": completion["completion_sha256"],
        "recovery_worker_lock_acquired": True,
        "recovery_worker_exit_proven": True,
        "safe_to_delete_temporary_admin": True,
        "finalized_at_unix": 1_000,
    })
    value.update(changes)
    return _self_digest(value, "receipt_sha256")


def _recovery_finalize_pending(completion, **changes: object):
    value = {
        "schema": launcher.RECOVERY_FINALIZE_PENDING_RECEIPT_SCHEMA,
        "ok": False,
        "state": "recovery_finalization_pending_no_secret",
        "release_sha": completion["release_sha"],
        "coordinator_input_sha256": completion["coordinator_input_sha256"],
        "credential_prepare_approval_sha256": completion[
            "credential_prepare_approval_sha256"
        ],
        "owner_subject_sha256": completion["owner_subject_sha256"],
        "ephemeral_admin_username": completion["ephemeral_admin_username"],
        "recovery_worker_completion_sha256": completion["completion_sha256"],
        **{
            name: completion[name]
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
        },
        "completion_admin_authority_may_have_been_used": True,
        "completion_admin_frame_zeroized": True,
        "completion_admin_session_closed": True,
        "worker_identity_state": "exact_alive",
        "target_signal_attempted_by_finalizer": True,
        "target_termination_proven_by_finalizer": False,
        "process_lock_acquired_by_finalizer": False,
        "completion_cas_attempted_by_finalizer": False,
        "completion_cas_succeeded_by_finalizer": False,
        "observed_journal_sha256": completion["completion_sha256"],
        "secret_gate_emitted_by_finalizer": False,
        "admin_frame_bytes_received_by_finalizer": 0,
        "admin_session_opened_by_finalizer": False,
        "admin_credential_mutation_performed_by_finalizer": False,
        "retryable": True,
        "completed_at_unix": 1_000,
    }
    value.update(changes)
    return _self_digest(value, "receipt_sha256")


class _RemoteSession:
    def __init__(self, gate, receipt, events, label, *, next_receipt=None) -> None:
        self.gate = gate
        self.receipt = receipt
        self.next_receipt = next_receipt
        self.events = events
        self.label = label
        self.frame: bytes | None = None
        self.raw_frame = None
        self.last_output = None
        self.termination_proven = False
        self.terminal_receipt_validated = False
        self.bounded_timeouts = []

    def read_gate(self):
        self.events.append(f"{self.label}_gate")
        self.last_output = self.gate
        return self.gate

    def finish(self, frame: bytes):
        self.events.append(f"{self.label}_finish")
        self.raw_frame = frame
        self.frame = bytes(frame)
        self.last_output = self.receipt
        return self.receipt

    def finish_before(self, frame, *, write_guard, on_first_write):
        write_guard()
        on_first_write()
        return self.finish(frame)

    def exchange(self, frame: bytes):
        raise AssertionError(f"{self.label} does not support a non-closing exchange")

    def cancel_no_secret(self):
        self.events.append(f"{self.label}_cancel")
        assert self.frame is None
        self.last_output = _approval_cancel_receipt()
        return self.last_output

    def read_next(self):
        self.events.append(f"{self.label}_next")
        self.last_output = self.next_receipt
        return self.next_receipt

    def read_next_bounded(self, timeout_seconds):
        self.events.append(f"{self.label}_bounded_next")
        self.bounded_timeouts.append(timeout_seconds)
        self.last_output = self.next_receipt
        return self.next_receipt

    def mark_validated(self, receipt):
        self.events.append(f"{self.label}_validated")
        assert receipt == self.last_output
        self.termination_proven = True
        self.terminal_receipt_validated = True

    def close(self):
        self.events.append(f"{self.label}_close")
        if not self.terminal_receipt_validated:
            raise OwnerLauncherError("remote_termination_unconfirmed")


class _FirstByteProbeSession(_RemoteSession):
    """Capture the mutable frame reference without claiming any byte write."""

    def __init__(self, *args, before_guard=lambda: None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.before_guard = before_guard
        self.attempted_frame = None
        self.first_write_called = False

    def finish_before(self, frame, *, write_guard, on_first_write):
        self.attempted_frame = frame
        self.before_guard()
        write_guard()
        self.first_write_called = True
        on_first_write()
        return self.finish(frame)


class _Transport:
    def __init__(self, gate: Mapping[str, object]) -> None:
        self.gate = gate
        self.events: list[str] = []
        self.discord_session = _RemoteSession(
            _discord_gate(), _discord_receipt(), self.events, "discord"
        )
        self.run_session = _RemoteSession(
            _coordinator_gate(),
            _approval_request(),
            self.events,
            "coordinator",
            next_receipt=_coordinator_receipt(),
        )
        self.approval_session = _RemoteSession(
            _approval_request(),
            _approval_install_receipt(),
            self.events,
            "approval",
        )
        self.retirement_session = _RemoteSession(
            _discord_retirement_gate(),
            _discord_retirement_receipt(),
            self.events,
            "retirement",
        )

    def preflight_recovery(self, release_sha: str) -> Mapping[str, object]:
        assert release_sha == RELEASE_SHA
        self.events.append("preflight_recovery")
        return _no_recovery_required()

    def preflight_owner_launch(self, release_sha: str) -> Mapping[str, object]:
        assert release_sha == RELEASE_SHA
        self.events.append("preflight")
        return self.gate

    def open_discord_install(self, release_sha: str):
        assert release_sha == RELEASE_SHA
        self.events.append("open_discord")
        return self.discord_session

    def open_run(self, release_sha: str):
        assert release_sha == RELEASE_SHA
        self.events.append("open_coordinator")
        return self.run_session

    def open_final_approval_install(self, release_sha: str):
        assert release_sha == RELEASE_SHA
        self.events.append("open_approval")
        return self.approval_session

    def open_discord_retirement(self, release_sha: str):
        assert release_sha == RELEASE_SHA
        self.events.append("open_retirement")
        return self.retirement_session

    def open_recovery(self, release_sha: str):
        raise AssertionError("stale process recovery was not expected")

    def open_recovery_finalizer(self, release_sha: str):
        raise AssertionError("recovery finalization was not expected")

    @property
    def admin_frame(self):
        return self.run_session.frame

    @property
    def discord_frame(self):
        return self.discord_session.frame


class _LostDiscordInstallResponseTransport(_Transport):
    def __init__(self, gate: Mapping[str, object]) -> None:
        super().__init__(gate)
        original_finish = self.discord_session.finish

        def finish_then_lose_response(frame):
            original_finish(frame)
            raise OwnerLauncherError("iap_ssh_ended_without_receipt")

        self.discord_session.finish = finish_then_lose_response


class _RecoverySession(_RemoteSession):
    def __init__(self, gate, events) -> None:
        super().__init__(gate, None, events, "recovery")
        self.ack = None
        self.ack_frame = None
        self.raw_ack_frame = None
        self.secret_gate = None
        self.completion = None

    def exchange(self, frame: bytes):
        self.events.append("recovery_exchange")
        self.raw_ack_frame = frame
        self.ack_frame = bytes(frame)
        assert frame.startswith(launcher.RECOVERY_ACK_FRAME_MAGIC)
        payload_length = struct.unpack(">I", frame[4:8])[0]
        payload_end = 8 + payload_length
        assert payload_end == len(frame)
        self.ack = json.loads(frame[8:payload_end])
        self.secret_gate = _recovery_secret_gate(self.gate, self.ack)
        self.last_output = self.secret_gate
        return self.secret_gate

    def finish(self, frame: bytes):
        self.events.append("recovery_finish")
        assert self.secret_gate is not None
        self.raw_frame = frame
        self.frame = bytes(frame)
        magic, gate_digest, nonce_digest, username_size, password_size = struct.unpack(
            "!4s32s32sHI", frame[:74]
        )
        assert magic == launcher.RECOVERY_ADMIN_FRAME_MAGIC
        assert gate_digest.hex() == self.secret_gate["gate_sha256"]
        assert nonce_digest.hex() == self.secret_gate["gate_nonce_sha256"]
        assert frame[74 : 74 + username_size].decode() == USERNAME
        assert len(frame) == 74 + username_size + password_size
        self.completion = _recovery_completion(
            self.gate,
            self.ack,
            self.secret_gate,
        )
        self.last_output = self.completion
        return self.completion


class _RecoveryFirstByteProbeSession(_RecoverySession):
    def __init__(self, gate, events, *, before_guard=lambda: None) -> None:
        super().__init__(gate, events)
        self.before_guard = before_guard
        self.attempted_frame = None
        self.first_write_called = False

    def finish_before(self, frame, *, write_guard, on_first_write):
        self.attempted_frame = frame
        self.before_guard()
        write_guard()
        self.first_write_called = True
        on_first_write()
        return self.finish(frame)


class _RecoveryFinalizerSession(_RemoteSession):
    def __init__(self, receipt, events) -> None:
        super().__init__(receipt, None, events, "recovery_finalizer")


class _RecoveryTransport(_Transport):
    def __init__(self, gate: Mapping[str, object]) -> None:
        super().__init__(gate)
        self.recovery_gate = _recovery_gate()
        self.recovery_session = _RecoverySession(
            self.recovery_gate,
            self.events,
        )

    def preflight_recovery(self, release_sha: str) -> Mapping[str, object]:
        assert release_sha == RELEASE_SHA
        self.events.append("preflight_recovery")
        return self.recovery_gate

    def open_recovery(self, release_sha: str):
        assert release_sha == RELEASE_SHA
        self.events.append("open_recovery")
        return self.recovery_session

    def open_recovery_finalizer(self, release_sha: str):
        assert release_sha == RELEASE_SHA
        self.events.append("open_recovery_finalizer")
        assert self.recovery_session.ack is not None
        assert self.recovery_session.secret_gate is not None
        assert self.recovery_session.completion is not None
        receipt = _recovery_receipt(
            self.recovery_session.gate,
            self.recovery_session.ack,
            secret_gate=self.recovery_session.secret_gate,
            completion=self.recovery_session.completion,
        )
        self.recovery_session.receipt = receipt
        return _RecoveryFinalizerSession(receipt, self.events)


class _PersistedRecoveryTransport(_Transport):
    def __init__(self, gate: Mapping[str, object]) -> None:
        super().__init__(gate)
        recovery_gate = _recovery_gate()
        ack = launcher.build_recovery_ack(
            recovery_gate,
            now_unix=1_000,
            nonce=b"persisted-recovery-nonce-value",
        )
        secret_gate = _recovery_secret_gate(recovery_gate, ack)
        completion = _recovery_completion(recovery_gate, ack, secret_gate)
        self.persisted_receipt = _recovery_receipt(
            recovery_gate,
            ack,
            secret_gate=secret_gate,
            completion=completion,
        )

    def preflight_recovery(self, release_sha: str) -> Mapping[str, object]:
        assert release_sha == RELEASE_SHA
        self.events.append("preflight_recovery")
        return self.persisted_receipt

    def open_recovery(self, release_sha: str):
        raise AssertionError("terminal recovery must not be repeated")

    def open_recovery_finalizer(self, release_sha: str):
        raise AssertionError("terminal recovery must not be finalized again")


class _PersistedCompletionTransport(_Transport):
    def __init__(self, gate: Mapping[str, object], *, pending: bool = False) -> None:
        super().__init__(gate)
        recovery_gate = _recovery_gate()
        ack = launcher.build_recovery_ack(
            recovery_gate,
            now_unix=1_000,
            nonce=b"persisted-completion-nonce-value",
        )
        secret_gate = _recovery_secret_gate(recovery_gate, ack)
        self.completion = _recovery_completion(recovery_gate, ack, secret_gate)
        self.final_receipt = _recovery_receipt(
            recovery_gate,
            ack,
            secret_gate=secret_gate,
            completion=self.completion,
        )
        self.pending = pending
        self.finalizer_calls = 0

    def preflight_recovery(self, release_sha: str) -> Mapping[str, object]:
        self.events.append("preflight_recovery")
        return self.completion

    def open_recovery(self, release_sha: str):
        raise AssertionError("persisted completion must use no-secret finalizer")

    def open_recovery_finalizer(self, release_sha: str):
        self.finalizer_calls += 1
        value = (
            _recovery_finalize_pending(self.completion)
            if self.pending
            else self.final_receipt
        )
        return _RecoveryFinalizerSession(value, self.events)


class _TokenOnlyRecoveryTransport(_Transport):
    def preflight_recovery(self, release_sha: str) -> Mapping[str, object]:
        assert release_sha == RELEASE_SHA
        self.events.append("preflight_recovery")
        return _token_only_recovery_required()


class _SecretReader:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.last: bytearray | None = None

    def read_discord_token(self) -> bytearray:
        self.events.append("read_discord")
        self.last = bytearray(DISCORD_TOKEN)
        return self.last


class _OwnerIdentity:
    def __init__(self, events: list[str], *, observed: str = OWNER_SUBJECT_SHA) -> None:
        self.events = events
        self.observed = observed
        self.bound: str | None = None

    def bind_approved_subject(self, expected_sha256: str) -> None:
        self.events.append("bind_owner")
        if expected_sha256 != self.observed:
            raise OwnerLauncherError("approved_owner_identity_mismatch")
        self.bound = expected_sha256

    def require_stable(self) -> None:
        self.events.append("owner_stable")
        if self.bound != self.observed:
            raise OwnerLauncherError("approved_owner_identity_changed")


class _FinalApprovalSource:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def read_final_approval(self, request):
        self.events.append("read_final_approval")
        assert request == _approval_request()
        return _final_approval()


class _SqlAdmin:
    def __init__(self, events: list[str], *, cleanup_fails: bool = False) -> None:
        self.events = events
        self.cleanup_fails = cleanup_fails
        self.password: str | None = None
        self.username: str | None = None
        self.ambiguity_observed = False
        self.reconciliation_proven = False

    def begin_mutation_observation(
        self,
        *,
        expected_owner_subject_sha256: str | None = None,
    ) -> None:
        assert expected_owner_subject_sha256 in {None, OWNER_SUBJECT_SHA}
        pass

    def mutation_reconciliation_required(self) -> bool:
        return False

    def require_current_authority(self, username: str) -> None:
        assert username == USERNAME

    def reconciliation_evidence(self):
        return {
            "mutation_ambiguity_observed": self.ambiguity_observed,
            "reconciliation_proven": self.reconciliation_proven,
            "reconciliation_evidence_sha256": (
                "a" * 64 if self.reconciliation_proven else None
            ),
            "quiet_window_seconds": 180.0 if self.reconciliation_proven else None,
            "response_known_candidate_observed": (
                False if self.reconciliation_proven else None
            ),
            "post_baseline_authority_operation_count": (
                0 if self.reconciliation_proven else None
            ),
        }

    def require_absent(self, username: str) -> None:
        self.events.append("require_absent")
        assert username == USERNAME

    def create(self, username: str, password: str) -> None:
        self.events.append("create_admin")
        self.username = username
        self.password = password

    def create_or_rotate_recovery(self, username: str, password: str) -> None:
        self.events.append("create_or_rotate_recovery_admin")
        self.username = username
        self.password = password

    def delete_and_confirm_absent(self, username: str) -> None:
        self.events.append("delete_admin")
        assert username == USERNAME
        if self.cleanup_fails:
            raise CleanupBlocked()
        self.reconciliation_proven = True

    def reconcile_ambiguous_mutation_and_confirm_absent(
        self,
        username: str,
    ) -> None:
        self.ambiguity_observed = True
        self.delete_and_confirm_absent(username)


class _AuthorityDriftSqlAdmin(_SqlAdmin):
    def __init__(self, events: list[str], *, drift_operation: str) -> None:
        super().__init__(events)
        self.drift_operation = drift_operation
        self.ambiguous = False
        self.candidate_observed: bool | None = None
        self.authority_operation_count: int | None = None

    def begin_mutation_observation(
        self,
        *,
        expected_owner_subject_sha256: str | None = None,
    ) -> None:
        assert expected_owner_subject_sha256 in {None, OWNER_SUBJECT_SHA}
        self.ambiguous = False
        self.ambiguity_observed = False
        self.reconciliation_proven = False
        self.candidate_observed = None
        self.authority_operation_count = None

    def mutation_reconciliation_required(self) -> bool:
        return self.ambiguous

    def require_current_authority(self, username: str) -> None:
        assert username == USERNAME
        self.events.append(f"inject_{self.drift_operation.lower()}")
        self.ambiguous = True
        self.ambiguity_observed = True
        raise OwnerLauncherError("cloud_sql_mutation_evidence_unconfirmed")

    def reconciliation_evidence(self):
        return {
            "mutation_ambiguity_observed": self.ambiguity_observed,
            "reconciliation_proven": self.reconciliation_proven,
            "reconciliation_evidence_sha256": (
                "d" * 64 if self.reconciliation_proven else None
            ),
            "quiet_window_seconds": 180.0 if self.reconciliation_proven else None,
            "response_known_candidate_observed": self.candidate_observed,
            "post_baseline_authority_operation_count": (self.authority_operation_count),
        }

    def delete_and_confirm_absent(self, username: str) -> None:
        self.events.append("delete_admin")
        assert username == USERNAME
        self.reconciliation_proven = True
        self.candidate_observed = False
        self.authority_operation_count = 0

    def reconcile_ambiguous_mutation_and_confirm_absent(
        self,
        username: str,
    ) -> None:
        assert username == USERNAME
        assert self.ambiguous is True
        self.events.append("reconcile_original_mutation_baseline")
        self.ambiguous = False
        self.ambiguity_observed = True
        self.reconciliation_proven = True
        self.candidate_observed = True
        self.authority_operation_count = (
            2 if self.drift_operation == "UPDATE_USER" else 1
        )


class _LostCandidateSqlAdmin(_AuthorityDriftSqlAdmin):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events, drift_operation="CREATE_USER")

    def create(self, username: str, password: str) -> None:
        self.events.append("create_response_lost_without_candidate")
        self.username = username
        self.password = password
        self.ambiguous = True
        self.ambiguity_observed = True
        raise OwnerLauncherError("google_api_unavailable")

    def reconcile_ambiguous_mutation_and_confirm_absent(
        self,
        username: str,
    ) -> None:
        assert username == USERNAME
        assert self.ambiguous is True
        self.events.append("reconcile_zero_candidate_absence")
        self.ambiguous = False
        self.ambiguity_observed = True
        self.reconciliation_proven = True
        self.candidate_observed = False
        self.authority_operation_count = 0


class _ScopedReconciliationSqlAdmin(_SqlAdmin):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.cleanup_number = 0
        self.current_digest: str | None = None
        self.phase_mutated = False
        self.candidate_observed: bool | None = None
        self.authority_operation_count: int | None = None

    def begin_mutation_observation(
        self,
        *,
        expected_owner_subject_sha256: str | None = None,
    ) -> None:
        assert expected_owner_subject_sha256 in {None, OWNER_SUBJECT_SHA}
        self.reconciliation_proven = False
        self.current_digest = None
        self.phase_mutated = False
        self.candidate_observed = None
        self.authority_operation_count = None

    def create(self, username: str, password: str) -> None:
        super().create(username, password)
        self.phase_mutated = True

    def create_or_rotate_recovery(self, username: str, password: str) -> None:
        super().create_or_rotate_recovery(username, password)
        self.phase_mutated = True

    def delete_and_confirm_absent(self, username: str) -> None:
        self.events.append("delete_admin")
        assert username == USERNAME
        self.cleanup_number += 1
        self.current_digest = f"{self.cleanup_number:x}" * 64
        self.reconciliation_proven = True
        self.candidate_observed = self.phase_mutated
        self.authority_operation_count = 1 if self.phase_mutated else 0

    def reconciliation_evidence(self):
        return {
            "mutation_ambiguity_observed": False,
            "reconciliation_proven": self.reconciliation_proven,
            "reconciliation_evidence_sha256": self.current_digest,
            "quiet_window_seconds": 180.0 if self.reconciliation_proven else None,
            "response_known_candidate_observed": self.candidate_observed,
            "post_baseline_authority_operation_count": (self.authority_operation_count),
        }


def test_gate_derives_exact_unique_username_and_self_digest():
    gate = validate_owner_launch_gate(
        _gate(), expected_release_sha=RELEASE_SHA, now_unix=1_000
    )

    assert gate["admin_username"] == USERNAME


def test_owner_approval_request_binds_exact_cutoff_and_all_expiry_authorities():
    request = _approval_request()

    assert (
        launcher.validate_owner_approval_request(
            request,
            gate=_coordinator_gate(),
            now_unix=1_000,
        )
        == request
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"owner_input_cutoff_unix": 1_209},
        {"final_approval_transmit_margin_seconds": 29},
        {"approval_deadline_unix": 1_239},
        {"credential_approval_expires_at_unix": 1_399},
        {"hba_expires_at_unix": 1_269},
        {"max_wait_seconds": 239},
    ],
)
def test_owner_approval_request_rejects_self_digested_timing_drift(changes):
    request = _approval_request()
    request.update(changes)
    request.pop("request_sha256")
    request = _self_digest(request, "request_sha256")

    with pytest.raises(OwnerLauncherError, match="invalid_owner_approval_request"):
        launcher.validate_owner_approval_request(
            request,
            gate=_coordinator_gate(),
            now_unix=1_000,
        )


def test_owner_approval_request_rejects_self_consistent_241_second_window():
    assert launcher.FINAL_APPROVAL_MAX_WAIT_SECONDS == 240
    request = _approval_request()
    request.update({
        "approval_deadline_unix": 1_241,
        "owner_input_cutoff_unix": 1_211,
        "max_wait_seconds": 241,
    })
    request.pop("request_sha256")
    request = _self_digest(request, "request_sha256")

    with pytest.raises(OwnerLauncherError, match="invalid_owner_approval_request"):
        launcher.validate_owner_approval_request(
            request,
            gate=_coordinator_gate(),
            now_unix=1_000,
        )


def test_final_approval_cancel_receipt_is_exact_no_secret_terminal_truth():
    receipt = _approval_cancel_receipt()

    assert (
        launcher.validate_final_approval_cancel_receipt(
            receipt,
            request=_approval_request(),
            now_unix=1_000,
        )
        == receipt
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"frame_bytes_received": 1},
        {"approval_path_matches_prior": False},
        {"new_owner_approval_installed": True},
        {"owner_approval_mutation_performed_by_this_helper": True},
        {"cancelled_at_unix": 1_241},
    ],
)
def test_final_approval_cancel_receipt_rejects_self_digested_false_proof(changes):
    receipt = _approval_cancel_receipt()
    receipt.update(changes)
    receipt.pop("receipt_sha256")
    receipt = _self_digest(receipt, "receipt_sha256")

    with pytest.raises(
        OwnerLauncherError,
        match="invalid_final_approval_cancel_receipt",
    ):
        launcher.validate_final_approval_cancel_receipt(
            receipt,
            request=_approval_request(),
            now_unix=1_241,
        )


@pytest.mark.parametrize(
    "receipt",
    [
        _approval_cancel_receipt(
            approval_request_artifact_state="matching_expired",
            approval_request_remains_active=False,
            cancelled_at_unix=1_241,
        ),
        _approval_cancel_receipt(
            approval_request_artifact_state="retired_absent",
            approval_request_present=False,
            approval_request_remains_active=False,
            observed_approval_request_file_sha256=None,
            staged_plan_artifact_state="retired_absent",
            staged_plan_present=False,
            observed_staged_plan_file_sha256=None,
            cancelled_at_unix=1_241,
        ),
    ],
)
def test_final_approval_cancel_accepts_expired_or_retired_state_after_deadline(
    receipt,
):
    assert (
        launcher.validate_final_approval_cancel_receipt(
            receipt,
            request=_approval_request(),
            now_unix=1_241,
        )
        == receipt
    )


def test_final_approval_cancel_validates_dedicated_state_conflict_truth():
    receipt = _approval_cancel_receipt(
        state="cancelled_no_secret_state_conflict",
        approval_request_artifact_state="superseded",
        approval_request_remains_active=False,
        observed_approval_request_file_sha256="f" * 64,
    )

    assert (
        launcher.validate_final_approval_cancel_receipt(
            receipt,
            request=_approval_request(),
            now_unix=1_000,
        )["state"]
        == "cancelled_no_secret_state_conflict"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {
            "staged_plan_artifact_state": "retired_absent",
            "staged_plan_present": False,
            "observed_staged_plan_file_sha256": None,
        },
        {
            "approval_request_artifact_state": "retired_absent",
            "approval_request_present": False,
            "approval_request_remains_active": False,
            "observed_approval_request_file_sha256": None,
        },
    ],
)
def test_final_approval_cancel_rejects_clean_state_for_incoherent_artifact_pair(
    changes,
):
    with pytest.raises(
        OwnerLauncherError,
        match="invalid_final_approval_cancel_receipt",
    ):
        launcher.validate_final_approval_cancel_receipt(
            _approval_cancel_receipt(**changes),
            request=_approval_request(),
            now_unix=1_000,
        )


def test_final_approval_cancel_accepts_incoherent_pair_only_as_state_conflict():
    receipt = _approval_cancel_receipt(
        state="cancelled_no_secret_state_conflict",
        staged_plan_artifact_state="retired_absent",
        staged_plan_present=False,
        observed_staged_plan_file_sha256=None,
    )

    assert (
        launcher.validate_final_approval_cancel_receipt(
            receipt,
            request=_approval_request(),
            now_unix=1_000,
        )
        == receipt
    )


def test_launch_records_cancel_state_conflict_without_false_remote_unconfirmed():
    transport = _Transport(_gate())

    def cancel_conflict():
        transport.events.append("approval_cancel")
        transport.approval_session.last_output = _approval_cancel_receipt(
            state="cancelled_no_secret_state_conflict",
            approval_request_artifact_state="superseded",
            approval_request_remains_active=False,
            observed_approval_request_file_sha256="f" * 64,
        )
        return transport.approval_session.last_output

    transport.approval_session.cancel_no_secret = cancel_conflict

    class FailingDecision:
        def read_final_approval(self, _request):
            raise OwnerLauncherError("owner_decision_unavailable")

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=FailingDecision(),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["primary_failure_code"] == "owner_decision_unavailable"
    assert receipt["failure_code"] == "final_approval_cancel_state_conflict"
    assert receipt["cleanup_failure_codes"] == ["final_approval_cancel_state_conflict"]
    assert receipt["final_approval_cancel_receipt_sha256"] is not None
    assert "remote_termination_unconfirmed" not in receipt["cleanup_failure_codes"]


@pytest.mark.parametrize(
    "changes",
    [
        {
            "state": "cancelled_no_secret_state_conflict",
            "approval_request_artifact_state": "superseded",
            "approval_request_remains_active": False,
        },
        {
            "state": "cancelled_no_secret_state_conflict",
            "approval_request_artifact_state": "superseded",
            "approval_request_present": False,
            "approval_request_remains_active": False,
            "observed_approval_request_file_sha256": None,
        },
        {
            "state": "cancelled_no_secret_state_conflict",
            "approval_request_artifact_state": "drifted",
            "approval_request_remains_active": False,
            "observed_approval_request_file_sha256": "f" * 64,
        },
        {
            "state": "cancelled_no_secret_state_conflict",
            "staged_plan_artifact_state": "superseded",
        },
        {
            "state": "cancelled_no_secret_state_conflict",
            "staged_plan_artifact_state": "drifted",
            "observed_staged_plan_file_sha256": "f" * 64,
        },
        {
            "state": "cancelled_no_secret_state_conflict",
            "owner_approval_artifact_state": "drifted",
            "approval_path_matches_prior": False,
            "new_owner_approval_installed": None,
        },
    ],
)
def test_final_approval_cancel_rejects_forged_conflict_artifact_matrix(changes):
    with pytest.raises(
        OwnerLauncherError,
        match="invalid_final_approval_cancel_receipt",
    ):
        launcher.validate_final_approval_cancel_receipt(
            _approval_cancel_receipt(**changes),
            request=_approval_request(),
            now_unix=1_000,
        )


def test_final_owner_approval_decision_after_input_cutoff_is_rejected():
    approval = _final_approval()
    approval["approved_at_unix"] = 1_211

    with pytest.raises(OwnerLauncherError, match="invalid_final_owner_approval"):
        launcher.validate_final_owner_approval(
            approval,
            request=_approval_request(),
            now_unix=1_211,
        )

    wrong_username = _gate(admin_username="muncho_canary_admin_ffffffffffffffff")
    with pytest.raises(OwnerLauncherError, match="invalid_owner_gate"):
        validate_owner_launch_gate(
            wrong_username, expected_release_sha=RELEASE_SHA, now_unix=1_000
        )


@pytest.mark.parametrize(
    "gate",
    [
        _gate(ok=False),
        _gate(expires_at_unix=1_000),
        _gate(release_sha="d" * 40),
        {**_gate(), "unexpected": True},
    ],
)
def test_no_fresh_mutation_or_secret_read_before_exact_approval_input(gate):
    transport = _Transport(gate)
    secret_reader = _SecretReader(transport.events)
    sql_admin = _SqlAdmin(transport.events)

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=secret_reader,
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: transport.events.append(
            "publish_approval_request"
        ),
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: transport.events.append("harden_secrets"),
        provenance_guard=lambda _release: None,
    )

    assert receipt["ok"] is False
    assert transport.events == [
        "harden_secrets",
        "preflight_recovery",
        "bind_owner",
        "owner_stable",
        "delete_admin",
        "owner_stable",
        "preflight",
    ]
    assert sql_admin.username is None
    assert secret_reader.last is None


def test_secrets_are_stdin_frames_only_and_absent_from_canonical_receipt():
    transport = _Transport(_gate())
    secret_reader = _SecretReader(transport.events)
    sql_admin = _SqlAdmin(transport.events)
    password = bytearray(PASSWORD)

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=secret_reader,
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: transport.events.append(
            "publish_approval_request"
        ),
        now=lambda: 1_000,
        password_factory=lambda: password,
        secret_hardener=lambda: transport.events.append("harden_secrets"),
        provenance_guard=lambda _release: None,
    )

    assert receipt["ok"] is True
    assert receipt["temporary_admin_absent"] is True
    assert transport.events == [
        "harden_secrets",
        "preflight_recovery",
        "bind_owner",
        "owner_stable",
        "delete_admin",
        "owner_stable",
        "preflight",
        "bind_owner",
        "owner_stable",
        "open_discord",
        "discord_gate",
        "owner_stable",
        "read_discord",
        "owner_stable",
        "discord_finish",
        "discord_validated",
        "owner_stable",
        "discord_close",
        "owner_stable",
        "require_absent",
        "owner_stable",
        "owner_stable",
        "create_admin",
        "owner_stable",
        "open_coordinator",
        "coordinator_gate",
        "owner_stable",
        "owner_stable",
        "owner_stable",
        "coordinator_finish",
        "owner_stable",
        "publish_approval_request",
        "open_approval",
        "approval_gate",
        "owner_stable",
        "read_final_approval",
        "owner_stable",
        "owner_stable",
        "approval_finish",
        "approval_validated",
        "approval_close",
        "owner_stable",
        "coordinator_next",
        "coordinator_validated",
        "owner_stable",
        "coordinator_close",
        "delete_admin",
        "owner_stable",
    ]
    assert transport.admin_frame is not None
    assert transport.admin_frame.startswith(ADMIN_FRAME_MAGIC)
    username_length = struct.unpack(">H", transport.admin_frame[4:6])[0]
    password_length = struct.unpack(">I", transport.admin_frame[6:10])[0]
    assert transport.admin_frame[10 : 10 + username_length] == USERNAME.encode()
    assert transport.admin_frame[10 + username_length :] == PASSWORD
    assert password_length == len(PASSWORD)
    assert (
        transport.discord_frame
        == DISCORD_FRAME_MAGIC + struct.pack(">I", len(DISCORD_TOKEN)) + DISCORD_TOKEN
    )
    assert isinstance(transport.discord_session.raw_frame, bytearray)
    assert not any(transport.discord_session.raw_frame)
    assert isinstance(transport.run_session.raw_frame, bytearray)
    assert not any(transport.run_session.raw_frame)

    rendered = _canonical(receipt)
    assert PASSWORD not in rendered
    assert DISCORD_TOKEN not in rendered
    assert password == bytearray(b"\x00" * len(PASSWORD))
    assert secret_reader.last == bytearray(b"\x00" * len(DISCORD_TOKEN))


def test_discord_gate_expiry_at_first_byte_guard_writes_zero_token_bytes():
    clock = {"now": 1_000}
    transport = _Transport(_gate())

    def expire_before_guard() -> None:
        clock["now"] = 1_371

    session = _FirstByteProbeSession(
        _discord_gate(),
        _discord_receipt(),
        transport.events,
        "discord",
        before_guard=expire_before_guard,
    )
    transport.discord_session = session
    token_reader = _SecretReader(transport.events)

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=token_reader,
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: clock["now"],
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["primary_failure_code"] == "discord_install_gate_expired"
    assert receipt["discord_frame_disclosed"] is False
    assert session.first_write_called is False
    assert session.frame is None
    assert session.attempted_frame is not None
    assert not any(session.attempted_frame)
    assert token_reader.last == bytearray(b"\x00" * len(DISCORD_TOKEN))


def test_mca2_authority_update_drift_before_first_byte_writes_zero_admin_bytes():
    transport = _Transport(_gate())
    probe = _FirstByteProbeSession(
        _coordinator_gate(),
        _approval_request(),
        transport.events,
        "coordinator",
        next_receipt=_coordinator_receipt(),
    )
    transport.run_session = probe
    sql_admin = _AuthorityDriftSqlAdmin(
        transport.events,
        drift_operation="UPDATE_USER",
    )
    password = bytearray(PASSWORD)

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: password,
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["primary_failure_code"] == (
        "cloud_sql_mutation_evidence_unconfirmed"
    )
    assert receipt["admin_frame_disclosed"] is False
    assert receipt["admin_mutation_ambiguity_observed"] is True
    assert receipt["admin_response_known_candidate_observed"] is True
    assert receipt["admin_post_baseline_authority_operation_count"] == 2
    assert "reconcile_original_mutation_baseline" in transport.events
    assert probe.first_write_called is False
    assert probe.frame is None
    assert probe.attempted_frame is not None
    assert not any(probe.attempted_frame)
    assert password == bytearray(b"\x00" * len(PASSWORD))


def test_mrc2_authority_delete_drift_before_first_byte_preserves_ambiguity():
    transport = _RecoveryTransport(_gate())
    probe = _RecoveryFirstByteProbeSession(
        transport.recovery_gate,
        transport.events,
    )
    transport.recovery_session = probe
    sql_admin = _AuthorityDriftSqlAdmin(
        transport.events,
        drift_operation="DELETE_USER",
    )
    password = bytearray(PASSWORD)

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: password,
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["primary_failure_code"] == (
        "cloud_sql_mutation_evidence_unconfirmed"
    )
    assert receipt["recovery_frame_disclosed_or_ambiguous"] is False
    assert receipt["recovery_admin_mutation_ambiguity_observed"] is True
    assert receipt["recovery_admin_response_known_candidate_observed"] is True
    assert receipt["recovery_admin_post_baseline_authority_operation_count"] == 1
    assert "reconcile_original_mutation_baseline" in transport.events
    assert probe.first_write_called is False
    assert probe.frame is None
    assert probe.attempted_frame is not None
    assert not any(probe.attempted_frame)
    assert password == bytearray(b"\x00" * len(PASSWORD))


def test_lost_create_response_with_zero_candidate_is_negative_proof_only():
    transport = _Transport(_gate())
    sql_admin = _LostCandidateSqlAdmin(transport.events)
    password = bytearray(PASSWORD)

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: password,
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["state"] == "blocked"
    assert receipt["primary_failure_code"] == "google_api_unavailable"
    assert receipt["temporary_admin_absent"] is True
    assert receipt["admin_cleanup_status"] == "absent_proven"
    assert receipt["admin_mutation_ambiguity_observed"] is True
    assert receipt["admin_response_known_candidate_observed"] is False
    assert receipt["admin_post_baseline_authority_operation_count"] == 0
    assert receipt["admin_frame_disclosed"] is False
    assert "open_coordinator" not in transport.events
    assert "reconcile_zero_candidate_absence" in transport.events
    assert password == bytearray(b"\x00" * len(PASSWORD))


def test_recovery_post_409_never_deletes_or_sends_mrc2():
    transport = _RecoveryTransport(_gate())
    sql_admin = _SqlAdmin(transport.events)
    password = bytearray(PASSWORD)

    def reject_recovery_create(username: str, secret: str) -> None:
        assert username == USERNAME
        assert secret == PASSWORD.decode()
        transport.events.append("recovery_post_409")
        raise launcher.CloudSqlCreateNotCommitted("google_api_rejected")

    sql_admin.create_or_rotate_recovery = reject_recovery_create  # type: ignore[method-assign]
    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: password,
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["primary_failure_code"] == "google_api_rejected"
    assert receipt["recovery_admin_mutation_explicitly_not_committed"] is True
    assert receipt["recovery_admin_delete_pending"] is False
    assert receipt["recovery_frame_disclosed_or_ambiguous"] is False
    assert receipt["recovery_required"] is True
    assert "delete_admin" not in transport.events
    assert "recovery_finish" not in transport.events
    assert transport.recovery_session.frame is None
    assert password == bytearray(b"\x00" * len(PASSWORD))


def test_recovery_and_fresh_reconciliation_digests_remain_phase_scoped():
    transport = _RecoveryTransport(_gate())
    sql_admin = _ScopedReconciliationSqlAdmin(transport.events)
    passwords = iter((bytearray(PASSWORD), bytearray(PASSWORD)))

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: next(passwords),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["ok"] is True
    assert receipt["recovery_admin_reconciliation_evidence_sha256"] == "1" * 64
    assert receipt["admin_reconciliation_evidence_sha256"] == "2" * 64
    assert (
        receipt["recovery_admin_reconciliation_evidence_sha256"]
        != (receipt["admin_reconciliation_evidence_sha256"])
    )
    assert receipt["recovery_admin_response_known_candidate_observed"] is True
    assert receipt["admin_response_known_candidate_observed"] is True
    assert receipt["recovery_admin_post_baseline_authority_operation_count"] == 1
    assert receipt["admin_post_baseline_authority_operation_count"] == 1


def test_final_approval_cutoff_cancels_zero_byte_session_before_mfa1_disclosure():
    transport = _Transport(_gate())
    clock = [1_000]

    class ApprovalArrivingAfterCutoff:
        def read_final_approval(self, request):
            assert request == _approval_request()
            clock[0] = 1_211
            return _final_approval()

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=ApprovalArrivingAfterCutoff(),
        approval_request_sink=lambda _request: None,
        now=lambda: clock[0],
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["ok"] is False
    assert receipt["failure_code"] == "final_approval_delivery_window_exhausted"
    assert receipt["final_approval_frame_disclosed"] is False
    assert receipt["final_approval_cancel_receipt_sha256"]
    assert transport.approval_session.frame is None
    assert "approval_cancel" in transport.events
    assert "approval_finish" not in transport.events
    assert transport.run_session.bounded_timeouts == [329.0]


def test_approval_file_arriving_during_remote_preopen_is_read_only_after_exact_gate():
    transport = _Transport(_gate())
    original_open = transport.open_final_approval_install
    approval_arrived = []

    def open_after_owner_file_arrives(release_sha):
        assert transport.events[-1] == "publish_approval_request"
        approval_arrived.append(True)
        return original_open(release_sha)

    transport.open_final_approval_install = open_after_owner_file_arrives

    class ApprovalAlreadyInstalled:
        def read_final_approval(self, request):
            transport.events.append("read_final_approval")
            assert approval_arrived == [True]
            assert transport.events.index("approval_gate") < transport.events.index(
                "read_final_approval"
            )
            assert request == _approval_request()
            return _final_approval()

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=ApprovalAlreadyInstalled(),
        approval_request_sink=lambda _request: transport.events.append(
            "publish_approval_request"
        ),
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["ok"] is True
    assert transport.events.index("publish_approval_request") < transport.events.index(
        "open_approval"
    )
    assert transport.approval_session.frame is not None


def test_remote_preopen_crossing_cutoff_sends_zero_mfa1_and_preserves_main_cleanup():
    transport = _Transport(_gate())
    clock = [1_000]
    terminal = _coordinator_failure()
    transport.run_session.next_receipt = terminal
    original_read_gate = transport.approval_session.read_gate

    def read_gate_after_cutoff():
        gate = original_read_gate()
        clock[0] = 1_211
        return gate

    transport.approval_session.read_gate = read_gate_after_cutoff

    class ApprovalMustNotBeRead:
        def read_final_approval(self, _request):
            raise AssertionError("approval cannot be read after the cutoff")

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=ApprovalMustNotBeRead(),
        approval_request_sink=lambda _request: transport.events.append(
            "publish_approval_request"
        ),
        now=lambda: clock[0],
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["failure_code"] == "final_approval_delivery_window_exhausted"
    assert receipt["final_approval_frame_disclosed"] is False
    assert receipt["final_approval_cancel_receipt_sha256"]
    assert transport.approval_session.frame is None
    assert "approval_cancel" in transport.events
    assert "read_final_approval" not in transport.events
    assert receipt["coordinator_receipt_sha256"] == terminal["receipt_sha256"]
    assert receipt["terminal_receipt_validated"] is True
    assert transport.run_session.bounded_timeouts == [329.0]


def test_remote_preopen_failure_sends_zero_mfa1_and_preserves_main_cleanup():
    transport = _Transport(_gate())
    terminal = _coordinator_failure()
    transport.run_session.next_receipt = terminal

    def fail_open(release_sha):
        assert release_sha == RELEASE_SHA
        transport.events.append("open_approval")
        raise OwnerLauncherError("final_approval_preopen_failed")

    transport.open_final_approval_install = fail_open

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: transport.events.append(
            "publish_approval_request"
        ),
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["failure_code"] == "final_approval_preopen_failed"
    assert receipt["final_approval_frame_disclosed"] is False
    assert receipt["final_approval_cancel_receipt_sha256"] is None
    assert transport.approval_session.frame is None
    assert "approval_cancel" not in transport.events
    assert "read_final_approval" not in transport.events
    assert receipt["coordinator_receipt_sha256"] == terminal["receipt_sha256"]
    assert receipt["terminal_receipt_validated"] is True


def test_local_final_approval_source_stops_at_request_cutoff_without_polling(tmp_path):
    sleeps = []
    source = launcher.FixedLocalFinalApprovalFile(
        tmp_path / "approval.json",
        now=lambda: 1_211,
        sleeper=lambda value: sleeps.append(value),
    )

    with pytest.raises(OwnerLauncherError, match="local_final_approval_timeout"):
        source.read_final_approval(_approval_request())
    assert sleeps == []


def test_cutoff_is_rechecked_after_owner_identity_preflight_before_request_sink():
    transport = _Transport(_gate())
    clock = [1_000]
    published = []

    class SlowIdentity(_OwnerIdentity):
        def require_stable(self):
            previous = self.events[-1] if self.events else None
            super().require_stable()
            if previous == "coordinator_finish":
                clock[0] = 1_211

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=SlowIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda request: published.append(request),
        now=lambda: clock[0],
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["failure_code"] == "final_approval_delivery_window_exhausted"
    assert published == []
    assert receipt["final_approval_frame_disclosed"] is False
    assert receipt["final_approval_cancel_receipt_sha256"] is None
    assert "open_approval" not in transport.events


def test_cutoff_is_rechecked_after_frame_build_before_mfa1_write(monkeypatch):
    transport = _Transport(_gate())
    clock = [1_000]
    original = launcher.build_final_approval_frame

    def slow_build(approval):
        frame = original(approval)
        clock[0] = 1_211
        return frame

    monkeypatch.setattr(launcher, "build_final_approval_frame", slow_build)
    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: clock[0],
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["failure_code"] == "final_approval_delivery_window_exhausted"
    assert receipt["final_approval_frame_disclosed"] is False
    assert transport.approval_session.frame is None
    assert receipt["final_approval_cancel_receipt_sha256"]


def test_stale_process_recovery_uses_open_mra1_then_gate_bound_mrc2():
    transport = _RecoveryTransport(_gate())
    sql_admin = _SqlAdmin(transport.events)
    recovery_password = bytearray(b"recovery-password-1234567890-abcdef")
    fresh_password = bytearray(PASSWORD)
    passwords = iter((recovery_password, fresh_password))

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: transport.events.append(
            "publish_approval_request"
        ),
        now=lambda: 1_000,
        password_factory=lambda: next(passwords),
        secret_hardener=lambda: transport.events.append("harden_secrets"),
        provenance_guard=lambda _release: None,
    )

    assert receipt["ok"] is True
    assert (
        receipt["recovery_receipt_sha256"]
        == transport.recovery_session.receipt["receipt_sha256"]
    )
    assert transport.recovery_session.ack["discord_token_state"] == "installed"
    assert transport.recovery_session.ack_frame.startswith(
        launcher.RECOVERY_ACK_FRAME_MAGIC
    )
    assert transport.recovery_session.frame.startswith(
        launcher.RECOVERY_ADMIN_FRAME_MAGIC
    )
    assert isinstance(transport.recovery_session.raw_ack_frame, bytearray)
    assert not any(transport.recovery_session.raw_ack_frame)
    assert isinstance(transport.recovery_session.raw_frame, bytearray)
    assert not any(transport.recovery_session.raw_frame)
    assert transport.events.index("recovery_exchange") < (
        transport.events.index("create_or_rotate_recovery_admin")
    )
    assert transport.events.index("create_or_rotate_recovery_admin") < (
        transport.events.index("recovery_finish")
    )
    first_delete = transport.events.index("delete_admin")
    assert transport.events.index("recovery_validated") < first_delete
    assert first_delete < transport.events.index("require_absent")
    assert transport.events.count("harden_secrets") == 1
    assert recovery_password == bytearray(b"\x00" * len(recovery_password))
    assert fresh_password == bytearray(b"\x00" * len(fresh_password))
    rendered = _canonical(receipt)
    assert b"recovery-password" not in rendered
    assert PASSWORD not in rendered


def test_recovery_failure_before_admin_preserves_active_recovery_truth():
    transport = _RecoveryTransport(_gate())

    def fail_password():
        raise OwnerLauncherError("recovery_password_unavailable")

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=fail_password,
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["primary_failure_code"] == "recovery_password_unavailable"
    assert receipt["recovery_required"] is True
    assert receipt["recovery_admin_mutation_attempted"] is False
    assert receipt["recovery_frame_write_attempted"] is False
    assert "create_or_rotate_recovery_admin" not in transport.events


def test_recovery_failure_before_ack_preserves_active_recovery_truth(monkeypatch):
    transport = _RecoveryTransport(_gate())

    def fail_ack(*_args, **_kwargs):
        raise OwnerLauncherError("recovery_ack_unavailable")

    monkeypatch.setattr(launcher, "build_recovery_ack", fail_ack)
    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["primary_failure_code"] == "recovery_ack_unavailable"
    assert receipt["recovery_required"] is True
    assert receipt["recovery_takeover_ack_sha256"] is None
    assert receipt["recovery_admin_mutation_attempted"] is False


@pytest.mark.parametrize("phase", ["open", "read"])
def test_recovery_transport_failure_before_remote_gate_keeps_preflight_truth(phase):
    class EarlyFailureTransport(_RecoveryTransport):
        def open_recovery(self, release_sha):
            if phase == "open":
                raise OwnerLauncherError("recovery_transport_open_failed")
            session = super().open_recovery(release_sha)

            def fail_read():
                raise OwnerLauncherError("recovery_transport_read_failed")

            session.read_gate = fail_read
            return session

    transport = EarlyFailureTransport(_gate())
    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["primary_failure_code"] == f"recovery_transport_{phase}_failed"
    assert receipt["recovery_gate_sha256"] is not None
    assert receipt["recovery_admin_username"] == USERNAME
    assert receipt["recovery_required"] is True
    assert receipt["recovery_admin_mutation_attempted"] is False


def test_recovery_state_absorb_cannot_erase_prior_recovery_requirement():
    prior = launcher._RecoveryAttemptState(recovery_required=True)
    prior.absorb(launcher._RecoveryAttemptState())

    assert prior.recovery_required is True


def test_session_close_failure_is_attached_without_replacing_primary():
    class FailingClose:
        def close(self):
            raise CleanupBlocked()

    primary = OwnerLauncherError("invalid_recovery_worker_completion")
    launcher._close_session_preserving_primary(FailingClose(), primary)

    assert primary.code == "invalid_recovery_worker_completion"
    assert launcher._attached_cleanup_failure_codes(primary) == ("cleanup_blocked",)


def test_post_rotation_pre_mrc2_failure_deletes_exact_admin_and_keeps_primary(
    monkeypatch,
):
    transport = _RecoveryTransport(_gate())
    sql_admin = _SqlAdmin(transport.events)
    recovery_password = bytearray(b"recovery-password-1234567890-abcdef")

    def fail_frame(*_args):
        raise OwnerLauncherError("recovery_stage_two_frame_build_failed")

    monkeypatch.setattr(launcher, "build_recovery_admin_frame", fail_frame)
    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: recovery_password,
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["primary_failure_code"] == "recovery_stage_two_frame_build_failed"
    assert receipt["recovery_admin_mutation_confirmed"] is True
    assert receipt["recovery_frame_write_attempted"] is False
    assert receipt["recovery_admin_delete_pending"] is False
    assert receipt["temporary_admin_absent"] is True
    assert receipt["recovery_required"] is True
    assert "delete_admin" in transport.events
    assert recovery_password == bytearray(b"\x00" * len(recovery_password))


def test_lost_mrc2_response_preserves_admin_and_primary_until_takeover_proof():
    class LostResponseTransport(_RecoveryTransport):
        def __init__(self, gate):
            super().__init__(gate)
            self.preflight_count = 0
            original_finish = self.recovery_session.finish

            def lose_after_write(frame):
                original_finish(frame)
                raise OwnerLauncherError("iap_ssh_ended_without_receipt")

            self.recovery_session.finish = lose_after_write

        def preflight_recovery(self, release_sha):
            self.preflight_count += 1
            if self.preflight_count == 1:
                return self.recovery_gate
            raise OwnerLauncherError("recovery_preflight_unavailable")

    transport = LostResponseTransport(_gate())
    sql_admin = _SqlAdmin(transport.events)
    recovery_password = bytearray(b"recovery-password-1234567890-abcdef")
    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: recovery_password,
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["primary_failure_code"] == "iap_ssh_ended_without_receipt"
    assert receipt["recovery_frame_write_attempted"] is True
    assert receipt["recovery_frame_disclosed_or_ambiguous"] is True
    assert receipt["recovery_admin_delete_pending"] is True
    assert receipt["temporary_admin_absent"] is False
    assert receipt["temporary_admin_preserved_for_recovery"] is True
    assert receipt["recovery_required"] is True
    assert "delete_admin" not in transport.events
    assert recovery_password == bytearray(b"\x00" * len(recovery_password))


def test_lost_completion_output_is_finalized_in_same_launch_without_new_secret():
    class LostCompletionTransport(_RecoveryTransport):
        def __init__(self, gate):
            super().__init__(gate)
            self.preflight_count = 0
            self.finalizer_calls = 0
            original_finish = self.recovery_session.finish

            def lose_completion_output(frame):
                original_finish(frame)
                raise OwnerLauncherError("iap_ssh_ended_without_receipt")

            self.recovery_session.finish = lose_completion_output

        def preflight_recovery(self, release_sha):
            self.preflight_count += 1
            if self.preflight_count == 1:
                return self.recovery_gate
            assert self.recovery_session.completion is not None
            return self.recovery_session.completion

        def open_recovery_finalizer(self, release_sha):
            self.finalizer_calls += 1
            receipt = _recovery_receipt(
                self.recovery_session.gate,
                self.recovery_session.ack,
                secret_gate=self.recovery_session.secret_gate,
                completion=self.recovery_session.completion,
            )
            self.recovery_session.receipt = receipt
            return _RecoveryFinalizerSession(receipt, self.events)

    transport = LostCompletionTransport(_gate())
    recovery_password = bytearray(b"recovery-password-1234567890-abcdef")
    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: recovery_password,
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["state"] == "blocked"
    assert receipt["primary_failure_code"] == "iap_ssh_ended_without_receipt"
    assert receipt["cleanup_failure_codes"] == []
    assert receipt["recovery_worker_completion_sha256"] is not None
    assert receipt["recovery_final_receipt_sha256"] is not None
    assert receipt["recovery_admin_delete_pending"] is False
    assert receipt["recovery_required"] is False
    assert receipt["temporary_admin_absent"] is True
    assert transport.finalizer_calls == 1
    assert recovery_password == bytearray(b"\x00" * len(recovery_password))


def test_subsequent_takeover_finalizes_lost_mrc2_then_deletes_admin():
    first_gate = _recovery_gate()
    second_gate = _recovery_gate(
        predecessor_kind="recovery_worker_lease",
        predecessor_schema=launcher.RECOVERY_WORKER_LEASE_SCHEMA,
        predecessor_journal_sha256="0" * 64,
        predecessor_generation=1,
        target_pid=5432,
        target_process_start_time_ticks=8765,
        target_boot_id_sha256="2" * 64,
        target_module_sha256="d" * 64,
        target_process_exe_sha256="e" * 64,
        target_process_cmdline_sha256="f" * 64,
    )

    class TakeoverTransport(_Transport):
        def __init__(self, gate):
            super().__init__(gate)
            self.preflight_count = 0
            self.open_count = 0
            self.first = _RecoverySession(first_gate, self.events)
            self.second = _RecoverySession(second_gate, self.events)
            original_finish = self.first.finish

            def lose_after_write(frame):
                original_finish(frame)
                raise OwnerLauncherError("iap_ssh_ended_without_receipt")

            self.first.finish = lose_after_write

        def preflight_recovery(self, release_sha):
            self.preflight_count += 1
            return first_gate if self.preflight_count == 1 else second_gate

        def open_recovery(self, release_sha):
            self.open_count += 1
            return self.first if self.open_count == 1 else self.second

        def open_recovery_finalizer(self, release_sha):
            receipt = _recovery_receipt(
                self.second.gate,
                self.second.ack,
                secret_gate=self.second.secret_gate,
                completion=self.second.completion,
            )
            self.second.receipt = receipt
            return _RecoveryFinalizerSession(receipt, self.events)

    transport = TakeoverTransport(_gate())
    sql_admin = _SqlAdmin(transport.events)
    passwords = iter((
        bytearray(b"first-recovery-password-1234567890"),
        bytearray(b"second-recovery-password-123456789"),
    ))
    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: next(passwords),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["state"] == "blocked"
    assert receipt["primary_failure_code"] == "iap_ssh_ended_without_receipt"
    assert receipt["cleanup_failure_codes"] == []
    assert receipt["recovery_final_receipt_sha256"] is not None
    assert receipt["recovery_admin_delete_pending"] is False
    assert receipt["recovery_required"] is False
    assert receipt["temporary_admin_absent"] is True
    assert transport.open_count == 2
    assert "delete_admin" in transport.events


def test_persisted_terminal_recovery_is_consumed_without_repeating_recovery():
    transport = _PersistedRecoveryTransport(_gate())

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: transport.events.append("harden_secrets"),
        provenance_guard=lambda _release: None,
    )

    assert receipt["ok"] is True
    assert (
        receipt["recovery_receipt_sha256"]
        == transport.persisted_receipt["receipt_sha256"]
    )
    assert "open_recovery" not in transport.events
    assert transport.events.index("delete_admin") < transport.events.index(
        "require_absent"
    )


def test_persisted_worker_completion_finalizes_without_secret_then_deletes():
    transport = _PersistedCompletionTransport(_gate())

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["ok"] is True
    assert (
        receipt["recovery_worker_completion_sha256"]
        == transport.completion["completion_sha256"]
    )
    assert (
        receipt["recovery_final_receipt_sha256"]
        == transport.final_receipt["receipt_sha256"]
    )
    assert receipt["recovery_admin_delete_pending"] is False
    assert receipt["recovery_required"] is False
    assert transport.finalizer_calls == 1
    assert transport.events.index("delete_admin") < transport.events.index(
        "require_absent"
    )


def test_persisted_worker_completion_rejects_self_consistent_different_final_receipt():
    class DriftedFinalReceiptTransport(_PersistedCompletionTransport):
        def __init__(self, gate):
            super().__init__(gate)
            forged = dict(self.final_receipt)
            forged["preplan_stopped_report_sha256"] = "0" * 64
            forged.pop("receipt_sha256")
            reconstructed = launcher._reconstruct_worker_completion_from_final_receipt(
                forged
            )
            forged["recovery_worker_completion_sha256"] = reconstructed[
                "completion_sha256"
            ]
            self.final_receipt = _self_digest(forged, "receipt_sha256")

    transport = DriftedFinalReceiptTransport(_gate())

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["state"] == "cleanup_blocked"
    assert (
        receipt["primary_failure_code"] == "recovery_final_receipt_completion_drifted"
    )
    assert receipt["recovery_final_receipt_sha256"] is None
    assert receipt["recovery_terminal_receipt_validated"] is False
    assert receipt["temporary_admin_absent"] is False
    assert receipt["recovery_admin_delete_pending"] is True
    assert receipt["recovery_required"] is True
    assert "delete_admin" not in transport.events


def test_persisted_worker_completion_pending_finalizer_preserves_admin():
    transport = _PersistedCompletionTransport(_gate(), pending=True)

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["state"] == "cleanup_blocked"
    assert receipt["recovery_finalize_pending_receipt_sha256"] is not None
    assert receipt["recovery_admin_delete_pending"] is True
    assert receipt["temporary_admin_absent"] is False
    assert receipt["recovery_required"] is True
    assert "delete_admin" not in transport.events
    assert transport.finalizer_calls == 2


def test_persisted_recovery_then_fresh_pre_run_failure_retires_new_token():
    transport = _PersistedRecoveryTransport(_gate())
    sql_admin = _SqlAdmin(transport.events)

    def fail_create(username: str, password: str) -> None:
        transport.events.append("create_admin_failed")
        raise OwnerLauncherError("google_api_unavailable")

    sql_admin.create = fail_create  # type: ignore[method-assign]
    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["state"] == "blocked"
    assert receipt["failure_code"] == "google_api_unavailable"
    assert receipt["discord_retirement_receipt_sha256"] is not None
    assert "retirement_validated" in transport.events
    assert "open_coordinator" not in transport.events


def test_token_only_recovery_deletes_stale_admin_and_completes_in_one_launch():
    transport = _TokenOnlyRecoveryTransport(_gate())

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["ok"] is True
    assert transport.events.index("retirement_validated") < transport.events.index(
        "delete_admin"
    )
    assert transport.events.index("delete_admin") < transport.events.index("preflight")
    assert transport.events.index("preflight") < transport.events.index(
        "require_absent"
    )
    assert transport.events.count("read_discord") == 1


def test_persisted_terminal_recovery_truth_does_not_expire_before_owner_delete():
    gate = _recovery_gate()
    ack = launcher.build_recovery_ack(
        gate,
        now_unix=1_000,
        nonce=b"persisted-old-recovery-nonce",
    )
    receipt = _recovery_receipt(gate, ack, cleanup_completed_at_unix=1_000)

    validated = launcher.validate_persisted_recovery_receipt(
        receipt,
        expected_release_sha=RELEASE_SHA,
        now_unix=100_000,
    )
    assert validated["safe_to_delete_temporary_admin"] is True


@pytest.mark.parametrize(
    "gate",
    [
        _recovery_gate(admin_credential_required=False),
        _recovery_gate(
            discord_token_state="installed",
            discord_retirement_receipt_sha256="9" * 64,
        ),
        _recovery_gate(
            discord_token_state="retired",
            discord_retirement_receipt_sha256=None,
        ),
    ],
)
def test_recovery_gate_rejects_missing_admin_or_inconsistent_token_truth(gate):
    with pytest.raises(OwnerLauncherError, match="invalid_recovery_gate"):
        launcher.validate_recovery_gate(
            gate,
            expected_release_sha=RELEASE_SHA,
            owner_gate=None,
            now_unix=1_000,
        )


@pytest.mark.parametrize(
    "gate",
    [
        _recovery_gate(
            predecessor_schema=launcher.RECOVERY_WORKER_LEASE_SCHEMA,
        ),
        _recovery_gate(predecessor_generation=1),
        _recovery_gate(
            predecessor_kind="recovery_worker_lease",
            predecessor_schema="muncho-full-canary-coordinator-process-lease.v1",
            predecessor_generation=1,
        ),
        _recovery_gate(
            predecessor_kind="recovery_worker_lease",
            predecessor_schema=launcher.RECOVERY_WORKER_LEASE_SCHEMA,
            predecessor_generation=0,
        ),
    ],
)
def test_recovery_gate_rejects_forged_predecessor_kind_schema_generation(gate):
    with pytest.raises(OwnerLauncherError, match="invalid_recovery_gate"):
        launcher.validate_recovery_gate(
            gate,
            expected_release_sha=RELEASE_SHA,
            owner_gate=None,
            now_unix=1_000,
        )


def test_recovery_ack_exactly_binds_retired_token_terminal_truth():
    gate = _recovery_gate(
        discord_token_state="retired",
        discord_retirement_receipt_sha256="9" * 64,
    )
    validated = launcher.validate_recovery_gate(
        gate,
        expected_release_sha=RELEASE_SHA,
        owner_gate=None,
        now_unix=1_000,
    )
    ack = launcher.build_recovery_ack(
        validated,
        now_unix=1_000,
        nonce=b"retired-token-recovery-nonce",
    )

    assert ack["discord_token_state"] == "retired"
    assert ack["discord_retirement_receipt_sha256"] == "9" * 64


@pytest.mark.parametrize("state", ["retirement_prepared", "retired"])
def test_recovery_ack_binds_monotonic_retirement_authority(state):
    gate = _recovery_gate(
        discord_token_state=state,
        discord_retirement_receipt_sha256="9" * 64,
    )
    validated = launcher.validate_recovery_gate(
        gate,
        expected_release_sha=RELEASE_SHA,
        owner_gate=None,
        now_unix=1_000,
    )
    ack = launcher.build_recovery_ack(
        validated,
        now_unix=1_000,
        nonce=b"monotonic-retirement-authority-nonce",
    )

    assert ack["discord_token_state"] == state
    assert ack["discord_retirement_receipt_sha256"] == "9" * 64
    assert ack["token_device"] == 10
    assert ack["token_inode"] == 11


def test_recovery_stage_two_frame_is_bound_to_exact_gate_and_nonce():
    gate = launcher.validate_recovery_gate(
        _recovery_gate(),
        expected_release_sha=RELEASE_SHA,
        owner_gate=None,
        now_unix=1_000,
    )
    ack = launcher.build_recovery_ack(
        gate,
        now_unix=1_000,
        nonce=b"stage-one-takeover-nonce-value",
    )
    secret_gate = launcher.validate_recovery_secret_gate(
        _recovery_secret_gate(gate, ack),
        takeover_gate=gate,
        takeover_ack=ack,
        now_unix=1_000,
    )
    password = bytearray(PASSWORD)
    frame = launcher.build_recovery_admin_frame(secret_gate, USERNAME, password)
    magic, gate_digest, nonce_digest, username_size, password_size = struct.unpack(
        "!4s32s32sHI", frame[:74]
    )

    assert magic == launcher.RECOVERY_ADMIN_FRAME_MAGIC
    assert gate_digest.hex() == secret_gate["gate_sha256"]
    assert nonce_digest.hex() == secret_gate["gate_nonce_sha256"]
    assert frame[74 : 74 + username_size] == USERNAME.encode()
    assert password_size == len(password)
    assert not frame.startswith(ADMIN_FRAME_MAGIC)


def test_recovery_stage_one_forbids_prebuffered_or_legacy_admin_credential():
    gate = _recovery_gate()
    ack = launcher.build_recovery_ack(
        gate,
        now_unix=1_000,
        nonce=b"stage-one-no-secret-nonce-value",
    )

    with pytest.raises(
        OwnerLauncherError,
        match="unexpected_recovery_admin_credential",
    ):
        launcher.build_recovery_ack_frame(
            gate,
            ack,
            password=bytearray(PASSWORD),
        )
    invalid_stage_two = _recovery_secret_gate(
        gate,
        ack,
        admin_frame_schema=launcher.ADMIN_FRAME_SCHEMA,
    )
    with pytest.raises(OwnerLauncherError, match="invalid_recovery_secret_gate"):
        launcher.validate_recovery_secret_gate(
            invalid_stage_two,
            takeover_gate=gate,
            takeover_ack=ack,
            now_unix=1_000,
        )


def test_recovery_retired_install_intent_accepts_only_exact_null_identity_pair():
    gate = _recovery_gate(
        discord_token_state="retired",
        discord_retirement_receipt_sha256="9" * 64,
        token_device=None,
        token_inode=None,
    )
    assert (
        launcher.validate_recovery_gate(
            gate,
            expected_release_sha=RELEASE_SHA,
            owner_gate=None,
            now_unix=1_000,
        )["discord_token_state"]
        == "retired"
    )

    invalid = _recovery_gate(
        discord_token_state="retired",
        discord_retirement_receipt_sha256="9" * 64,
        token_device=None,
        token_inode=11,
    )
    with pytest.raises(OwnerLauncherError, match="invalid_recovery_gate"):
        launcher.validate_recovery_gate(
            invalid,
            expected_release_sha=RELEASE_SHA,
            owner_gate=None,
            now_unix=1_000,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"discord_retirement_receipt_sha256": None},
        {"safe_to_delete_temporary_admin": False},
    ],
)
def test_recovery_receipt_requires_terminal_retirement_and_safe_delete(changes):
    gate = _recovery_gate()
    ack = launcher.build_recovery_ack(
        gate,
        now_unix=1_000,
        nonce=b"recovery-receipt-negative-nonce",
    )
    secret_gate = _recovery_secret_gate(gate, ack)
    completion = _recovery_completion(gate, ack, secret_gate)
    receipt = _recovery_receipt(
        gate,
        ack,
        secret_gate=secret_gate,
        completion=completion,
        **changes,
    )

    with pytest.raises(OwnerLauncherError, match="invalid_recovery_receipt"):
        launcher.validate_recovery_receipt(
            receipt,
            gate=gate,
            ack=ack,
            secret_gate=secret_gate,
            completion=completion,
            password=bytearray(PASSWORD),
            now_unix=1_000,
        )


def test_recovery_final_receipt_must_be_exact_completion_projection():
    gate = _recovery_gate()
    ack = launcher.build_recovery_ack(
        gate,
        now_unix=1_000,
        nonce=b"completion-projection-negative",
    )
    secret_gate = _recovery_secret_gate(gate, ack)
    completion = _recovery_completion(gate, ack, secret_gate)
    receipt = _recovery_receipt(
        gate,
        ack,
        secret_gate=secret_gate,
        completion=completion,
    )
    receipt["preplan_stopped_report_sha256"] = "0" * 64
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical({
            key: item for key, item in receipt.items() if key != "receipt_sha256"
        })
    ).hexdigest()

    with pytest.raises(OwnerLauncherError, match="invalid_recovery_receipt"):
        launcher.validate_recovery_receipt(
            receipt,
            gate=gate,
            ack=ack,
            secret_gate=secret_gate,
            completion=completion,
            password=bytearray(PASSWORD),
            now_unix=1_000,
        )


def test_persisted_recovery_rejects_skipped_generation_even_with_rebound_digest():
    gate = _recovery_gate()
    ack = launcher.build_recovery_ack(
        gate,
        now_unix=1_000,
        nonce=b"skipped-generation-negative",
    )
    receipt = _recovery_receipt(gate, ack)
    receipt["recovery_generation"] = 2
    reconstructed = launcher._reconstruct_worker_completion_from_final_receipt(receipt)
    receipt["recovery_worker_completion_sha256"] = reconstructed["completion_sha256"]
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical({
            key: item for key, item in receipt.items() if key != "receipt_sha256"
        })
    ).hexdigest()

    with pytest.raises(OwnerLauncherError, match="invalid_recovery_receipt"):
        launcher.validate_persisted_recovery_receipt(
            receipt,
            expected_release_sha=RELEASE_SHA,
            now_unix=1_000,
        )


def test_persisted_completion_rejects_run_kind_with_nonzero_predecessor_generation():
    gate = _recovery_gate()
    ack = launcher.build_recovery_ack(
        gate,
        now_unix=1_000,
        nonce=b"forged-run-generation-completion",
    )
    secret_gate = _recovery_secret_gate(gate, ack)
    completion = _recovery_completion(gate, ack, secret_gate)
    completion["predecessor_generation"] = 1
    completion["recovery_generation"] = 2
    completion["completion_sha256"] = hashlib.sha256(
        _canonical({
            key: item for key, item in completion.items() if key != "completion_sha256"
        })
    ).hexdigest()

    with pytest.raises(
        OwnerLauncherError,
        match="invalid_recovery_worker_completion",
    ):
        launcher.validate_persisted_recovery_worker_completion(
            completion,
            expected_release_sha=RELEASE_SHA,
            now_unix=1_000,
        )


def test_persisted_final_rejects_worker_kind_with_zero_predecessor_generation():
    gate = _recovery_gate()
    ack = launcher.build_recovery_ack(
        gate,
        now_unix=1_000,
        nonce=b"forged-worker-generation-final",
    )
    receipt = _recovery_receipt(gate, ack)
    receipt["predecessor_kind"] = "recovery_worker_lease"
    reconstructed = launcher._reconstruct_worker_completion_from_final_receipt(receipt)
    receipt["recovery_worker_completion_sha256"] = reconstructed["completion_sha256"]
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical({
            key: item for key, item in receipt.items() if key != "receipt_sha256"
        })
    ).hexdigest()

    with pytest.raises(OwnerLauncherError, match="invalid_recovery_receipt"):
        launcher.validate_persisted_recovery_receipt(
            receipt,
            expected_release_sha=RELEASE_SHA,
            now_unix=1_000,
        )


def test_recovery_rejects_process_executable_drift_before_admin_authority():
    transport = _RecoveryTransport(_gate())
    transport.recovery_session.gate = _recovery_gate(target_process_exe_sha256="f" * 64)
    transport.recovery_session.close = lambda: transport.events.append(
        "recovery_close_unvalidated"
    )

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["failure_code"] == "recovery_gate_drifted"
    assert "create_or_rotate_recovery_admin" not in transport.events


def test_unconfirmed_delete_overrides_success_with_cleanup_blocked():
    transport = _Transport(_gate())
    sql_admin = _SqlAdmin(transport.events, cleanup_fails=True)

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: transport.events.append(
            "publish_approval_request"
        ),
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: transport.events.append("harden_secrets"),
        provenance_guard=lambda _release: None,
    )

    assert receipt["ok"] is False
    assert receipt["state"] == "cleanup_blocked"
    assert receipt["failure_code"] == "admin_cleanup_blocked"
    assert receipt["temporary_admin_absent"] is False
    assert transport.events[-1] == "delete_admin"


def test_run_terminal_failure_before_admin_frame_is_proven_and_admin_deleted():
    transport = _Transport(_gate())
    transport.run_session.gate = _early_terminal_failure(bound=True)
    sql_admin = _SqlAdmin(transport.events)

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["state"] == "blocked"
    assert receipt["failure_code"] == "coordinator_input_invalid"
    assert receipt["admin_frame_disclosed"] is False
    assert receipt["remote_coordinator_terminated"] is True
    assert receipt["terminal_receipt_validated"] is True
    assert receipt["temporary_admin_absent"] is True
    assert "coordinator_finish" not in transport.events
    assert "delete_admin" in transport.events


def test_run_terminal_failure_after_admin_frame_is_not_misread_as_approval_request():
    transport = _Transport(_gate())
    transport.run_session.receipt = _early_terminal_failure(
        bound=True,
        admin_frame_disclosed=True,
    )

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["state"] == "blocked"
    assert receipt["admin_frame_disclosed"] is True
    assert receipt["remote_coordinator_terminated"] is True
    assert receipt["temporary_admin_absent"] is True
    assert "publish_approval_request" not in transport.events


def test_secret_derived_nested_digest_is_rejected_before_receipt_hash_trust():
    receipt = _coordinator_receipt()
    receipt["live_driver_result"]["offline_invariant_receipt"][  # type: ignore[index]
        "token_sha256"
    ] = "f" * 64

    with pytest.raises(OwnerLauncherError, match="invalid_coordinator_receipt"):
        validate_coordinator_receipt(
            receipt,
            gate=_coordinator_gate(),
            password=bytearray(PASSWORD),
            now_unix=1_000,
        )


@pytest.mark.parametrize(
    "phase",
    ["discord_token_retirement", "root_snapshot_cleanup", "secret_cleanup"],
)
def test_producer_cleanup_phases_remain_exact_valid_terminal_failures(phase):
    receipt = _coordinator_failure()
    receipt.update({
        "phase": phase,
        "full_canary_plan_sha256": None,
        "cleanup_status": "cleanup_blocked",
        "recovery_material_preserved": True,
        "discord_token_removed": False,
        "coordinator_process_lease_retired": False,
    })
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical({
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        })
    ).hexdigest()

    validated = validate_coordinator_receipt(
        receipt,
        gate=_coordinator_gate(),
        password=bytearray(PASSWORD),
        now_unix=1_000,
    )
    assert validated["cleanup_status"] == "cleanup_blocked"


def test_discord_post_frame_terminal_failure_is_validated_at_that_boundary():
    transport = _Transport(_gate())
    transport.discord_session.receipt = _early_terminal_failure(bound=False)

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["state"] == "blocked"
    assert receipt["discord_frame_disclosed"] is True
    assert receipt["remote_failure_receipt_sha256"] is not None
    assert "require_absent" not in transport.events


def test_lost_discord_install_response_attempts_dra_cleanup_in_same_launch():
    transport = _LostDiscordInstallResponseTransport(_gate())

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["discord_frame_disclosed"] is True
    assert receipt["discord_install_receipt_sha256"] is None
    assert receipt["discord_retirement_receipt_sha256"] is not None
    assert receipt["discord_token_retired"] is True
    assert "open_retirement" in transport.events
    assert "retirement_validated" in transport.events


def test_terminal_first_failure_accepts_true_ambient_bootstrap_absence_truth():
    failure = _early_terminal_failure(bound=False)
    failure["bootstrap_credential_removed"] = True
    failure["receipt_sha256"] = hashlib.sha256(
        _canonical({
            key: value for key, value in failure.items() if key != "receipt_sha256"
        })
    ).hexdigest()

    validated = launcher.validate_terminal_first_failure(
        failure,
        owner_gate=None,
        token_lease_expected=False,
        process_lease_expected=False,
    )
    assert validated["bootstrap_credential_removed"] is True


def test_ambiguous_create_failure_still_attempts_cleanup():
    transport = _Transport(_gate())
    sql_admin = _SqlAdmin(transport.events)

    def fail_after_possible_mutation(username: str, password: str) -> None:
        sql_admin.events.append("create_admin")
        raise OwnerLauncherError("google_api_unavailable")

    sql_admin.create = fail_after_possible_mutation  # type: ignore[method-assign]

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: transport.events.append(
            "publish_approval_request"
        ),
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: transport.events.append("harden_secrets"),
        provenance_guard=lambda _release: None,
    )

    assert receipt["state"] == "blocked"
    assert receipt["failure_code"] == "google_api_unavailable"
    assert receipt["temporary_admin_absent"] is True
    assert transport.events[-2:] == ["delete_admin", "owner_stable"]
    assert "retirement_validated" in transport.events


def test_explicit_create_rejection_never_deletes_concurrent_same_name_account():
    transport = _Transport(_gate())
    sql_admin = _SqlAdmin(transport.events)

    def reject_create(username: str, password: str) -> None:
        transport.events.append("create_admin_rejected")
        raise launcher.CloudSqlCreateNotCommitted("google_api_rejected")

    sql_admin.create = reject_create  # type: ignore[method-assign]
    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=sql_admin,
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=_FinalApprovalSource(transport.events),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["failure_code"] == "google_api_rejected"
    # The only delete is the exact no-journal stale-account reconciliation
    # before require_absent; no delete follows the explicit rejected POST.
    assert transport.events.count("delete_admin") == 1
    assert transport.events.index("delete_admin") < transport.events.index(
        "require_absent"
    )
    assert "retirement_validated" in transport.events


def test_discord_token_is_read_from_exact_owner_fd0_frame_only():
    frame = (
        OWNER_DISCORD_INPUT_MAGIC
        + struct.pack(">I", len(DISCORD_TOKEN))
        + DISCORD_TOKEN
    )
    reader = OwnerStdinDiscordTokenReader(io.BytesIO(frame))

    assert reader.read_discord_token() == bytearray(DISCORD_TOKEN)


class _TtyInput:
    def isatty(self):
        return True


def test_discord_token_tty_uses_masked_reader_only():
    calls = []
    reader = OwnerDiscordTokenReader(
        _TtyInput(),
        masked_reader=lambda prompt: calls.append(prompt) or DISCORD_TOKEN.decode(),
    )

    assert reader.read_discord_token() == bytearray(DISCORD_TOKEN)
    assert calls == ["Canary Discord bot token: "]


def test_discord_token_non_tty_requires_exact_mdo1_frame():
    frame = (
        OWNER_DISCORD_INPUT_MAGIC
        + struct.pack(">I", len(DISCORD_TOKEN))
        + DISCORD_TOKEN
    )

    assert OwnerDiscordTokenReader(io.BytesIO(frame)).read_discord_token() == bytearray(
        DISCORD_TOKEN
    )


@pytest.mark.parametrize(
    "frame",
    [
        b"BAD!" + struct.pack(">I", len(DISCORD_TOKEN)) + DISCORD_TOKEN,
        OWNER_DISCORD_INPUT_MAGIC
        + struct.pack(">I", len(DISCORD_TOKEN) + 1)
        + DISCORD_TOKEN,
        OWNER_DISCORD_INPUT_MAGIC
        + struct.pack(">I", len(DISCORD_TOKEN))
        + DISCORD_TOKEN
        + b"x",
    ],
)
def test_discord_owner_fd0_frame_rejects_wrong_magic_length_or_trailing_bytes(frame):
    with pytest.raises(OwnerLauncherError, match="invalid_owner_discord_frame"):
        OwnerStdinDiscordTokenReader(io.BytesIO(frame)).read_discord_token()


def test_discord_retirement_receipt_binds_exact_installed_inode_and_absence_proof():
    receipt = validate_discord_retirement_receipt(
        _discord_retirement_receipt(),
        owner_gate=_gate(),
        install_receipt=_discord_receipt(),
        now_unix=1_000,
    )

    assert receipt["state"] == "retired"
    drifted = _discord_retirement_receipt()
    drifted["token_inode"] = 12
    drifted["receipt_sha256"] = hashlib.sha256(
        _canonical({
            key: value for key, value in drifted.items() if key != "receipt_sha256"
        })
    ).hexdigest()
    with pytest.raises(OwnerLauncherError, match="invalid_discord_retirement_receipt"):
        validate_discord_retirement_receipt(
            drifted,
            owner_gate=_gate(),
            install_receipt=_discord_receipt(),
            now_unix=1_000,
        )


def test_discord_retirement_install_intent_null_identity_round_trip():
    gate = _discord_retirement_gate(token_device=None, token_inode=None)
    validated_gate = launcher.validate_discord_retirement_gate(
        gate,
        owner_gate=_gate(),
        install_receipt=None,
        now_unix=1_000,
    )
    ack = launcher.build_discord_retirement_ack(
        validated_gate,
        now_unix=1_000,
        nonce=b"install-intent-retirement-nonce",
    )
    receipt = _discord_retirement_receipt(
        token_device=None,
        token_inode=None,
    )

    assert ack["token_device"] is None
    assert ack["token_inode"] is None
    assert (
        validate_discord_retirement_receipt(
            receipt,
            owner_gate=_gate(),
            install_receipt=None,
            retirement_gate=validated_gate,
            now_unix=1_000,
        )["receipt_sha256"]
        == receipt["receipt_sha256"]
    )


def test_google_rest_boundary_forbids_secret_manager_and_production_secret_names():
    called = []
    client = GoogleRestClient(
        lambda: "owner-access-token",
        requester=lambda *args: called.append(args),
    )

    with pytest.raises(OwnerLauncherError, match="forbidden_google_api_url"):
        client.request_json(
            "GET",
            "https://secretmanager.googleapis.com/v1/projects/"
            f"{PROJECT}/secrets/ai-platform-discord-bot-token/versions/latest:access",
        )

    assert called == []


def test_http_response_repr_never_contains_a_secret_bearing_body():
    response = HttpResponse(200, b"sensitive-response-body")

    assert "sensitive-response-body" not in repr(response)


def test_owner_access_token_uses_strict_argv_and_sanitized_environment(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "must-not-be-inherited")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/forbidden/key.json")
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[len(GCLOUD_COMMAND_PREFIX) :][:2] == ("auth", "list"):
            return subprocess.CompletedProcess(argv, 0, b"owner@example.com\n", b"")
        return subprocess.CompletedProcess(argv, 0, b"opaque-owner-access-token\n", b"")

    provider = GcloudOwnerAccessToken(
        gcloud_executable=_StableExecutable(),
        gcloud_configuration=_StableGcloudConfiguration(),
        runner=runner,
    )

    provider.bind_approved_subject(OWNER_SUBJECT_SHA)
    provider.require_stable()
    assert provider() == "opaque-owner-access-token"
    assert (
        provider.owner_subject_sha256
        == hashlib.sha256(b"owner@example.com").hexdigest()
    )
    assert [call[0] for call in calls] == [
        (
            *GCLOUD_COMMAND_PREFIX,
            "auth",
            "list",
            "--filter=status:ACTIVE",
            "--format=value(account)",
            "--limit=2",
            "--quiet",
        ),
        (
            *GCLOUD_COMMAND_PREFIX,
            "auth",
            "list",
            "--filter=status:ACTIVE",
            "--format=value(account)",
            "--limit=2",
            "--quiet",
        ),
        (
            *GCLOUD_COMMAND_PREFIX,
            "auth",
            "print-access-token",
            "--account=owner@example.com",
            "--quiet",
        ),
    ]
    for _argv, kwargs in calls:
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["timeout"] == 20.0
        assert "DISCORD_BOT_TOKEN" not in kwargs["env"]
        assert "GOOGLE_APPLICATION_CREDENTIALS" not in kwargs["env"]
        assert kwargs["env"]["CLOUDSDK_CORE_PROJECT"] == PROJECT
        assert kwargs["env"]["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
        assert kwargs["env"]["CLOUDSDK_CORE_LOG_HTTP"] == "0"
        assert kwargs["env"]["CLOUDSDK_CORE_DISABLE_FILE_LOGGING"] == "1"
        assert kwargs["env"]["CLOUDSDK_PYTHON_ARGS"] == (
            "-I -S -B -X pycache_prefix=/var/empty/muncho-canary"
        )
        assert not any("PROXY" in name for name in kwargs["env"])
        assert "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT" not in kwargs["env"]


def test_trusted_gcloud_ignores_hostile_path_and_pins_executable_bytes(
    tmp_path,
    monkeypatch,
):
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    hostile_gcloud = hostile / "gcloud"
    hostile_gcloud.write_text("#!/bin/sh\nexit 99\n")
    hostile_gcloud.chmod(0o755)
    monkeypatch.chdir(hostile)
    monkeypatch.setenv("PATH", f".:{hostile}")

    sdk = tmp_path / "sealed-sdk"
    gcloud = sdk / "bin/gcloud"
    module = sdk / "lib/gcloud.py"
    gcloud.parent.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    gcloud.write_text("#!/bin/sh\nexit 0\n")
    gcloud.chmod(0o755)
    module.write_text("VALUE = 1\n")
    (sdk / "VERSION").write_text("fixed\n")
    python = tmp_path / "python/bin/python3.13"
    python.parent.mkdir(parents=True)
    python.write_text("fixed-python\n")
    python.chmod(0o755)

    class StablePath:
        def __init__(self, selected, **_kwargs):
            self.selected = selected

        def absolute_path(self):
            return self.selected

    monkeypatch.setattr(launcher, "_PinnedExecutablePath", StablePath)
    executable = launcher.TrustedGcloudExecutable(
        candidates=(str(gcloud),),
        python_candidates=(str(python),),
    )
    assert executable.trusted_command_prefix() == (
        str(python),
        "-I",
        "-S",
        "-B",
        "-X",
        "pycache_prefix=/var/empty/muncho-canary",
        str(module),
    )

    module.write_text("VALUE = 2\n")
    with pytest.raises(OwnerLauncherError, match="trusted_gcloud_sdk_changed"):
        executable.trusted_command_prefix()


def test_pinned_executable_rejects_identity_drift(monkeypatch):
    executable = launcher._PinnedExecutablePath(
        "/usr/bin/true",
        invalid_code="trusted_gcloud_invalid",
        changed_code="trusted_gcloud_changed",
    )
    original_capture = executable._capture

    def changed(path):
        fingerprint, resolved = original_capture(path)
        return (fingerprint, "changed"), resolved

    monkeypatch.setattr(executable, "_capture", changed)
    with pytest.raises(OwnerLauncherError, match="trusted_gcloud_changed"):
        executable.absolute_path()


def test_pinned_file_reader_accepts_stable_nonempty_and_explicit_empty(tmp_path):
    nonempty = tmp_path / "nonempty"
    nonempty.write_bytes(b"fixed")
    empty = tmp_path / "empty"
    empty.write_bytes(b"")

    fingerprint, payload = launcher._read_pinned_regular_file(
        str(nonempty),
        maximum=64,
        unavailable_code="unavailable",
        invalid_code="invalid",
        changed_code="changed",
        allowed_owners=frozenset({launcher.os.getuid()}),
    )
    assert payload == b"fixed"
    assert fingerprint[-1] == hashlib.sha256(b"fixed").hexdigest()
    with pytest.raises(OwnerLauncherError, match="invalid"):
        launcher._read_pinned_regular_file(
            str(empty),
            maximum=64,
            unavailable_code="unavailable",
            invalid_code="invalid",
            changed_code="changed",
            allowed_owners=frozenset({launcher.os.getuid()}),
        )
    assert (
        launcher._read_pinned_regular_file(
            str(empty),
            maximum=64,
            unavailable_code="unavailable",
            invalid_code="invalid",
            changed_code="changed",
            allowed_owners=frozenset({launcher.os.getuid()}),
            allow_empty=True,
        )[1]
        == b""
    )


def _tiny_sdk_archive(
    *,
    forbidden_name: str | None = None,
    include_directories: bool = True,
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        if include_directories:
            for name in (
                "google-cloud-sdk",
                "google-cloud-sdk/bin",
                "google-cloud-sdk/lib",
            ):
                member = tarfile.TarInfo(name)
                member.type = tarfile.DIRTYPE
                member.mode = 0o755
                archive.addfile(member)
        files = {
            "google-cloud-sdk/VERSION": b"569.0.0\n",
            "google-cloud-sdk/bin/gcloud": b"#!/bin/sh\nexit 0\n",
            "google-cloud-sdk/lib/gcloud.py": b"VALUE = 1\n",
        }
        if forbidden_name is not None:
            files[forbidden_name] = b"forbidden"
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mode = 0o755 if name.endswith("/gcloud") else 0o644
            archive.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def _tiny_bootstrap_fixture(tmp_path, monkeypatch):
    home = tmp_path / "owner"
    hermes = home / ".hermes"
    hermes.mkdir(parents=True)
    hermes.chmod(0o700)
    archive = _tiny_sdk_archive(include_directories=False)
    downloads = []
    validations = []

    class StablePath:
        def __init__(self, selected, **_kwargs):
            self.selected = selected

        def absolute_path(self):
            return self.selected

    python = StablePath(str(home / "python/bin/python3.11"))
    monkeypatch.setattr(launcher, "_canonical_owner_home", lambda **_kwargs: str(home))
    monkeypatch.setattr(launcher, "_PinnedExecutablePath", StablePath)
    monkeypatch.setattr(launcher, "_GCLOUD_SDK_ARCHIVE_BYTES", len(archive))
    monkeypatch.setattr(
        launcher,
        "_GCLOUD_SDK_ARCHIVE_SHA256",
        hashlib.sha256(archive).hexdigest(),
    )

    def portable_no_replace(source, destination, *, exists_code, failed_code):
        # These tests exercise the bootstrap recovery state machine, not the
        # Darwin syscall contract.  The dedicated Darwin-only test below uses
        # the real renamex_np(RENAME_EXCL) implementation.
        if os.path.lexists(destination):
            raise OwnerLauncherError(exists_code)
        try:
            os.rename(source, destination)
        except OSError:
            raise OwnerLauncherError(failed_code) from None

    monkeypatch.setattr(
        launcher,
        "_darwin_rename_no_replace",
        portable_no_replace,
    )

    def downloader(path):
        downloads.append(path)
        launcher.Path(path).write_bytes(archive)

    def python_snapshot():
        return (
            python,
            str(home / "python"),
            (3, 100, "a" * 64),
            "3.11.15",
            ("/usr/lib/x",),
        )

    def run(now=1_000):
        return launcher.bootstrap_trusted_gcloud_runtime(
            RELEASE_SHA,
            now_unix=now,
            archive_downloader=downloader,
            python_snapshot=python_snapshot,
            runtime_validator=lambda release: validations.append(release),
        )

    return home, archive, downloads, validations, run


def test_packaged_trusted_runtime_bootstrap_is_no_replace_and_retry_safe(
    tmp_path,
    monkeypatch,
):
    home, _archive, downloads, validations, run = _tiny_bootstrap_fixture(
        tmp_path,
        monkeypatch,
    )
    receipt = run(1_000)
    retried = run(2_000)

    sdk = home / ".hermes/trusted/google-cloud-sdk-569.0.0"
    assert receipt == retried
    assert receipt["sdk_tree_sha256"]
    assert (sdk / "lib/gcloud.py").read_bytes() == b"VALUE = 1\n"
    assert len(downloads) == 1
    assert validations == [RELEASE_SHA, RELEASE_SHA]
    assert not list(sdk.rglob("*.pyc"))
    assert not list(sdk.rglob("__pycache__"))
    assert receipt["launcher_sha256"] == launcher._current_launcher_sha256()
    assert receipt["python_version"] == "3.11.15"
    assert receipt["sdk_publication_intent_sha256"]


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin renamex_np contract")
@pytest.mark.parametrize("competitor_kind", ["file", "directory", "symlink"])
def test_atomic_no_replace_preserves_every_existing_competitor(
    tmp_path,
    competitor_kind,
):
    source_file = tmp_path / "source-file"
    source_file.write_bytes(b"source")
    source_directory = tmp_path / "source-directory"
    source_directory.mkdir()
    (source_directory / "payload").write_bytes(b"source-directory")
    target_file = tmp_path / "target-file"
    target_directory = tmp_path / "target-directory"
    target_directory.mkdir()
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"sentinel")
    if competitor_kind == "file":
        target_file.write_bytes(b"competitor")
    elif competitor_kind == "directory":
        target_file.mkdir()
        (target_file / "competitor").write_bytes(b"directory")
    else:
        target_file.symlink_to(sentinel)

    with pytest.raises(OwnerLauncherError, match="receipt_exists"):
        launcher._publish_regular_no_replace(source_file, target_file)
    with pytest.raises(OwnerLauncherError, match="destination_exists"):
        launcher._rename_directory_no_replace(source_directory, target_directory)

    assert source_file.read_bytes() == b"source"
    assert (source_directory / "payload").read_bytes() == b"source-directory"
    if competitor_kind == "file":
        assert target_file.read_bytes() == b"competitor"
    elif competitor_kind == "directory":
        assert (target_file / "competitor").read_bytes() == b"directory"
    else:
        assert target_file.is_symlink()
        assert os.readlink(target_file) == str(sentinel)
    assert not list(target_directory.iterdir())


def test_bootstrap_recovers_from_durable_intent_before_sdk_publication(
    tmp_path,
    monkeypatch,
):
    home, _archive, downloads, _validations, run = _tiny_bootstrap_fixture(
        tmp_path,
        monkeypatch,
    )
    original = launcher._rename_directory_no_replace
    calls = 0

    def interrupt_before_rename(source, destination):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OwnerLauncherError("simulated_crash_before_sdk_publish")
        return original(source, destination)

    monkeypatch.setattr(
        launcher,
        "_rename_directory_no_replace",
        interrupt_before_rename,
    )
    with pytest.raises(
        OwnerLauncherError,
        match="simulated_crash_before_sdk_publish",
    ):
        run()
    intent_path = home / launcher._TRUSTED_SDK_PUBLICATION_INTENT_RELATIVE
    sdk = home / launcher._TRUSTED_SDK_RELATIVE
    assert intent_path.is_file()
    assert not sdk.exists()

    receipt = run(1_001)

    assert receipt["ok"] is True
    assert sdk.is_dir()
    assert len(downloads) == 2


@pytest.mark.parametrize("fail_fsync_call", [2, 3])
def test_bootstrap_recovers_after_sdk_or_receipt_publication_fsync_boundary(
    tmp_path,
    monkeypatch,
    fail_fsync_call,
):
    home, _archive, downloads, _validations, run = _tiny_bootstrap_fixture(
        tmp_path,
        monkeypatch,
    )
    original = launcher._fsync_directory
    calls = 0

    def fail_once(path, *, error_code):
        nonlocal calls
        calls += 1
        if calls == fail_fsync_call:
            raise OwnerLauncherError(f"simulated_fsync_{fail_fsync_call}")
        return original(path, error_code=error_code)

    monkeypatch.setattr(launcher, "_fsync_directory", fail_once)
    with pytest.raises(
        OwnerLauncherError,
        match=f"simulated_fsync_{fail_fsync_call}",
    ):
        run()

    sdk = home / launcher._TRUSTED_SDK_RELATIVE
    receipt_path = (
        home / f".hermes/trusted/trusted-runtime-bootstrap-{RELEASE_SHA}.json"
    )
    assert sdk.is_dir()
    assert receipt_path.exists() is (fail_fsync_call == 3)
    receipt = run(1_001)
    assert receipt["ok"] is True
    assert len(downloads) == 1


def test_existing_sdk_without_exact_intent_or_with_tree_drift_is_denied(
    tmp_path,
    monkeypatch,
):
    home, _archive, _downloads, _validations, run = _tiny_bootstrap_fixture(
        tmp_path,
        monkeypatch,
    )
    receipt = run()
    receipt_path = (
        home / f".hermes/trusted/trusted-runtime-bootstrap-{RELEASE_SHA}.json"
    )
    receipt_path.unlink()
    sdk_module = home / launcher._TRUSTED_SDK_RELATIVE / "lib/gcloud.py"
    sdk_module.write_bytes(b"DRIFTED = True\n")

    with pytest.raises(
        OwnerLauncherError,
        match="trusted_runtime_publication_intent_invalid",
    ):
        run(1_001)
    assert receipt["sdk_publication_intent_sha256"]
    assert not receipt_path.exists()

    intent_path = home / launcher._TRUSTED_SDK_PUBLICATION_INTENT_RELATIVE
    intent_path.unlink()
    with pytest.raises(
        OwnerLauncherError,
        match="trusted_runtime_publication_intent_unavailable",
    ):
        run(1_002)


def test_two_bootstraps_reconcile_the_identical_no_replace_receipt_winner(
    tmp_path,
    monkeypatch,
):
    home, _archive, downloads, validations, run = _tiny_bootstrap_fixture(
        tmp_path,
        monkeypatch,
    )
    original = launcher._write_owner_file_no_replace
    raced = False

    def publish_winner_then_report_exists(destination, payload, **kwargs):
        nonlocal raced
        if "trusted-runtime-bootstrap-" in os.path.basename(destination) and not raced:
            raced = True
            original(destination, payload, **kwargs)
            raise OwnerLauncherError("trusted_runtime_bootstrap_receipt_exists")
        return original(destination, payload, **kwargs)

    monkeypatch.setattr(
        launcher,
        "_write_owner_file_no_replace",
        publish_winner_then_report_exists,
    )
    receipt = run()

    stored = json.loads(
        (
            home / f".hermes/trusted/trusted-runtime-bootstrap-{RELEASE_SHA}.json"
        ).read_text()
    )
    assert receipt == stored
    assert raced is True
    assert len(downloads) == 1
    assert validations == [RELEASE_SHA]


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "../escape",
        "google-cloud-sdk/lib/__pycache__/evil.pyc",
        "google-cloud-sdk/lib/evil.pyo",
    ],
)
def test_trusted_runtime_archive_rejects_traversal_and_bytecode(
    tmp_path,
    forbidden_name,
):
    archive_path = tmp_path / "sdk.tar.gz"
    archive_path.write_bytes(_tiny_sdk_archive(forbidden_name=forbidden_name))
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(
        OwnerLauncherError, match="trusted_runtime_archive_member_invalid"
    ):
        launcher._safe_extract_gcloud_archive(str(archive_path), str(stage))


def test_trusted_sdk_tree_rejects_adjacent_python_bytecode(tmp_path):
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "module.pyc").write_bytes(b"untrusted-bytecode")

    with pytest.raises(
        OwnerLauncherError,
        match="trusted_gcloud_sdk_bytecode_forbidden",
    ):
        launcher.TrustedGcloudExecutable._capture_tree(str(sdk), scope="sdk")


def test_actual_pinned_google_archive_bootstraps_intent_sdk_receipt_and_retries(
    tmp_path,
    monkeypatch,
):
    fixed_python = launcher.Path.home() / launcher._TRUSTED_PYTHON_RELATIVE
    archive_path = launcher.Path("/tmp/google-cloud-cli-569.0.0-darwin-arm.tar.gz")
    if not archive_path.is_file():
        pytest.skip("retained pinned Google archive fixture is not available")
    payload = archive_path.read_bytes()
    assert len(payload) == launcher._GCLOUD_SDK_ARCHIVE_BYTES
    assert hashlib.sha256(payload).hexdigest() == launcher._GCLOUD_SDK_ARCHIVE_SHA256
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
    assert sum(member.isdir() for member in members) == 1

    home = tmp_path / "owner"
    (home / ".hermes").mkdir(parents=True)
    (home / ".hermes").chmod(0o700)

    class StablePath:
        def __init__(self, selected, **_kwargs):
            self.selected = selected

        def absolute_path(self):
            return self.selected

    python = StablePath(str(home / "python/bin/python3.11"))
    monkeypatch.setattr(launcher, "_canonical_owner_home", lambda **_kwargs: str(home))
    monkeypatch.setattr(launcher, "_PinnedExecutablePath", StablePath)
    downloads = []

    def downloader(destination):
        downloads.append(destination)
        launcher.Path(destination).write_bytes(payload)

    def python_snapshot():
        return (
            python,
            str(home / "python"),
            (3, 100, "a" * 64),
            "3.11.15",
            ("/usr/lib/x",),
        )

    receipt = launcher.bootstrap_trusted_gcloud_runtime(
        RELEASE_SHA,
        now_unix=1_000,
        archive_downloader=downloader,
        python_snapshot=python_snapshot,
        runtime_validator=lambda _release: None,
    )
    retried = launcher.bootstrap_trusted_gcloud_runtime(
        RELEASE_SHA,
        now_unix=1_001,
        archive_downloader=downloader,
        python_snapshot=python_snapshot,
        runtime_validator=lambda _release: None,
    )
    sdk = home / launcher._TRUSTED_SDK_RELATIVE
    tree = launcher.TrustedGcloudExecutable._capture_tree(str(sdk), scope="sdk")

    assert tree[0] > len(members)
    assert (sdk / "bin/gcloud").is_file()
    assert (sdk / "lib/gcloud.py").is_file()
    assert not list(sdk.rglob("*.pyc"))
    assert not list(sdk.rglob("__pycache__"))
    assert receipt == retried
    assert receipt["sdk_publication_intent_sha256"]
    assert len(downloads) == 1
    if not fixed_python.is_file():
        pytest.skip("fixed uv Python execution anchor is not available")
    before_execution = launcher._capture_sdk_publication_tree(str(sdk))
    config_root = home / "offline-gcloud-config"
    config_root.mkdir(mode=0o700)
    completed = subprocess.run(
        (
            str(fixed_python),
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/var/empty/muncho-canary",
            str(sdk / "lib/gcloud.py"),
            "version",
            "--format=json",
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "HOME": str(home),
            "CLOUDSDK_CONFIG": str(config_root),
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "TMPDIR": "/tmp",
            "LANG": "C",
            "LC_ALL": "C",
            "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
            "CLOUDSDK_CORE_DISABLE_USAGE_REPORTING": "1",
            "CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK": "1",
            "CLOUDSDK_CORE_DISABLE_FILE_LOGGING": "1",
            "CLOUDSDK_PYTHON": str(fixed_python),
            "CLOUDSDK_PYTHON_ARGS": (
                "-I -S -B -X pycache_prefix=/var/empty/muncho-canary"
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        timeout=30.0,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    version = json.loads(completed.stdout)
    assert version["Google Cloud SDK"] == "569.0.0"
    assert launcher._capture_sdk_publication_tree(str(sdk)) == before_execution


def test_release_bound_runtime_receipt_and_full_trees_are_rechecked(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "owner"
    sdk = home / ".hermes/trusted/google-cloud-sdk-569.0.0"
    python_root = home / ".local/share/uv/python/cpython-3.11.15-macos-aarch64-none"
    (sdk / "bin").mkdir(parents=True)
    (sdk / "lib").mkdir()
    (python_root / "bin").mkdir(parents=True)
    for directory in (
        home / ".hermes",
        home / ".hermes/trusted",
        sdk,
        sdk / "bin",
        sdk / "lib",
        home / ".local",
        home / ".local/share",
        home / ".local/share/uv",
        home / ".local/share/uv/python",
        python_root,
        python_root / "bin",
    ):
        directory.chmod(0o700)
    gcloud = sdk / "bin/gcloud"
    gcloud.write_text("#!/bin/sh\nexit 0\n")
    gcloud.chmod(0o700)
    module = sdk / "lib/gcloud.py"
    module.write_text("VALUE = 1\n")
    module.chmod(0o600)
    (sdk / "VERSION").write_text("569.0.0\n")
    (sdk / "VERSION").chmod(0o600)
    python = python_root / "bin/python3.11"
    python.write_text("fixed-python\n")
    python.chmod(0o700)

    class StablePath:
        def __init__(self, selected, **_kwargs):
            self.selected = selected

        def absolute_path(self):
            return self.selected

    monkeypatch.setattr(launcher, "_canonical_owner_home", lambda **_kwargs: str(home))
    monkeypatch.setattr(launcher, "_PinnedExecutablePath", StablePath)
    monkeypatch.setattr(
        launcher.TrustedGcloudExecutable,
        "_capture_python_dependencies",
        lambda _self: launcher._TRUSTED_PYTHON_DEPENDENCIES,
    )
    python_version = ["3.11.15"]
    monkeypatch.setattr(
        launcher.TrustedGcloudExecutable,
        "_capture_python_version",
        lambda _self: python_version[0],
    )
    sdk_tree = launcher.TrustedGcloudExecutable._capture_tree(str(sdk), scope="sdk")
    python_tree = launcher.TrustedGcloudExecutable._capture_tree(
        str(python_root),
        scope="python_tree",
    )
    publication_tree = launcher._capture_sdk_publication_tree(str(sdk))
    launcher_sha256 = launcher._current_launcher_sha256()
    intent_unsigned = {
        "schema": launcher.TRUSTED_SDK_PUBLICATION_INTENT_SCHEMA,
        "ok": True,
        "state": "trusted_sdk_publication_prepared",
        "publication_release_sha": RELEASE_SHA,
        "launcher_sha256": launcher_sha256,
        "sdk_archive_url": launcher._GCLOUD_SDK_ARCHIVE_URL,
        "sdk_archive_bytes": launcher._GCLOUD_SDK_ARCHIVE_BYTES,
        "sdk_archive_sha256": launcher._GCLOUD_SDK_ARCHIVE_SHA256,
        "sdk_version": launcher._GCLOUD_SDK_VERSION,
        "sdk_root": str(sdk),
        "sdk_tree_entries": publication_tree[0],
        "sdk_tree_bytes": publication_tree[1],
        "sdk_tree_sha256": publication_tree[2],
        "prepared_at_unix": 900,
    }
    intent = {
        **intent_unsigned,
        "intent_sha256": hashlib.sha256(_canonical(intent_unsigned)).hexdigest(),
    }
    intent_path = home / launcher._TRUSTED_SDK_PUBLICATION_INTENT_RELATIVE
    intent_path.write_bytes(_canonical(intent) + b"\n")
    intent_path.chmod(0o600)
    unsigned = {
        "schema": launcher.TRUSTED_RUNTIME_BOOTSTRAP_RECEIPT_SCHEMA,
        "ok": True,
        "state": "trusted_runtime_ready",
        "release_sha": RELEASE_SHA,
        "launcher_sha256": launcher_sha256,
        "sdk_archive_url": launcher._GCLOUD_SDK_ARCHIVE_URL,
        "sdk_archive_bytes": launcher._GCLOUD_SDK_ARCHIVE_BYTES,
        "sdk_archive_sha256": launcher._GCLOUD_SDK_ARCHIVE_SHA256,
        "sdk_version": launcher._GCLOUD_SDK_VERSION,
        "sdk_root": str(sdk),
        "sdk_tree_entries": sdk_tree[0],
        "sdk_tree_bytes": sdk_tree[1],
        "sdk_tree_sha256": sdk_tree[2],
        "sdk_publication_release_sha": RELEASE_SHA,
        "sdk_publication_intent_sha256": intent["intent_sha256"],
        "sdk_publication_tree_entries": publication_tree[0],
        "sdk_publication_tree_bytes": publication_tree[1],
        "sdk_publication_tree_sha256": publication_tree[2],
        "python_root": str(python_root),
        "python_executable": str(python),
        "python_version": "3.11.15",
        "python_tree_entries": python_tree[0],
        "python_tree_bytes": python_tree[1],
        "python_tree_sha256": python_tree[2],
        "python_dependencies": list(launcher._TRUSTED_PYTHON_DEPENDENCIES),
        "created_at_unix": 1_000,
    }
    receipt = {
        **unsigned,
        "receipt_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }
    receipt_path = (
        home / f".hermes/trusted/trusted-runtime-bootstrap-{RELEASE_SHA}.json"
    )
    receipt_path.write_bytes(_canonical(receipt) + b"\n")
    receipt_path.chmod(0o600)

    runtime = launcher.TrustedGcloudExecutable(release_sha=RELEASE_SHA)
    assert runtime.trusted_command_prefix()[-1] == str(module)

    python_version[0] = "3.11.14"
    with pytest.raises(OwnerLauncherError, match="trusted_python_version_changed"):
        runtime.trusted_command_prefix()
    python_version[0] = "3.11.15"

    receipt["created_at_unix"] = 1_001
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical({
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        })
    ).hexdigest()
    receipt_path.write_bytes(_canonical(receipt) + b"\n")
    with pytest.raises(
        OwnerLauncherError,
        match="trusted_runtime_bootstrap_receipt_changed",
    ):
        runtime.trusted_command_prefix()


def _write_gcloud_configuration(home, *, extra: str = ""):
    root = home / ".config/gcloud"
    configurations = root / "configurations"
    configurations.mkdir(parents=True)
    (home / ".config").chmod(0o700)
    (root / "active_config").write_text("adventico-ai-platform-admin")
    (configurations / "config_adventico-ai-platform-admin").write_text(
        "[core]\n"
        "project = adventico-ai-platform\n"
        "account = owner@example.com\n\n"
        "[compute]\n"
        "zone = europe-west3-a\n"
        f"{extra}"
    )
    return root


def test_gcloud_configuration_is_parsed_without_gcloud_and_pinned(tmp_path):
    root = _write_gcloud_configuration(tmp_path)
    configuration = launcher.PinnedGcloudConfiguration(
        owner_home=tmp_path,
        environment={},
    )

    assert configuration.account == "owner@example.com"
    assert configuration.environment_values() == {
        "HOME": str(tmp_path),
        "CLOUDSDK_CONFIG": str(root),
    }

    config_file = root / "configurations/config_adventico-ai-platform-admin"
    config_file.write_text(config_file.read_text().replace(PROJECT, "attacker-project"))
    with pytest.raises(
        OwnerLauncherError,
        match="trusted_gcloud_config_(invalid|changed)",
    ):
        configuration.assert_stable()


def test_gcloud_auth_root_requires_private_dot_config_ancestry(tmp_path):
    _write_gcloud_configuration(tmp_path)
    (tmp_path / ".config").chmod(0o755)

    with pytest.raises(OwnerLauncherError, match="trusted_gcloud_config_invalid"):
        launcher.PinnedGcloudConfiguration(
            owner_home=tmp_path,
            environment={},
        )


@pytest.mark.parametrize(
    "extra",
    [
        "impersonate_service_account = attacker@example.com\n",
        "proxy/type = http\n",
        "custom_ca_certs_file = /tmp/attacker.pem\n",
        "api_endpoint_overrides/sql = https://attacker.invalid\n",
        "log_http = true\n",
        "\n[auth]\naccess_token_file = /tmp/attacker-token\n",
    ],
)
def test_gcloud_configuration_rejects_every_extra_effective_property(
    tmp_path,
    extra,
):
    _write_gcloud_configuration(tmp_path, extra=extra)

    with pytest.raises(OwnerLauncherError, match="trusted_gcloud_config_invalid"):
        launcher.PinnedGcloudConfiguration(
            owner_home=tmp_path,
            environment={},
        )


def test_hostile_home_and_alternate_cloudsdk_config_fail_before_use(tmp_path):
    attacker_home = tmp_path / "attacker-home"
    attacker_home.mkdir()
    _write_gcloud_configuration(attacker_home)

    with pytest.raises(OwnerLauncherError, match="ambient_owner_home_forbidden"):
        launcher._canonical_owner_home(environment={"HOME": str(attacker_home)})

    clean_home = tmp_path / "clean-home"
    clean_home.mkdir()
    _write_gcloud_configuration(clean_home)
    with pytest.raises(OwnerLauncherError, match="ambient_gcloud_config_forbidden"):
        launcher.PinnedGcloudConfiguration(
            owner_home=clean_home,
            environment={"CLOUDSDK_CONFIG": str(attacker_home / ".config/gcloud")},
        )


def test_gcloud_runtime_and_config_are_attested_before_and_after_auth_call():
    events = []

    class Runtime:
        def trusted_command_prefix(self):
            events.append("runtime")
            return GCLOUD_COMMAND_PREFIX

    class Configuration(_StableGcloudConfiguration):
        @property
        def account(self):
            events.append("config_account")
            return "owner@example.com"

        def assert_stable(self):
            events.append("config")

        def environment_values(self):
            self.assert_stable()
            return super().environment_values()

    def runner(argv, **_kwargs):
        events.append("subprocess")
        return subprocess.CompletedProcess(argv, 0, b"owner@example.com\n", b"")

    provider = GcloudOwnerAccessToken(
        gcloud_executable=Runtime(),
        gcloud_configuration=Configuration(),
        runner=runner,
    )
    provider.account_for_read_only_preflight()

    assert events.index("runtime") < events.index("subprocess")
    assert events.index("config") < events.index("subprocess")
    assert events[-2:] == ["config", "config_account"]


def _known_host_line(host=None):
    target = host or f"compute.{launcher.VM_INSTANCE_ID}"
    algorithm = b"ssh-ed25519"
    blob = (
        struct.pack(">I", len(algorithm))
        + algorithm
        + struct.pack(">I", 32)
        + b"k" * 32
    )
    return f"{target} ssh-ed25519 {launcher.base64.b64encode(blob).decode()}\n"


def test_google_compute_known_hosts_is_pinned_and_drift_rejected(tmp_path):
    tmp_path.chmod(0o700)
    known_hosts_path = tmp_path / "google_compute_known_hosts"
    known_hosts_path.write_text(_known_host_line())
    known_hosts_path.chmod(0o644)
    private_key = tmp_path / "google_compute_engine"
    private_key.write_text("fixed-private-key\n")
    private_key.chmod(0o600)
    public_key = tmp_path / "google_compute_engine.pub"
    public_key.write_text("ssh-ed25519 fixed-public-key owner@example.com\n")
    public_key.chmod(0o644)
    known_hosts = launcher.PinnedGoogleComputeKnownHosts(known_hosts_path)

    assert known_hosts.absolute_path() == str(known_hosts_path)
    assert known_hosts.private_key_path() == str(private_key)
    assert (
        known_hosts.public_key_line()
        == "ssh-ed25519 fixed-public-key owner@example.com"
    )
    known_hosts_path.write_text(_known_host_line("compute.999"))
    with pytest.raises(OwnerLauncherError, match="trusted_known_hosts_changed"):
        known_hosts.absolute_path()


def test_google_compute_private_key_is_fixed_and_rechecked(tmp_path):
    tmp_path.chmod(0o700)
    known_hosts = tmp_path / "google_compute_known_hosts"
    known_hosts.write_text(_known_host_line())
    known_hosts.chmod(0o644)
    private_key = tmp_path / "google_compute_engine"
    private_key.write_text("fixed-private-key\n")
    private_key.chmod(0o600)
    public_key = tmp_path / "google_compute_engine.pub"
    public_key.write_text("ssh-ed25519 fixed-public-key owner@example.com\n")
    public_key.chmod(0o644)
    material = launcher.PinnedGoogleComputeKnownHosts(known_hosts)

    private_key.write_text("drifted-private-key\n")
    with pytest.raises(OwnerLauncherError, match="trusted_known_hosts_changed"):
        material.private_key_path()

    tmp_path.chmod(0o755)
    with pytest.raises(OwnerLauncherError, match="trusted_known_hosts_invalid"):
        launcher.PinnedGoogleComputeKnownHosts(known_hosts)


@pytest.mark.parametrize(
    "invalid_line",
    [
        "@cert-authority * ssh-ed25519 AAAA\n",
        "* ssh-ed25519 AAAA\n",
        "|1|hash|value ssh-ed25519 AAAA\n",
        "!compute.1 ssh-ed25519 AAAA\n",
        "compute.1,compute.2 ssh-ed25519 AAAA\n",
        f"compute.{launcher.VM_INSTANCE_ID} ssh-rsa AAAA\n",
        f"compute.{launcher.VM_INSTANCE_ID} ssh-ed25519 not-base64!\n",
        "# comment\n",
    ],
)
def test_known_hosts_semantics_reject_broad_or_malformed_trust(
    tmp_path,
    invalid_line,
):
    tmp_path.chmod(0o700)
    known_hosts = tmp_path / "google_compute_known_hosts"
    known_hosts.write_text(invalid_line)
    known_hosts.chmod(0o644)
    private_key = tmp_path / "google_compute_engine"
    private_key.write_text("fixed-private-key\n")
    private_key.chmod(0o600)
    public_key = tmp_path / "google_compute_engine.pub"
    public_key.write_text("ssh-ed25519 fixed-public-key owner@example.com\n")
    public_key.chmod(0o644)

    with pytest.raises(OwnerLauncherError, match="trusted_known_hosts_invalid"):
        launcher.PinnedGoogleComputeKnownHosts(known_hosts)


def test_known_hosts_requires_one_unique_exact_canary_target(tmp_path):
    tmp_path.chmod(0o700)
    known_hosts = tmp_path / "google_compute_known_hosts"
    target = _known_host_line()
    known_hosts.write_text(target + target)
    known_hosts.chmod(0o644)
    private_key = tmp_path / "google_compute_engine"
    private_key.write_text("fixed-private-key\n")
    private_key.chmod(0o600)
    public_key = tmp_path / "google_compute_engine.pub"
    public_key.write_text("ssh-ed25519 fixed-public-key owner@example.com\n")
    public_key.chmod(0o644)

    with pytest.raises(OwnerLauncherError, match="trusted_known_hosts_invalid"):
        launcher.PinnedGoogleComputeKnownHosts(known_hosts)


def test_known_hosts_drift_fails_before_owner_auth_or_iap():
    events = []

    class Identity:
        def account_for_read_only_preflight(self):
            events.append("auth")
            return "owner@example.com"

    class DriftedKnownHosts:
        def absolute_path(self):
            events.append("known_hosts")
            raise OwnerLauncherError("trusted_known_hosts_changed")

    transport = IapCoordinatorTransport(
        Identity(),
        gcloud_executable=_StableExecutable(),
        gcloud_configuration=_StableGcloudConfiguration(),
        known_hosts=DriftedKnownHosts(),
    )

    with pytest.raises(OwnerLauncherError, match="trusted_known_hosts_changed"):
        transport._argv(RELEASE_SHA, "preflight-recovery", approved=False)
    assert events == ["known_hosts"]


def test_custom_ca_environment_is_rejected_before_token_or_https(monkeypatch):
    calls = []
    monkeypatch.setenv("SSL_CERT_FILE", "/tmp/attacker-ca.pem")
    client = GoogleRestClient(
        lambda: calls.append("token") or "owner-access-token",
        requester=lambda *args: calls.append("request"),
    )

    with pytest.raises(OwnerLauncherError, match="custom_ca_bundle_forbidden"):
        client.request_json(
            "GET",
            f"https://sqladmin.googleapis.com/sql/v1beta4/projects/{PROJECT}/"
            f"instances/{SQL_INSTANCE}/users",
        )
    with pytest.raises(OwnerLauncherError, match="custom_ca_bundle_forbidden"):
        launcher._owner_gcloud_environment(
            _StableGcloudConfiguration(),
            "/trusted/bin/python3.13",
        )
    assert calls == []


def test_owner_access_token_rejects_service_account_without_token_command():
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv,
            0,
            b"runtime@project.iam.gserviceaccount.com\n",
            b"",
        )

    provider = GcloudOwnerAccessToken(
        gcloud_executable=_StableExecutable(),
        gcloud_configuration=_StableGcloudConfiguration(),
        runner=runner,
    )
    with pytest.raises(OwnerLauncherError, match="active_owner_identity_unavailable"):
        provider.bind_approved_subject(OWNER_SUBJECT_SHA)

    assert len(calls) == 1


def _initialize_launcher_repo(tmp_path, *, tracked: bool = True):
    repository = tmp_path / ("tracked" if tracked else "untracked")
    module = repository / "scripts/canary/full_canary_owner_launcher.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(b"#!/usr/bin/env python3\nVALUE = 1\n")
    subprocess.run(("/usr/bin/git", "init", "-q", str(repository)), check=True)
    if tracked:
        subprocess.run(
            ("/usr/bin/git", "-C", str(repository), "add", str(module)),
            check=True,
        )
    else:
        marker = repository / "marker"
        marker.write_text("tracked\n")
        subprocess.run(
            ("/usr/bin/git", "-C", str(repository), "add", str(marker)),
            check=True,
        )
    subprocess.run(
        (
            "/usr/bin/git",
            "-C",
            str(repository),
            "-c",
            "user.name=Canary Test",
            "-c",
            "user.email=canary@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ),
        check=True,
    )
    sha = (
        subprocess
        .run(
            ("/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"),
            check=True,
            stdout=subprocess.PIPE,
        )
        .stdout.decode("ascii")
        .strip()
    )
    return module, sha


def test_local_launcher_provenance_accepts_exact_clean_tracked_release(tmp_path):
    module, sha = _initialize_launcher_repo(tmp_path)

    observed = launcher.LocalLauncherProvenance(module_path=module)(sha)

    assert observed == hashlib.sha256(module.read_bytes()).hexdigest()


def test_local_launcher_provenance_rejects_dirty_module(tmp_path):
    module, sha = _initialize_launcher_repo(tmp_path)
    module.write_bytes(module.read_bytes() + b"DIRTY = True\n")

    with pytest.raises(OwnerLauncherError, match="local_launcher_dirty"):
        launcher.LocalLauncherProvenance(module_path=module)(sha)


def test_local_launcher_provenance_rejects_untracked_module(tmp_path):
    module, sha = _initialize_launcher_repo(tmp_path, tracked=False)

    with pytest.raises(OwnerLauncherError, match="local_launcher_untracked"):
        launcher.LocalLauncherProvenance(module_path=module)(sha)


def test_local_launcher_provenance_rejects_wrong_head(tmp_path):
    module, _sha = _initialize_launcher_repo(tmp_path)

    with pytest.raises(OwnerLauncherError, match="local_launcher_release_mismatch"):
        launcher.LocalLauncherProvenance(module_path=module)("f" * 40)


def test_owner_interpreter_requires_every_isolation_flag(monkeypatch):
    isolated_flags = SimpleNamespace(
        isolated=1,
        no_site=1,
        dont_write_bytecode=1,
        no_user_site=1,
        ignore_environment=1,
        safe_path=True,
    )
    fake_sys = SimpleNamespace(
        executable=sys.executable,
        flags=isolated_flags,
        pycache_prefix="/var/empty/muncho-canary",
    )
    monkeypatch.setattr(launcher, "sys", fake_sys)

    launcher._validate_owner_interpreter_invocation(sys.executable)
    isolated_flags.safe_path = False
    with pytest.raises(OwnerLauncherError, match="trusted_owner_interpreter_invalid"):
        launcher._validate_owner_interpreter_invocation(sys.executable)


def test_cli_interpreter_then_provenance_precede_gcloud_or_secret_construction(
    monkeypatch,
    capfd,
):
    events = []

    def reject(_release):
        events.append("provenance")
        raise OwnerLauncherError("local_launcher_dirty")

    class Runtime:
        pass

    def trusted_runtime(_release):
        events.append("interpreter")
        return Runtime()

    monkeypatch.setattr(launcher, "require_trusted_owner_runtime", trusted_runtime)
    monkeypatch.setattr(launcher, "require_local_launcher_provenance", reject)

    assert launcher.main(("--release-sha", RELEASE_SHA)) == 2
    assert events == ["interpreter", "provenance"]
    output = json.loads(capfd.readouterr().out)
    assert output["error_code"] == "local_launcher_dirty"


def test_cli_terminal_provenance_recheck_runs_after_raised_launch_and_wins(
    monkeypatch,
    capfd,
):
    events = []

    class Runtime:
        def trusted_command_prefix(self):
            events.append("runtime")
            return GCLOUD_COMMAND_PREFIX

    provenance_calls = 0

    def provenance(_release):
        nonlocal provenance_calls
        provenance_calls += 1
        events.append(f"provenance_{provenance_calls}")
        if provenance_calls == 2:
            raise OwnerLauncherError("local_launcher_changed")
        return "a" * 64

    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_runtime",
        lambda _release: Runtime(),
    )
    monkeypatch.setattr(launcher, "require_local_launcher_provenance", provenance)
    monkeypatch.setattr(
        launcher,
        "_validate_owner_interpreter_invocation",
        lambda _path: events.append("interpreter"),
    )
    monkeypatch.setattr(launcher, "PinnedGcloudConfiguration", lambda: object())
    monkeypatch.setattr(launcher, "GcloudOwnerAccessToken", lambda *_a, **_k: object())
    monkeypatch.setattr(launcher, "IapCoordinatorTransport", lambda *_a, **_k: object())
    monkeypatch.setattr(launcher, "GoogleRestClient", lambda *_a, **_k: object())
    monkeypatch.setattr(launcher, "CloudSqlTemporaryAdmin", lambda *_a, **_k: object())
    monkeypatch.setattr(launcher, "OwnerDiscordTokenReader", lambda: object())
    monkeypatch.setattr(launcher, "FixedLocalFinalApprovalFile", lambda: object())

    def raised_launch(**_kwargs):
        events.append("launch")
        raise OwnerLauncherError("primary_launch_failure")

    monkeypatch.setattr(launcher, "launch_full_canary", raised_launch)

    assert launcher.main(("--release-sha", RELEASE_SHA)) == 2
    output = json.loads(capfd.readouterr().out)
    assert output["error_code"] == "local_launcher_changed"
    assert events == [
        "provenance_1",
        "launch",
        "runtime",
        "interpreter",
        "provenance_2",
    ]


class _CloudSqlClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def _cloud_sql_user_item(name: str = USERNAME) -> dict[str, object]:
    return {
        "kind": "sql#user",
        "name": name,
        "project": PROJECT,
        "instance": SQL_INSTANCE,
        "type": "BUILT_IN",
    }


def _cloud_sql_users_payload(names=()) -> dict[str, object]:
    return {
        "kind": "sql#usersList",
        "items": [_cloud_sql_user_item(name) for name in names],
    }


def _cloud_sql_operation_item(
    name: str,
    operation_type: str,
    *,
    status: str = "DONE",
) -> dict[str, object]:
    return {
        "kind": "sql#operation",
        "name": name,
        "operationType": operation_type,
        "status": status,
        "targetProject": PROJECT,
        "targetId": SQL_INSTANCE,
        "selfLink": (
            "https://sqladmin.googleapis.com/sql/v1beta4/projects/"
            f"{PROJECT}/operations/{name}"
        ),
        "targetLink": (
            "https://sqladmin.googleapis.com/sql/v1beta4/projects/"
            f"{PROJECT}/instances/{SQL_INSTANCE}"
        ),
        "user": "owner@example.com",
    }


def _cloud_sql_operations_payload(items=()) -> dict[str, object]:
    return {"kind": "sql#operationsList", "items": list(items)}


def test_actual_http_error_499_is_ambiguous_and_never_reads_response_body(
    monkeypatch,
):
    class SecretErrorBody:
        def __init__(self) -> None:
            self.closed = False

        def read(self, *_args):
            raise AssertionError("HTTP error body must never be read")

        def close(self):
            self.closed = True

    secret_body = SecretErrorBody()

    class Opener:
        def open(self, request, *, timeout):
            raise launcher.urllib.error.HTTPError(
                request.full_url,
                499,
                "secret-bearing error",
                {},
                secret_body,
            )

    monkeypatch.setattr(
        launcher.urllib.request,
        "build_opener",
        lambda *_handlers: Opener(),
    )
    client = GoogleRestClient(
        lambda: "owner-access-token",
        requester=launcher._default_http_request,
    )

    with pytest.raises(OwnerLauncherError, match="google_api_ambiguous_status"):
        client.request_json(
            "POST",
            (
                "https://sqladmin.googleapis.com/sql/v1beta4/projects/"
                f"{PROJECT}/instances/{SQL_INSTANCE}/users"
            ),
            body={"password": PASSWORD.decode()},
        )

    assert secret_body.closed is True


def test_cloud_sql_operations_pagination_cycle_is_evidence_incomplete():
    def request(method, url, headers, body, timeout):
        assert method == "GET"
        token = launcher.urllib.parse.parse_qs(
            launcher.urllib.parse.urlsplit(url).query
        ).get("pageToken", [None])[0]
        payload = _cloud_sql_operations_payload()
        payload["nextPageToken"] = "page-2" if token is None else "page-2"
        return HttpResponse(200, _canonical(payload))

    admin = CloudSqlTemporaryAdmin(
        GoogleRestClient(lambda: "owner-access-token", requester=request)
    )

    with pytest.raises(
        OwnerLauncherError,
        match="cloud_sql_operations_evidence_incomplete",
    ):
        admin.begin_mutation_observation()


def test_cloud_sql_users_page_one_shift_at_end_fence_is_rejected():
    first_page_reads = 0

    def request(method, url, headers, body, timeout):
        nonlocal first_page_reads
        assert method == "GET"
        token = launcher.urllib.parse.parse_qs(
            launcher.urllib.parse.urlsplit(url).query
        ).get("pageToken", [None])[0]
        if token == "page-2":
            return HttpResponse(200, _canonical(_cloud_sql_users_payload()))
        first_page_reads += 1
        payload = _cloud_sql_users_payload([USERNAME] if first_page_reads > 1 else [])
        payload["nextPageToken"] = "page-2"
        return HttpResponse(200, _canonical(payload))

    admin = CloudSqlTemporaryAdmin(
        GoogleRestClient(lambda: "owner-access-token", requester=request)
    )

    with pytest.raises(OwnerLauncherError, match="invalid_cloud_sql_users"):
        admin.require_absent(USERNAME)


@pytest.mark.parametrize(
    "second_snapshot",
    [
        {"authority-op": ("CREATE_USER", "DONE", "f" * 64, True)},
        {"authority-op": ("CREATE_USER", "DONE", OWNER_SUBJECT_SHA, False)},
        {},
    ],
    ids=("actor-drift", "outcome-drift", "done-disappeared"),
)
def test_cloud_sql_done_authority_drift_or_disappearance_fails_closed(
    second_snapshot,
):
    admin = CloudSqlTemporaryAdmin(GoogleRestClient(lambda: "owner-access-token"))
    expected = {
        "authority-op": (
            "CREATE_USER",
            "DONE",
            OWNER_SUBJECT_SHA,
            True,
        )
    }
    snapshots = iter((expected, second_snapshot))
    admin._mutation_operation_baseline = frozenset()  # type: ignore[attr-defined]
    admin._mutation_relevant_baseline = {}  # type: ignore[attr-defined]
    admin._confirmed_authority_operation_name = "authority-op"  # type: ignore[attr-defined]
    admin._confirmed_authority_operation_type = "CREATE_USER"  # type: ignore[attr-defined]
    admin._expected_owner_subject_sha256 = OWNER_SUBJECT_SHA  # type: ignore[attr-defined]
    admin._instance_operations = lambda: next(snapshots)  # type: ignore[method-assign]
    admin._user_names = (  # type: ignore[method-assign]
        lambda *, exact_admin_username=None: frozenset({USERNAME})
    )

    with pytest.raises(
        OwnerLauncherError,
        match="cloud_sql_mutation_evidence_unconfirmed",
    ):
        admin.require_current_authority(USERNAME)

    assert admin.mutation_reconciliation_required() is True


def test_cloud_sql_password_exists_only_in_https_json_body_and_delete_is_proven():
    calls: list[tuple[str, str, bytes | None]] = []
    users = set()
    operation = 0
    operations: list[dict[str, object]] = []

    def request(method, url, headers, body, timeout):
        nonlocal operation
        calls.append((method, url, body))
        if method == "GET" and "/operations?" in url:
            return HttpResponse(
                200, _canonical(_cloud_sql_operations_payload(operations))
            )
        if method == "GET" and url.endswith("/users"):
            return HttpResponse(
                200,
                _canonical(_cloud_sql_users_payload(sorted(users))),
            )
        if method == "POST" and url.endswith("/users"):
            decoded = json.loads(body)
            assert decoded == {
                "instance": SQL_INSTANCE,
                "name": USERNAME,
                "password": PASSWORD.decode(),
                "project": PROJECT,
                "type": "BUILT_IN",
            }
            users.add(USERNAME)
            operation += 1
            operations.append(
                _cloud_sql_operation_item(f"operation-{operation}", "CREATE_USER")
            )
            return HttpResponse(200, _canonical({"name": f"operation-{operation}"}))
        if method == "DELETE" and "/users?" in url:
            users.discard(USERNAME)
            operation += 1
            operations.append(
                _cloud_sql_operation_item(f"operation-{operation}", "DELETE_USER")
            )
            return HttpResponse(200, _canonical({"name": f"operation-{operation}"}))
        if method == "GET" and "/operations/" in url:
            name = url.rsplit("/", 1)[-1]
            operation_type = "CREATE_USER" if name == "operation-1" else "DELETE_USER"
            return HttpResponse(
                200,
                _canonical(_cloud_sql_operation_item(name, operation_type)),
            )
        raise AssertionError((method, url))

    clock = _CloudSqlClock()
    admin = CloudSqlTemporaryAdmin(
        GoogleRestClient(lambda: "owner-access-token", requester=request),
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
        operation_timeout_seconds=2.0,
    )
    admin.require_absent(USERNAME)
    admin.create(USERNAME, PASSWORD.decode())
    admin.delete_and_confirm_absent(USERNAME)

    password_calls = [call for call in calls if call[2] and PASSWORD in call[2]]
    assert len(password_calls) == 1
    assert password_calls[0][0] == "POST"
    assert all(PASSWORD.decode() not in url for _method, url, _body in calls)
    assert all(
        "--password" not in url and "password=" not in url
        for _method, url, _body in calls
    )
    assert USERNAME not in users


def test_cloud_sql_delete_without_absent_proof_is_cleanup_blocked():
    def request(method, url, headers, body, timeout):
        if method == "DELETE":
            return HttpResponse(200, _canonical({"name": "delete-operation"}))
        if method == "GET" and "/operations?" in url:
            return HttpResponse(200, _canonical(_cloud_sql_operations_payload()))
        if "/operations/" in url:
            return HttpResponse(
                200,
                _canonical(
                    _cloud_sql_operation_item("delete-operation", "DELETE_USER")
                ),
            )
        if method == "GET" and "/users" in url:
            return HttpResponse(200, _canonical(_cloud_sql_users_payload([USERNAME])))
        raise AssertionError((method, url))

    clock = _CloudSqlClock()
    admin = CloudSqlTemporaryAdmin(
        GoogleRestClient(lambda: "owner-access-token", requester=request),
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
        operation_timeout_seconds=2.0,
    )

    with pytest.raises(CleanupBlocked):
        admin.delete_and_confirm_absent(USERNAME)


def test_cloud_sql_ambiguous_create_is_never_retried_or_accepted_in_call():
    calls = []

    def request(method, url, headers, body, timeout):
        calls.append(method)
        if method == "GET" and "/operations?" in url:
            return HttpResponse(200, _canonical(_cloud_sql_operations_payload()))
        if method == "POST":
            raise OwnerLauncherError("google_api_unavailable")
        raise AssertionError((method, url))

    admin = CloudSqlTemporaryAdmin(
        GoogleRestClient(lambda: "owner-access-token", requester=request),
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(OwnerLauncherError, match="google_api_unavailable"):
        admin.create(USERNAME, PASSWORD.decode())
    assert admin.mutation_reconciliation_required() is True
    assert calls.count("POST") == 1
    assert calls[-1] == "POST"
    assert set(calls[:-1]) == {"GET"}
    assert "PUT" not in calls


def test_cloud_sql_explicit_create_rejection_never_claims_concurrent_presence():
    calls = []

    def request(method, url, headers, body, timeout):
        calls.append(method)
        if method == "GET" and "/operations?" in url:
            return HttpResponse(200, _canonical(_cloud_sql_operations_payload()))
        if method == "POST":
            return HttpResponse(409, b"{}")
        if method == "GET" and url.endswith("/users"):
            return HttpResponse(200, _canonical({"items": [{"name": USERNAME}]}))
        raise AssertionError((method, url))

    admin = CloudSqlTemporaryAdmin(
        GoogleRestClient(lambda: "owner-access-token", requester=request),
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(OwnerLauncherError, match="google_api_rejected"):
        admin.create(USERNAME, PASSWORD.decode())
    assert calls.count("POST") == 1
    assert calls[-1] == "POST"
    assert set(calls[:-1]) == {"GET"}
    assert "PUT" not in calls


def test_cloud_sql_operation_timeout_requires_confirmed_password_reset():
    calls = []
    monotonic_values = iter((0.0, 1.0, 2.0))

    def request(method, url, headers, body, timeout):
        calls.append((method, url))
        if method == "GET" and "/operations?" in url:
            return HttpResponse(200, _canonical(_cloud_sql_operations_payload()))
        if method == "POST":
            return HttpResponse(200, _canonical({"name": "create-operation"}))
        if method == "GET" and url.endswith("/operations/create-operation"):
            return HttpResponse(
                200,
                _canonical(
                    _cloud_sql_operation_item(
                        "create-operation", "CREATE_USER", status="RUNNING"
                    )
                ),
            )
        raise AssertionError((method, url))

    admin = CloudSqlTemporaryAdmin(
        GoogleRestClient(lambda: "owner-access-token", requester=request),
        monotonic=lambda: next(monotonic_values),
        sleeper=lambda _seconds: None,
        operation_timeout_seconds=0.5,
    )

    with pytest.raises(OwnerLauncherError, match="cloud_sql_operation_timeout"):
        admin.create(USERNAME, PASSWORD.decode())
    methods = [method for method, _url in calls]
    assert methods.count("POST") == 1
    assert methods.count("PUT") == 0
    assert (
        sum(
            method == "GET" and url.endswith("/operations/create-operation")
            for method, url in calls
        )
        == 1
    )


@pytest.mark.parametrize("initially_present", [False, True])
def test_cloud_sql_recovery_creates_or_rotates_exact_admin(initially_present):
    present = initially_present
    calls = []
    operations = []

    def request(method, url, headers, body, timeout):
        nonlocal present
        calls.append(method)
        if method == "GET" and "/operations?" in url:
            return HttpResponse(
                200, _canonical(_cloud_sql_operations_payload(operations))
            )
        if method == "GET" and url.endswith("/users"):
            names = [USERNAME] if present else []
            return HttpResponse(200, _canonical(_cloud_sql_users_payload(names)))
        if method == "POST" and url.endswith("/users"):
            assert not present
            assert json.loads(body)["password"] == PASSWORD.decode()
            present = True
            operations.append(
                _cloud_sql_operation_item("create-operation", "CREATE_USER")
            )
            return HttpResponse(200, _canonical({"name": "create-operation"}))
        if method == "PUT" and "/users?" in url:
            assert present
            assert json.loads(body)["password"] == PASSWORD.decode()
            operations.append(
                _cloud_sql_operation_item("rotate-operation", "UPDATE_USER")
            )
            return HttpResponse(200, _canonical({"name": "rotate-operation"}))
        if method == "GET" and "/operations/" in url:
            name = url.rsplit("/", 1)[-1]
            operation_type = (
                "CREATE_USER" if name == "create-operation" else "UPDATE_USER"
            )
            return HttpResponse(
                200,
                _canonical(_cloud_sql_operation_item(name, operation_type)),
            )
        raise AssertionError((method, url))

    admin = CloudSqlTemporaryAdmin(
        GoogleRestClient(lambda: "owner-access-token", requester=request),
        sleeper=lambda _seconds: None,
    )
    admin.create_or_rotate_recovery(USERNAME, PASSWORD.decode())

    assert present is True
    if initially_present:
        assert calls.count("PUT") == 1
        assert calls.count("POST") == 0
    else:
        assert calls.count("POST") == 1
        assert calls.count("PUT") == 0
    assert set(calls) <= {"GET", "POST", "PUT"}


def test_cloud_sql_delete_skips_delete_when_initial_list_proves_absence():
    calls = []

    def request(method, url, headers, body, timeout):
        calls.append(method)
        assert method == "GET"
        payload = (
            _cloud_sql_operations_payload()
            if "/operations?" in url
            else _cloud_sql_users_payload()
        )
        return HttpResponse(200, _canonical(payload))

    clock = _CloudSqlClock()
    admin = CloudSqlTemporaryAdmin(
        GoogleRestClient(lambda: "owner-access-token", requester=request),
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
        operation_timeout_seconds=2.0,
    )

    admin.delete_and_confirm_absent(USERNAME)
    assert set(calls) == {"GET"}
    # A/U/B fencing plus the short test quiet window stays bounded; production
    # uses a five-second cadence to remain comfortably below LIST quota.
    assert calls.count("GET") <= 30
    assert admin.reconciliation_evidence()["reconciliation_proven"] is True


def test_cloud_sql_delete_error_succeeds_only_after_fresh_absence_proof():
    present = True
    calls = []

    def request(method, url, headers, body, timeout):
        nonlocal present
        calls.append(method)
        if method == "GET" and "/operations?" in url:
            return HttpResponse(200, _canonical(_cloud_sql_operations_payload()))
        if method == "GET" and url.endswith("/users"):
            names = [USERNAME] if present else []
            return HttpResponse(200, _canonical(_cloud_sql_users_payload(names)))
        if method == "DELETE":
            present = False
            raise OwnerLauncherError("google_api_unavailable")
        raise AssertionError((method, url))

    clock = _CloudSqlClock()
    admin = CloudSqlTemporaryAdmin(
        GoogleRestClient(lambda: "owner-access-token", requester=request),
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
        operation_timeout_seconds=2.0,
    )

    admin.delete_and_confirm_absent(USERNAME)
    assert calls.count("DELETE") == 1
    assert admin.reconciliation_evidence() == {
        "mutation_ambiguity_observed": True,
        "reconciliation_proven": True,
        "reconciliation_evidence_sha256": admin.reconciliation_evidence()[
            "reconciliation_evidence_sha256"
        ],
        "quiet_window_seconds": 2.0,
        "response_known_candidate_observed": False,
        "post_baseline_authority_operation_count": 0,
    }


def test_default_http_request_disables_proxies_and_rejects_redirect(monkeypatch):
    handlers = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self):
            return "https://attacker.invalid/redirected"

        def read(self, _maximum):
            return b"{}"

    class Opener:
        def open(self, request, *, timeout):
            return Response()

    def build_opener(*items):
        handlers.extend(items)
        return Opener()

    monkeypatch.setattr(launcher.urllib.request, "build_opener", build_opener)

    with pytest.raises(OwnerLauncherError, match="google_api_redirect_forbidden"):
        launcher._default_http_request(
            "GET",
            "https://sqladmin.googleapis.com/fixed",
            {},
            None,
            1.0,
        )

    proxy = next(
        item
        for item in handlers
        if isinstance(item, launcher.urllib.request.ProxyHandler)
    )
    redirect = next(
        item
        for item in handlers
        if isinstance(item, launcher.urllib.request.HTTPRedirectHandler)
    )
    assert proxy.proxies == {}
    assert (
        redirect.redirect_request(None, None, 302, "redirect", {}, "https://x") is None
    )


def test_iap_session_force_stops_hung_child_and_reports_unconfirmed():
    code = "import sys,time;sys.stdout.write('{}\\n');sys.stdout.flush();time.sleep(60)"

    def popen_factory(_argv, **kwargs):
        return subprocess.Popen((sys.executable, "-c", code), **kwargs)

    session = launcher._IapRemoteSession(
        (*GCLOUD_COMMAND_PREFIX, "fixed"),
        environment={},
        popen_factory=popen_factory,
        gate_timeout_seconds=1.0,
        termination_timeout_seconds=0.1,
    )
    assert session.read_gate() == {}

    with pytest.raises(launcher.RemoteTerminationUnconfirmed):
        session.close()


def test_iap_session_half_closed_child_times_out_without_fabricating_terminal():
    code = (
        "import sys,time;sys.stdout.write('{}\\n');sys.stdout.flush();"
        "sys.stdin.buffer.read();time.sleep(60)"
    )

    def popen_factory(_argv, **kwargs):
        return subprocess.Popen((sys.executable, "-c", code), **kwargs)

    session = launcher._IapRemoteSession(
        (*GCLOUD_COMMAND_PREFIX, "fixed"),
        environment={},
        popen_factory=popen_factory,
        gate_timeout_seconds=1.0,
        post_frame_timeout_seconds=0.05,
        termination_timeout_seconds=0.1,
    )
    assert session.read_gate() == {}
    with pytest.raises(OwnerLauncherError, match="iap_ssh_read_timeout"):
        session.finish(b"fixed-non-secret-frame")
    with pytest.raises(launcher.RemoteTerminationUnconfirmed):
        session.close()


def test_iap_session_accepts_exact_failure_only_with_eof_and_exit_two():
    failure = _early_terminal_failure(bound=False)
    code = (
        "import sys;"
        f"sys.stdout.buffer.write({(_canonical(failure) + bytes((10,)))!r});"
        "sys.stdout.flush();sys.exit(2)"
    )

    def popen_factory(_argv, **kwargs):
        return subprocess.Popen((sys.executable, "-c", code), **kwargs)

    session = launcher._IapRemoteSession(
        (*GCLOUD_COMMAND_PREFIX, "fixed"),
        environment={},
        popen_factory=popen_factory,
        gate_timeout_seconds=1.0,
        termination_timeout_seconds=1.0,
    )
    terminal = launcher.validate_terminal_first_failure(
        session.read_gate(),
        owner_gate=None,
        token_lease_expected=False,
        process_lease_expected=False,
    )
    session.mark_validated(terminal)
    session.close()
    assert session.termination_proven is True


def test_iap_session_runs_postflight_only_after_exact_terminal_exit_and_propagates():
    failure = _early_terminal_failure(bound=False)
    code = (
        "import sys;"
        f"sys.stdout.buffer.write({(_canonical(failure) + bytes((10,)))!r});"
        "sys.stdout.flush();sys.exit(2)"
    )

    def popen_factory(_argv, **kwargs):
        return subprocess.Popen((sys.executable, "-c", code), **kwargs)

    events = []
    session = None

    def postflight():
        events.append(("postflight", session._process.poll()))
        raise OwnerLauncherError("trusted_private_key_changed")

    session = launcher._IapRemoteSession(
        (*GCLOUD_COMMAND_PREFIX, "fixed"),
        environment={},
        popen_factory=popen_factory,
        gate_timeout_seconds=1.0,
        termination_timeout_seconds=1.0,
        postflight_guard=postflight,
    )
    terminal = launcher.validate_terminal_first_failure(
        session.read_gate(),
        owner_gate=None,
        token_lease_expected=False,
        process_lease_expected=False,
    )
    with pytest.raises(OwnerLauncherError, match="trusted_private_key_changed"):
        session.mark_validated(terminal)

    assert session.termination_proven is True
    assert events == [("postflight", 2)]


def test_iap_session_zero_byte_cancel_validates_exact_receipt_and_exit_two():
    gate = _approval_request()
    cancel = _approval_cancel_receipt()
    code = (
        "import sys;"
        f"sys.stdout.buffer.write({(_canonical(gate) + bytes((10,)))!r});"
        "sys.stdout.flush();"
        "data=sys.stdin.buffer.read();"
        "sys.exit(3) if data else None;"
        f"sys.stdout.buffer.write({(_canonical(cancel) + bytes((10,)))!r});"
        "sys.stdout.flush();sys.exit(2)"
    )

    def popen_factory(_argv, **kwargs):
        return subprocess.Popen((sys.executable, "-c", code), **kwargs)

    session = launcher._IapRemoteSession(
        (*GCLOUD_COMMAND_PREFIX, "fixed"),
        environment={},
        popen_factory=popen_factory,
        gate_timeout_seconds=1.0,
        post_frame_timeout_seconds=1.0,
        termination_timeout_seconds=1.0,
    )
    assert session.read_gate() == gate
    observed = session.cancel_no_secret()
    validated = launcher.validate_final_approval_cancel_receipt(
        observed,
        request=gate,
        now_unix=1_000,
    )
    session.mark_validated(validated)
    session.close()

    assert session.termination_proven is True
    assert session.terminal_receipt_validated is True


def test_iap_session_rejects_failure_receipt_with_success_exit_code():
    failure = _early_terminal_failure(bound=False)
    code = (
        "import sys;"
        f"sys.stdout.buffer.write({(_canonical(failure) + bytes((10,)))!r});"
        "sys.stdout.flush()"
    )

    def popen_factory(_argv, **kwargs):
        return subprocess.Popen((sys.executable, "-c", code), **kwargs)

    session = launcher._IapRemoteSession(
        (*GCLOUD_COMMAND_PREFIX, "fixed"),
        environment={},
        popen_factory=popen_factory,
        gate_timeout_seconds=1.0,
        termination_timeout_seconds=1.0,
    )
    terminal = launcher.validate_terminal_first_failure(
        session.read_gate(),
        owner_gate=None,
        token_lease_expected=False,
        process_lease_expected=False,
    )
    with pytest.raises(OwnerLauncherError, match="remote_terminal_exit_mismatch"):
        session.mark_validated(terminal)
    with pytest.raises(launcher.RemoteTerminationUnconfirmed):
        session.close()


def test_iap_transport_is_release_addressed_and_recovery_requires_approved_owner():
    class Identity:
        approved_account = "owner@example.com"

        def account_for_read_only_preflight(self):
            return "owner@example.com"

    transport = IapCoordinatorTransport(
        Identity(),
        gcloud_executable=_StableExecutable(),
        gcloud_configuration=_StableGcloudConfiguration(),
        known_hosts=_StableExecutable("/trusted/google_compute_known_hosts"),
    )
    argv = transport._argv(
        RELEASE_SHA,
        "stop-and-retire-discord-token",
        approved=True,
    )
    rendered = " ".join(argv)

    assert argv[: len(GCLOUD_COMMAND_PREFIX) + 3] == (
        *GCLOUD_COMMAND_PREFIX,
        "compute",
        "ssh",
        "lomliev_adventico_com@muncho-canary-v2-01",
    )
    assert f"--project={PROJECT}" in argv
    assert "--zone=europe-west3-a" in argv
    assert "--account=owner@example.com" in argv
    assert "--plain" in argv
    assert "--tunnel-through-iap" in argv
    assert "-oStrictHostKeyChecking=yes" in rendered
    assert "/usr/bin/sudo --non-interactive --" in rendered
    assert {
        "--ssh-flag=-F/dev/null",
        "--ssh-flag=-oPermitLocalCommand=no",
        "--ssh-flag=-oClearAllForwardings=yes",
        "--ssh-flag=-oControlMaster=no",
        "--ssh-flag=-oControlPath=none",
        "--ssh-flag=-oKnownHostsCommand=none",
        "--ssh-flag=-oCanonicalizeHostname=no",
        "--ssh-flag=-oForwardAgent=no",
        "--ssh-flag=-oIdentitiesOnly=yes",
        "--ssh-flag=-oIdentityAgent=none",
        "--ssh-flag=-oCertificateFile=none",
        "--ssh-flag=-oPreferredAuthentications=publickey",
        "--ssh-flag=-oPubkeyAuthentication=yes",
        "--ssh-flag=-oPasswordAuthentication=no",
        "--ssh-flag=-oKbdInteractiveAuthentication=no",
        "--ssh-flag=-oGSSAPIAuthentication=no",
        "--ssh-flag=-oHostbasedAuthentication=no",
        "--ssh-flag=-oEscapeChar=none",
        "--ssh-flag=-oUserKnownHostsFile=/trusted/google_compute_known_hosts",
        "--ssh-flag=-oGlobalKnownHostsFile=none",
        "--ssh-flag=-oUpdateHostKeys=no",
        "--ssh-flag=-oVerifyHostKeyDNS=no",
    }.issubset(set(argv))
    assert not any(item.startswith("--ssh-key-file=") for item in argv)
    assert "--ssh-flag=-i/trusted/google_compute_engine" in argv
    assert not any("ProxyCommand=" in item or "ProxyJump=" in item for item in argv)
    assert f"/opt/muncho-canary-releases/{RELEASE_SHA}/venv/bin/python" in rendered
    assert "stop-and-retire-discord-token" in rendered
    assert PASSWORD.decode() not in rendered
    assert DISCORD_TOKEN.decode() not in rendered


def _stopped_release_service_states():
    return [
        {
            "unit": unit,
            "state": "absent",
            "properties": {
                "LoadState": "not-found",
                "ActiveState": "inactive",
                "SubState": "dead",
                "UnitFileState": "",
                "MainPID": "0",
                "FragmentPath": "",
                "DropInPaths": "",
            },
        }
        for unit in launcher._STOPPED_RELEASE_UNITS
    ]


def _stopped_release_plan():
    machine = "1" * 64
    hostname = "2" * 64
    gce = {
        "project_id": PROJECT,
        "project_number": "39589465056",
        "zone": launcher.ZONE,
        "instance_name": launcher.VM_NAME,
        "instance_id": launcher.VM_INSTANCE_ID,
        "service_account_email": (
            "muncho-canary-v2-runtime@adventico-ai-platform.iam.gserviceaccount.com"
        ),
    }
    release_root = f"/opt/muncho-canary-releases/{RELEASE_SHA}"
    unsigned = {
        "schema": launcher.STOPPED_RELEASE_PLAN_SCHEMA,
        "revision": RELEASE_SHA,
        "source": {
            "repository": launcher.STOPPED_RELEASE_SOURCE_REPOSITORY,
            "root": f"{launcher.STOPPED_RELEASE_SOURCE_BASE}/{RELEASE_SHA}",
            "head_sha": RELEASE_SHA,
            "tree_sha": "b" * 40,
        },
        "release_root": release_root,
        "release_manifest_path": f"{release_root}/release-manifest.json",
        "evidence_receipt_path": (
            f"{launcher.STOPPED_RELEASE_EVIDENCE_BASE}/{RELEASE_SHA}/"
            "stopped-release-publication.json"
        ),
        "host_identity_receipt_path": launcher.STOPPED_RELEASE_HOST_RECEIPT_PATH,
        "python_version": launcher.STOPPED_RELEASE_PYTHON_VERSION,
        "interpreter": f"{release_root}/venv/bin/python",
        "tools": {
            "git": "/usr/bin/git",
            "systemctl": "/usr/bin/systemctl",
            "uv": "/usr/local/bin/uv",
            "uv_cache": "/var/cache/muncho-writer-release",
        },
        "dedicated_host": {
            **gce,
            "gce_identity_sha256": hashlib.sha256(_canonical(gce)).hexdigest(),
            "machine_id_sha256": machine,
            "hostname_sha256": hostname,
            "host_identity_sha256": hashlib.sha256(
                _canonical({
                    "machine_id_sha256": machine,
                    "hostname_sha256": hostname,
                })
            ).hexdigest(),
            "boot_id_sha256": "3" * 64,
        },
        "activation_inventory": [
            {"path": path, "state": "absent"}
            for path in launcher._STOPPED_RELEASE_ACTIVATION_PATHS
        ],
        "service_states": _stopped_release_service_states(),
    }
    return {
        **unsigned,
        "plan_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }


def _stopped_release_receipt(plan):
    release_root = f"/opt/muncho-canary-releases/{RELEASE_SHA}"
    unsigned = {
        "schema": launcher.STOPPED_RELEASE_RECEIPT_SCHEMA,
        "ok": True,
        "state": "published_services_stopped",
        "release_revision": RELEASE_SHA,
        "plan_sha256": plan["plan_sha256"],
        "source": plan["source"],
        "dedicated_host": plan["dedicated_host"],
        "activation_inventory": plan["activation_inventory"],
        "service_state_before": plan["service_states"],
        "service_state_after": plan["service_states"],
        "services_stopped_and_disabled": True,
        "tools": plan["tools"],
        "release_root": release_root,
        "release_manifest_path": f"{release_root}/release-manifest.json",
        "release_manifest_file_sha256": "4" * 64,
        "release_artifact_sha256": "5" * 64,
        "interpreter": f"{release_root}/venv/bin/python",
        "interpreter_sha256": "6" * 64,
        "python_version": launcher.STOPPED_RELEASE_PYTHON_VERSION,
        "retained_wheel_path": f"{release_root}/artifacts/hermes_agent.whl",
        "retained_wheel_sha256": "7" * 64,
        "build_constraints_sha256": "8" * 64,
        "host_identity_receipt_path": launcher.STOPPED_RELEASE_HOST_RECEIPT_PATH,
        "host_identity_receipt_file_sha256": "9" * 64,
        "host_identity_receipt_sha256": "a" * 64,
        "receipt_path": (
            f"{launcher.STOPPED_RELEASE_EVIDENCE_BASE}/{RELEASE_SHA}/"
            "stopped-release-publication.json"
        ),
        "created_at_unix": 1_000,
    }
    return {
        **unsigned,
        "receipt_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }


def test_stopped_release_owner_validators_bind_host_receipt_path_exactly():
    plan = _stopped_release_plan()
    assert (
        launcher.validate_stopped_release_plan(
            plan,
            expected_release_sha=RELEASE_SHA,
        )["plan_sha256"]
        == plan["plan_sha256"]
    )

    drifted_plan = json.loads(json.dumps(plan))
    drifted_plan["host_identity_receipt_path"] = "/tmp/host-identity.json"
    drifted_plan["plan_sha256"] = hashlib.sha256(
        _canonical({
            name: value for name, value in drifted_plan.items() if name != "plan_sha256"
        })
    ).hexdigest()
    with pytest.raises(OwnerLauncherError, match="stopped_release_plan_invalid"):
        launcher.validate_stopped_release_plan(
            drifted_plan,
            expected_release_sha=RELEASE_SHA,
        )

    receipt = _stopped_release_receipt(plan)
    assert (
        launcher.validate_stopped_release_receipt(receipt, plan=plan)["receipt_sha256"]
        == receipt["receipt_sha256"]
    )
    drifted_receipt = json.loads(json.dumps(receipt))
    drifted_receipt["host_identity_receipt_path"] = "/tmp/host-identity.json"
    drifted_receipt["receipt_sha256"] = hashlib.sha256(
        _canonical({
            name: value
            for name, value in drifted_receipt.items()
            if name != "receipt_sha256"
        })
    ).hexdigest()
    with pytest.raises(OwnerLauncherError, match="stopped_release_receipt_invalid"):
        launcher.validate_stopped_release_receipt(drifted_receipt, plan=plan)


def test_stopped_release_transport_renders_only_fixed_canary_iap_argv():
    identity = SimpleNamespace(
        account_for_read_only_preflight=lambda: "owner@example.com",
    )
    transport = IapStoppedReleaseTransport(
        identity,
        gcloud_executable=_StableExecutable(),
        gcloud_configuration=_StableGcloudConfiguration(),
        known_hosts=_StableExecutable("/trusted/google_compute_known_hosts"),
    )
    source_root = f"{launcher.STOPPED_RELEASE_SOURCE_BASE}/{RELEASE_SHA}"
    remote = (
        *transport._fixed_remote_environment(chdir=source_root),
        "/usr/bin/python3",
        "-B",
        "-E",
        "-s",
        "-m",
        "scripts.canary.writer_release",
        "plan",
        "--revision",
        RELEASE_SHA,
    )

    argv = transport._remote_argv(remote, account="owner@example.com")
    rendered = " ".join(argv)

    assert argv[: len(GCLOUD_COMMAND_PREFIX) + 3] == (
        *GCLOUD_COMMAND_PREFIX,
        "compute",
        "ssh",
        "lomliev_adventico_com@muncho-canary-v2-01",
    )
    assert f"--project={PROJECT}" in argv
    assert "--zone=europe-west3-a" in argv
    assert "--tunnel-through-iap" in argv
    assert f"--chdir={source_root}" in rendered
    assert "scripts.canary.writer_release plan" in rendered
    assert "python -c" not in rendered
    assert "heredoc" not in rendered
    assert "scp" not in rendered
    assert "skyai-runtime-prod-01" not in rendered
    assert "ai-platform-runtime-01" not in rendered
    assert PASSWORD.decode() not in rendered
    assert DISCORD_TOKEN.decode() not in rendered


def test_stopped_release_source_publication_is_revision_addressed_and_no_shell():
    transport = IapStoppedReleaseTransport(
        SimpleNamespace(
            account_for_read_only_preflight=lambda: "owner@example.com",
        ),
        gcloud_executable=_StableExecutable(),
        gcloud_configuration=_StableGcloudConfiguration(),
        known_hosts=_StableExecutable("/trusted/google_compute_known_hosts"),
    )
    source_root = f"{launcher.STOPPED_RELEASE_SOURCE_BASE}/{RELEASE_SHA}"
    commands = []
    transport._revision_source_exists = lambda _release, **_kwargs: False

    def run_remote(argv, **_kwargs):
        commands.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    transport._run_remote = run_remote

    assert (
        transport._prepare_source(
            RELEASE_SHA,
            account="owner@example.com",
        )
        == source_root
    )
    assert len(commands) == 2
    assert commands[0][-6:] == (
        "/usr/bin/git",
        "clone",
        "--no-checkout",
        "--no-tags",
        launcher.STOPPED_RELEASE_SOURCE_REPOSITORY,
        source_root,
    )
    assert commands[1][-6:] == (
        "/usr/bin/git",
        "-C",
        source_root,
        "checkout",
        "--detach",
        RELEASE_SHA,
    )
    assert all(
        item not in {"sh", "bash", "/bin/sh", "/bin/bash"}
        for command in commands
        for item in command
    )


def test_stopped_release_source_absence_requires_successful_bounded_find():
    transport = IapStoppedReleaseTransport(
        SimpleNamespace(
            account_for_read_only_preflight=lambda: "owner@example.com",
        ),
        gcloud_executable=_StableExecutable(),
        gcloud_configuration=_StableGcloudConfiguration(),
        known_hosts=_StableExecutable("/trusted/google_compute_known_hosts"),
    )
    outputs = iter((b"", RELEASE_SHA.encode("ascii"), b"unexpected"))

    def run_remote(argv, **kwargs):
        assert argv == (
            "/usr/bin/find",
            launcher.STOPPED_RELEASE_SOURCE_BASE,
            "-mindepth",
            "1",
            "-maxdepth",
            "1",
            "-name",
            RELEASE_SHA,
            "-printf",
            "%f",
        )
        assert kwargs["maximum_output_bytes"] == 40
        return subprocess.CompletedProcess(argv, 0, next(outputs), b"")

    transport._run_remote = run_remote

    assert (
        transport._revision_source_exists(
            RELEASE_SHA,
            account="owner@example.com",
        )
        is False
    )
    assert (
        transport._revision_source_exists(
            RELEASE_SHA,
            account="owner@example.com",
        )
        is True
    )
    with pytest.raises(OwnerLauncherError, match="path_probe_invalid"):
        transport._revision_source_exists(
            RELEASE_SHA,
            account="owner@example.com",
        )


def test_stopped_release_transport_binds_plan_and_terminal_receipt():
    transport = IapStoppedReleaseTransport(
        SimpleNamespace(
            account_for_read_only_preflight=lambda: "owner@example.com",
        ),
        gcloud_executable=_StableExecutable(),
        gcloud_configuration=_StableGcloudConfiguration(),
        known_hosts=_StableExecutable("/trusted/google_compute_known_hosts"),
    )
    plan = _stopped_release_plan()
    plan_sha256 = plan["plan_sha256"]
    receipt = _stopped_release_receipt(plan)
    observed = []
    transport._prepare_source = lambda release, **_kwargs: (
        observed.append(("source", release)) or "source"
    )

    def run_release(release, command, **kwargs):
        observed.append((command, release, kwargs.get("approved_plan_sha256")))
        return plan if command == "plan" else receipt

    transport._run_release_command = run_release

    assert transport.publish(RELEASE_SHA) == receipt
    assert observed == [
        ("source", RELEASE_SHA),
        ("plan", RELEASE_SHA, None),
        ("apply", RELEASE_SHA, plan_sha256),
    ]


def test_cli_stopped_release_action_never_constructs_live_secret_boundaries(
    monkeypatch,
    capfd,
):
    events = []

    class Runtime:
        def trusted_command_prefix(self):
            events.append("runtime")
            return GCLOUD_COMMAND_PREFIX

    receipt = {
        "schema": launcher.STOPPED_RELEASE_RECEIPT_SCHEMA,
        "ok": True,
        "release_revision": RELEASE_SHA,
    }
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_runtime",
        lambda _release: Runtime(),
    )
    monkeypatch.setattr(
        launcher,
        "require_local_launcher_provenance",
        lambda _release: events.append("provenance") or "a" * 64,
    )
    monkeypatch.setattr(
        launcher,
        "_validate_owner_interpreter_invocation",
        lambda _path: events.append("interpreter"),
    )
    monkeypatch.setattr(launcher, "PinnedGcloudConfiguration", lambda: object())
    monkeypatch.setattr(launcher, "GcloudOwnerAccessToken", lambda *_a, **_k: object())

    class Transport:
        def __init__(self, *_args, **_kwargs):
            events.append("transport")

        def publish(self, release):
            events.append(("publish", release))
            return receipt

    monkeypatch.setattr(launcher, "IapStoppedReleaseTransport", Transport)
    monkeypatch.setattr(
        launcher,
        "IapCoordinatorTransport",
        lambda *_a, **_k: pytest.fail("live coordinator must not be constructed"),
    )
    monkeypatch.setattr(
        launcher,
        "CloudSqlTemporaryAdmin",
        lambda *_a, **_k: pytest.fail("SQL admin must not be constructed"),
    )
    monkeypatch.setattr(
        launcher,
        "OwnerDiscordTokenReader",
        lambda: pytest.fail("Discord token reader must not be constructed"),
    )

    assert (
        launcher.main((
            "--release-sha",
            RELEASE_SHA,
            "--publish-stopped-release",
        ))
        == 0
    )
    assert json.loads(capfd.readouterr().out) == receipt
    assert events == [
        "provenance",
        "transport",
        ("publish", RELEASE_SHA),
        "runtime",
        "interpreter",
        "provenance",
    ]


def test_cli_stopped_release_action_is_mutually_exclusive_with_runtime_bootstrap():
    with pytest.raises(SystemExit):
        launcher._cli_parser().parse_args((
            "--release-sha",
            RELEASE_SHA,
            "--bootstrap-trusted-runtime",
            "--publish-stopped-release",
        ))


def _iap_authorization_documents(public_key):
    fingerprint = "f" * 64
    instance = {
        "id": launcher.VM_INSTANCE_ID,
        "name": launcher.VM_NAME,
        "zone": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{PROJECT}/zones/{launcher.ZONE}"
        ),
        "metadata": {
            "items": [
                {"key": "disable-legacy-endpoints", "value": "TRUE"},
                {"key": "enable-oslogin", "value": "TRUE"},
            ]
        },
    }
    project = {
        "name": PROJECT,
        "commonInstanceMetadata": {
            "items": [{"key": "ssh-keys", "value": "preexisting-project-key"}]
        },
    }
    profile = {
        "name": launcher.OS_LOGIN_PROFILE_ID,
        "posixAccounts": [
            {
                "username": launcher.OS_LOGIN_USERNAME,
                "primary": True,
                "operatingSystemType": "LINUX",
                "homeDirectory": f"/home/{launcher.OS_LOGIN_USERNAME}",
                "uid": "1000000001",
                "gid": "1000000001",
            }
        ],
        "sshPublicKeys": {
            fingerprint: {
                "fingerprint": fingerprint,
                "key": public_key,
                "name": f"users/fixed/sshPublicKeys/{fingerprint}",
            }
        },
    }
    return instance, project, profile


def test_iap_authorization_preflight_is_read_only_exact_and_drift_sensitive():
    calls = []
    public_key = _StableExecutable().public_key_line()
    instance, project, profile = _iap_authorization_documents(public_key)
    documents = [instance, project, profile]

    def runner(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        if "instances" in argv:
            value = documents[0]
        elif "project-info" in argv:
            value = documents[1]
        elif "os-login" in argv:
            value = documents[2]
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, _canonical(value), b"")

    transport = IapCoordinatorTransport(
        SimpleNamespace(
            approved_account="owner@example.com",
            account_for_read_only_preflight=lambda: "owner@example.com",
        ),
        gcloud_executable=_StableExecutable(),
        gcloud_configuration=_StableGcloudConfiguration(),
        known_hosts=_StableExecutable("/trusted/google_compute_known_hosts"),
        preflight_runner=runner,
    )
    before = transport._authorization_snapshot("owner@example.com")
    assert transport._authorization_snapshot("owner@example.com") == before
    assert len(calls) == 6
    for argv, kwargs in calls:
        rendered = " ".join(argv)
        assert "add-ssh-key" not in rendered
        assert "add-metadata" not in rendered
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["shell"] is False
        assert kwargs["env"]["CLOUDSDK_CORE_DISABLE_USAGE_REPORTING"] == "1"
        assert kwargs["env"]["CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK"] == "1"
        assert "SSH_AUTH_SOCK" not in kwargs["env"]
        assert "SSH_ASKPASS" not in kwargs["env"]
        assert "DISPLAY" not in kwargs["env"]

    documents[2] = {**profile, "sshPublicKeys": {}}
    with pytest.raises(OwnerLauncherError, match="iap_ssh_authorization_invalid"):
        transport._authorization_snapshot("owner@example.com")


def test_iap_plain_dry_run_requires_exact_trusted_nested_proxy_and_ssh_command():
    observed = []
    transport = None

    def runner(argv, **kwargs):
        observed.append((tuple(argv), kwargs))
        actual = tuple(argv[:-1])
        remote = next(
            item.split("=", 1)[1] for item in actual if item.startswith("--command=")
        )
        private_key = "/trusted/google_compute_engine"
        known_hosts = "/trusted/google_compute_known_hosts"
        proxy = "ProxyCommand " + " ".join((
            *GCLOUD_COMMAND_PREFIX,
            "compute",
            "start-iap-tunnel",
            launcher.VM_NAME,
            "%p",
            "--listen-on-stdin",
            f"--project={PROJECT}",
            f"--zone={launcher.ZONE}",
            "--verbosity=error",
        ))
        tokens = (
            "/usr/bin/ssh",
            "-T",
            "-o",
            proxy,
            "-o",
            "ProxyUseFdpass=no",
            *(
                item.removeprefix("--ssh-flag=")
                for item in transport._ssh_flags(known_hosts, private_key)
            ),
            f"{launcher.OS_LOGIN_USERNAME}@compute.{launcher.VM_INSTANCE_ID}",
            "--",
            *remote.split(" "),
        )
        return subprocess.CompletedProcess(
            argv,
            0,
            (shlex.join(tokens) + "\n").encode(),
            b"",
        )

    identity = SimpleNamespace(
        approved_account="owner@example.com",
        account_for_read_only_preflight=lambda: "owner@example.com",
    )
    transport = IapCoordinatorTransport(
        identity,
        gcloud_executable=_StableExecutable(),
        gcloud_configuration=_StableGcloudConfiguration(),
        known_hosts=_StableExecutable("/trusted/google_compute_known_hosts"),
        preflight_runner=runner,
    )
    argv = transport._argv(RELEASE_SHA, "preflight-owner-launch", approved=False)
    transport._validate_dry_run(argv)

    dry_argv, kwargs = observed[0]
    assert dry_argv[-1] == "--dry-run"
    assert kwargs["env"]["CLOUDSDK_PYTHON"] == "/trusted/bin/python3.13"
    assert kwargs["env"]["CLOUDSDK_PYTHON_ARGS"] == (
        "-I -S -B -X pycache_prefix=/var/empty/muncho-canary"
    )
    assert "SSH_AUTH_SOCK" not in kwargs["env"]

    def hostile_runner(argv, **_kwargs):
        result = runner(argv, **_kwargs)
        return subprocess.CompletedProcess(
            argv,
            0,
            result.stdout.replace(
                b"/trusted/bin/python3.13",
                b"/opt/homebrew/bin/python3",
            ),
            b"",
        )

    transport._preflight_runner = hostile_runner
    with pytest.raises(OwnerLauncherError, match="iap_ssh_dry_run_invalid"):
        transport._validate_dry_run(argv)


def test_real_gcloud_569_plain_dry_run_preserves_closed_nested_runtime():
    home = launcher.Path.home()
    python = home / launcher._TRUSTED_PYTHON_RELATIVE
    sdk = launcher.Path("/opt/homebrew/share/google-cloud-sdk")
    module = sdk / "lib/gcloud.py"
    if not python.is_file() or not module.is_file() or not (sdk / "VERSION").is_file():
        pytest.skip("local gcloud 569 and fixed Python trust anchors are unavailable")
    if (sdk / "VERSION").read_text() != "569.0.0\n":
        pytest.skip("local gcloud SDK is not the reviewed 569.0.0 release")

    class Runtime:
        def trusted_command_prefix(self):
            return (
                str(python),
                "-I",
                "-S",
                "-B",
                "-X",
                "pycache_prefix=/var/empty/muncho-canary",
                str(module),
            )

    try:
        configuration = launcher.PinnedGcloudConfiguration()
        known_hosts = launcher.PinnedGoogleComputeKnownHosts()
    except OwnerLauncherError:
        pytest.skip("exact local gcloud/SSH operator state is unavailable")
    if configuration.account != "lomliev@adventico.com":
        pytest.skip("reviewed owner gcloud account is not active")
    identity = SimpleNamespace(
        approved_account=configuration.account,
        account_for_read_only_preflight=lambda: configuration.account,
    )
    transport = IapCoordinatorTransport(
        identity,
        gcloud_executable=Runtime(),
        gcloud_configuration=configuration,
        known_hosts=known_hosts,
    )
    argv = transport._argv(RELEASE_SHA, "preflight-owner-launch", approved=False)

    transport._validate_dry_run(argv)


def test_concrete_iap_subprocesses_complete_the_full_cross_process_protocol(tmp_path):
    marker = tmp_path / "approval-installed"
    calls = []

    def emit_code(value):
        payload = _canonical(value) + bytes((10,))
        return f"sys.stdout.buffer.write({payload!r});sys.stdout.flush();"

    def popen_factory(argv, **kwargs):
        calls.append((argv, kwargs))
        remote = next(item for item in argv if item.startswith("--command="))
        prefix = "import os,sys,time;"
        if remote.endswith(" preflight-recovery"):
            code = prefix + emit_code(_no_recovery_required()) + "sys.exit(2)"
        elif remote.endswith(" preflight-owner-launch"):
            code = prefix + emit_code(_gate())
        elif remote.endswith(" install-discord-token"):
            code = (
                prefix
                + emit_code(_discord_gate())
                + "data=sys.stdin.buffer.read();"
                + "sys.exit(3) if not data.startswith(b'DCT1') else None;"
                + emit_code(_discord_receipt())
            )
        elif remote.endswith(" install-final-approval"):
            code = (
                prefix
                + emit_code(_approval_request())
                + "data=sys.stdin.buffer.read();"
                + "sys.exit(3) if not data.startswith(b'MFA1') else None;"
                + f"open({str(marker)!r},'xb').close();"
                + emit_code(_approval_install_receipt())
            )
        elif remote.endswith(" run"):
            code = (
                prefix
                + emit_code(_coordinator_gate())
                + "data=sys.stdin.buffer.read();"
                + "sys.exit(3) if not data.startswith(b'MCA2') else None;"
                + emit_code(_approval_request())
                + f"deadline=time.time()+5;path={str(marker)!r};"
                + "\nwhile not os.path.exists(path) and time.time()<deadline:time.sleep(.01)\n"
                + "sys.exit(4) if not os.path.exists(path) else None;"
                + emit_code(_coordinator_receipt())
            )
        elif remote.endswith(" stop-and-retire-discord-token"):
            code = (
                prefix
                + emit_code(_discord_retirement_gate())
                + "data=sys.stdin.buffer.read();"
                + "sys.exit(3) if not data.startswith(b'DRA1') else None;"
                + emit_code(_discord_retirement_receipt())
            )
        else:
            raise AssertionError(remote)
        return subprocess.Popen((sys.executable, "-c", code), **kwargs)

    class ConcreteIdentity(_OwnerIdentity):
        def account_for_read_only_preflight(self):
            return "owner@example.com"

        @property
        def approved_account(self):
            if self.bound is None:
                raise AssertionError("owner must be bound before mutation")
            return "owner@example.com"

    events = []
    identity = ConcreteIdentity(events)
    transport = IapCoordinatorTransport(
        identity,
        gcloud_executable=_StableExecutable(),
        gcloud_configuration=_StableGcloudConfiguration(),
        known_hosts=_StableExecutable("/trusted/google_compute_known_hosts"),
        popen_factory=popen_factory,
    )
    transport._authorization_snapshot = lambda _account: ("i", "p", "o")
    transport._validate_dry_run = lambda _argv: None
    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(events),
        sql_admin=_SqlAdmin(events),
        owner_identity=identity,
        final_approval_source=_FinalApprovalSource(events),
        approval_request_sink=lambda _request: events.append("request_published"),
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: events.append("hardened"),
        provenance_guard=lambda _release: None,
    )

    assert receipt["ok"] is True
    assert receipt["remote_coordinator_terminated"] is True
    assert receipt["temporary_admin_absent"] is True
    assert marker.exists()
    assert len(calls) == 5
    for argv, kwargs in calls:
        assert argv[: len(GCLOUD_COMMAND_PREFIX)] == GCLOUD_COMMAND_PREFIX
        assert kwargs["shell"] is False
        assert kwargs["start_new_session"] is True
        rendered = " ".join(argv)
        assert PASSWORD.decode() not in rendered
        assert DISCORD_TOKEN.decode() not in rendered


def test_real_cli_invalid_release_emits_one_canonical_secret_free_failure():
    repo = launcher.os.path.dirname(
        launcher.os.path.dirname(launcher.os.path.dirname(launcher.__file__))
    )
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "scripts.canary.full_canary_owner_launcher",
            "--release-sha",
            "not-a-release",
        ),
        cwd=repo,
        input=b"",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 2
    lines = completed.stdout.splitlines()
    assert len(lines) == 1
    value = json.loads(lines[0])
    assert value["error_code"] == "invalid_release_sha"
    assert _canonical(value) == lines[0]
    assert PASSWORD not in completed.stdout + completed.stderr
    assert DISCORD_TOKEN not in completed.stdout + completed.stderr


def test_second_signal_during_cleanup_is_suppressed_until_all_cleanup_finishes(
    monkeypatch,
):
    installed = {}

    monkeypatch.setattr(launcher.signal, "getsignal", lambda _number: signal.SIG_DFL)

    def install(number, handler):
        installed[number] = handler

    monkeypatch.setattr(launcher.signal, "signal", install)
    transport = _Transport(_gate())
    original_close = transport.run_session.close

    def close_with_second_signal():
        installed[signal.SIGTERM](signal.SIGTERM, None)
        original_close()

    transport.run_session.close = close_with_second_signal

    class FirstSignalApprovalSource:
        def read_final_approval(self, _request):
            installed[signal.SIGTERM](signal.SIGTERM, None)
            raise AssertionError("signal handler must interrupt")

    receipt = launch_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_SecretReader(transport.events),
        sql_admin=_SqlAdmin(transport.events),
        owner_identity=_OwnerIdentity(transport.events),
        final_approval_source=FirstSignalApprovalSource(),
        approval_request_sink=lambda _request: None,
        now=lambda: 1_000,
        password_factory=lambda: bytearray(PASSWORD),
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt["failure_code"] == "owner_launcher_interrupted"
    assert receipt["temporary_admin_absent"] is True
    assert "delete_admin" in transport.events


def test_first_signal_enters_non_reentrant_state_before_exception_unwinds(
    monkeypatch,
):
    installed = {}
    monkeypatch.setattr(launcher.signal, "getsignal", lambda _number: signal.SIG_DFL)
    monkeypatch.setattr(
        launcher.signal,
        "signal",
        lambda number, handler: installed.__setitem__(number, handler),
    )
    fence = launcher._OwnerSignalFence()
    fence.install()

    with pytest.raises(launcher._OwnerLaunchSignal):
        try:
            installed[signal.SIGTERM](signal.SIGTERM, None)
        except launcher._OwnerLaunchSignal:
            # This arrives before launch_full_canary's finally block can call
            # begin_cleanup(); it must not replace the first interruption.
            installed[signal.SIGHUP](signal.SIGHUP, None)
            raise

    assert fence.cleaning is True
    assert fence.received is True


def test_signal_restore_blocks_all_guarded_signals_until_every_handler_is_restored(
    monkeypatch,
):
    events = []
    current = {
        signal.SIGINT: "old-int",
        signal.SIGTERM: "old-term",
        signal.SIGHUP: "old-hup",
    }

    monkeypatch.setattr(launcher.signal, "getsignal", lambda number: current[number])

    def install(number, handler):
        current[number] = handler
        events.append(("handler", number, handler))

    def mask(operation, values):
        events.append(("mask", operation, frozenset(values)))
        if operation == signal.SIG_UNBLOCK:
            assert all(current[number] == signal.SIG_IGN for number in values)
            events.append(("pending_signal_discarded",))
        return frozenset()

    monkeypatch.setattr(launcher.signal, "signal", install)
    monkeypatch.setattr(launcher.signal, "pthread_sigmask", mask)
    fence = launcher._OwnerSignalFence()
    fence.install()
    events.clear()

    fence.begin_cleanup()
    fence.restore()

    assert events[0] == (
        "mask",
        signal.SIG_BLOCK,
        frozenset({signal.SIGINT, signal.SIGTERM, signal.SIGHUP}),
    )
    assert [item[0] for item in events] == [
        "mask",
        "handler",
        "handler",
        "handler",
        "mask",
        "pending_signal_discarded",
        "mask",
        "handler",
        "handler",
        "handler",
        "mask",
    ]
    assert events[-1] == ("mask", signal.SIG_SETMASK, frozenset())
