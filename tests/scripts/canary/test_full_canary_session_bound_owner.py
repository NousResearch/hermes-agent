from __future__ import annotations

import inspect

import hashlib
import json
import os
import signal
import subprocess
import sys
import urllib.parse
from collections.abc import Mapping

import pytest
from gateway import canonical_writer_foundation_phase_b as foundation_phase_b
import gateway.canonical_writer_preflight_publisher as preflight_publisher
from gateway import canonical_writer_schema_reconciliation_bootstrap as reconciliation_bootstrap
from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import writer_release


RELEASE_SHA = "a" * 40
OWNER_SHA = hashlib.sha256(b"owner@example.com").hexdigest()
NOW = 1_000
TOKEN = b"isolated-discord-token"


def test_owner_launch_signal_preserves_the_received_signal_number():
    fence = launcher._OwnerSignalFence()

    with pytest.raises(launcher._OwnerLaunchSignal) as caught:
        fence._handle(signal.SIGTERM, None)

    assert caught.value.signum == signal.SIGTERM


def test_schema_reconciliation_owner_wire_loads_under_exact_isolation_flags():
    probe = f"""
import runpy

namespace = runpy.run_path({launcher.__file__!r})
OwnerLauncherError = namespace["OwnerLauncherError"]

def stop_before_cloud(_release_sha):
    raise OwnerLauncherError("isolated_probe_stop")

try:
    namespace["reconcile_legacy_canary_schema"](
        release_sha={RELEASE_SHA!r},
        transport=object(),
        cloud_sql_client=object(),
        owner_identity=object(),
        provenance_guard=stop_before_cloud,
    )
except OwnerLauncherError as error:
    if error.code != "isolated_probe_stop":
        raise
else:
    raise RuntimeError("isolated reconciliation probe unexpectedly continued")
print("isolated_probe_stop")
"""

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/var/empty/muncho-canary",
            "-c",
            probe,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    assert completed.stdout == b"isolated_probe_stop\n"


def test_stopped_release_activation_inventory_matches_writer_release_contract():
    writer_paths = tuple(str(path) for path in writer_release._ACTIVATION_PATHS)

    assert launcher._STOPPED_RELEASE_ACTIVATION_PATHS == writer_paths
    assert tuple(str(path) for path in preflight_publisher._ACTIVATION_PATHS) == (
        writer_paths
    )


def test_stopped_release_service_inventory_matches_writer_release_contract():
    assert launcher._STOPPED_RELEASE_UNITS == writer_release._STOPPED_SERVICE_UNITS
    assert launcher._STOPPED_RELEASE_UNITS == foundation_phase_b.SERVICE_UNITS
    assert launcher._STOPPED_RELEASE_UNITS == (
        "muncho-canary-discord-edge.service",
        "muncho-discord-egress.service",
        "muncho-canonical-writer.service",
        "muncho-canonical-writer-phase-b-readiness.service",
        "muncho-canonical-writer-export.service",
        "muncho-canonical-writer-export.timer",
        "hermes-cloud-gateway.service",
    )


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _self_digest(value: dict[str, object], name: str) -> dict[str, object]:
    value[name] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def _writer_preflight_service_state() -> dict[str, dict[str, str]]:
    absent = {
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "SubState": "dead",
        "MainPID": "0",
        "UnitFileState": "",
        "FragmentPath": "",
        "DropInPaths": "",
        "NeedDaemonReload": "no",
    }
    return {
        unit: dict(absent) for unit in launcher._WRITER_PREFLIGHT_SERVICE_PATHS
    }


def _writer_preflight_plan() -> dict[str, object]:
    release_root = f"/opt/muncho-canary-releases/{RELEASE_SHA}"
    value: dict[str, object] = {
        "schema": launcher.WRITER_PREFLIGHT_PLAN_SCHEMA,
        "revision": RELEASE_SHA,
        "stopped_release_receipt_path": (
            f"{launcher.STOPPED_RELEASE_EVIDENCE_BASE}/{RELEASE_SHA}/"
            "stopped-release-publication.json"
        ),
        "stopped_release_receipt_file_sha256": "1" * 64,
        "stopped_release_receipt_sha256": "2" * 64,
        "release_root": release_root,
        "release_artifact_sha256": "3" * 64,
        "release_manifest_path": f"{release_root}/release-manifest.json",
        "release_manifest_file_sha256": "4" * 64,
        "host_identity_receipt_path": launcher.STOPPED_RELEASE_HOST_RECEIPT_PATH,
        "host_identity_receipt_file_sha256": "5" * 64,
        "host_identity_receipt_sha256": "6" * 64,
        "host_identity_sha256": "7" * 64,
        "boot_id_sha256": "8" * 64,
        "database": {
            "host": launcher.DATABASE_HOST,
            "port": launcher.DATABASE_PORT,
            "database": launcher.DATABASE_NAME,
            "user": "muncho_canary_writer_login",
            "tls_server_name": launcher.WRITER_PREFLIGHT_DATABASE_TLS_SERVER_NAME,
            "ca_path": "/etc/muncho/trust/cloudsql-server-ca.pem",
            "ca_sha256": "9" * 64,
        },
        "credential_provenance": {
            "path": "/etc/muncho/credentials/canonical-writer-db-password",
            "device": 1,
            "inode": 2,
            "owner_uid": 999,
            "group_gid": 994,
            "mode": "0400",
            "link_count": 1,
            "modification_time_ns": 3,
            "change_time_ns": 4,
            "content_or_digest_recorded": False,
        },
        "owner_discord_user_ids": [
            launcher.WRITER_PREFLIGHT_OWNER_DISCORD_USER_ID
        ],
        "external_iam_policy_sha256": "b" * 64,
        "service_state": _writer_preflight_service_state(),
        "fixed_output_paths": {
            "writer_config": str(
                preflight_publisher.DEFAULT_WRITER_CONFIG_SOURCE_PATH
            ),
            "gateway_config": str(
                preflight_publisher.DEFAULT_GATEWAY_CONFIG_SOURCE_PATH
            ),
            "writer_unit": str(
                preflight_publisher.DEFAULT_STAGED_WRITER_UNIT_PATH
            ),
            "phase_b_readiness_unit": str(
                preflight_publisher.DEFAULT_STAGED_PHASE_B_READINESS_UNIT_PATH
            ),
            "gateway_unit": str(
                preflight_publisher.DEFAULT_STAGED_GATEWAY_UNIT_PATH
            ),
            "native_observation_plan": str(
                preflight_publisher.DEFAULT_STAGED_NATIVE_PLAN_PATH
            ),
            "publication_evidence_root": str(
                preflight_publisher.PUBLICATION_EVIDENCE_ROOT
            ),
        },
        "invariants": {
            "services_started": False,
            "units_installed": False,
            "daemon_reloaded": False,
            "approval_created": False,
            "discord_started": False,
            "credential_content_or_digest_recorded": False,
        },
    }
    return _self_digest(value, "plan_sha256")


def _writer_preflight_receipt(plan: Mapping[str, object]) -> dict[str, object]:
    artifacts = {
        "writer_config": {
            "path": plan["fixed_output_paths"]["writer_config"],
            "sha256": "c" * 64,
        },
        "gateway_config": {
            "path": plan["fixed_output_paths"]["gateway_config"],
            "sha256": "d" * 64,
        },
        "writer_unit": {
            "path": plan["fixed_output_paths"]["writer_unit"],
            "sha256": "e" * 64,
        },
        "phase_b_readiness_unit": {
            "path": plan["fixed_output_paths"]["phase_b_readiness_unit"],
            "sha256": "f" * 64,
        },
        "gateway_unit": {
            "path": plan["fixed_output_paths"]["gateway_unit"],
            "sha256": "0" * 64,
        },
        "native_observation_plan": {
            "path": plan["fixed_output_paths"]["native_observation_plan"],
            "sha256": "a" * 64,
        },
    }
    collector_sha256 = "1" * 64
    report_sha256 = "2" * 64
    report_file_sha256 = "3" * 64
    hba_observed_at = 1_000
    collector_collected_at = 1_050
    observed_at = 1_100
    hba_expires_at = 1_300
    time_envelope = {
        "config_collector_receipt_sha256": collector_sha256,
        "native_observation_plan_sha256": artifacts[
            "native_observation_plan"
        ]["sha256"],
        "preflight_report_sha256": report_sha256,
        "collector_hba_observed_at_unix": hba_observed_at,
        "collector_collected_at_unix": collector_collected_at,
        "observed_at_unix": observed_at,
        "collector_hba_expires_at_unix": hba_expires_at,
    }
    time_envelope_sha256 = hashlib.sha256(_canonical(time_envelope)).hexdigest()
    provenance = {
        "approved_plan_sha256": plan["plan_sha256"],
        "release_artifact_sha256": plan["release_artifact_sha256"],
        "release_manifest_file_sha256": plan["release_manifest_file_sha256"],
        "database_ca_sha256": plan["database"]["ca_sha256"],
        "config_collector_receipt_sha256": collector_sha256,
        "config_collector_receipt_file_sha256": "4" * 64,
        "collector_writer_config_sha256": artifacts["writer_config"]["sha256"],
        "collector_gateway_config_sha256": artifacts["gateway_config"]["sha256"],
        "native_observation_plan_sha256": artifacts[
            "native_observation_plan"
        ]["sha256"],
        "native_writer_config_sha256": artifacts["writer_config"]["sha256"],
        "native_gateway_config_sha256": artifacts["gateway_config"]["sha256"],
        "native_writer_unit_sha256": artifacts["writer_unit"]["sha256"],
        "staged_phase_b_readiness_unit_sha256": artifacts[
            "phase_b_readiness_unit"
        ]["sha256"],
        "native_gateway_unit_sha256": artifacts["gateway_unit"]["sha256"],
        "preflight_report_sha256": report_sha256,
        "preflight_report_file_sha256": report_file_sha256,
        "preflight_time_envelope_sha256": time_envelope_sha256,
    }
    value: dict[str, object] = {
        "schema": launcher.WRITER_PREFLIGHT_RECEIPT_SCHEMA,
        "ok": True,
        "state": "staged_preflight_passed_services_stopped",
        "revision": RELEASE_SHA,
        "approved_plan_sha256": plan["plan_sha256"],
        "stopped_release_receipt_sha256": plan[
            "stopped_release_receipt_sha256"
        ],
        "release_artifact_sha256": plan["release_artifact_sha256"],
        "release_manifest_file_sha256": plan["release_manifest_file_sha256"],
        "host_identity_receipt_sha256": plan["host_identity_receipt_sha256"],
        "config_collector_receipt_path": (
            "/var/lib/muncho-writer-canary-evidence/config-collector/"
            f"{RELEASE_SHA}/{collector_sha256}.json"
        ),
        "config_collector_receipt_sha256": collector_sha256,
        "config_collector_receipt_file_sha256": "4" * 64,
        "native_observation_plan_sha256": artifacts[
            "native_observation_plan"
        ]["sha256"],
        "external_iam_policy_sha256": plan["external_iam_policy_sha256"],
        "preflight_report_path": (
            f"{launcher.WRITER_PREFLIGHT_EVIDENCE_BASE}/{RELEASE_SHA}/"
            f"{plan['plan_sha256']}/reports/{report_sha256}.json"
        ),
        "preflight_report_file_sha256": report_file_sha256,
        "preflight_report_sha256": report_sha256,
        "preflight_observed_at_unix": observed_at,
        "preflight_collector_hba_observed_at_unix": hba_observed_at,
        "preflight_collector_collected_at_unix": collector_collected_at,
        "preflight_collector_hba_expires_at_unix": hba_expires_at,
        "preflight_time_envelope_sha256": time_envelope_sha256,
        "preflight_fresh_at_seal": True,
        "service_state_before": plan["service_state"],
        "service_state_after": plan["service_state"],
        "artifacts": artifacts,
        "provenance": provenance,
        "invariants": plan["invariants"],
        "sealed_at_unix": 1_200,
        "receipt_path": (
            f"{launcher.WRITER_PREFLIGHT_EVIDENCE_BASE}/{RELEASE_SHA}/"
            f"{plan['plan_sha256']}/publication.json"
        ),
    }
    return _self_digest(value, "receipt_sha256")


def test_writer_preflight_publisher_contract_is_accepted_by_owner_validator():
    plan = _writer_preflight_plan()
    receipt = _writer_preflight_receipt(plan)

    assert set(receipt["provenance"]) == (
        preflight_publisher._PUBLICATION_PROVENANCE_FIELDS
    )
    assert launcher.validate_writer_preflight_plan(
        plan,
        expected_release_sha=RELEASE_SHA,
        expected_external_iam_policy_sha256="b" * 64,
    ) == plan
    assert launcher.validate_writer_preflight_receipt(receipt, plan=plan) == receipt


@pytest.mark.parametrize(
    ("command", "error_type", "expected_code"),
    (
        ("plan", "RuntimeError", "writer_preflight_plan_runtime_failed"),
        ("apply", "ValueError", "writer_preflight_apply_validation_failed"),
        ("apply", "UnexpectedError", "writer_preflight_apply_remote_failed"),
    ),
)
def test_writer_preflight_transport_preserves_safe_remote_failure_stage(
    monkeypatch,
    command,
    error_type,
    expected_code,
):
    failure = {
        "schema": launcher.WRITER_PREFLIGHT_FAILURE_SCHEMA,
        "ok": False,
        "error_code": "writer_preflight_publication_failed",
        "error_type": error_type,
    }
    observed = {}
    transport = object.__new__(launcher.IapWriterPreflightTransport)

    def run_remote(_remote, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            args=(),
            returncode=2,
            stdout=_canonical(failure) + b"\n",
        )

    monkeypatch.setattr(transport, "_run_remote", run_remote)

    with pytest.raises(launcher.OwnerLauncherError) as raised:
        transport._run_writer_preflight_command(
            RELEASE_SHA,
            command,
            account="owner@example.com",
            external_iam_policy_sha256="b" * 64,
            approved_plan_sha256=("c" * 64 if command == "apply" else None),
        )

    assert raised.value.code == expected_code
    assert observed["allowed_returncodes"] == frozenset({0, 2})


@pytest.mark.parametrize(
    ("returncode", "payload"),
    (
        (2, _writer_preflight_plan()),
        (
            0,
            {
                "schema": launcher.WRITER_PREFLIGHT_FAILURE_SCHEMA,
                "ok": False,
                "error_code": "writer_preflight_publication_failed",
                "error_type": "RuntimeError",
            },
        ),
        (
            2,
            {
                "schema": launcher.WRITER_PREFLIGHT_FAILURE_SCHEMA,
                "ok": False,
                "error_code": "writer_preflight_publication_failed",
                "error_type": "RuntimeError",
                "unexpected": "must-not-pass",
            },
        ),
    ),
)
def test_writer_preflight_transport_rejects_returncode_payload_mismatch(
    monkeypatch,
    returncode,
    payload,
):
    transport = object.__new__(launcher.IapWriterPreflightTransport)
    monkeypatch.setattr(
        transport,
        "_run_remote",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=(),
            returncode=returncode,
            stdout=_canonical(payload) + b"\n",
        ),
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="writer_preflight_output_invalid",
    ):
        transport._run_writer_preflight_command(
            RELEASE_SHA,
            "plan",
            account="owner@example.com",
            external_iam_policy_sha256="b" * 64,
        )


def _anchor() -> dict[str, object]:
    return {
        "phase_b_release_revision": RELEASE_SHA,
        "phase_b_plan_sha256": "1" * 64,
        "phase_b_approval_sha256": "2" * 64,
        "phase_b_terminal_receipt_sha256": "3" * 64,
        "phase_b_foundation_generation_sha256": "4" * 64,
        "phase_b_readiness_receipt_sha256": "5" * 64,
        "phase_b_readiness_handoff_file_sha256": "6" * 64,
        "phase_b_readiness_sequence": 7,
    }


def _live_gate() -> dict[str, object]:
    anchor = _anchor()
    return _self_digest(
        {
            "schema": launcher.PHASE_B_LIVE_GATE_SCHEMA,
            "ok": True,
            "state": "phase_b_terminal_ready",
            "release_sha": RELEASE_SHA,
            "coordinator_input_sha256": "c" * 64,
            "owner_subject_sha256": OWNER_SHA,
            "approval_source_sha256": launcher.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256,
            "phase_b_readiness_anchor": anchor,
            "phase_b_readiness_anchor_sha256": hashlib.sha256(
                _canonical(anchor)
            ).hexdigest(),
            "issued_at_unix": NOW,
            "expires_at_unix": NOW + 300,
        },
        "gate_sha256",
    )


def _discord_gate() -> dict[str, object]:
    return _self_digest(
        {
            "schema": launcher.DISCORD_INSTALL_GATE_SCHEMA,
            "ok": True,
            "state": "token_install_authorized",
            "coordinator_input_sha256": "c" * 64,
            "discord_token_install_approval_sha256": "d" * 64,
            "owner_subject_sha256": OWNER_SHA,
            "release_sha": RELEASE_SHA,
            "token_path": launcher.DISCORD_TOKEN_PATH,
            "edge_uid": 1001,
            "edge_gid": 1002,
            "expires_at_unix": NOW + 300,
            "frame_schema": launcher.DISCORD_FRAME_SCHEMA,
        },
        "gate_sha256",
    )


def _discord_receipt() -> dict[str, object]:
    return _self_digest(
        {
            "schema": launcher.DISCORD_INSTALL_RECEIPT_SCHEMA,
            "ok": True,
            "release_sha": RELEASE_SHA,
            "coordinator_input_sha256": "c" * 64,
            "discord_token_install_approval_sha256": "d" * 64,
            "owner_subject_sha256": OWNER_SHA,
            "token_path": launcher.DISCORD_TOKEN_PATH,
            "device": 10,
            "inode": 11,
            "owner_uid": 1001,
            "group_gid": 1002,
            "mode": "0400",
            "size": len(TOKEN),
            "link_count": 1,
            "content_or_digest_recorded": False,
            "installed_at_unix": NOW,
        },
        "receipt_sha256",
    )


def _request() -> dict[str, object]:
    return _self_digest(
        {
            "schema": launcher.SESSION_BOUND_APPROVAL_REQUEST_SCHEMA,
            "ok": True,
            "state": "awaiting_session_bound_owner_approval",
            "release_sha": RELEASE_SHA,
            "coordinator_input_sha256": "c" * 64,
            "full_canary_plan_sha256": "a" * 64,
            "staged_plan_path": "/etc/muncho/full-canary/staged/runtime-plan.json",
            "staged_plan_file_sha256": "b" * 64,
            "fixture_sha256": "f" * 64,
            "phase_b_readiness_anchor_sha256": _live_gate()[
                "phase_b_readiness_anchor_sha256"
            ],
            "phase_b_approval_sha256": "2" * 64,
            "owner_subject_sha256": OWNER_SHA,
            "approval_source_sha256": launcher.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256,
            "requested_at_unix": NOW,
            "owner_input_cutoff_unix": NOW + 55,
            "approval_deadline_unix": NOW + 60,
            "approval_path": None,
            "final_approval_frame_schema": launcher.FINAL_APPROVAL_FRAME_SCHEMA,
        },
        "request_sha256",
    )


def _approval() -> dict[str, object]:
    return {
        "schema": "muncho-full-canary-owner-approval.v1",
        "scope": "full_canary_runtime_start",
        "plan_sha256": "a" * 64,
        "authority_kind": "trusted_root_bootstrap_out_of_band_owner",
        "cryptographic_owner_proof": False,
        "owner_subject_sha256": OWNER_SHA,
        "approval_source_sha256": launcher.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256,
        "nonce_sha256": "9" * 64,
        "approved_at_unix": NOW,
        "expires_at_unix": NOW + 50,
    }


def _coordinator_receipt() -> dict[str, object]:
    result = {
        "schema": "muncho-full-canary-live-driver.v1",
        "ok": True,
        "release_sha": RELEASE_SHA,
        "full_canary_plan_sha256": "a" * 64,
        "canary_run_id": "run-1",
        "evidence_path": (
            f"/var/lib/muncho-full-canary/plans/{RELEASE_SHA}/"
            f"{'a' * 64}/live/evidence.json"
        ),
        "evidence_sha256": "e" * 64,
        "offline_invariant_receipt": {"verified": True},
        "lifecycle_verification_receipt": {"verified": True},
        "discord_ingress_claimed": False,
    }
    return _self_digest(
        {
            "schema": launcher.SESSION_BOUND_COORDINATOR_RECEIPT_SCHEMA,
            "ok": True,
            "state": "verified_stopped_and_credentials_retired",
            "release_sha": RELEASE_SHA,
            "coordinator_input_sha256": "c" * 64,
            "full_canary_plan_sha256": "a" * 64,
            "owner_approval_sha256": hashlib.sha256(
                _canonical(_approval())
            ).hexdigest(),
            "phase_b_readiness_anchor_sha256": _live_gate()[
                "phase_b_readiness_anchor_sha256"
            ],
            "api_session_key_sha256": "8" * 64,
            "fixture_sha256": "f" * 64,
            "live_driver_result": result,
            "live_driver_receipt_sha256": hashlib.sha256(
                _canonical(result)
            ).hexdigest(),
            "services_stopped": True,
            "discord_token_retired": True,
            "temporary_admin_created": False,
            "bootstrap_credential_created": False,
            "completed_at_unix": NOW,
        },
        "receipt_sha256",
    )


class _Session:
    def __init__(self, gate, terminal):
        self.gate = gate
        self.terminal = terminal
        self.frames: list[bytes] = []
        self.validated = None
        self.closed = False

    termination_proven = True
    terminal_receipt_validated = True

    def read_gate(self):
        return self.gate

    def finish_before(self, frame, *, write_guard, on_first_write):
        write_guard()
        on_first_write()
        self.frames.append(bytes(frame))
        return self.terminal

    def finish(self, frame):
        self.frames.append(bytes(frame))
        return self.terminal

    def mark_validated(self, receipt):
        assert receipt == self.terminal
        self.validated = receipt

    def close(self):
        self.closed = True


class _Transport:
    def __init__(self):
        self.discord = _Session(_discord_gate(), _discord_receipt())
        self.run = _Session(_request(), _coordinator_receipt())

    def preflight_phase_b_live_run(self, release_sha):
        assert release_sha == RELEASE_SHA
        return _live_gate()

    def open_discord_install(self, release_sha):
        assert release_sha == RELEASE_SHA
        return self.discord

    def open_run(self, release_sha):
        assert release_sha == RELEASE_SHA
        return self.run

    def open_discord_retirement(self, _release_sha):
        raise AssertionError("happy path must retire inside the coordinator")


class _OwnerIdentity:
    def __init__(self):
        self.bound = None
        self.stability_checks = 0

    def bind_approved_subject(self, value):
        self.bound = value

    def require_stable(self):
        self.stability_checks += 1


class _TokenSource:
    def read_discord_token(self):
        return bytearray(TOKEN)


class _ApprovalSource:
    def __init__(self):
        self.request = None

    def read_final_approval(self, request):
        self.request = request
        return _approval()


class _SignalFence:
    def install(self):
        pass

    def begin_cleanup(self):
        pass

    def restore(self):
        pass


def test_session_bound_owner_uses_one_run_and_no_admin_or_bootstrap(monkeypatch):
    monkeypatch.setattr(launcher, "_OwnerSignalFence", _SignalFence)
    transport = _Transport()
    identity = _OwnerIdentity()
    approval_source = _ApprovalSource()
    emitted = []

    receipt = launcher.launch_session_bound_full_canary(
        release_sha=RELEASE_SHA,
        transport=transport,
        token_source=_TokenSource(),
        owner_identity=identity,
        final_approval_source=approval_source,
        approval_request_sink=emitted.append,
        now=lambda: NOW,
        secret_hardener=lambda: None,
        provenance_guard=lambda release: None,
    )

    assert receipt["schema"] == launcher.SESSION_BOUND_OWNER_RECEIPT_SCHEMA
    assert receipt["temporary_admin_created"] is False
    assert receipt["bootstrap_credential_created"] is False
    assert receipt["discord_token_retired"] is True
    assert emitted == [_request()]
    assert approval_source.request == _request()
    assert transport.discord.frames[0].startswith(launcher.DISCORD_FRAME_MAGIC)
    assert transport.run.frames[0].startswith(launcher.FINAL_APPROVAL_FRAME_MAGIC)
    assert transport.discord.closed and transport.run.closed
    assert identity.bound == OWNER_SHA
    assert identity.stability_checks >= 5
    parameters = inspect.signature(
        launcher.launch_session_bound_full_canary
    ).parameters
    assert "sql_admin" not in parameters
    assert "bootstrap_login_factory" not in parameters
    assert "password_factory" not in parameters


def test_current_phase_b_gate_rejects_unpinned_approval_source():
    gate = _live_gate()
    gate["approval_source_sha256"] = "0" * 64
    gate.pop("gate_sha256")
    _self_digest(gate, "gate_sha256")

    try:
        launcher.validate_current_phase_b_live_gate(
            gate,
            expected_release_sha=RELEASE_SHA,
            now_unix=NOW,
        )
    except launcher.OwnerLauncherError as exc:
        assert exc.code == "invalid_phase_b_live_gate"
    else:
        raise AssertionError("unpinned owner lineage was accepted")


def test_session_bound_request_has_no_external_approval_artifact():
    request = launcher.validate_session_bound_approval_request(
        _request(),
        live_gate=_live_gate(),
        now_unix=NOW,
    )

    assert request["approval_path"] is None


def test_session_bound_terminal_failure_validates_without_legacy_authority():
    failure = _self_digest(
        {
            "schema": launcher.COORDINATOR_FAILURE_SCHEMA,
            "ok": False,
            "phase": "command",
            "command": "run",
            "error_code": "coordinator_failed",
            "release_sha": RELEASE_SHA,
            "coordinator_input_sha256": "c" * 64,
            "cleanup_status": "complete",
            "discord_token_removed": True,
            "services_stopped": True,
            "obsolete_process_journal_absent": True,
            "completed_at_unix": NOW,
        },
        "receipt_sha256",
    )

    validated = launcher.validate_terminal_first_failure(
        failure,
        owner_gate=_live_gate(),
    )

    assert validated == failure
    assert set(validated) == {
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
    }


def test_schema_reconciliation_remote_exchange_is_exactly_three_frames():
    session = object.__new__(launcher._IapRemoteSession)
    session._stdin_closed = False
    session._frames_written = 0
    session._messages_read = 1
    calls: list[tuple[bytes, bool]] = []

    def write(frame, *, close_stdin):
        calls.append((bytes(frame), close_stdin))
        session._frames_written += 1
        session._messages_read += 1
        session._stdin_closed = close_stdin
        return {"round": session._frames_written}

    session._write_frame = write

    a1 = (
        launcher.SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_MAGIC
        + (2).to_bytes(4, "big")
        + b"{}"
        + b"x" * launcher.SCHEMA_RECONCILIATION_CREDENTIAL_BYTES
    )
    a2 = (
        launcher.SCHEMA_RECONCILIATION_PREFLIGHT_AUTHORIZATION_MAGIC
        + (2).to_bytes(4, "big")
        + b"{}"
    )
    c3 = (
        launcher.SCHEMA_RECONCILIATION_ADMIN_CLEANUP_MAGIC
        + (2).to_bytes(4, "big")
        + b"{}"
    )
    assert session.schema_reconciliation_exchange(a1, terminal=False) == {
        "round": 1
    }
    assert session.schema_reconciliation_exchange(a2, terminal=False) == {
        "round": 2
    }
    assert session.schema_reconciliation_exchange(c3, terminal=True) == {
        "round": 3
    }
    assert calls == [(a1, False), (a2, False), (c3, True)]
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="schema_reconciliation_remote_frame_state_invalid",
    ):
        session.schema_reconciliation_exchange(c3, terminal=True)


def test_schema_reconciliation_remote_exchange_rejects_early_terminal():
    session = object.__new__(launcher._IapRemoteSession)
    session._stdin_closed = False
    session._frames_written = 0
    session._messages_read = 1

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="schema_reconciliation_remote_frame_state_invalid",
    ):
        session.schema_reconciliation_exchange(
            launcher.SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_MAGIC
            + (2).to_bytes(4, "big")
            + b"{}"
            + b"x" * launcher.SCHEMA_RECONCILIATION_CREDENTIAL_BYTES,
            terminal=True,
        )


def test_schema_reconciliation_first_frame_rechecks_authority_before_write():
    session = object.__new__(launcher._IapRemoteSession)
    session._stdin_closed = False
    session._frames_written = 0
    session._messages_read = 1
    calls: list[object] = []

    def write(
        frame,
        *,
        close_stdin,
        write_guard,
        on_first_write,
        on_write_complete,
    ):
        calls.append((bytes(frame), close_stdin))
        write_guard()
        on_first_write()
        calls.append("write")
        on_write_complete()
        return {"schema": "preflight"}

    session._write_frame = write
    a1 = (
        launcher.SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_MAGIC
        + (2).to_bytes(4, "big")
        + b"{}"
        + b"x" * launcher.SCHEMA_RECONCILIATION_CREDENTIAL_BYTES
    )
    assert session.schema_reconciliation_exchange_before(
        a1,
        write_guard=lambda: calls.append("guard"),
        on_first_write=lambda: calls.append("first"),
        on_write_complete=lambda: calls.append("complete"),
    ) == {"schema": "preflight"}
    assert calls == [
        (a1, False),
        "guard",
        "first",
        "write",
        "complete",
    ]


def test_remote_session_current_authority_guard_is_repeatable():
    session = object.__new__(launcher._IapRemoteSession)
    calls: list[str] = []
    session._authority_guard = lambda: calls.append("authority")

    session.require_current_authority()
    session.require_current_authority()

    assert calls == ["authority", "authority"]


def test_remote_session_abort_requires_exact_remote_exit_two():
    read_descriptor, write_descriptor = os.pipe()
    os.close(write_descriptor)
    stdout = os.fdopen(read_descriptor, "rb", buffering=0)

    class _ExitedProcess:
        def __init__(self):
            self.stdout = stdout

        def wait(self, _timeout):
            return 2

    session = object.__new__(launcher._IapRemoteSession)
    session._process = _ExitedProcess()
    session._termination_timeout = 1.0
    session._buffer = bytearray()
    session._stdout_eof = False
    session._stdin_closed = True
    session._termination_proven = False
    postflight = []
    session._run_postflight = lambda: postflight.append("postflight")
    session._force_local_stop = lambda: pytest.fail("must not force stop")
    try:
        session.abort_and_prove_terminated()
    finally:
        stdout.close()

    assert session._termination_proven is True
    assert postflight == ["postflight"]


@pytest.mark.parametrize("returncode", (0, 2))
def test_remote_failure_receipt_requires_exact_exit_two(returncode):
    read_descriptor, write_descriptor = os.pipe()
    os.close(write_descriptor)
    stdout = os.fdopen(read_descriptor, "rb", buffering=0)
    receipt = {
        "schema": launcher.SCHEMA_RECONCILIATION_REMOTE_FAILURE_SCHEMA,
        "ok": False,
    }

    class _ExitedProcess:
        def __init__(self):
            self.stdout = stdout

        def wait(self, _timeout):
            return returncode

    session = object.__new__(launcher._IapRemoteSession)
    session._process = _ExitedProcess()
    session._termination_timeout = 1.0
    session._buffer = bytearray()
    session._stdout_eof = False
    session._stdin_closed = True
    session._last_mapping = receipt
    session._validated_terminal = False
    session._termination_proven = False
    postflight = []
    session._run_postflight = lambda: postflight.append("postflight")
    try:
        if returncode == 2:
            session.mark_validated(receipt)
            assert session._validated_terminal is True
            assert session._termination_proven is True
            assert postflight == ["postflight"]
        else:
            with pytest.raises(
                launcher.OwnerLauncherError,
                match="remote_terminal_exit_mismatch",
            ):
                session.mark_validated(receipt)
            assert session._validated_terminal is False
            assert session._termination_proven is False
            assert postflight == []
    finally:
        stdout.close()


def test_schema_reconciliation_first_frame_guard_failure_writes_nothing():
    session = object.__new__(launcher._IapRemoteSession)
    session._stdin_closed = False
    session._frames_written = 0
    session._messages_read = 1
    calls: list[str] = []

    def reject() -> None:
        raise launcher.OwnerLauncherError("authority_changed")

    def write(
        _frame,
        *,
        close_stdin,
        write_guard,
        on_first_write,
        on_write_complete,
    ):
        assert close_stdin is False
        write_guard()
        on_first_write()
        calls.append("write")
        on_write_complete()
        return {"schema": "preflight"}

    session._write_frame = write
    a1 = (
        launcher.SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_MAGIC
        + (2).to_bytes(4, "big")
        + b"{}"
        + b"x" * launcher.SCHEMA_RECONCILIATION_CREDENTIAL_BYTES
    )
    with pytest.raises(launcher.OwnerLauncherError, match="authority_changed"):
        session.schema_reconciliation_exchange_before(
            a1,
            write_guard=reject,
            on_first_write=lambda: None,
            on_write_complete=lambda: None,
        )
    assert calls == []


def test_schema_reconciliation_remote_exchange_rejects_wrong_frame_shape():
    session = object.__new__(launcher._IapRemoteSession)
    session._stdin_closed = False
    session._frames_written = 0
    session._messages_read = 1

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="schema_reconciliation_remote_frame_invalid",
    ):
        session.schema_reconciliation_exchange(
            launcher.SCHEMA_RECONCILIATION_PREFLIGHT_AUTHORIZATION_MAGIC
            + (2).to_bytes(4, "big")
            + b"{}",
            terminal=False,
        )


def test_schema_reconciliation_control_frames_are_two_stage_and_exact():
    credential = bytearray(b"q" * 64)
    install = launcher._schema_reconciliation_control_frame(
        launcher.SCHEMA_RECONCILIATION_CONTROL_INSTALL_MAGIC,
        {"ok": True},
        credential=credential,
    )
    cleanup = launcher._schema_reconciliation_control_frame(
        launcher.SCHEMA_RECONCILIATION_CONTROL_CLEANUP_MAGIC,
        {"ok": True},
    )

    launcher._IapRemoteSession._validate_schema_reconciliation_control_frame(
        install,
        sequence=0,
    )
    launcher._IapRemoteSession._validate_schema_reconciliation_control_frame(
        cleanup,
        sequence=1,
    )
    assert install[:4] == b"MCB1"
    assert cleanup[:4] == b"MCC1"
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="schema_reconciliation_control_remote_frame_invalid",
    ):
        launcher._IapRemoteSession._validate_schema_reconciliation_control_frame(
            install,
            sequence=1,
        )


def test_schema_reconciliation_transport_and_signatures_are_domain_separated():
    assert launcher.IapSchemaReconciliationTransport._MODULE == (
        "gateway.canonical_writer_schema_reconciliation_bootstrap"
    )
    assert launcher.IapSchemaReconciliationTransport._COMMANDS == {"run"}
    namespaces = {
        launcher.SCHEMA_RECONCILIATION_EXECUTOR_PREFLIGHT_SSHSIG_NAMESPACE,
        launcher.SCHEMA_RECONCILIATION_PREFLIGHT_AUTHORIZATION_SSHSIG_NAMESPACE,
        launcher.SCHEMA_RECONCILIATION_EXECUTOR_CLEANUP_SSHSIG_NAMESPACE,
    }
    assert len(namespaces) == 3
    assert all(value.endswith(("-v2", "-v3")) for value in namespaces)
    assert {
        launcher.SCHEMA_RECONCILIATION_CONTROL_INSTALL_SSHSIG_NAMESPACE,
        launcher.SCHEMA_RECONCILIATION_CONTROL_CLEANUP_SSHSIG_NAMESPACE,
    } == {
        "muncho-canonical-writer-schema-reconciliation-control-install-owner-v1",
        "muncho-canonical-writer-schema-reconciliation-control-cleanup-owner-v1",
    }
    assert (
        launcher.IapSchemaReconciliationControlBootstrapTransport._MODULE
        == "gateway.canonical_writer_schema_reconciliation_control_bootstrap"
    )
    assert (
        launcher.IapSchemaReconciliationControlBootstrapTransport._COMMANDS
        == {"install"}
    )


def test_schema_reconciliation_cli_action_is_mutually_exclusive():
    arguments = launcher._cli_parser().parse_args([
        "--release-sha",
        RELEASE_SHA,
        "--reconcile-legacy-canary-db",
    ])

    assert arguments.reconcile_legacy_canary_db is True
    assert arguments.bootstrap_schema_reconciliation_control is False
    assert arguments.publish_writer_preflight is False
    assert arguments.apply_phase_b_foundation is False
    with pytest.raises(SystemExit):
        launcher._cli_parser().parse_args([
            "--release-sha",
            RELEASE_SHA,
            "--reconcile-legacy-canary-db",
            "--publish-writer-preflight",
        ])
    control_arguments = launcher._cli_parser().parse_args([
        "--release-sha",
        RELEASE_SHA,
        "--bootstrap-schema-reconciliation-control",
    ])
    assert control_arguments.bootstrap_schema_reconciliation_control is True
    assert control_arguments.reconcile_legacy_canary_db is False
    with pytest.raises(SystemExit):
        launcher._cli_parser().parse_args([
            "--release-sha",
            RELEASE_SHA,
            "--bootstrap-schema-reconciliation-control",
            "--reconcile-legacy-canary-db",
        ])


def test_schema_reconciliation_cli_dispatches_exact_owner_boundaries(monkeypatch):
    calls = []
    emitted = []
    owner_identity = object()
    gcloud_configuration = object()
    database_client = object()
    transport = object()
    receipt = {"ok": True, "terminal_sha256": "e" * 64}

    class _TrustedExecutable:
        def trusted_command_prefix(self):
            calls.append("trusted_command_prefix")
            return ("/trusted/python",)

    executable = _TrustedExecutable()
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_runtime",
        lambda release: calls.append(("runtime", release)) or executable,
    )
    monkeypatch.setattr(
        launcher,
        "activate_trusted_owner_support",
        lambda runtime, *, release_sha: calls.append(
            ("support_activate", runtime, release_sha)
        ),
    )
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_support_activation",
        lambda runtime, *, release_sha: calls.append(
            ("support_revalidate", runtime, release_sha)
        ),
    )
    monkeypatch.setattr(
        launcher,
        "require_local_launcher_provenance",
        lambda release: calls.append(("provenance", release)) or "f" * 64,
    )
    monkeypatch.setattr(
        launcher,
        "_validate_owner_interpreter_invocation",
        lambda path: calls.append(("interpreter", path)),
    )
    monkeypatch.setattr(
        launcher,
        "PinnedGcloudConfiguration",
        lambda: gcloud_configuration,
    )
    monkeypatch.setattr(
        launcher,
        "GcloudOwnerAccessToken",
        lambda **kwargs: (
            calls.append(("identity", kwargs)) or owner_identity
        ),
    )
    monkeypatch.setattr(
        launcher,
        "IapSchemaReconciliationTransport",
        lambda identity, **kwargs: (
            calls.append(("transport", identity, kwargs)) or transport
        ),
    )
    monkeypatch.setattr(
        launcher,
        "GoogleRestClient",
        lambda identity: (
            calls.append(("database_client", identity)) or database_client
        ),
    )

    def reconcile(**kwargs):
        calls.append(("reconcile", kwargs))
        kwargs["provenance_guard"](kwargs["release_sha"])
        return receipt

    monkeypatch.setattr(launcher, "reconcile_legacy_canary_schema", reconcile)
    monkeypatch.setattr(launcher, "_emit_canonical_line", emitted.append)

    result = launcher.main([
        "--release-sha",
        RELEASE_SHA,
        "--reconcile-legacy-canary-db",
    ])

    assert result == 0
    assert emitted == [receipt]
    reconcile_call = next(item for item in calls if item[0] == "reconcile")
    assert reconcile_call[1]["release_sha"] == RELEASE_SHA
    assert reconcile_call[1]["transport"] is transport
    assert reconcile_call[1]["cloud_sql_client"] is database_client
    assert reconcile_call[1]["owner_identity"] is owner_identity
    assert ("support_activate", executable, RELEASE_SHA) in calls
    assert ("support_revalidate", executable, RELEASE_SHA) in calls
    assert not any(
        item[0] in {"writer_preflight", "phase_b"}
        for item in calls
        if isinstance(item, tuple)
    )


def test_schema_reconciliation_control_cli_dispatches_separate_bootstrap(
    monkeypatch,
):
    calls = []
    emitted = []
    owner_identity = object()
    gcloud_configuration = object()
    database_client = object()
    transport = object()
    receipt = {"ok": True, "terminal_sha256": "e" * 64}

    class _TrustedExecutable:
        def trusted_command_prefix(self):
            calls.append("trusted_command_prefix")
            return ("/trusted/python",)

    executable = _TrustedExecutable()
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_runtime",
        lambda release: calls.append(("runtime", release)) or executable,
    )
    monkeypatch.setattr(
        launcher,
        "activate_trusted_owner_support",
        lambda runtime, *, release_sha: calls.append(
            ("support_activate", runtime, release_sha)
        ),
    )
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_support_activation",
        lambda runtime, *, release_sha: calls.append(
            ("support_revalidate", runtime, release_sha)
        ),
    )
    monkeypatch.setattr(
        launcher,
        "require_local_launcher_provenance",
        lambda release: calls.append(("provenance", release)) or "f" * 64,
    )
    monkeypatch.setattr(
        launcher,
        "_validate_owner_interpreter_invocation",
        lambda path: calls.append(("interpreter", path)),
    )
    monkeypatch.setattr(
        launcher,
        "PinnedGcloudConfiguration",
        lambda: gcloud_configuration,
    )
    monkeypatch.setattr(
        launcher,
        "GcloudOwnerAccessToken",
        lambda **kwargs: calls.append(("identity", kwargs)) or owner_identity,
    )
    monkeypatch.setattr(
        launcher,
        "IapSchemaReconciliationControlBootstrapTransport",
        lambda identity, **kwargs: (
            calls.append(("control_transport", identity, kwargs)) or transport
        ),
    )
    monkeypatch.setattr(
        launcher,
        "GoogleRestClient",
        lambda identity: (
            calls.append(("database_client", identity)) or database_client
        ),
    )

    def bootstrap(**kwargs):
        calls.append(("control_bootstrap", kwargs))
        kwargs["provenance_guard"](kwargs["release_sha"])
        return receipt

    monkeypatch.setattr(
        launcher,
        "bootstrap_schema_reconciliation_control",
        bootstrap,
    )
    monkeypatch.setattr(launcher, "_emit_canonical_line", emitted.append)

    result = launcher.main([
        "--release-sha",
        RELEASE_SHA,
        "--bootstrap-schema-reconciliation-control",
    ])

    assert result == 0
    assert emitted == [receipt]
    call = next(item for item in calls if item[0] == "control_bootstrap")
    assert call[1]["release_sha"] == RELEASE_SHA
    assert call[1]["transport"] is transport
    assert call[1]["cloud_sql_client"] is database_client
    assert call[1]["owner_identity"] is owner_identity
    assert ("support_activate", executable, RELEASE_SHA) in calls
    assert ("support_revalidate", executable, RELEASE_SHA) in calls


@pytest.mark.parametrize(
    "action",
    (
        "--reconcile-legacy-canary-db",
        "--bootstrap-schema-reconciliation-control",
    ),
)
def test_schema_reconciliation_cli_rejects_external_iam_before_runtime(
    monkeypatch,
    action,
):
    emitted = []
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_runtime",
        lambda _release: pytest.fail("invalid CLI reached trusted runtime"),
    )
    monkeypatch.setattr(launcher, "_emit_canonical_line", emitted.append)

    result = launcher.main([
        "--release-sha",
        RELEASE_SHA,
        action,
        "--external-iam-policy-sha256",
        "f" * 64,
    ])

    assert result == 2
    assert emitted[0]["error_code"] == "schema_reconciliation_owner_cli_invalid"


class _RecordingReconciliationSigner:
    def __init__(self):
        self.calls = []

    def sign(self, message, *, namespace, expected_authority):
        self.calls.append((message, namespace, expected_authority))
        return "-----BEGIN SSH SIGNATURE-----\nfixed\n-----END SSH SIGNATURE-----\n"


def _schema_reconciliation_gate():
    return {
        "issued_at_unix": NOW - 10,
        "expires_at_unix": NOW + 1_000,
        "gate_sha256": "1" * 64,
        "release_revision": RELEASE_SHA,
        "plan_sha256": "2" * 64,
        "temporary_executor_username_sha256": "3" * 64,
        "owner_subject_sha256": OWNER_SHA,
        "owner_key_id": "4" * 64,
        "journal_head": {
            "state": "empty",
            "authorized_intent_sha256": None,
            "terminal_receipt_sha256": None,
            "head_sha256": "5" * 64,
        },
    }


def _schema_reconciliation_remote_failure(
    gate,
    *,
    wire_stage,
    transcript_head_sha256,
    error_code="schema_reconciliation_runtime_post_cleanup_invalid",
):
    return _self_digest(
        {
            "schema": launcher.SCHEMA_RECONCILIATION_REMOTE_FAILURE_SCHEMA,
            "ok": False,
            "wire_stage": wire_stage,
            "error_code": error_code,
            "gate_sha256": gate["gate_sha256"],
            "release_revision": gate["release_revision"],
            "plan_sha256": gate["plan_sha256"],
            "transcript_head_sha256": transcript_head_sha256,
            "secret_material_recorded": False,
        },
        "receipt_sha256",
    )


def test_schema_reconciliation_remote_failure_validator_binds_exact_prefix():
    gate = _schema_reconciliation_gate()
    receipt = _schema_reconciliation_remote_failure(
        gate,
        wire_stage="a2_to_i2",
        transcript_head_sha256="7" * 64,
    )

    assert launcher.validate_schema_reconciliation_remote_failure(
        receipt,
        gate=gate,
        expected_wire_stage="a2_to_i2",
        expected_transcript_head_sha256="7" * 64,
    ) == receipt

    for field, replacement in (
        ("wire_stage", "c3_to_t3"),
        ("error_code", "not_schema_reconciliation_code"),
        ("gate_sha256", "8" * 64),
        ("release_revision", "b" * 40),
        ("plan_sha256", "9" * 64),
        ("transcript_head_sha256", "a" * 64),
        ("secret_material_recorded", True),
    ):
        changed = dict(receipt)
        changed[field] = replacement
        changed.pop("receipt_sha256")
        changed = _self_digest(changed, "receipt_sha256")
        with pytest.raises(
            launcher.OwnerLauncherError,
            match="schema_reconciliation_remote_failure_invalid",
        ):
            launcher.validate_schema_reconciliation_remote_failure(
                changed,
                gate=gate,
                expected_wire_stage="a2_to_i2",
                expected_transcript_head_sha256="7" * 64,
            )


class _RoleBoundReconciliationClient:
    def __init__(self, username: str, database_roles: list[str]) -> None:
        self.username = username
        self.calls: list[tuple[str, str, object]] = []
        self.list_etag = "non-authoritative-list-etag"
        self.describe_etag = "exact-etag-v1"
        self.list_payload = {
            "kind": "sql#usersList",
            "items": [
                {
                    "kind": "sql#user",
                    "name": username,
                    "project": launcher.PROJECT,
                    "instance": launcher.SQL_INSTANCE,
                    "host": "",
                    # Deliberately no type or databaseRoles: this matches the
                    # real users.list projection and must remain presence-only.
                    "etag": self.list_etag,
                }
            ],
        }
        self.describe_payload = {
            "kind": "sql#user",
            "databaseRoles": list(database_roles),
            "etag": self.describe_etag,
            "host": "",
            "instance": launcher.SQL_INSTANCE,
            "name": username,
            "project": launcher.PROJECT,
            # Cloud SQL users.get omits the default BUILT_IN type for real
            # PostgreSQL users.  The boundary must canonicalize that API
            # default without accepting an explicitly different type.
        }

    def request_json(self, method, url, *, body=None):
        self.calls.append((method, url, body))
        assert method == "GET"
        assert body is None
        if url == (
            f"{launcher.CloudSqlTemporaryAdmin._BASE}/instances/"
            f"{launcher.SQL_INSTANCE}/users"
        ):
            return self.list_payload
        encoded_username = urllib.parse.quote(self.username, safe="")
        if url == (
            f"{launcher.CloudSqlTemporaryAdmin._BASE}/instances/"
            f"{launcher.SQL_INSTANCE}/users/{encoded_username}?host="
        ):
            return self.describe_payload
        raise AssertionError(f"unexpected Cloud SQL request: {method} {url}")


def _role_bound_schema_reconciliation_cloud_boundary(
    database_roles: list[str],
) -> launcher.CloudSqlSchemaReconciliationExecutor:
    username = "muncho_canary_reconciler_" + "a" * 16
    operation = ("CREATE_USER", "DONE", OWNER_SHA, True)
    operations = {"authority-operation": operation}
    client = _RoleBoundReconciliationClient(username, database_roles)
    boundary = launcher.CloudSqlSchemaReconciliationExecutor(client)
    boundary._mutation_operation_baseline = frozenset()
    boundary._mutation_relevant_baseline = {}
    boundary._confirmed_authority_operation_name = "authority-operation"
    boundary._confirmed_authority_operation_type = "CREATE_USER"
    boundary._expected_owner_subject_sha256 = OWNER_SHA
    boundary._expected_mutation_context_sha256 = "f" * 64
    boundary._instance_operations = lambda: operations
    boundary._stable_instance_operations = lambda: operations
    return boundary


def test_schema_reconciliation_cloud_boundary_uses_fenced_describes_for_partial_list() -> None:
    username = "muncho_canary_reconciler_" + "a" * 16
    boundary = _role_bound_schema_reconciliation_cloud_boundary(
        list(launcher.SCHEMA_RECONCILIATION_DATABASE_ROLES)
    )
    operation_snapshots = []
    operations = boundary._instance_operations()
    boundary._instance_operations = lambda: (
        operation_snapshots.append(dict(operations)) or operations
    )

    resource = boundary._role_bound_resource(
        username,
        require_exact_roles=True,
    )

    assert resource is not None
    assert resource["databaseRoles"] == list(
        launcher.SCHEMA_RECONCILIATION_DATABASE_ROLES
    )
    assert resource["etag"] == "exact-etag-v1"
    assert resource["host"] == ""
    assert resource["type"] == "BUILT_IN"
    client = boundary._client
    assert isinstance(client, _RoleBoundReconciliationClient)
    list_calls = [call for call in client.calls if call[1] == boundary._users_url]
    describe_calls = [
        call for call in client.calls if call[1] == boundary._user_url(username)
    ]
    assert len(operation_snapshots) == 4
    assert len(list_calls) == 4
    assert len(describe_calls) == 2
    assert client.list_etag != resource["etag"]


def test_schema_reconciliation_cloud_boundary_accepts_explicit_built_in_type() -> None:
    username = "muncho_canary_reconciler_" + "a" * 16
    boundary = _role_bound_schema_reconciliation_cloud_boundary(
        list(launcher.SCHEMA_RECONCILIATION_DATABASE_ROLES)
    )
    client = boundary._client
    assert isinstance(client, _RoleBoundReconciliationClient)
    client.describe_payload["type"] = "BUILT_IN"

    resource = boundary._role_bound_resource(
        username,
        require_exact_roles=True,
    )

    assert resource is not None
    assert resource["type"] == "BUILT_IN"


@pytest.mark.parametrize(
    "resource_type",
    [
        None,
        "",
        0,
        "UNKNOWN",
        "CLOUD_IAM_USER",
        "CLOUD_IAM_SERVICE_ACCOUNT",
        "CLOUD_IAM_GROUP",
        "CLOUD_IAM_GROUP_USER",
        "CLOUD_IAM_GROUP_SERVICE_ACCOUNT",
    ],
)
def test_schema_reconciliation_cloud_boundary_rejects_explicit_non_built_in_type(
    resource_type: object,
) -> None:
    username = "muncho_canary_reconciler_" + "a" * 16
    boundary = _role_bound_schema_reconciliation_cloud_boundary(
        list(launcher.SCHEMA_RECONCILIATION_DATABASE_ROLES)
    )
    client = boundary._client
    assert isinstance(client, _RoleBoundReconciliationClient)
    client.describe_payload["type"] = resource_type

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="cloud_sql_schema_reconciliation_executor_resource_invalid",
    ):
        boundary._role_bound_resource(
            username,
            require_exact_roles=True,
        )


@pytest.mark.parametrize(
    ("etag_present", "etag"),
    [
        (False, None),
        (True, None),
        (True, ""),
        (True, "invalid etag"),
        (True, 0),
    ],
)
def test_schema_reconciliation_cloud_boundary_keeps_etag_mandatory_when_type_omitted(
    etag_present: bool,
    etag: object,
) -> None:
    username = "muncho_canary_reconciler_" + "a" * 16
    boundary = _role_bound_schema_reconciliation_cloud_boundary(
        list(launcher.SCHEMA_RECONCILIATION_DATABASE_ROLES)
    )
    client = boundary._client
    assert isinstance(client, _RoleBoundReconciliationClient)
    if etag_present:
        client.describe_payload["etag"] = etag
    else:
        client.describe_payload.pop("etag")

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="cloud_sql_schema_reconciliation_executor_resource_invalid",
    ):
        boundary._role_bound_resource(
            username,
            require_exact_roles=True,
        )


def test_schema_reconciliation_cloud_boundary_rejects_describe_drift() -> None:
    username = "muncho_canary_reconciler_" + "a" * 16
    boundary = _role_bound_schema_reconciliation_cloud_boundary(
        list(launcher.SCHEMA_RECONCILIATION_DATABASE_ROLES)
    )
    client = boundary._client
    assert isinstance(client, _RoleBoundReconciliationClient)
    original_request = client.request_json
    describe_calls = 0

    def drifting_request(method, url, *, body=None):
        nonlocal describe_calls
        payload = original_request(method, url, body=body)
        if url == boundary._user_url(username):
            describe_calls += 1
            if describe_calls == 2:
                return {**payload, "etag": "drifted-describe-etag"}
        return payload

    client.request_json = drifting_request

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="cloud_sql_schema_reconciliation_executor_resource_drifted",
    ):
        boundary._role_bound_resource(
            username,
            require_exact_roles=True,
        )


def test_schema_reconciliation_cloud_boundary_is_exact_custom_role_only() -> None:
    username = "muncho_canary_reconciler_" + "a" * 16
    boundary = _role_bound_schema_reconciliation_cloud_boundary(
        list(launcher.SCHEMA_RECONCILIATION_DATABASE_ROLES)
    )

    assert launcher.SCHEMA_RECONCILIATION_DATABASE_ROLES == (
        "canonical_brain_schema_reconciler",
    )
    assert launcher.SCHEMA_RECONCILIATION_EXECUTOR_USERNAME_PREFIX == (
        "muncho_canary_reconciler_"
    )
    assert boundary._create_user_body(username, "q" * 64) == {
        "databaseRoles": list(launcher.SCHEMA_RECONCILIATION_DATABASE_ROLES),
        "instance": launcher.SQL_INSTANCE,
        "name": username,
        "password": "q" * 64,
        "project": launcher.PROJECT,
        "type": "BUILT_IN",
    }
    assert boundary._update_user_query_values(username) == {
        "databaseRoles": list(launcher.SCHEMA_RECONCILIATION_DATABASE_ROLES),
        "host": "",
        "name": username,
        "revokeExistingRoles": "true",
    }
    receipt = boundary.temporary_executor_authority_receipt(username)
    assert receipt["schema"] == (
        launcher.SCHEMA_RECONCILIATION_EXECUTOR_AUTHORITY_RECEIPT_SCHEMA
    )
    assert receipt["type"] == "BUILT_IN"
    assert receipt["database_roles"] == list(
        launcher.SCHEMA_RECONCILIATION_DATABASE_ROLES
    )
    assert receipt["cloudsqlsuperuser_absent"] is True
    assert receipt["resource_etag_sha256"] == hashlib.sha256(
        b"exact-etag-v1"
    ).hexdigest()
    assert boundary._absence_evidence_identity() == {
        "schema": (
            launcher.SCHEMA_RECONCILIATION_EXECUTOR_ABSENCE_RECEIPT_SCHEMA
        ),
        "temporary_executor_absent": True,
    }
    assert receipt["receipt_sha256"] == hashlib.sha256(
        _canonical({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    ).hexdigest()


def test_schema_reconciliation_cloud_boundary_rejects_extra_cloudsqlsuperuser() -> None:
    username = "muncho_canary_reconciler_" + "a" * 16
    boundary = _role_bound_schema_reconciliation_cloud_boundary([
        *launcher.SCHEMA_RECONCILIATION_DATABASE_ROLES,
        "cloudsqlsuperuser",
    ])

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="cloud_sql_schema_reconciliation_executor_resource_invalid",
    ):
        boundary.temporary_executor_authority_receipt(username)


def test_schema_reconciliation_cloud_update_revokes_every_stale_role_in_query() -> None:
    username = "muncho_canary_reconciler_" + "a" * 16

    class Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, Mapping[str, object]]] = []

        def request_json(self, method, url, *, body=None):
            self.calls.append((method, url, body))
            return {"name": "role-replacement-operation"}

    client = Client()
    boundary = launcher.CloudSqlSchemaReconciliationExecutor(client)
    boundary._fixed_update_etag = "stale-user-etag"
    boundary._ensure_mutation_observation = lambda: None
    boundary._settle_user_presence = lambda *_args, **_kwargs: True
    boundary._record_operation_name = lambda *_args, **_kwargs: (
        "role-replacement-operation"
    )
    boundary._wait_operation = lambda *_args, **_kwargs: None
    boundary._confirm_direct_mutation_operation = lambda *_args, **_kwargs: None

    boundary.rotate_existing(username, "q" * 64)

    method, url, body = client.calls[-1]
    query = urllib.parse.parse_qs(
        urllib.parse.urlsplit(url).query,
        keep_blank_values=True,
    )
    assert method == "PUT"
    assert query == {
        "databaseRoles": list(launcher.SCHEMA_RECONCILIATION_DATABASE_ROLES),
        "host": [""],
        "name": [username],
        "revokeExistingRoles": ["true"],
    }
    assert body["etag"] == "stale-user-etag"
    assert body["revokeExistingRoles"] is True


def test_schema_reconciliation_control_admin_is_separate_broad_one_time_login():
    username = "muncho_canary_control_" + "a" * 16
    operation = ("CREATE_USER", "DONE", OWNER_SHA, True)
    boundary = launcher.CloudSqlSchemaReconciliationControlAdmin(object())
    boundary._mutation_operation_baseline = frozenset()
    boundary._mutation_relevant_baseline = {}
    boundary._confirmed_authority_operation_name = "authority-operation"
    boundary._confirmed_authority_operation_type = "CREATE_USER"
    boundary._expected_owner_subject_sha256 = OWNER_SHA
    boundary._expected_mutation_context_sha256 = "f" * 64
    boundary._stable_instance_operations = lambda: {
        "authority-operation": operation
    }
    boundary.require_current_authority = lambda value: (
        None if value == username else pytest.fail("wrong control username")
    )

    assert boundary._create_user_body(username, "q" * 64) == {
        "instance": launcher.SQL_INSTANCE,
        "name": username,
        "password": "q" * 64,
        "project": launcher.PROJECT,
        "type": "BUILT_IN",
    }
    receipt = boundary.temporary_control_admin_authority_receipt(username)
    assert receipt["schema"] == (
        launcher.SCHEMA_RECONCILIATION_CONTROL_ADMIN_AUTHORITY_RECEIPT_SCHEMA
    )
    assert receipt["broad_bootstrap_authority"] is True
    assert receipt["database_roles_requested"] == []
    assert receipt["normal_reconciliation_executor"] is False
    assert receipt["receipt_sha256"] == hashlib.sha256(
        _canonical({
            key: value
            for key, value in receipt.items()
            if key != "receipt_sha256"
        })
    ).hexdigest()
    assert boundary._absence_evidence_identity() == {
        "schema": (
            launcher.SCHEMA_RECONCILIATION_CONTROL_ADMIN_ABSENCE_RECEIPT_SCHEMA
        ),
        "temporary_control_admin_absent": True,
    }


def test_schema_reconciliation_uses_role_bound_cloud_boundary_by_default() -> None:
    default = inspect.signature(
        launcher.reconcile_legacy_canary_schema
    ).parameters["boundary_factory"].default
    assert default is launcher.CloudSqlSchemaReconciliationExecutor


def test_schema_reconciliation_owner_builders_bind_all_three_signed_frames():
    signer = _RecordingReconciliationSigner()
    authority = object()
    gate = _schema_reconciliation_gate()
    credential = bytearray(b"c" * 64)
    cloud_authority = {"receipt_sha256": "6" * 64}
    admin = launcher.build_schema_reconciliation_executor_preflight(
        gate=gate,
        cloud_sql_authority_receipt=cloud_authority,
        credential=credential,
        signer=signer,
        owner_authority=authority,
        now_unix=NOW,
        nonce_factory=lambda size: b"a" * size,
    )
    admin_unsigned = {
        key: value
        for key, value in admin.items()
        if key not in {"authority_claim_sha256", "signature_sshsig"}
    }
    assert admin["authority_claim_sha256"] == hashlib.sha256(
        _canonical(admin_unsigned)
    ).hexdigest()
    assert admin["action"] == "authorize_temporary_executor_locked_preflight"
    assert admin["temporary_executor_username_sha256"] == "3" * 64
    assert signer.calls[-1][0] == (
        reconciliation_bootstrap.admin_preflight_signature_payload(admin)
    )
    assert signer.calls[-1][1] == (
        launcher.SCHEMA_RECONCILIATION_EXECUTOR_PREFLIGHT_SSHSIG_NAMESPACE
    )

    challenge = {
        "preflight_challenge_sha256": "7" * 64,
        "preflight": {
            "state": "exact_old_missing_one_helper",
            "preflight_sha256": "8" * 64,
            "observed_contract_sha256": "9" * 64,
            "truth_receipt_sha256": "a" * 64,
        },
    }
    authorization = launcher.build_schema_reconciliation_preflight_authorization(
        gate=gate,
        admin_preflight=admin,
        challenge=challenge,
        post_hba_temporary_executor_authority_receipt=cloud_authority,
        signer=signer,
        owner_authority=authority,
        now_unix=NOW + 1,
        nonce_factory=lambda size: b"b" * size,
    )
    authorization_unsigned = {
        key: value
        for key, value in authorization.items()
        if key not in {"preflight_authorization_claim_sha256", "signature_sshsig"}
    }
    assert authorization["execution_mode"] == "reconcile_missing_helper"
    assert authorization[
        "post_hba_temporary_executor_authority_receipt"
    ] == cloud_authority
    assert authorization[
        "post_hba_temporary_executor_authority_receipt_sha256"
    ] == cloud_authority["receipt_sha256"]
    assert authorization["preflight_authorization_claim_sha256"] == hashlib.sha256(
        _canonical(authorization_unsigned)
    ).hexdigest()
    assert signer.calls[-1][0] == (
        reconciliation_bootstrap.preflight_authorization_signature_payload(
            authorization
        )
    )

    intermediate = {"database_intermediate_sha256": "b" * 64}
    absence = {"evidence_sha256": "c" * 64}
    cleanup = launcher.build_schema_reconciliation_executor_cleanup(
        gate=gate,
        admin_preflight=admin,
        challenge=challenge,
        authorization=authorization,
        intermediate=intermediate,
        cloud_sql_absence_receipt=absence,
        signer=signer,
        owner_authority=authority,
        now_unix=NOW + 2,
        nonce_factory=lambda size: b"d" * size,
    )
    cleanup_unsigned = {
        key: value
        for key, value in cleanup.items()
        if key not in {"cleanup_claim_sha256", "signature_sshsig"}
    }
    assert cleanup["cleanup_claim_sha256"] == hashlib.sha256(
        _canonical(cleanup_unsigned)
    ).hexdigest()
    assert cleanup["action"] == "confirm_temporary_executor_absence"
    assert cleanup["temporary_executor_username_sha256"] == "3" * 64
    assert signer.calls[-1][0] == (
        reconciliation_bootstrap.admin_cleanup_signature_payload(cleanup)
    )
    assert [call[1] for call in signer.calls] == [
        launcher.SCHEMA_RECONCILIATION_EXECUTOR_PREFLIGHT_SSHSIG_NAMESPACE,
        reconciliation_bootstrap.PREFLIGHT_AUTHORIZATION_OWNER_SSHSIG_NAMESPACE,
        launcher.SCHEMA_RECONCILIATION_EXECUTOR_CLEANUP_SSHSIG_NAMESPACE,
    ]


def test_schema_reconciliation_a1_frame_carries_only_exact_opaque_credential():
    credential = bytearray(b"z" * 64)
    claim = {"schema": "fixed"}
    frame = launcher._schema_reconciliation_frame(
        launcher.SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_MAGIC,
        claim,
        credential=credential,
    )
    payload = _canonical(claim)

    assert bytes(frame[:4]) == launcher.SCHEMA_RECONCILIATION_ADMIN_PREFLIGHT_MAGIC
    assert int.from_bytes(frame[4:8], "big") == len(payload)
    assert bytes(frame[8 : 8 + len(payload)]) == payload
    assert bytes(frame[-64:]) == bytes(credential)
    assert hashlib.sha256(bytes(credential)).hexdigest().encode() not in frame


def _fixed_owner_authority():
    public = bytes.fromhex(
        "4f928fd117e2e62f1e52b0095d6ab552"
        "4370707f5e9b295efdc62479ce887e26"
    )
    return launcher._PhaseBOwnerPublicAuthority(
        public_key_ed25519_hex=public.hex(),
        key_id=hashlib.sha256(public).hexdigest(),
        public_key_file_sha256="d" * 64,
        public_fingerprint=launcher.PHASE_B_OWNER_PUBLIC_KEY_FINGERPRINT,
        public_key_source={
            "path": "fixed",
            "file_sha256": "d" * 64,
            "device": 1,
            "inode": 2,
            "uid": 501,
            "gid": 20,
            "mode": "0600",
            "size": 100,
        },
        _public_blob=b"public",
        _public_item_identity=(1,) * 10,
        _private_item_identity=(2,) * 10,
    )


class _FixedSchemaReconciliationSigner(_RecordingReconciliationSigner):
    def __init__(self, authority):
        super().__init__()
        self.authority = authority

    def inspect(self):
        return self.authority


class _SchemaReconciliationOwnerIdentity:
    def __init__(self, account):
        self.account = account
        self.owner_subject_sha256 = None
        self.bound = False

    def account_for_read_only_preflight(self):
        return self.account

    def bind_approved_subject(self, expected):
        assert expected == hashlib.sha256(self.account.encode()).hexdigest()
        self.owner_subject_sha256 = expected
        self.bound = True

    def require_stable(self):
        assert self.bound


class _SchemaReconciliationBoundary:
    def __init__(
        self,
        events,
        *,
        cleanup_fails=False,
        create_fails=False,
        authority_fails=False,
    ):
        self.events = events
        self.cleanup_fails = cleanup_fails
        self.create_fails = create_fails
        self.authority_fails = authority_fails
        self.ambiguous = False

    def begin_mutation_observation(self, **values):
        self.events.append(("begin", values))

    def create_or_rotate_recovery(self, username, password):
        self.events.append(("create", username, len(password)))
        if self.create_fails:
            raise launcher.OwnerLauncherError("cloud_sql_operation_failed")

    def temporary_executor_authority_receipt(self, username):
        self.events.append(("authority", username))
        return {"receipt_sha256": "6" * 64}

    def require_current_authority(self, username):
        self.events.append(("require", username))
        if self.authority_fails:
            raise launcher.OwnerLauncherError(
                "cloud_sql_mutation_evidence_unconfirmed"
            )

    def delete_and_confirm_absent(self, username):
        self.events.append(("delete", username))
        if self.cleanup_fails:
            raise launcher.CleanupBlocked("delete_unconfirmed")

    def reconciliation_receipt(self):
        return {"evidence_sha256": "c" * 64}

    def mutation_reconciliation_required(self):
        return self.ambiguous

    def reconcile_ambiguous_mutation_and_confirm_absent(self, username):
        self.events.append(("reconcile", username))


class _SchemaReconciliationSession:
    def __init__(
        self,
        gate,
        challenge,
        intermediate,
        terminal,
        events,
        *,
        fail_a2=False,
        abort_fails=False,
        failure_stage=None,
    ):
        self.gate = gate
        self.challenge = challenge
        self.intermediate = intermediate
        self.terminal = terminal
        self.events = events
        self.fail_a2 = fail_a2
        self.abort_fails = abort_fails
        self.failure_stage = failure_stage
        self.round = 0

    def _failure(self, wire_stage):
        head = {
            "a1_to_p1": self.gate["gate_sha256"],
            "a2_to_i2": self.challenge["preflight_challenge_sha256"],
            "c3_to_t3": self.intermediate["database_intermediate_sha256"],
        }[wire_stage]
        return _schema_reconciliation_remote_failure(
            self.gate,
            wire_stage=wire_stage,
            transcript_head_sha256=head,
        )

    def read_gate(self):
        return self.gate

    def require_current_authority(self):
        self.events.append("iap")

    def schema_reconciliation_exchange_before(
        self,
        frame,
        *,
        write_guard,
        on_first_write,
        on_write_complete,
    ):
        self.events.append(("a1", bytes(frame[:4])))
        write_guard()
        on_first_write()
        on_write_complete()
        if self.failure_stage == "a1_to_p1":
            return self._failure("a1_to_p1")
        return self.challenge

    def schema_reconciliation_exchange(self, frame, *, terminal):
        self.round += 1
        self.events.append(("exchange", self.round, bytes(frame[:4]), terminal))
        if self.round == 1:
            if self.fail_a2:
                raise launcher.OwnerLauncherError("remote_apply_failed")
            if self.failure_stage == "a2_to_i2":
                return self._failure("a2_to_i2")
            return self.intermediate
        if self.failure_stage == "c3_to_t3":
            return self._failure("c3_to_t3")
        return self.terminal

    def mark_validated(self, terminal):
        assert terminal == self.terminal or terminal == self._failure(
            self.failure_stage
        )
        self.events.append("validated")

    def abort_and_prove_terminated(self):
        self.events.append("aborted")
        if self.abort_fails:
            raise launcher.RemoteTerminationUnconfirmed()

    def close(self):
        self.events.append("closed")


class _SchemaReconciliationTransport:
    def __init__(self, session):
        self.session = session

    def open_reconciliation(self, release_sha):
        assert release_sha == RELEASE_SHA
        return self.session


def _patch_schema_reconciliation_owner_validators(monkeypatch):
    monkeypatch.setattr(
        reconciliation_bootstrap,
        "validate_gate",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        reconciliation_bootstrap,
        "validate_preflight_challenge_for_owner",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        reconciliation_bootstrap,
        "validate_database_intermediate_for_owner",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        reconciliation_bootstrap,
        "validate_terminal_for_owner",
        lambda value, **_kwargs: value,
    )


def _schema_reconciliation_owner_case(
    events,
    *,
    fail_a2=False,
    cleanup_fails=False,
    create_fails=False,
    authority_fails=False,
    abort_fails=False,
    failure_stage=None,
):
    authority = _fixed_owner_authority()
    account = "owner@example.com"
    owner_subject = hashlib.sha256(account.encode()).hexdigest()
    gate = {
        **_schema_reconciliation_gate(),
        "release_revision": RELEASE_SHA,
        "owner_subject_sha256": owner_subject,
        "owner_public_key_ed25519_hex": authority.public_key_ed25519_hex,
        "owner_key_id": authority.key_id,
        "owner_public_fingerprint": authority.public_fingerprint,
        "temporary_executor_username": (
            launcher.SCHEMA_RECONCILIATION_EXECUTOR_USERNAME_PREFIX
            + ("2" * 64)[:16]
        ),
        "expires_at_unix": NOW + 1_200,
    }
    challenge = {
        "preflight_challenge_sha256": "7" * 64,
        "preflight": {
            "state": "exact_old_missing_one_helper",
            "preflight_sha256": "8" * 64,
            "observed_contract_sha256": "9" * 64,
            "truth_receipt_sha256": "a" * 64,
        },
    }
    intermediate = {"database_intermediate_sha256": "b" * 64}
    terminal = {"ok": True, "terminal_sha256": "e" * 64}
    session = _SchemaReconciliationSession(
        gate,
        challenge,
        intermediate,
        terminal,
        events,
        fail_a2=fail_a2,
        abort_fails=abort_fails,
        failure_stage=failure_stage,
    )
    boundary = _SchemaReconciliationBoundary(
        events,
        cleanup_fails=cleanup_fails,
        create_fails=create_fails,
        authority_fails=authority_fails,
    )
    return {
        "authority": authority,
        "identity": _SchemaReconciliationOwnerIdentity(account),
        "session": session,
        "transport": _SchemaReconciliationTransport(session),
        "boundary": boundary,
    }


def test_schema_reconciliation_owner_orchestrates_and_cleans_before_c3(monkeypatch):
    _patch_schema_reconciliation_owner_validators(monkeypatch)
    events = []
    case = _schema_reconciliation_owner_case(events)
    credential = bytearray(b"q" * 64)

    receipt = launcher.reconcile_legacy_canary_schema(
        release_sha=RELEASE_SHA,
        transport=case["transport"],
        cloud_sql_client=object(),
        owner_identity=case["identity"],
        now=lambda: NOW,
        password_factory=lambda: credential,
        nonce_factory=lambda size: b"n" * size,
        signer=_FixedSchemaReconciliationSigner(case["authority"]),
        boundary_factory=lambda _client: case["boundary"],
        secret_hardener=lambda: None,
        provenance_guard=lambda _revision: None,
    )

    assert receipt == case["session"].terminal
    assert credential == bytearray(64)
    authority_indexes = [
        index
        for index, item in enumerate(events)
        if isinstance(item, tuple) and item[0] == "authority"
    ]
    assert len(authority_indexes) == 2
    p1_index = next(
        index
        for index, item in enumerate(events)
        if isinstance(item, tuple) and item[0] == "a1"
    )
    a2_index = next(
        index
        for index, item in enumerate(events)
        if isinstance(item, tuple)
        and item[:3]
        == (
            "exchange",
            1,
            launcher.SCHEMA_RECONCILIATION_PREFLIGHT_AUTHORIZATION_MAGIC,
        )
    )
    require_index = max(
        index
        for index, item in enumerate(events[: authority_indexes[1]])
        if isinstance(item, tuple) and item[0] == "require"
    )
    assert p1_index < require_index < authority_indexes[1] < a2_index
    delete_index = next(i for i, item in enumerate(events) if item[0] == "delete")
    c3_index = next(
        i
        for i, item in enumerate(events)
        if isinstance(item, tuple)
        and item[:3]
        == (
            "exchange",
            2,
            launcher.SCHEMA_RECONCILIATION_ADMIN_CLEANUP_MAGIC,
        )
    )
    assert delete_index < c3_index
    assert events[-2:] == ["validated", "closed"]


def test_schema_reconciliation_owner_rejects_post_hba_authority_drift_before_a2(
    monkeypatch,
):
    _patch_schema_reconciliation_owner_validators(monkeypatch)
    events = []
    case = _schema_reconciliation_owner_case(events)
    authority_call_count = 0

    def changed_authority_receipt(username):
        nonlocal authority_call_count
        authority_call_count += 1
        events.append(("authority", username))
        digest = "6" * 64 if authority_call_count == 1 else "7" * 64
        return {"receipt_sha256": digest}

    case["boundary"].temporary_executor_authority_receipt = (
        changed_authority_receipt
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="schema_reconciliation_owner_frame_invalid",
    ):
        launcher.reconcile_legacy_canary_schema(
            release_sha=RELEASE_SHA,
            transport=case["transport"],
            cloud_sql_client=object(),
            owner_identity=case["identity"],
            now=lambda: NOW,
            password_factory=lambda: bytearray(b"w" * 64),
            nonce_factory=lambda size: b"n" * size,
            signer=_FixedSchemaReconciliationSigner(case["authority"]),
            boundary_factory=lambda _client: case["boundary"],
            secret_hardener=lambda: None,
            provenance_guard=lambda _revision: None,
        )

    assert authority_call_count == 2
    assert not any(
        isinstance(item, tuple) and item[0] == "exchange"
        for item in events
    )
    assert "aborted" in events
    assert any(
        isinstance(item, tuple) and item[0] == "delete"
        for item in events
    )


@pytest.mark.parametrize(
    "failure_stage",
    ("a1_to_p1", "a2_to_i2", "c3_to_t3"),
)
def test_schema_reconciliation_owner_validates_phase_failure_before_cleanup(
    monkeypatch,
    failure_stage,
):
    _patch_schema_reconciliation_owner_validators(monkeypatch)
    events = []
    case = _schema_reconciliation_owner_case(
        events,
        failure_stage=failure_stage,
    )

    with pytest.raises(launcher.RemoteCommandFailed) as caught:
        launcher.reconcile_legacy_canary_schema(
            release_sha=RELEASE_SHA,
            transport=case["transport"],
            cloud_sql_client=object(),
            owner_identity=case["identity"],
            now=lambda: NOW,
            password_factory=lambda: bytearray(b"v" * 64),
            nonce_factory=lambda size: b"n" * size,
            signer=_FixedSchemaReconciliationSigner(case["authority"]),
            boundary_factory=lambda _client: case["boundary"],
            secret_hardener=lambda: None,
            provenance_guard=lambda _revision: None,
        )

    assert caught.value.code == (
        "schema_reconciliation_runtime_post_cleanup_invalid"
    )
    assert caught.value.receipt["wire_stage"] == failure_stage
    assert "validated" in events
    assert sum(
        isinstance(item, tuple) and item[0] == "delete" for item in events
    ) == 1
    validated_index = events.index("validated")
    delete_index = next(i for i, item in enumerate(events) if item[0] == "delete")
    if failure_stage == "c3_to_t3":
        assert delete_index < validated_index
    else:
        assert validated_index < delete_index


def test_schema_reconciliation_owner_failure_still_deletes_executor(monkeypatch):
    _patch_schema_reconciliation_owner_validators(monkeypatch)
    events = []
    case = _schema_reconciliation_owner_case(events, fail_a2=True)
    credential = bytearray(b"r" * 64)

    with pytest.raises(launcher.OwnerLauncherError, match="remote_apply_failed"):
        launcher.reconcile_legacy_canary_schema(
            release_sha=RELEASE_SHA,
            transport=case["transport"],
            cloud_sql_client=object(),
            owner_identity=case["identity"],
            now=lambda: NOW,
            password_factory=lambda: credential,
            nonce_factory=lambda size: b"n" * size,
            signer=_FixedSchemaReconciliationSigner(case["authority"]),
            boundary_factory=lambda _client: case["boundary"],
            secret_hardener=lambda: None,
            provenance_guard=lambda _revision: None,
        )

    assert credential == bytearray(64)
    abort_index = events.index("aborted")
    delete_index = next(i for i, item in enumerate(events) if item[0] == "delete")
    assert abort_index < delete_index


def test_schema_reconciliation_unconfirmed_abort_defers_executor_delete(monkeypatch):
    _patch_schema_reconciliation_owner_validators(monkeypatch)
    events = []
    case = _schema_reconciliation_owner_case(
        events,
        fail_a2=True,
        abort_fails=True,
    )

    with pytest.raises(launcher.CleanupBlocked) as error:
        launcher.reconcile_legacy_canary_schema(
            release_sha=RELEASE_SHA,
            transport=case["transport"],
            cloud_sql_client=object(),
            owner_identity=case["identity"],
            now=lambda: NOW,
            password_factory=lambda: bytearray(b"u" * 64),
            nonce_factory=lambda size: b"n" * size,
            signer=_FixedSchemaReconciliationSigner(case["authority"]),
            boundary_factory=lambda _client: case["boundary"],
            secret_hardener=lambda: None,
            provenance_guard=lambda _revision: None,
        )

    assert error.value.cause_code == "remote_termination_unconfirmed"
    assert "aborted" in events
    assert not any(
        isinstance(item, tuple) and item[0] == "delete" for item in events
    )
    assert events[-1] == "closed"


def test_schema_reconciliation_definitive_rotation_failure_still_proves_absence(
    monkeypatch,
):
    _patch_schema_reconciliation_owner_validators(monkeypatch)
    events = []
    case = _schema_reconciliation_owner_case(
        events,
        create_fails=True,
        authority_fails=True,
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="cloud_sql_operation_failed",
    ):
        launcher.reconcile_legacy_canary_schema(
            release_sha=RELEASE_SHA,
            transport=case["transport"],
            cloud_sql_client=object(),
            owner_identity=case["identity"],
            now=lambda: NOW,
            password_factory=lambda: bytearray(b"t" * 64),
            nonce_factory=lambda size: b"n" * size,
            signer=_FixedSchemaReconciliationSigner(case["authority"]),
            boundary_factory=lambda _client: case["boundary"],
            secret_hardener=lambda: None,
            provenance_guard=lambda _revision: None,
        )

    abort_index = events.index("aborted")
    delete_index = next(i for i, item in enumerate(events) if item[0] == "delete")
    assert abort_index < delete_index


def test_schema_reconciliation_cleanup_failure_overrides_remote_failure(monkeypatch):
    _patch_schema_reconciliation_owner_validators(monkeypatch)
    events = []
    case = _schema_reconciliation_owner_case(
        events,
        fail_a2=True,
        cleanup_fails=True,
    )

    with pytest.raises(launcher.CleanupBlocked):
        launcher.reconcile_legacy_canary_schema(
            release_sha=RELEASE_SHA,
            transport=case["transport"],
            cloud_sql_client=object(),
            owner_identity=case["identity"],
            now=lambda: NOW,
            password_factory=lambda: bytearray(b"s" * 64),
            nonce_factory=lambda size: b"n" * size,
            signer=_FixedSchemaReconciliationSigner(case["authority"]),
            boundary_factory=lambda _client: case["boundary"],
            secret_hardener=lambda: None,
            provenance_guard=lambda _revision: None,
        )


def test_schema_reconciliation_cleanup_state_failure_does_not_skip_session_close(
    monkeypatch,
):
    _patch_schema_reconciliation_owner_validators(monkeypatch)
    events = []
    case = _schema_reconciliation_owner_case(events)

    class _UnavailableStateBoundary(_SchemaReconciliationBoundary):
        def begin_mutation_observation(self, **values):
            self.events.append(("begin", values))
            raise launcher.OwnerLauncherError("begin_observation_failed")

        def mutation_reconciliation_required(self):
            raise RuntimeError("state unavailable")

    boundary = _UnavailableStateBoundary(events)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="begin_observation_failed",
    ) as error:
        launcher.reconcile_legacy_canary_schema(
            release_sha=RELEASE_SHA,
            transport=case["transport"],
            cloud_sql_client=object(),
            owner_identity=case["identity"],
            now=lambda: NOW,
            password_factory=lambda: bytearray(b"s" * 64),
            nonce_factory=lambda size: b"n" * size,
            signer=_FixedSchemaReconciliationSigner(case["authority"]),
            boundary_factory=lambda _client: boundary,
            secret_hardener=lambda: None,
            provenance_guard=lambda _revision: None,
        )

    assert events[-1] == "closed"
    assert launcher._attached_cleanup_failure_codes(error.value) == (
        "schema_reconciliation_executor_cleanup_state_failed",
    )
