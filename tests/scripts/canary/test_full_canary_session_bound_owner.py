from __future__ import annotations

import inspect

import hashlib
import json
from collections.abc import Mapping

import gateway.canonical_writer_preflight_publisher as preflight_publisher
from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import writer_release


RELEASE_SHA = "a" * 40
OWNER_SHA = hashlib.sha256(b"owner@example.com").hexdigest()
NOW = 1_000
TOKEN = b"isolated-discord-token"


def test_stopped_release_activation_inventory_matches_writer_release_contract():
    writer_paths = tuple(str(path) for path in writer_release._ACTIVATION_PATHS)

    assert launcher._STOPPED_RELEASE_ACTIVATION_PATHS == writer_paths
    assert tuple(str(path) for path in preflight_publisher._ACTIVATION_PATHS) == (
        writer_paths
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
