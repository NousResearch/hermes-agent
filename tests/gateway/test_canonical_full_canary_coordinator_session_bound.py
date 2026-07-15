from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import canonical_full_canary_coordinator as coordinator
from gateway.canonical_full_canary_runtime import FullCanaryOwnerApproval


RELEASE = "a" * 40
OWNER = "b" * 64
INPUT_SHA = "c" * 64
PLAN_SHA = "d" * 64
FIXTURE_SHA = "e" * 64


class _Plan:
    sha256 = PLAN_SHA
    artifacts = {"e2e_fixture": SimpleNamespace(sha256=FIXTURE_SHA)}

    def to_mapping(self):
        return {"full_canary_plan_sha256": self.sha256}


class _Prepared:
    def __init__(self, plan, approval):
        self.plan = plan
        self.approval = approval
        self.session_key_sha256 = "8" * 64
        self.fixture_sha256 = FIXTURE_SHA
        self.discarded = False

    def discard_session_key(self):
        self.discarded = True


def _approval():
    return {
        "schema": "muncho-full-canary-owner-approval.v1",
        "scope": "full_canary_runtime_start",
        "plan_sha256": PLAN_SHA,
        "authority_kind": "trusted_root_bootstrap_out_of_band_owner",
        "cryptographic_owner_proof": False,
        "owner_subject_sha256": OWNER,
        "approval_source_sha256": coordinator.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256,
        "nonce_sha256": "9" * 64,
        "approved_at_unix": 1_000,
        "expires_at_unix": 1_100,
    }


def test_session_bound_run_has_no_admin_bootstrap_or_semantic_scope(monkeypatch):
    identities = SimpleNamespace(writer_gid=1001, gateway_gid=1002)
    input_value = {
        "writer_config": {"database": {"user": "writer"}},
        "writer_activation_receipt": {"ok": True},
        "writer_activation_receipt_file_sha256": "1" * 64,
    }
    canary_input = SimpleNamespace(
        revision=RELEASE,
        sha256=INPUT_SHA,
        identities=identities,
        value=input_value,
        writer_activation_plan=object(),
        artifacts={
            "gateway_config": object(),
            "edge_config": object(),
            "host_identity_receipt": object(),
        },
    )
    anchor = {
        "phase_b_release_revision": RELEASE,
        "phase_b_plan_sha256": "2" * 64,
        "phase_b_approval_sha256": "3" * 64,
        "phase_b_terminal_receipt_sha256": "4" * 64,
        "phase_b_foundation_generation_sha256": "5" * 64,
        "phase_b_readiness_receipt_sha256": "6" * 64,
        "phase_b_readiness_handoff_file_sha256": "7" * 64,
        "phase_b_readiness_sequence": 1,
    }
    foundation = {
        "approval": {
            "owner_subject_sha256": OWNER,
            "approval_source_sha256": coordinator.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256,
            "approval_sha256": "3" * 64,
        }
    }
    base_fixture = {
        "valid_until_unix_ms": 2_000_000,
        "release_sha": RELEASE,
    }
    install_receipt = {"receipt_sha256": "a" * 64}
    installed = SimpleNamespace(st_dev=1, st_ino=2)
    install_snapshot = object()
    emitted = []
    observed = {}
    prepared_holder = {}
    stopped = {"value": True}

    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(coordinator, "_harden_secret_process", lambda: None)
    monkeypatch.setattr(coordinator, "load_coordinator_input", lambda: canary_input)
    monkeypatch.setattr(
        coordinator,
        "_services_are_exactly_stopped_and_disabled",
        lambda: stopped["value"],
    )
    monkeypatch.setattr(
        coordinator,
        "_load_fixed_phase_b_live_authority",
        lambda _value: (foundation, anchor),
    )
    monkeypatch.setattr(
        coordinator,
        "_validated_base_e2e_fixture",
        lambda _value: base_fixture,
    )
    monkeypatch.setattr(
        coordinator,
        "load_discord_token_install_receipt",
        lambda _value: (install_receipt, installed, install_snapshot),
    )
    monkeypatch.setattr(coordinator, "_capture_writer_snapshot", lambda _gid: None)
    monkeypatch.setattr(
        coordinator,
        "_read_staged_writer_config",
        lambda path, **_kwargs: (
            (b'{"database":{"user":"writer"}}', object())
            if path == coordinator.DEFAULT_WRITER_CONFIG_SOURCE
            else (b'{"api_session_key_sha256":"8"}', object())
        ),
    )
    monkeypatch.setattr(coordinator.os.path, "lexists", lambda _path: False)
    monkeypatch.setattr(
        coordinator,
        "_publish_root_payload",
        lambda *_args, **_kwargs: SimpleNamespace(after=SimpleNamespace(sha256="f" * 64)),
    )
    monkeypatch.setattr(coordinator, "build_full_canary_plan", lambda **_kwargs: _Plan())
    monkeypatch.setattr(coordinator, "validate_dedicated_canary_host", lambda _plan: None)
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_000.0)

    def prepare(**kwargs):
        observed.update(kwargs)
        assert "bootstrap_provisioner" not in kwargs
        plan = kwargs["plan_builder"]()
        approval = kwargs["approval_provider"](plan)
        prepared = _Prepared(plan, approval)
        prepared_holder["value"] = prepared
        return prepared

    monkeypatch.setattr(coordinator, "prepare_session_bound_plan", prepare)

    live_result = {
        "schema": "muncho-full-canary-live-driver.v1",
        "ok": True,
        "release_sha": RELEASE,
        "full_canary_plan_sha256": PLAN_SHA,
    }

    class _Driver:
        def __init__(self, prepared):
            assert prepared is prepared_holder["value"]

        def run(self):
            return live_result

    def retire(**kwargs):
        assert kwargs["install_receipt"] == install_receipt
        return {
            "state": "retired",
            "token_removed": True,
            "install_receipt_removed": True,
        }

    monkeypatch.setattr(coordinator, "_retire_discord_token_lease", retire)

    receipt = coordinator.run_session_bound_full_canary(
        frame_emitter=emitted.append,
        final_approval_frame_reader=_approval,
        driver_factory=_Driver,
    )

    assert receipt["schema"] == coordinator.SESSION_BOUND_COORDINATOR_RECEIPT_SCHEMA
    assert receipt["temporary_admin_created"] is False
    assert receipt["bootstrap_credential_created"] is False
    assert receipt["services_stopped"] is True
    assert receipt["discord_token_retired"] is True
    assert prepared_holder["value"].discarded is True
    assert len(emitted) == 1
    assert emitted[0]["approval_path"] is None
    assert emitted[0]["approval_source_sha256"] == (
        coordinator.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
    )
    assert observed["writer_config"] == input_value["writer_config"]


def test_coordinator_cli_has_only_target_session_bound_surface():
    assert coordinator._ATTESTED_CLI_COMMANDS == {
        "publish-coordinator-input",
        "preflight-phase-b-apply",
        "preflight-phase-b-live-run",
        "phase-b-apply",
        "install-discord-token",
        "run",
        "stop-and-retire-discord-token",
    }


def test_phase_b_owner_lineage_requires_agreeing_pinned_receipts():
    activation = SimpleNamespace(
        value={
            "owner_subject_sha256": OWNER,
            "approval_source_sha256": (
                coordinator.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
            ),
        }
    )
    native = SimpleNamespace(value=dict(activation.value))

    assert coordinator._phase_b_owner_lineage(
        activation_approval=activation,
        native_approval=native,
    ) == {
        "owner_subject_sha256": OWNER,
        "approval_source_sha256": (
            coordinator.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
        ),
    }

    native.value["owner_subject_sha256"] = "0" * 64
    with pytest.raises(coordinator.CoordinatorError, match="authority_drifted"):
        coordinator._phase_b_owner_lineage(
            activation_approval=activation,
            native_approval=native,
        )


def test_initial_phase_b_gate_is_readiness_without_credential_file(
    monkeypatch,
    tmp_path: Path,
):
    canary_input = SimpleNamespace(revision=RELEASE, sha256=INPUT_SHA)
    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(coordinator, "_harden_secret_process", lambda: None)
    monkeypatch.setattr(coordinator, "load_coordinator_input", lambda: canary_input)
    monkeypatch.setattr(
        coordinator,
        "_require_phase_b_pristine_live_boundary",
        lambda: None,
    )
    monkeypatch.setattr(
        coordinator,
        "_phase_b_authority_provenance",
        lambda _value: {
            "owner_subject_sha256": OWNER,
            "approval_source_sha256": (
                coordinator.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
            ),
        },
    )
    monkeypatch.setattr(coordinator, "PHASE_B_AUTHORITY_ROOT", tmp_path / "absent")
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_000.0)

    gate = coordinator.preflight_phase_b_apply()

    assert gate["schema"] == "muncho-canonical-writer-phase-b-owner-gate.v2"
    assert gate["state"] == "initial_apply_ready"
    assert gate["owner_subject_sha256"] == OWNER
    assert gate["approval_source_sha256"] == (
        coordinator.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
    )
    assert gate["authority_present"] is False
    assert gate["expires_at_unix"] == 1_900
    assert not any("credential_prepare" in key for key in gate)


def test_phase_b_sidecar_contains_signed_lineage_not_legacy_credential():
    plan = SimpleNamespace(sha256=PLAN_SHA)
    approval_value = {
        "approval_source_sha256": coordinator.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256,
        "owner_subject_sha256": OWNER,
        "issued_at_unix": 1_000,
        "expires_at_unix": 1_100,
    }
    approval = SimpleNamespace(
        sha256="f" * 64,
        to_mapping=lambda: dict(approval_value),
    )
    sidecar = coordinator._phase_b_authority_sidecar(
        provenance={
            "owner_subject_sha256": OWNER,
            "approval_source_sha256": (
                coordinator.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
            ),
            "authority_sources": {},
            "authority_sources_sha256": hashlib.sha256(b"{}").hexdigest(),
        },
        plan=plan,
        approval=approval,
    )

    assert sidecar["schema"] == "muncho-canonical-writer-phase-b-authority.v2"
    assert not any("credential_prepare" in key for key in sidecar)


def test_terminal_failure_has_no_legacy_admin_or_bootstrap_claims(monkeypatch):
    monkeypatch.setattr(
        coordinator,
        "_stable_root_read",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(coordinator.os.path, "lexists", lambda _path: False)
    monkeypatch.setattr(
        coordinator,
        "_services_are_exactly_stopped_and_disabled",
        lambda: True,
    )
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_000.0)

    receipt = coordinator._unbound_failure(RuntimeError("boom"), command="run")

    assert receipt["schema"] == coordinator.COORDINATOR_FAILURE_SCHEMA
    assert receipt["cleanup_status"] == "complete"
    assert receipt["discord_token_removed"] is True
    assert receipt["services_stopped"] is True
    assert receipt["obsolete_process_journal_absent"] is True
    assert set(receipt) == {
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
