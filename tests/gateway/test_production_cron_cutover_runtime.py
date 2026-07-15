from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from gateway import production_cron_cutover_runtime as runtime
from gateway.operational_edge_readiness import (
    OPERATIONAL_EDGE_READINESS_SCHEMA,
)


PLAN_SHA = "1" * 64
RELEASE_REVISION = "a" * 40
EDGE_BOOT_ID_SHA256 = "c" * 64
EDGE_OBSERVED_AT_UNIX = 1_800_000_000
EDGE_COLLECTOR_NONCE = "12345678-1234-4123-8123-123456789abc"


def _runtime_readiness(_context: runtime.RuntimeContext) -> tuple[dict, dict]:
    return (
        {
            "schema": OPERATIONAL_EDGE_READINESS_SCHEMA,
            "release_revision": RELEASE_REVISION,
            "receipt_sha256": "a" * 64,
            "boot_id_sha256": EDGE_BOOT_ID_SHA256,
            "observed_at_unix": EDGE_OBSERVED_AT_UNIX,
            "maximum_age_seconds": 120,
            "collector_nonce": EDGE_COLLECTOR_NONCE,
            "required_job_count": 14,
            "job_count": 14,
        },
        {
            "readiness_sha256": "b" * 64,
            "activation_ready": True,
            "scoped_execution_edge_receipt_sha256": "a" * 64,
            "scoped_execution_edge_meaningful_packet_count": 14,
        },
    )


def _preflight_receipt() -> dict:
    return runtime._receipt(
        {
            "schema": runtime.PREFLIGHT_SCHEMA,
            "created_at": "2026-07-15T00:00:00Z",
            "cutover_plan_sha256": PLAN_SHA,
            "continuity_plan_sha256": "2" * 64,
            "artifact_index_sha256": "3" * 64,
            "source_store_sha256": "4" * 64,
            "expected_target_store_sha256": "5" * 64,
            "collector_timer_count": 21,
            "gateway_writer_connector_stopped": True,
            "artifacts_valid": True,
            "source_store_unchanged": True,
            "prepared_recovery_sha256": "6" * 64,
            "jobs_archive_sha256": "4" * 64,
            "host_snapshot_sha256": "7" * 64,
            "spool_prestate_sha256": "8" * 64,
            "manifest_directory_prestate_sha256": "9" * 64,
            "legacy_auto_sync_disabled": True,
            "legacy_auto_sync_no_active_claim": True,
            "legacy_auto_sync_next_run_reconciled": True,
            "collector_execution_readiness_sha256": "b" * 64,
            "operational_edge_readiness_receipt_sha256": "a" * 64,
            "operational_edge_boot_id_sha256": EDGE_BOOT_ID_SHA256,
            "operational_edge_observed_at_unix": EDGE_OBSERVED_AT_UNIX,
            "operational_edge_maximum_age_seconds": 120,
            "operational_edge_collector_nonce": EDGE_COLLECTOR_NONCE,
            "operational_edge_meaningful_packet_count": 14,
            "collector_execution_ready": True,
            "recovery_evidence_persisted": True,
            "runtime_target_mutation_performed": False,
            "provider_or_model_invoked": False,
            "discord_delivery_attempted": False,
            "secret_material_recorded": False,
        }
    )


def test_receipt_validator_is_exact_self_digesting_and_semantic_free() -> None:
    receipt = _preflight_receipt()

    assert runtime.validate_cutover_receipt(
        receipt,
        action="preflight",
        plan_sha256=PLAN_SHA,
        expected_sha256=receipt["receipt_sha256"],
    ) == receipt

    for mutate in (
        lambda value: value.update(extra=True),
        lambda value: value.update(provider_or_model_invoked=True),
        lambda value: value.update(collector_timer_count=20),
        lambda value: value.update(source_store_sha256="f" * 64),
    ):
        drifted = copy.deepcopy(receipt)
        mutate(drifted)
        with pytest.raises(
            runtime.ProductionCronCutoverRuntimeError,
            match="receipt_invalid",
        ):
            runtime.validate_cutover_receipt(
                drifted,
                action="preflight",
                plan_sha256=PLAN_SHA,
            )


def test_runtime_readiness_collects_and_publishes_live_instead_of_loading_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import operational_edge_readiness as edge_readiness

    context = runtime.RuntimeContext(
        cutover_plan={"plan_sha256": PLAN_SHA},
        inventory={},
        continuity_plan={},
        replacement_bundle={},
        collector_package={"release_revision": RELEASE_REVISION},
        artifact_index={},
    )
    operational, execution = _runtime_readiness(context)
    calls: list[tuple[str, dict[str, str]]] = []

    def collect_live(**kwargs):
        calls.append(("live", dict(kwargs["required_jobs"])))
        assert kwargs["revision"] == RELEASE_REVISION
        return operational

    monkeypatch.setattr(
        edge_readiness,
        "collect_and_publish_operational_edge_readiness",
        collect_live,
    )
    monkeypatch.setattr(
        edge_readiness,
        "load_operational_edge_readiness",
        lambda **_kwargs: pytest.fail("stale readiness loader used"),
    )
    monkeypatch.setattr(
        runtime.collector_rail,
        "collect_execution_readiness",
        lambda *_args, **_kwargs: execution,
    )
    monkeypatch.setattr(
        runtime.collector_rail,
        "validate_execution_readiness",
        lambda value, **_kwargs: value,
    )

    assert runtime._collect_runtime_execution_readiness(context) == (
        operational,
        execution,
    )
    assert len(calls) == 1
    assert len(calls[0][1]) == 14


def test_activation_authority_binds_terminal_forward_only_journal() -> None:
    authority = runtime.build_activation_authority(
        cutover_plan_sha256=PLAN_SHA,
        cron_postflight_receipt_sha256="2" * 64,
        database_terminal_entry_sha256="3" * 64,
        activation_commit_intent_entry_sha256="4" * 64,
        boot_committed_entry_sha256="5" * 64,
        gateway_started_entry_sha256="6" * 64,
    )

    assert runtime.validate_activation_authority(
        authority,
        cutover_plan_sha256=PLAN_SHA,
        cron_postflight_receipt_sha256="2" * 64,
        expected_authority_sha256=authority["authority_sha256"],
    ) == authority
    assert authority["forward_recovery_only"] is True
    assert authority["secret_material_recorded"] is False

    drifted = copy.deepcopy(authority)
    drifted["gateway_started"] = False
    with pytest.raises(
        runtime.ProductionCronCutoverRuntimeError,
        match="activation_authority_invalid",
    ):
        runtime.validate_activation_authority(
            drifted,
            cutover_plan_sha256=PLAN_SHA,
            cron_postflight_receipt_sha256="2" * 64,
            expected_authority_sha256=authority["authority_sha256"],
        )


def test_directory_prestate_restores_absence_or_exact_metadata(
    tmp_path: Path,
) -> None:
    absent = tmp_path / "absent"
    absent_state = runtime._directory_prestate(absent)
    absent.mkdir(mode=0o750)
    runtime._restore_directory_prestate(absent_state, path=absent)
    assert not absent.exists()

    present = tmp_path / "present"
    present.mkdir(mode=0o710)
    os.chmod(present, 0o710)
    present_state = runtime._directory_prestate(present)
    os.chmod(present, 0o755)
    runtime._restore_directory_prestate(present_state, path=present)
    metadata = present.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o710
    assert metadata.st_uid == present_state["uid"]
    assert metadata.st_gid == present_state["gid"]


def test_rollback_is_forward_only_as_soon_as_activation_authority_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = runtime.RuntimeContext(
        cutover_plan={"plan_sha256": PLAN_SHA},
        inventory={},
        continuity_plan={},
        replacement_bundle={},
        collector_package={"release_revision": RELEASE_REVISION},
        artifact_index={},
    )
    monkeypatch.setattr(runtime, "load_runtime_context", lambda **_kwargs: context)
    authority = tmp_path / "activation-authority.json"
    authority.write_text("partially-written-is-still-forward-only", encoding="utf-8")

    with pytest.raises(
        runtime.ProductionCronCutoverRuntimeError,
        match="forward_recovery_required",
    ):
        runtime.rollback(
            expected_cutover_plan_sha256=PLAN_SHA,
            evidence_root=tmp_path / "evidence",
            activation_authority_path=authority,
            services_stopped=lambda: True,
        )


def test_preflight_replay_revalidates_source_and_prepared_host_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = b'{"jobs":[]}\n'
    source_sha = hashlib.sha256(source).hexdigest()
    target_sha = "5" * 64
    unsigned = {
        key: value
        for key, value in _preflight_receipt().items()
        if key != "receipt_sha256"
    }
    unsigned.update(
        source_store_sha256=source_sha,
        jobs_archive_sha256=source_sha,
        expected_target_store_sha256=target_sha,
    )
    receipt = runtime._receipt(unsigned)
    archive = tmp_path / "archive.json"
    archive.write_bytes(source)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps({"snapshot_sha256": "7" * 64}),
        encoding="utf-8",
    )
    jobs = tmp_path / "jobs.json"
    jobs.write_bytes(source)
    prepared = {
        "prepared_sha256": receipt["prepared_recovery_sha256"],
        "source_store_sha256": source_sha,
        "expected_target_store_sha256": target_sha,
        "jobs_archive_path": str(archive),
        "host_snapshot_path": str(snapshot_path),
    }
    context = runtime.RuntimeContext(
        cutover_plan={"plan_sha256": PLAN_SHA},
        inventory={},
        continuity_plan={},
        replacement_bundle={},
        collector_package={"release_revision": RELEASE_REVISION},
        artifact_index={},
    )
    monkeypatch.setattr(runtime, "load_runtime_context", lambda **_kwargs: context)
    monkeypatch.setattr(runtime, "_existing_receipt", lambda **_kwargs: receipt)
    monkeypatch.setattr(runtime, "_load_prepared", lambda **_kwargs: prepared)
    monkeypatch.setattr(runtime, "_timers_match", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runtime, "_live_snapshot_matches", lambda *_args: True)
    monkeypatch.setattr(
        runtime, "_installed_package_files_match", lambda *_args: False
    )
    monkeypatch.setattr(runtime, "_packet_root_gateway_readable", lambda: False)

    assert runtime.preflight(
        expected_cutover_plan_sha256=PLAN_SHA,
        jobs_path=jobs,
        evidence_root=tmp_path / "evidence",
        activation_authority_path=tmp_path / "authority.json",
        services_stopped=lambda: True,
        execution_readiness_collector=_runtime_readiness,
    ) == receipt

    jobs.write_bytes(b'{"jobs":[{"id":"drift"}]}\n')
    with pytest.raises(
        runtime.ProductionCronCutoverRuntimeError,
        match="preflight_replay_drifted",
    ):
        runtime.preflight(
            expected_cutover_plan_sha256=PLAN_SHA,
            jobs_path=jobs,
            evidence_root=tmp_path / "evidence",
            activation_authority_path=tmp_path / "authority.json",
            services_stopped=lambda: True,
            execution_readiness_collector=_runtime_readiness,
        )


def test_apply_refuses_unexplained_host_drift_before_any_target_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = b'{"jobs":[]}\n'
    target = b'{"jobs":[{"id":"target"}]}\n'
    source_sha = hashlib.sha256(source).hexdigest()
    target_sha = hashlib.sha256(target).hexdigest()
    archive = tmp_path / "archive.json"
    archive.write_bytes(source)
    jobs = tmp_path / "jobs.json"
    jobs.write_bytes(source)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot = {"snapshot_sha256": "7" * 64}
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    prior_unsigned = {
        key: value
        for key, value in _preflight_receipt().items()
        if key != "receipt_sha256"
    }
    prior_unsigned.update(
        source_store_sha256=source_sha,
        jobs_archive_sha256=source_sha,
        expected_target_store_sha256=target_sha,
    )
    prior = runtime._receipt(prior_unsigned)
    prepared = {
        "prepared_sha256": prior["prepared_recovery_sha256"],
        "source_store_sha256": source_sha,
        "expected_target_store_sha256": target_sha,
        "jobs_archive_path": str(archive),
        "host_snapshot_path": str(snapshot_path),
        "host_snapshot_sha256": snapshot["snapshot_sha256"],
    }
    context = runtime.RuntimeContext(
        cutover_plan={"plan_sha256": PLAN_SHA},
        inventory={},
        continuity_plan={},
        replacement_bundle={},
        collector_package={"release_revision": RELEASE_REVISION},
        artifact_index={},
    )
    monkeypatch.setattr(runtime, "load_runtime_context", lambda **_kwargs: context)
    monkeypatch.setattr(runtime, "_prior_receipt", lambda **_kwargs: prior)
    monkeypatch.setattr(runtime, "_existing_receipt", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "_load_prepared", lambda **_kwargs: prepared)
    monkeypatch.setattr(
        runtime, "build_target_jobs_bytes", lambda **_kwargs: target
    )
    monkeypatch.setattr(runtime, "_validate_snapshot", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr(runtime, "_live_snapshot_matches", lambda *_args: False)
    monkeypatch.setattr(
        runtime, "_installed_package_files_match", lambda *_args: False
    )
    monkeypatch.setattr(runtime, "_packet_root_gateway_readable", lambda: False)
    monkeypatch.setattr(runtime, "_unit_paths", lambda *_args: {})

    with pytest.raises(
        runtime.ProductionCronCutoverRuntimeError,
        match="host_prestate_drifted",
    ):
        runtime.apply(
            expected_cutover_plan_sha256=PLAN_SHA,
            expected_preflight_receipt_sha256=prior["receipt_sha256"],
            jobs_path=jobs,
            evidence_root=tmp_path / "evidence",
            activation_authority_path=tmp_path / "authority.json",
            services_stopped=lambda: True,
            execution_readiness_collector=_runtime_readiness,
        )
    assert jobs.read_bytes() == source


def test_isolated_entrypoint_attests_itself_and_runtime(tmp_path: Path) -> None:
    release = tmp_path / "hermes-agent-aaaaaaaaaaaa"
    entrypoint = release / "scripts/canary/production_cron_cutover_entrypoint.py"
    runtime_path = release / "gateway/production_cron_cutover_runtime.py"
    entrypoint.parent.mkdir(parents=True)
    runtime_path.parent.mkdir(parents=True)
    shutil.copyfile(
        Path(runtime.__file__).parents[1]
        / "scripts/canary/production_cron_cutover_entrypoint.py",
        entrypoint,
    )
    (runtime_path.parent / "__init__.py").write_text("", encoding="utf-8")
    runtime_path.write_text(
        "import json\n"
        "def main(argv):\n"
        "    print(json.dumps({'argv': argv}, sort_keys=True))\n"
        "    return 0\n",
        encoding="utf-8",
    )
    entrypoint_sha = hashlib.sha256(entrypoint.read_bytes()).hexdigest()
    runtime_sha = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    command = [
        sys.executable,
        "-I",
        "-B",
        str(entrypoint),
        "preflight",
        "--expected-cutover-plan-sha256",
        PLAN_SHA,
        "--expected-entrypoint-sha256",
        entrypoint_sha,
        "--expected-runtime-sha256",
        runtime_sha,
    ]

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "argv": [
            "preflight",
            "--expected-cutover-plan-sha256",
            PLAN_SHA,
        ]
    }

    command[-1] = "f" * 64
    drifted = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert drifted.returncode == 1
    assert "launcher_identity_drifted" in drifted.stderr
