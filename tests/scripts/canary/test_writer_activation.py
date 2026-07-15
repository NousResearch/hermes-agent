"""The source canary entrypoint is only a packaged-planner delegate."""

from __future__ import annotations

import json

from gateway import canonical_writer_planner as planner
from scripts.canary import writer_activation


REVISION = "a" * 40


def test_source_entrypoint_delegates_all_production_planning_contracts():
    for name in (
        "build_activation_plan",
        "build_and_stage_final_activation_plan",
        "build_and_stage_native_observation_plan",
        "build_native_observation_plan",
        "load_release_manifest",
        "main",
    ):
        assert getattr(writer_activation, name) is getattr(planner, name)

    for legacy_name in (
        "ActivationPreview",
        "build_activation_preview",
        "write_root_activation_preview",
        "write_root_collector_manifest_preview",
    ):
        assert not hasattr(writer_activation, legacy_name)


def test_source_entrypoint_native_cli_is_digest_only(monkeypatch, capsys):
    result = {
        "artifact_sha256": "1" * 64,
        "native_observation_plan_sha256": "2" * 64,
    }
    monkeypatch.setattr(
        planner,
        "build_and_stage_native_observation_plan",
        lambda **_arguments: result,
    )

    assert writer_activation.main([
        "build-native-plan",
        "--revision",
        REVISION,
        "--external-iam-policy-sha256",
        "3" * 64,
        "--config-collector-receipt-sha256",
        "4" * 64,
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)

    assert emitted == result
    assert all(name.endswith("_sha256") for name in emitted)


def test_source_entrypoint_final_cli_is_digest_only(monkeypatch, capsys):
    result = {
        "activation_plan_sha256": "5" * 64,
        "native_observation_receipt_sha256": "6" * 64,
    }
    monkeypatch.setattr(
        planner,
        "build_and_stage_final_activation_plan",
        lambda **_arguments: result,
    )

    assert writer_activation.main([
        "build-final-plan",
        "--native-observation-receipt-sha256",
        "6" * 64,
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)

    assert emitted == result
    assert all(name.endswith("_sha256") for name in emitted)
