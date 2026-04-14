"""Tests for Hermes-native release evidence ladder generation and gating."""

from __future__ import annotations

import json
from pathlib import Path

from gateway import status
from gateway.evidence_ladder import build_release_evidence_bundle, evaluate_release_evidence_ladder


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_release_evidence_bundle_emits_hermes_runtime_refs_and_passes_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status.write_pid_file()
    status.write_runtime_status(
        gateway_state="running",
        active_agents=2,
        platform="telegram",
        platform_state="connected",
        error_code=None,
        error_message=None,
    )
    status.append_runtime_evidence("gateway_running", details={"source": "test"})

    bundle = build_release_evidence_bundle(
        release_id="rel_wave3_smoke",
        activation_mode="shadow",
        repo_root=REPO_ROOT,
    )
    decision = evaluate_release_evidence_ladder(bundle=bundle, repo_root=REPO_ROOT)

    assert bundle["schema_version"] == "clawd.release_evidence_bundle.v1"
    assert bundle["activation_mode"] == "shadow"
    assert [row["stage"] for row in bundle["stages"]] == [
        "local_determinism",
        "presubmit",
        "integration_replay",
        "shadow",
    ]
    assert bundle["stages"][0]["evidence_refs"][0].endswith("gateway_state.json")
    assert bundle["stages"][0]["evidence_refs"][1].endswith("gateway.pid")
    assert bundle["rollback_proof"]["state_rollback_ref"].endswith("gateway_runtime_events.jsonl")
    assert bundle["compatibility_lifecycle"]["register_ref"].endswith("compatibility_lifecycle.json")

    assert decision["verdict"] == "pass"
    assert [row["status"] for row in decision["gate_results"]] == ["pass", "pass", "pass", "pass"]
    log_path = tmp_path / "release_governance" / "release_evidence_ladder_decisions.jsonl"
    assert log_path.exists()
    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["release_id"] == "rel_wave3_smoke"


def test_build_release_evidence_bundle_blocks_when_runtime_is_invalid(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status.write_pid_file()
    (tmp_path / "gateway_state.json").write_text(json.dumps({"pid": "oops", "gateway_state": "mystery", "platforms": []}))
    status.append_runtime_evidence("gateway_startup_failed", details={"source": "test"})

    bundle = build_release_evidence_bundle(
        release_id="rel_wave3_broken",
        activation_mode="shadow",
        repo_root=REPO_ROOT,
    )
    decision = evaluate_release_evidence_ladder(bundle=bundle, repo_root=REPO_ROOT)

    assert bundle["stages"][0]["status"] == "block"
    assert decision["verdict"] == "block"
    stage_coverage = next(row for row in decision["gate_results"] if row["gate_id"] == "stage_coverage")
    assert stage_coverage["status"] == "block"
    assert "blocked_stages" in stage_coverage["details"]
