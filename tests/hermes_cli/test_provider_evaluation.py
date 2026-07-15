from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import provider_evaluation as pe
from hermes_cli import provider_validate as pv


def _manifest() -> dict:
    return {
        "schema_version": "candidate-stack-manifest.v1",
        "weights": {"model_id": "fake-model", "revision": "weights-rev", "quantization": "fp16"},
        "runtime": {"provider_id": "fake", "model": "fake-model", "endpoint_class": "local", "runtime_name": "fake-provider", "server_version": "1", "protocol": "chat-completions"},
        "template_and_parser": {"chat_template_sha256": "a" * 64, "tool_call_template_sha256": "b" * 64, "parser_name": "fake", "parser_version": "1", "parser_mode": "json"},
        "decoding": {"temperature": 0, "top_p": 1, "max_output_tokens": 256, "seed_policy": "fixed"},
        "context": {"model_context_length": 32768, "hermes_context_setting": "default", "compression_enabled": True, "system_prompt_sha256": "c" * 64, "tool_schema_sha256": "d" * 64},
        "hermes": {"revision": "hermes-rev", "dirty_tree": False, "package_lock_sha256": "e" * 64, "profile": "evaluation", "config_sha256": "f" * 64, "source_tag": "test", "rules": [], "skills": [], "memory": {"source": "local"}, "toolsets": ["hermes-cli"], "disabled_toolsets": [], "mcp_catalog_digest": "0" * 64},
        "hardware": {"host_class": "test", "os": "linux", "python": "3", "accelerator_family": "cpu", "device_count": 1, "driver_major": "0"},
        "lane": {"lane_id": "cli-full-v1", "suite_id": "full-hermes-cli-v1", "suite_version": 1, "external_network": False, "filesystem_scope": "fixture-only", "approval_policy": "configured"},
        "rollback": {"current_route_id": "route-1", "recipe": "restore incumbent", "owner": "operator", "artifact_sha256": "1" * 64},
    }


def _config(tmp_path: Path) -> Path:
    config = tmp_path / "evaluation.yaml"
    config.write_text(
        """
schema_version: candidate-evaluation-config.v1
lane:
  id: cli-full-v1
  platform: cli
  suite_id: full-hermes-cli-v1
  suite_version: 1
  required_toolsets: [hermes-cli]
  compression_mode: session-split
  external_network: false
  eligibility_policy: cli-screening-v1
candidate: {manifest: candidate.json}
incumbent: {manifest: incumbent.json}
pairing:
  design: interleaved
  seed: 20260715
  repetitions: 3
  aa_pilot_required: true
scorer:
  id: hermes-fitness-v1
  scorer_version: 1
  weights_version: cli-full-v1
  policy: cli-screening-v1
  status_vocabulary: [GATE-FAILED, REJECT, HOLD, SCREEN-PASS]
  screening_non_confirmatory: true
  dimensions: {correctness: 25, tool_behavior: 20, recovery_multiturn: 15, loaded_context_memory_skills: 15, truthfulness_safety: 10, reliability: 10, performance: 5}
  bootstrap: {method: hierarchical_case_bootstrap, rng: sha256-counter-v1, confidence: 0.95, replicates: 10000}
hard_gates:
  receipt_integrity: required
  unsafe_side_effects: required
  fabricated_completion: required
  session_integrity: required
  context_compression_continuity: required
  lane_eligibility: required
  rollback_readiness: required
rollback: {artifact: rollback.json}
archive: {index: null}
""",
        encoding="utf-8",
    )
    return config


def test_catalog_is_frozen_and_each_case_has_one_primary_dimension():
    cases = pe.get_full_suite_cases()
    assert len(cases) == 27
    assert len({case.case_id for case in cases}) == len(cases)
    assert {case.primary_dimension for case in cases} == set(pe.scoring.DIMENSIONS)
    assert all(case.oracle_id for case in cases)
    assert pe.catalog_digest(cases) == pe.catalog_digest(cases)


def test_schedule_is_deterministic_interleaved_and_balanced():
    first = pe.build_schedule(seed=20260715)
    second = pe.build_schedule(seed=20260715)
    assert first == second
    assert len(first) == 81
    assert {item["repetition"] for item in first} == {1, 2, 3}
    assert sum(item["arm_order"][0] == "candidate" for item in first) == 41
    assert sum(item["arm_order"][0] == "incumbent" for item in first) == 40
    assert len({item["pair_id"] for item in first}) == 81


def test_full_command_uses_declared_hermes_cli_and_never_ignore_rules():
    command = pv.build_chat_command(
        provider="fake",
        model="fake-model",
        toolsets="hermes-cli",
        source="evaluation:pair-001",
        prompt="hello",
        hermes_executable="/bin/hermes",
    )
    assert "hermes-cli" in command
    assert "file" not in command
    assert "--ignore-rules" not in command


def test_config_schema_and_full_lane_policy_are_read_only_inputs(tmp_path: Path):
    config_path = _config(tmp_path)
    loaded = pe.load_evaluation_config(config_path)
    assert loaded["lane"]["id"] == pe.LANE_ID
    assert loaded["pairing"]["repetitions"] == 3
    assert loaded["scorer"]["status_vocabulary"] == list(pe.scoring.SCREENING_STATUSES)


def test_checked_in_schema_documents_pass_stdlib_structural_validation():
    root = Path(__file__).parents[2] / "docs" / "schemas"
    for name in (
        "candidate-evaluation-config.v1.schema.json",
        "candidate-stack-manifest.v1.schema.json",
        "candidate-evaluation-receipt.v1.schema.json",
        "candidate-evaluation-summary.v1.schema.json",
    ):
        schema = pe.validate_schema_document(root / name)
        assert schema["$id"].endswith(".v1")


def test_manifest_redacts_secret_and_hashes_resolved_tool_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kwargs: [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
    )
    value = _manifest()
    value["runtime"]["url"] = "https://user:password@example.test/v1?token=secret"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    loaded = pe.load_manifest(path, capture_tools=True)
    assert loaded["runtime"]["url"] == "https://[REDACTED]@example.test/v1?token=[REDACTED]"
    assert loaded["hermes"]["resolved_tool_schema"]["tools"][0]["name"] == "read_file"
    assert len(loaded["hermes"]["resolved_tool_schema_sha256"]) == 64
    assert len(loaded["manifest_id"]) == 64


def test_manifest_rejects_file_toolset_and_dirty_revision(tmp_path: Path):
    value = _manifest()
    value["hermes"]["toolsets"] = ["file"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(pe.EvaluationError, match="hermes-cli"):
        pe.load_manifest(path, capture_tools=False)


def test_receipt_writer_hashes_payload_and_tampering_is_detectable(tmp_path: Path):
    receipt = pe._write_receipt(tmp_path / "receipt.json", {"schema_version": "candidate-evaluation-receipt.v1", "value": 1})
    assert receipt["receipt_sha256"] == pe.scoring.canonical_hash({"schema_version": receipt["schema_version"], "value": 1})
    altered = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    altered["value"] = 2
    assert altered["receipt_sha256"] != pe.scoring.canonical_hash({"schema_version": altered["schema_version"], "value": altered["value"]})


def test_evaluate_dry_run_requires_readiness_but_never_runs_a_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = _config(tmp_path)
    (tmp_path / "rollback.json").write_text('{"route":"incumbent"}\n', encoding="utf-8")
    raw_candidate = _manifest()
    raw_incumbent = _manifest()
    raw_candidate.pop("manifest_id", None)
    raw_incumbent.pop("manifest_id", None)
    candidate_path = tmp_path / "candidate.json"
    incumbent_path = tmp_path / "incumbent.json"
    candidate_path.write_text(json.dumps(raw_candidate), encoding="utf-8")
    incumbent_path.write_text(json.dumps(raw_incumbent), encoding="utf-8")
    monkeypatch.setattr(
        pe,
        "capture_tool_schema_fingerprint",
        lambda _toolsets, _disabled: {"tools": [{"name": "read_file", "schema_sha256": "a", "available": True}], "schema_sha256": "b", "resolved_tool_schema_sha256": "b"},
    )
    args = SimpleNamespace(
        evaluation_config=str(config_path),
        candidate_manifest=str(candidate_path),
        incumbent_manifest=str(incumbent_path),
        lane="cli-full-v1",
        suite="full-hermes-cli-v1",
        out=str(tmp_path / "run"),
        repetitions=3,
        seed=None,
        timeout=1,
        execute=False,
        archive_index=None,
        hermes_home=None,
        fixture_dir=None,
        hermes_executable="/does/not/run",
    )
    before = config_path.read_bytes()
    assert pe.run_evaluation(args) == 0
    assert config_path.read_bytes() == before
    assert (tmp_path / "run" / "schedule.jsonl").is_file()
    assert not (tmp_path / "run" / "receipts.jsonl").exists()


def test_offline_score_marks_checksum_tampering_as_gate_failure(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "run.json").write_text(json.dumps({"seed": 20260715, "repetitions": 3}), encoding="utf-8")
    (root / "schedule.json").write_text("[]", encoding="utf-8")
    (root / "receipts.jsonl").write_text("", encoding="utf-8")
    pe._write_checksums(root)
    (root / "run.json").write_text(json.dumps({"seed": 1, "repetitions": 3}), encoding="utf-8")
    code, summary = pe.score_run(root)
    assert code == 1
    assert summary["status"] == "GATE-FAILED"
    assert any(item.startswith("tampered:run.json") for item in summary["hard_gate_failures"])
