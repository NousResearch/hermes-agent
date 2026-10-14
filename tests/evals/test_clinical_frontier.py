"""Offline contracts for synthetic clinical evaluation; no model requests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from evals.clinical_frontier import run_eval, score_eval

ROOT = run_eval.ROOT


def test_dry_run_preserves_eight_synthetic_cases_and_disables_providers(capsys):
    run_eval.main(["--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["cases"] == 8
    assert result["enabled_providers"] == []
    assert result["model_calls"] == 0


def test_default_invocation_never_starts_processes(monkeypatch):
    monkeypatch.setattr(run_eval.subprocess, "run", lambda *a, **k: pytest.fail("unexpected process"))
    with pytest.raises(SystemExit):
        run_eval.main([])
    with pytest.raises(SystemExit, match="disabled"):
        run_eval.main(["--execute"])


def test_model_commands_match_current_cli_parser():
    from hermes_cli._parser import build_top_level_parser
    config = json.loads((ROOT / "models.json").read_text())
    for provider in config["providers"]:
        parser, _, _ = build_top_level_parser()
        args = parser.parse_args(provider["command"][3:])
        assert args.query_file == "{prompt_file}"
        assert args.oneshot_exit is True
        assert args.provider in {"openai-codex", "gemini"}
        if provider["id"] == "alfred_openai":
            assert not args.ignore_rules
            assert args.skills


def test_forbidden_provider_rejected_even_in_dry_run(tmp_path):
    config = tmp_path / "models.json"
    config.write_text(json.dumps({"providers": [{"id": "claude_bad", "enabled": True,
        "model": "fixture", "command": ["anthropic", "{prompt_file}"]}]}))
    with pytest.raises(ValueError, match="forbidden"):
        run_eval.main(["--models", str(config), "--dry-run"])


def test_fixture_preserves_question_blinds_identity_and_isolates_home(tmp_path, capsys):
    dataset, models = tmp_path / "cases.jsonl", tmp_path / "models.json"
    question = "¿Texto exacto $(no shell) 38,1 °C?"
    dataset.write_text(json.dumps({"id": "c", "question": question}) + "\n", encoding="utf-8")
    code = ("import os,pathlib,sys; "
            "assert os.environ['HERMES_HOME'] == os.environ['HOME']; "
            "assert not os.environ.get('OPENAI_API_KEY'); "
            "print(pathlib.Path(sys.argv[1]).read_text())")
    models.write_text(json.dumps({"providers": [{"id": "fixture", "enabled": True, "model": "fixture",
        "command": [sys.executable, "-c", code, "{prompt_file}"]}]}))
    run_eval.main(["--dataset", str(dataset), "--models", str(models), "--out", str(tmp_path / "runs"), "--execute"])
    run = next((tmp_path / "runs").iterdir())
    raw = run_eval.load_jsonl(run / "responses.jsonl")[0]
    assert raw["response"] == question
    assert raw["exit_code"] == 0
    assert raw["reported_cost_usd"] is None
    blind = run_eval.load_jsonl(run / "blind_responses.jsonl")[0]
    assert blind["question"] == question
    assert "provider_id" not in blind and "model" not in blind
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["comparison_complete"] is False
    assert len(manifest["dataset_sha256"]) == 64
    assert not list(run.glob("*.txt"))
    capsys.readouterr()


def rating(response="R1", rater="human1"):
    return {"response_id": response, "rater_id": rater,
            "clinical": {dim: 4 for dim in score_eval.DIMS},
            "gates": {gate: False for gate in score_eval.GATES}}


KEY = {"R1": {"case_id": "synthetic", "provider_id": "fixture", "model": "fixture"}}


@pytest.mark.parametrize("mutation", ["null_gate", "missing_rater", "bool_score", "unknown_response", "nan_score"])
def test_scorer_rejects_incomplete_or_invalid_clinical_ratings(mutation):
    row = rating()
    mutations = {
        "null_gate": (row["gates"], "critical_safety_failure", None),
        "missing_rater": (row, "rater_id", ""),
        "bool_score": (row["clinical"], "safety", True),
        "unknown_response": (row, "response_id", "unknown"),
        "nan_score": (row["clinical"], "safety", float("nan")),
    }
    target, key, value = mutations[mutation]
    target[key] = value
    with pytest.raises(ValueError):
        score_eval.aggregate([row], KEY)


def test_scorer_requires_distinct_ratings_and_never_certifies_comparison():
    with pytest.raises(ValueError, match="duplicate"):
        score_eval.aggregate([rating(), rating()], KEY)
    report = score_eval.aggregate([rating(), rating(rater="human2")], KEY)
    assert report["minimum_two_raters"] is True
    assert report["comparison_complete"] is False
    assert report["status"] == "descriptive_only"
    assert report["providers"]["fixture"]["mean_weighted_0_to_4"] == 4


def test_fabricated_source_forces_evidence_and_traceability_to_zero():
    row = rating()
    row["gates"]["fabricated_source"] = True
    report = score_eval.aggregate([row], KEY)
    provider = report["providers"]["fixture"]
    assert provider["mean_weighted_0_to_4"] == 3
    assert provider["fabricated_source_count"] == 1


def test_partial_coverage_is_reported_without_winner():
    key = dict(KEY, R2={"case_id": "other", "provider_id": "fixture", "model": "fixture"})
    report = score_eval.aggregate([rating()], key)
    assert report["missing_response_ids"] == ["R2"]
    assert report["minimum_two_raters"] is False
    assert report["comparison_complete"] is False


@pytest.mark.parametrize("failure", ["launch", "timeout"])
def test_failed_fixture_execution_is_recorded_without_success(tmp_path, monkeypatch, capsys, failure):
    import subprocess
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"providers": [{"id": "fixture", "enabled": True, "model": "fixture",
        "command": ["fixture-not-executed", "{prompt_file}"]}]}))
    error = (FileNotFoundError("fixture") if failure == "launch" else
             subprocess.TimeoutExpired("fixture", 1, output=b"partial", stderr=b"timeout"))

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(run_eval.subprocess, "run", fail)
    run_eval.main(["--models", str(models), "--case", "urgent_chest_pain", "--out", str(tmp_path / "runs"), "--execute"])
    run = next((tmp_path / "runs").iterdir())
    row = run_eval.load_jsonl(run / "responses.jsonl")[0]
    assert row["exit_code"] is None
    assert row["timeout"] is (failure == "timeout")
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["completed_ok"] == 0
    assert manifest["comparison_complete"] is False
    assert not list(run.glob("*.txt"))
    capsys.readouterr()
