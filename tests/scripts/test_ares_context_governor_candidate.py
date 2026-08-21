"""Failure-evidence contract tests for the pre-activation candidate harness."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ares_context_governor_candidate.py"
)
SPEC = importlib.util.spec_from_file_location("ares_cg_candidate", SCRIPT)
assert SPEC and SPEC.loader
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


def valid_sample(**overrides):
    sample = {
        "prompt_visible_provenance_bytes": 64,
        "prompt_visible_provenance_tokens": 16,
        "authoritative_provenance_bytes": 1024,
        "receipt_bytes": 2048,
        "cumulative_receipt_store_bytes": 4096,
        "compaction_latency_ms": 10.0,
        "restart_load_latency_ms": 10.0,
        "exact_expansion_latency_ms": 10.0,
        "input_tokens": 1024,
        "output_tokens": 512,
        "net_token_savings": 512,
        "budget_decision": "admit",
        "exact_expansion_hash": "a" * 64,
        "exact_expansion_result": "PASS",
        "hmac_verification_result": "PASS",
        "key_id": "b" * 64,
    }
    sample.update(overrides)
    return sample


def run(monkeypatch, sample):
    checkpoints = []
    monkeypatch.setattr(candidate, "_certification_sample", lambda *_: sample)
    result = candidate.certify(
        Path("binary"),
        Path("stage"),
        "core-id",
        generations=(16,),
        persist=checkpoints.append,
    )
    return result, checkpoints


def test_failing_metric_is_persisted_before_harness_returns(monkeypatch, tmp_path):
    result, checkpoints = run(monkeypatch, valid_sample(receipt_bytes=524289))

    assert not result["pass"]
    assert checkpoints
    artifact = candidate.write_non_authorizing_certification(
        tmp_path, result, failed=True
    )
    persisted = candidate.json.loads(artifact.read_text())
    assert persisted["candidate_id"] == "core-id"
    assert persisted["terminal_outcome"] == "HARD_FAILURE"
    assert "receipt_bytes" in persisted["generations"][0]["failing_metric_ids"]


def test_multiple_hard_failures_and_raw_samples_survive(monkeypatch):
    sample = valid_sample(
        receipt_bytes=524289, net_token_savings=0, budget_decision="reject"
    )
    result, checkpoints = run(monkeypatch, sample)

    record = result["generations"][0]
    assert {"receipt_bytes", "net_token_savings", "budget_decision"} <= set(
        record["failing_metric_ids"]
    )
    assert len(record["raw_measurement_samples"]) == 13
    assert record["raw_measurement_samples"][0]["metrics"]["receipt_bytes"] == 524289
    assert checkpoints[-1]["candidate_id"] == "core-id"


def test_failure_issues_no_authorizing_identity(monkeypatch):
    result, _ = run(monkeypatch, valid_sample(exact_expansion_result="FAIL"))

    assert not result["pass"]
    assert result["non_authorizing"] is True
    assert result["candidate_id"] == "core-id"
    assert "certification_set_id" not in result
    assert "sealed_candidate_id" not in result


def test_soft_warning_remains_distinct_from_hard_failure(monkeypatch):
    result, _ = run(monkeypatch, valid_sample(receipt_bytes=400000))

    record = result["generations"][0]
    assert result["pass"] is True
    assert record["hard_pass"] is True
    assert any(
        warning["metric_id"] == "receipt_bytes_soft_warning" and warning["triggered"]
        for warning in record["soft_warning_evaluations"]
    )
    assert "receipt_bytes" not in record["failing_metric_ids"]


def test_successful_certification_remains_explicitly_non_authorizing(monkeypatch):
    result, checkpoints = run(monkeypatch, valid_sample())

    assert result["pass"] is True
    assert result["terminal_outcome"] == "PASS"
    assert result["authorization_state"] == "NON_AUTHORIZING"
    assert result["non_authorizing"] is True
    assert len(result["generations"][0]["raw_measurement_samples"]) == 13
    assert checkpoints[-1]["terminal_outcome"] == "IN_PROGRESS"


def test_successful_staged_recertification_persists_and_reports_one_non_authorizing_state(
    monkeypatch, tmp_path
):
    """Regression for Sol's finding: old code persisted `false` on success."""
    result, _ = run(monkeypatch, valid_sample())
    artifact = candidate.write_non_authorizing_certification(tmp_path, result)
    persisted = candidate.json.loads(artifact.read_text())
    stdout = candidate.staged_certification_status(persisted)

    assert persisted["pass"] is True
    assert persisted["authorization_state"] == "NON_AUTHORIZING"
    assert persisted["non_authorizing"] is True
    assert stdout == {
        "terminal_outcome": "PASS",
        "authorization_state": "NON_AUTHORIZING",
        "non_authorizing": True,
        "candidate_core_id": "core-id",
    }
    assert "activation_authorization_id" not in persisted
    assert "certification_set_id" not in persisted
    assert "sealed_candidate_id" not in persisted


@pytest.mark.parametrize(
    "terminal_outcome",
    ["IN_PROGRESS", "HARD_FAILURE", "PASS"],
)
def test_all_staged_certification_outcomes_are_non_authorizing(terminal_outcome):
    result = candidate._certification_result("core-id", [], terminal_outcome)

    assert result["authorization_state"] == "NON_AUTHORIZING"
    assert result["non_authorizing"] is True


def test_contradictory_staged_status_fails_closed():
    with pytest.raises(RuntimeError, match="AuthorizationStateContradiction"):
        candidate.staged_certification_status({
            "candidate_id": "core-id",
            "terminal_outcome": "PASS",
            "authorization_state": "NON_AUTHORIZING",
            "non_authorizing": False,
        })


def test_generated_python_cache_never_enters_candidate_file_map(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("source\n")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "source.cpython-314.pyc").write_bytes(b"generated")

    assert [entry["path"] for entry in candidate.file_map(tmp_path)] == ["source.py"]


def test_runtime_activation_owner_is_declared_candidate_scope():
    assert (
        candidate.classification("ares", "ares_runtime/activation.py")
        == candidate.REQUIRED
    )
    assert (
        candidate.classification("ares", "tests/ares_runtime/test_activation.py")
        == candidate.REQUIRED
    )


def test_candidate_builder_rejects_unpinned_or_unformatted_python_payload(
    monkeypatch, tmp_path
):
    payload = tmp_path / "payload"
    ares = payload / "ares"
    ares.mkdir(parents=True)
    (ares / "candidate.py").write_text("x=1\n", encoding="utf-8")
    (ares / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")

    monkeypatch.setattr(
        candidate.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, "ruff 0.15.0\n", ""
        ),
    )
    with pytest.raises(RuntimeError, match="Ruff 0.15.10"):
        candidate.require_candidate_python_format(payload, ["candidate.py"])


def test_candidate_builder_formats_only_ledger_declared_python_paths(
    monkeypatch, tmp_path
):
    payload = tmp_path / "payload"
    ares = payload / "ares"
    ares.mkdir(parents=True)
    declared = ares / "declared.py"
    declared.write_text("x = 1\n", encoding="utf-8")
    (ares / "legacy.py").write_text("x=1\n", encoding="utf-8")
    (ares / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    commands: list[list[str]] = []

    def run(args, **kwargs):
        commands.append(args)
        if args[1] == "--version":
            return subprocess.CompletedProcess(args, 0, "ruff 0.15.10\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(candidate.shutil, "which", lambda _name: "/tmp/ruff")
    monkeypatch.setattr(candidate.subprocess, "run", run)

    candidate.require_candidate_python_format(payload, ["declared.py"])

    checked = " ".join(" ".join(command) for command in commands[1:])
    assert str(declared) in checked
    assert str(ares / "legacy.py") not in checked


def test_candidate_builder_rejects_missing_ledger_python_path(tmp_path):
    payload = tmp_path / "payload"
    ares = payload / "ares"
    ares.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="missing Ares Python path"):
        candidate.require_candidate_python_format(payload, ["missing.py"])


def test_candidate_builder_formats_python_beneath_ledger_directory(
    monkeypatch, tmp_path
):
    payload = tmp_path / "payload"
    ares = payload / "ares"
    runtime = ares / "ares_runtime"
    runtime.mkdir(parents=True)
    declared = runtime / "activation.py"
    declared.write_text("x = 1\n", encoding="utf-8")
    (ares / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    commands: list[list[str]] = []

    def run(args, **kwargs):
        commands.append(args)
        if args[1] == "--version":
            return subprocess.CompletedProcess(args, 0, "ruff 0.15.10\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(candidate.shutil, "which", lambda _name: "/tmp/ruff")
    monkeypatch.setattr(candidate.subprocess, "run", run)

    candidate.require_candidate_python_format(payload, ["ares_runtime/"])

    assert str(declared) in " ".join(" ".join(command) for command in commands)
