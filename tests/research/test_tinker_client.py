"""Tests for research.tinker_client."""

import json

import pytest

from research.models import RunSpec
from research.state_store import ResearchStateStore
from research.tinker_client import build_runner_script, build_runtime_paths, normalize_run_spec


def test_normalize_run_spec_sft(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"prompt":"hi","completion":"hello"}\n', encoding="utf-8")
    spec = normalize_run_spec(
        {"dataset_path": str(dataset), "method": "sft"},
        project_id="proj_test",
        cwd=tmp_path,
    )
    assert spec.method == "sft"
    assert spec.project_id == "proj_test"
    assert spec.field_map["prompt"] == "prompt"
    assert spec.field_map["completion"] == "completion"


def test_normalize_run_spec_dpo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    spec = normalize_run_spec(
        {"dataset_path": str(dataset), "method": "dpo"},
        project_id="proj_test",
        cwd=tmp_path,
    )
    assert spec.method == "dpo"
    assert "chosen" in spec.field_map
    assert "rejected" in spec.field_map


def test_normalize_run_spec_ppo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    spec = normalize_run_spec(
        {"dataset_path": str(dataset), "method": "ppo"},
        project_id="proj_test",
        cwd=tmp_path,
    )
    assert spec.method == "ppo"
    assert "logprobs" in spec.field_map


def test_normalize_run_spec_missing_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="dataset_path is required"):
        normalize_run_spec({}, project_id="proj_test", cwd=tmp_path)


def test_normalize_run_spec_unsupported_method(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported training method"):
        normalize_run_spec(
            {"dataset_path": str(dataset), "method": "grpo"},
            project_id="proj_test",
            cwd=tmp_path,
        )


def test_normalize_run_spec_relative_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    spec = normalize_run_spec(
        {"dataset_path": "data.jsonl"},
        project_id="proj_test",
        cwd=tmp_path,
    )
    assert spec.dataset_path == str(dataset)


def test_normalize_run_spec_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    spec = normalize_run_spec(
        {"dataset_path": str(dataset)},
        project_id="proj_test",
        cwd=tmp_path,
    )
    # Defaults from config
    assert spec.base_model == "meta-llama/Llama-3.1-8B"
    assert spec.lora_rank == 32
    assert spec.learning_rate == 1e-4
    assert spec.batch_size == 32
    assert spec.num_epochs == 1


def test_build_runtime_paths(tmp_path):
    paths = build_runtime_paths(tmp_path, "run_abc")
    assert paths.runtime_dir == tmp_path / "runtime" / "run_abc"
    assert paths.config_path.name == "config.json"
    assert paths.runner_path.name == "runner.py"
    assert paths.status_path.name == "status.json"
    assert paths.runtime_dir.is_dir()


def test_build_runner_script(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"prompt":"hi","completion":"hello"}\n', encoding="utf-8")
    spec = normalize_run_spec(
        {"dataset_path": str(dataset), "method": "sft"},
        project_id="proj_test",
        cwd=tmp_path,
    )
    paths = build_runtime_paths(tmp_path, spec.run_id)
    script = build_runner_script(spec, paths)
    assert "#!/usr/bin/env python3" in script
    assert "import tinker" in script
    assert "cross_entropy" in script
    assert spec.run_id in script
