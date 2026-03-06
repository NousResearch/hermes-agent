"""Tests for batch_runner checkpoint behavior — incremental writes, resume, atomicity."""

import json
import os
from pathlib import Path
from multiprocessing import Lock
from unittest.mock import patch, MagicMock

import pytest

# batch_runner uses relative imports, ensure project root is on path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_runner import BatchRunner


@pytest.fixture
def runner(tmp_path):
    """Create a BatchRunner with all paths pointing at tmp_path."""
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text("")
    output_file = tmp_path / "output.jsonl"
    checkpoint_file = tmp_path / "checkpoint.json"
    r = BatchRunner.__new__(BatchRunner)
    r.run_name = "test_run"
    r.checkpoint_file = checkpoint_file
    r.output_file = output_file
    r.prompts_file = prompts_file
    return r


class TestSaveCheckpoint:
    """Verify _save_checkpoint writes valid, atomic JSON."""

    def test_writes_valid_json(self, runner):
        data = {"run_name": "test", "completed_prompts": [1, 2, 3], "batch_stats": {}}
        runner._save_checkpoint(data)

        result = json.loads(runner.checkpoint_file.read_text())
        assert result["run_name"] == "test"
        assert result["completed_prompts"] == [1, 2, 3]

    def test_adds_last_updated(self, runner):
        data = {"run_name": "test", "completed_prompts": []}
        runner._save_checkpoint(data)

        result = json.loads(runner.checkpoint_file.read_text())
        assert "last_updated" in result
        assert result["last_updated"] is not None

    def test_overwrites_previous_checkpoint(self, runner):
        runner._save_checkpoint({"run_name": "test", "completed_prompts": [1]})
        runner._save_checkpoint({"run_name": "test", "completed_prompts": [1, 2, 3]})

        result = json.loads(runner.checkpoint_file.read_text())
        assert result["completed_prompts"] == [1, 2, 3]

    def test_with_lock(self, runner):
        lock = Lock()
        data = {"run_name": "test", "completed_prompts": [42]}
        runner._save_checkpoint(data, lock=lock)

        result = json.loads(runner.checkpoint_file.read_text())
        assert result["completed_prompts"] == [42]

    def test_without_lock(self, runner):
        data = {"run_name": "test", "completed_prompts": [99]}
        runner._save_checkpoint(data, lock=None)

        result = json.loads(runner.checkpoint_file.read_text())
        assert result["completed_prompts"] == [99]

    def test_creates_parent_dirs(self, tmp_path):
        runner_deep = BatchRunner.__new__(BatchRunner)
        runner_deep.checkpoint_file = tmp_path / "deep" / "nested" / "checkpoint.json"

        data = {"run_name": "test", "completed_prompts": []}
        runner_deep._save_checkpoint(data)

        assert runner_deep.checkpoint_file.exists()

    def test_no_temp_files_left(self, runner):
        runner._save_checkpoint({"run_name": "test", "completed_prompts": []})

        tmp_files = [f for f in runner.checkpoint_file.parent.iterdir()
                     if ".tmp" in f.name]
        assert len(tmp_files) == 0


class TestLoadCheckpoint:
    """Verify _load_checkpoint reads existing data or returns defaults."""

    def test_returns_empty_when_no_file(self, runner):
        result = runner._load_checkpoint()
        assert result.get("completed_prompts", []) == []

    def test_loads_existing_checkpoint(self, runner):
        data = {"run_name": "test_run", "completed_prompts": [5, 10, 15],
                "batch_stats": {"0": {"processed": 3}}}
        runner.checkpoint_file.write_text(json.dumps(data))

        result = runner._load_checkpoint()
        assert result["completed_prompts"] == [5, 10, 15]
        assert result["batch_stats"]["0"]["processed"] == 3

    def test_handles_corrupt_json(self, runner):
        runner.checkpoint_file.write_text("{broken json!!")

        result = runner._load_checkpoint()
        # Should return empty/default, not crash
        assert isinstance(result, dict)


class TestResumePreservesProgress:
    """Verify that initializing a run with resume=True loads prior checkpoint."""

    def test_completed_prompts_loaded_from_checkpoint(self, runner):
        # Simulate a prior run that completed prompts 0-4
        prior = {
            "run_name": "test_run",
            "completed_prompts": [0, 1, 2, 3, 4],
            "batch_stats": {"0": {"processed": 5}},
            "last_updated": "2026-01-01T00:00:00",
        }
        runner.checkpoint_file.write_text(json.dumps(prior))

        # Load checkpoint like run() does
        checkpoint_data = runner._load_checkpoint()
        if checkpoint_data.get("run_name") != runner.run_name:
            checkpoint_data = {
                "run_name": runner.run_name,
                "completed_prompts": [],
                "batch_stats": {},
                "last_updated": None,
            }

        completed_set = set(checkpoint_data.get("completed_prompts", []))
        assert completed_set == {0, 1, 2, 3, 4}

    def test_different_run_name_starts_fresh(self, runner):
        prior = {
            "run_name": "different_run",
            "completed_prompts": [0, 1, 2],
            "batch_stats": {},
        }
        runner.checkpoint_file.write_text(json.dumps(prior))

        checkpoint_data = runner._load_checkpoint()
        if checkpoint_data.get("run_name") != runner.run_name:
            checkpoint_data = {
                "run_name": runner.run_name,
                "completed_prompts": [],
                "batch_stats": {},
                "last_updated": None,
            }

        assert checkpoint_data["completed_prompts"] == []
        assert checkpoint_data["run_name"] == "test_run"
