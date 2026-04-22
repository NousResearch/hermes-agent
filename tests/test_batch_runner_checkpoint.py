"""Tests for batch_runner checkpoint behavior — incremental writes, resume, atomicity."""

import json
import os
from pathlib import Path
from threading import Lock
from unittest.mock import patch, MagicMock

import pytest

# batch_runner uses relative imports, ensure project root is on path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_runner import BatchRunner, _process_batch_worker


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


class TestBatchWorkerResumeBehavior:
    def test_discarded_no_reasoning_prompts_are_marked_completed(self, tmp_path, monkeypatch):
        batch_file = tmp_path / "batch_1.jsonl"
        prompt_result = {
            "success": True,
            "trajectory": [{"role": "assistant", "content": "x"}],
            "reasoning_stats": {"has_any_reasoning": False},
            "tool_stats": {},
            "metadata": {},
            "completed": True,
            "api_calls": 1,
            "toolsets_used": [],
        }

        monkeypatch.setattr("batch_runner._process_single_prompt", lambda *args, **kwargs: prompt_result)

        result = _process_batch_worker((
            1,
            [(0, {"prompt": "hi"})],
            tmp_path,
            set(),
            {"verbose": False},
        ))

        assert result["discarded_no_reasoning"] == 1
        assert result["completed_prompts"] == [0]
        assert not batch_file.exists() or batch_file.read_text() == ""


class _FakePool:
    def __init__(self, results):
        self._results = results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap_unordered(self, func, tasks):
        return iter(self._results)


class _FakeProgress:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_task(self, *args, **kwargs):
        return "task"

    def update(self, *args, **kwargs):
        return None


class TestRunCheckpointAggregation:
    def test_final_checkpoint_keeps_completed_prompts_unique(self, tmp_path, monkeypatch):
        runner = BatchRunner.__new__(BatchRunner)
        runner.run_name = "test_run"
        runner.distribution = "default"
        runner.max_iterations = 1
        runner.base_url = None
        runner.api_key = None
        runner.model = "test-model"
        runner.num_workers = 1
        runner.verbose = False
        runner.ephemeral_system_prompt = None
        runner.log_prefix_chars = 0
        runner.providers_allowed = None
        runner.providers_ignored = None
        runner.providers_order = None
        runner.provider_sort = None
        runner.max_tokens = None
        runner.reasoning_config = None
        runner.prefill_messages = None
        runner.batch_size = 1
        runner.dataset = [{"prompt": "hello"}]
        runner.batches = [[(0, {"prompt": "hello"})]]
        runner.output_dir = tmp_path / "out"
        runner.output_dir.mkdir()
        runner.checkpoint_file = runner.output_dir / "checkpoint.json"
        runner.stats_file = runner.output_dir / "statistics.json"

        fake_results = [{
            "batch_num": 0,
            "completed_prompts": [0],
            "processed": 1,
            "skipped": 0,
            "discarded_no_reasoning": 0,
            "tool_stats": {},
            "reasoning_stats": {},
        }]

        monkeypatch.setattr("batch_runner.Pool", lambda *args, **kwargs: _FakePool(fake_results))
        monkeypatch.setattr("batch_runner.Console", lambda *args, **kwargs: None)
        monkeypatch.setattr("batch_runner.Progress", _FakeProgress)

        runner.run()

        checkpoint_data = json.loads(runner.checkpoint_file.read_text())
        assert checkpoint_data["completed_prompts"] == [0]
