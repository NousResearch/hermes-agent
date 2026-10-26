"""Tests for batch_runner checkpoint behavior — incremental writes, resume, atomicity."""

import json
from pathlib import Path

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

    def test_overwrites_previous_checkpoint(self, runner):
        runner._save_checkpoint({"run_name": "test", "completed_prompts": [1]})
        runner._save_checkpoint({"run_name": "test", "completed_prompts": [1, 2, 3]})

        result = json.loads(runner.checkpoint_file.read_text())
        assert result["completed_prompts"] == [1, 2, 3]

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

class TestBatchWorkerResumeBehavior:
    def test_discarded_no_reasoning_prompts_are_marked_completed(self, tmp_path, monkeypatch):
        batch_file = tmp_path / "batch_1.jsonl"
        prompt_result = {
            "success": True,
            "trajectory": [{"from": "human", "value": "hi"},
                            {"role": "assistant", "content": "x"}],
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

        # A tombstone row must be written so the content-based resume scan
        # can see this prompt was already processed and discarded.
        assert batch_file.exists()
        lines = [l for l in batch_file.read_text(encoding="utf-8").strip().split("\n") if l]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["discarded"] == "no_reasoning"
        # The lightweight tombstone carries the human prompt text so the
        # content scan can match it without a full trajectory payload.
        assert entry["prompt"] == "hi"

    def test_resume_after_all_discarded_batch_reruns_zero_prompts(self, tmp_path, monkeypatch):
        """Regression for the issue: a resumed run must not re-execute
        prompts that were already processed and discarded for having no
        reasoning — the content-based scan must see the discard tombstone.
        """
        prompt_result = {
            "success": True,
            "trajectory": [{"from": "human", "value": "hi"},
                            {"role": "assistant", "content": "x"}],
            "reasoning_stats": {"has_any_reasoning": False},
            "tool_stats": {},
            "metadata": {},
            "completed": True,
            "api_calls": 1,
            "toolsets_used": [],
        }
        monkeypatch.setattr("batch_runner._process_single_prompt", lambda *args, **kwargs: prompt_result)

        # First run: prompt 0 gets processed and discarded, writing its
        # tombstone row into batch_1.jsonl.
        _process_batch_worker((1, [(0, {"prompt": "hi"})], tmp_path, set(), {"verbose": False}))

        # Simulate a fresh resume: scan batch files by content, exactly as
        # BatchRunner.run() does.
        r = BatchRunner.__new__(BatchRunner)
        r.output_dir = tmp_path
        completed_prompt_texts = r._scan_completed_prompts_by_content()

        assert "hi" in completed_prompt_texts, (
            "discarded prompt is invisible to the content-based resume scan"
        )

        r.dataset = [{"prompt": "hi"}]
        filtered_entries, skipped_indices = r._filter_dataset_by_completed(completed_prompt_texts)

        assert filtered_entries == [], "discarded prompt was rescheduled on resume"
        assert skipped_indices == [0]
