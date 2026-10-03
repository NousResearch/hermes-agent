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


class _InProcessPool:
    """Stand-in for multiprocessing.Pool: runs batch tasks in this process."""

    def __init__(self, processes=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, fn, tasks):
        return map(fn, tasks)


class TestResumeRowIdentity:
    """--resume must run every dataset row that has no completed record, whatever its index."""

    @pytest.fixture
    def run_dataset(self, tmp_path, monkeypatch):
        import batch_runner

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(batch_runner, "Pool", _InProcessPool)
        calls, failing = [], set()

        def fake_single(idx, entry, batch_num, config):
            calls.append((idx, entry.get("image", entry["prompt"])))
            if entry.get("image") in failing:
                return batch_runner._failure_result(idx, batch_num, "transient 503")
            return {"success": True, "prompt_index": idx, "completed": True, "partial": False,
                    "trajectory": [{"from": "human", "value": entry["prompt"]}, {"from": "gpt", "value": "ok"}],
                    "reasoning_stats": {"has_any_reasoning": True}, "tool_stats": {}, "metadata": {},
                    "api_calls": 1, "toolsets_used": []}

        monkeypatch.setattr(batch_runner, "_process_single_prompt", fake_single)
        dataset = tmp_path / "ds.jsonl"

        def run(entries, *, resume, fail=()):
            dataset.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
            calls.clear()
            failing.clear()
            failing.update(fail)
            BatchRunner(dataset_file=str(dataset), batch_size=2, run_name="rows", num_workers=1).run(resume=resume)
            return list(calls)

        return run

    def test_resume_runs_prompts_added_or_moved_onto_completed_indices(self, run_dataset):
        run_dataset([{"prompt": "A"}, {"prompt": "B"}], resume=False)  # checkpoint: indices 0, 1

        # Same run, dataset edited: new prompts now sit at the checkpointed indices.
        calls = run_dataset([{"prompt": p} for p in ("C", "D", "A", "B")], resume=True)

        assert sorted(calls) == [(0, "C"), (1, "D")]

    def test_resume_retries_the_failed_row_among_rows_sharing_a_prompt(self, run_dataset):
        rows = [{"prompt": "Fix the failing tests in /repo.", "image": img} for img in ("img-a", "img-b", "img-c")]
        run_dataset(rows, resume=False, fail={"img-b"})

        assert run_dataset(rows, resume=True) == [(1, "img-b")]

    def test_resume_tells_rows_sharing_a_prompt_apart_after_a_reorder(self, run_dataset):
        """The completed img-a row now sits where the failed img-b row was recorded: resume must
        still run img-b (and not img-a again), so identity includes the row's own data."""
        img_a, img_b = ({"prompt": "Fix the failing tests in /repo.", "image": img} for img in ("img-a", "img-b"))
        run_dataset([img_a, img_b], resume=False, fail={"img-b"})

        assert run_dataset([img_b, img_a], resume=True) == [(0, "img-b")]
