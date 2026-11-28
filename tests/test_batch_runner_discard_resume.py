"""Discarded samples must be visible to --resume (#93527).

The no-reasoning discard path used to mark prompts completed only in the
checkpoint index while writing no batch_*.jsonl row. Resume filters
exclusively by scanning those files for prompt content, so every
discarded sample was re-run at full cost on each restart. The fix writes
a tombstone row that the content scan treats as completed and the
trajectories.jsonl merge excludes.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import batch_runner
from agent.agent_runtime_helpers import convert_to_trajectory_format
from batch_runner import (
    BatchRunner,
    _entry_prompt_text,
)


# ─────────────────────────────────────────────────────────────────────
# Worker: discard leaves a tombstone row
# ─────────────────────────────────────────────────────────────────────


def _discarded_result():
    return {
        "success": True,
        "trajectory": [{"role": "assistant", "content": "x"}],
        "reasoning_stats": {"has_any_reasoning": False},
        "tool_stats": {},
        "metadata": {},
        "completed": True,
        "api_calls": 1,
        "toolsets_used": [],
    }




# ─────────────────────────────────────────────────────────────────────
# Worker: a failed agent run stays retryable
# ─────────────────────────────────────────────────────────────────────

_REASONING_TOOL_TURN = [
    {
        "role": "assistant",
        "content": "<REASONING_SCRATCHPAD>read it first</REASONING_SCRATCHPAD>",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "c1", "content": '{"content": "x"}'},
]


@pytest.mark.parametrize("turns", [[], _REASONING_TOOL_TURN], ids=["before-any-answer", "after-a-tool-turn"])
def test_failed_agent_run_is_retried_on_resume(tmp_path, monkeypatch, turns):
    """run_conversation reports provider failures (credits exhausted, rate limit,
    outage) as ``failed`` instead of raising. Such a run is neither a no-reasoning
    discard (tombstoned, never retried) nor a finished sample (truncated trajectory)."""

    class FailedRunAgent:
        _convert_to_trajectory_format = convert_to_trajectory_format

        def __init__(self, **kwargs):
            pass

        def _format_tools_for_system_message(self):
            return "[]"

        def run_conversation(self, prompt, task_id=None):
            return {
                "messages": [{"role": "user", "content": prompt}, *turns],
                "completed": False, "failed": True, "api_calls": 1,
                "error": "HTTP 402: credits exhausted",
            }

        def close(self):
            pass

    monkeypatch.setattr(batch_runner, "AIAgent", FailedRunAgent)
    monkeypatch.setattr(batch_runner, "sample_toolsets_from_distribution", lambda name: [])
    config = {"distribution": "default", "model": "m", "max_iterations": 2, "verbose": False}

    result = batch_runner._process_batch_worker((0, [(0, {"prompt": "q"})], str(tmp_path), set(), config))

    assert result["completed_prompts"] == []
    assert _scan_runner(tmp_path)._scan_completed_prompts_by_content() == set()


# ─────────────────────────────────────────────────────────────────────
# Content scan: tombstones count as completed
# ─────────────────────────────────────────────────────────────────────


def _scan_runner(tmp_path):
    runner = BatchRunner.__new__(BatchRunner)
    runner.output_dir = tmp_path
    return runner


def test_content_scan_treats_tombstone_as_completed(tmp_path):
    (tmp_path / "batch_1.jsonl").write_text(
        json.dumps({"prompt_index": 0, "discarded": "no_reasoning", "prompt": "tombstoned q"})
        + "\n"
        + json.dumps({
            "conversations": [{"from": "human", "value": "normal q"}],
            "completed": True,
        })
        + "\n",
        encoding="utf-8",
    )

    completed = _scan_runner(tmp_path)._scan_completed_prompts_by_content()

    assert completed == {"tombstoned q", "normal q"}


def test_content_scan_still_skips_failed_rows(tmp_path):
    (tmp_path / "batch_1.jsonl").write_text(
        json.dumps({"failed": True, "conversations": [{"from": "human", "value": "retry me"}]})
        + "\n",
        encoding="utf-8",
    )

    assert _scan_runner(tmp_path)._scan_completed_prompts_by_content() == set()


# ─────────────────────────────────────────────────────────────────────
# Merge: tombstones never enter trajectories.jsonl
# ─────────────────────────────────────────────────────────────────────


def _make_real_runner(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"prompt": "hi"}) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return BatchRunner(
        dataset_file=str(dataset),
        batch_size=1,
        run_name="discard-resume-test",
        num_workers=1,
    )


def _fake_pool(batch_results):
    pool = MagicMock()
    pool.imap_unordered.return_value = iter(batch_results)
    pool_cm = MagicMock()
    pool_cm.__enter__ = MagicMock(return_value=pool)
    pool_cm.__exit__ = MagicMock(return_value=False)
    return pool_cm


def test_merge_excludes_tombstones_from_trajectories(tmp_path, monkeypatch):
    # Pre-existing output from an earlier session: one real trajectory +
    # one tombstone. run(resume=False) re-processes its batches through the
    # patched Pool, then merges ALL batch files on disk.
    out_dir = tmp_path / "data" / "discard-resume-test"
    out_dir.mkdir(parents=True)
    (out_dir / "batch_1.jsonl").write_text(
        json.dumps({
            "prompt_index": 0,
            "conversations": [{"from": "human", "value": "real q"}],
            "completed": True,
            "tool_stats": {},
        })
        + "\n"
        + json.dumps({"prompt_index": 1, "discarded": "no_reasoning", "prompt": "dropped q"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["batch_runner.py"])

    runner = _make_real_runner(tmp_path, monkeypatch)
    # Point the runner at the pre-populated directory instead of cwd/data.
    runner.output_dir = out_dir

    batch_result = {
        "batch_num": 1,
        "processed": 0,
        "skipped": 0,
        "tool_stats": {},
        "reasoning_stats": {},
        "discarded_no_reasoning": 0,
        "completed_prompts": [],
    }
    with patch.object(batch_runner, "Pool", return_value=_fake_pool([batch_result])):
        runner.run()

    merged = (out_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in merged if line.strip()]
    assert len(parsed) == 1
    assert "discarded" not in parsed[0]
    stats = json.loads((out_dir / "statistics.json").read_text(encoding="utf-8"))
    assert "discarded_no_reasoning" in stats


# ─────────────────────────────────────────────────────────────────────
# Prompt-text extraction shapes
# ─────────────────────────────────────────────────────────────────────


def test_entry_prompt_text_shapes():
    assert _entry_prompt_text({"prompt": "flat"}) == "flat"
    assert _entry_prompt_text({"conversations": [{"from": "human", "value": "sharegpt"}]}) == "sharegpt"
    assert _entry_prompt_text({"conversations": [{"role": "user", "content": "chat"}]}) == "chat"
    assert _entry_prompt_text({"messages": [{"role": "user", "content": "msgs"}]}) == "msgs"
    assert _entry_prompt_text({"prompt": "  padded  ", "discarded": "x"}) == "padded"
    assert _entry_prompt_text({}) == ""
    assert _entry_prompt_text("not-a-dict") == ""


def test_scans_skip_non_dict_lines(tmp_path):
    """A parseable-but-non-object line used to raise AttributeError/TypeError
    mid-scan (resume scan and dataset load); it is skipped like invalid JSON."""
    (tmp_path / "batch_1.jsonl").write_text(
        '"scalar row"\n'
        + json.dumps({"prompt": "ok q", "completed": True}) + "\n",
        encoding="utf-8",
    )
    runner = _scan_runner(tmp_path)
    assert runner._scan_completed_prompts_by_content() == {"ok q"}
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('"not an entry"\n' + json.dumps({"prompt": "real"}) + "\n", encoding="utf-8")
    runner.dataset_file = dataset
    assert runner._load_dataset() == [{"prompt": "real"}]


def test_combine_batch_files_skips_non_dict_lines(tmp_path):
    (tmp_path / "batch_1.jsonl").write_text(
        '42\n'
        + json.dumps({"conversations": [{"from": "human", "value": "kept"}]}) + "\n",
        encoding="utf-8",
    )
    runner = _scan_runner(tmp_path)
    kept, _found = runner._combine_batch_files()
    assert kept == 1
    out = (tmp_path / "trajectories.jsonl").read_text(encoding="utf-8")
    assert "kept" in out and "42" not in out
