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

import batch_runner
from batch_runner import (
    BatchRunner,
    _entry_prompt_text,
    _process_batch_worker,
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


def test_discard_writes_tombstone_row(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "batch_runner._process_single_prompt", lambda *a, **kw: _discarded_result()
    )

    _process_batch_worker((1, [(0, {"prompt": "hi"})], tmp_path, set(), {"verbose": False}))

    batch_file = tmp_path / "batch_1.jsonl"
    rows = [json.loads(line) for line in batch_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["discarded"] == "no_reasoning"
    assert rows[0]["prompt_index"] == 0
    assert rows[0]["prompt"] == "hi"


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
