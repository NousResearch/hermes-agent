"""Resume-integrity invariants for batch_runner's ``--resume`` (#95322).

Three defects in the resume machinery, one invariant test each:

1. The checkpoint's *index* set was re-applied to the batches resume rebuilt
   from content-filtered rows carrying current-file indices — a never-completed
   prompt whose new index collided with a stale one was silently skipped by the
   worker, reintroducing inside the worker exactly the index-drift bug the
   content scan exists to fix.
2. Resume renumbered its shards from 0, appending new rows into the previous
   run's ``batch_*.jsonl`` files and overwriting that run's per-shard
   ``batch_stats`` in ``checkpoint.json`` with counts for a different subset.
3. A non-string ``prompt`` value (the loader accepts any JSON type) crashed the
   resume content filter with ``AttributeError: 'int' object has no attribute
   'strip'`` after possibly hours of scanning.
"""

import json
from unittest.mock import MagicMock, patch

import batch_runner
from batch_runner import BatchRunner

RUN_NAME = "resume-integrity"


# ─────────────────────────────────────────────────────────────────────
# Harness: inline pool + recorded per-prompt results
# ─────────────────────────────────────────────────────────────────────

def _fake_pool_ctx():
    """Pool stand-in that runs every batch task inline instead of forking."""
    pool = MagicMock()
    pool.imap_unordered.side_effect = lambda fn, tasks: iter([fn(task) for task in tasks])
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=pool)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _record_processed(monkeypatch, processed):
    """Replace the agent call with a successful result that records prompt indices."""

    def _fake_process(prompt_index, prompt_data, batch_num, config):
        processed.append(prompt_index)
        return {
            "success": True,
            "prompt_index": prompt_index,
            "trajectory": [
                {"from": "human", "value": str(prompt_data.get("prompt"))},
                {"role": "assistant", "content": "ok"},
            ],
            "tool_stats": {},
            "reasoning_stats": {
                "total_assistant_turns": 1,
                "turns_with_reasoning": 1,
                "turns_without_reasoning": 0,
                "has_any_reasoning": True,
            },
            "completed": True,
            "partial": False,
            "api_calls": 1,
            "toolsets_used": [],
            "metadata": {},
        }

    monkeypatch.setattr("batch_runner._process_single_prompt", _fake_process)


def _setup(tmp_path, monkeypatch, *, rows, batch_files, checkpoint):
    """Build a dataset + pre-populated output dir, return a real BatchRunner."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["batch_runner.py"])
    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    out_dir = tmp_path / "data" / RUN_NAME
    out_dir.mkdir(parents=True)
    for name, content in batch_files.items():
        (out_dir / name).write_text(content, encoding="utf-8")
    (out_dir / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    runner = BatchRunner(
        dataset_file=str(dataset_file),
        batch_size=1,
        run_name=RUN_NAME,
        num_workers=1,
    )
    return runner, out_dir


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────
# 1. Stale checkpoint indices must not filter resume's re-indexed batches
# ─────────────────────────────────────────────────────────────────────

def test_resume_does_not_reapply_stale_checkpoint_indices(tmp_path, monkeypatch):
    # 'done' sat at index 1 when the interrupted run completed it. The dataset
    # was edited before resuming, so 'done' is now at index 0 and a
    # never-completed prompt has taken over index 1 — the exact index the
    # checkpoint still remembers as completed.
    runner, out_dir = _setup(
        tmp_path,
        monkeypatch,
        rows=[{"prompt": "done"}, {"prompt": "never-done"}],
        batch_files={
            "batch_0.jsonl": json.dumps(
                {"prompt_index": 1, "discarded": "no_reasoning", "prompt": "done"}
            )
            + "\n",
        },
        checkpoint={
            "run_name": RUN_NAME,
            "completed_prompts": [1],  # stale: 'done' has moved to index 0
            "batch_stats": {},
        },
    )
    processed = []
    _record_processed(monkeypatch, processed)

    with patch.object(batch_runner, "Pool", return_value=_fake_pool_ctx()):
        runner.run(resume=True)

    assert processed == [1], (
        "resume re-applied the stale checkpoint index set and silently skipped "
        "the never-completed prompt at index 1"
    )
    checkpoint = _load_json(out_dir / "checkpoint.json")
    # Fresh current-file index for 'done' (0) plus this run's 'never-done' (1):
    # the pre-edit index must not survive into the checkpoint either.
    assert checkpoint["completed_prompts"] == [0, 1]


# ─────────────────────────────────────────────────────────────────────
# 2. Resume shards continue past the previous run's numbering
# ─────────────────────────────────────────────────────────────────────

def test_resume_numbers_new_shards_after_existing_ones(tmp_path, monkeypatch):
    old_stats = {
        "0": {"processed": 7, "skipped": 3, "discarded_no_reasoning": 2},
        "1": {"processed": 9, "skipped": 1, "discarded_no_reasoning": 0},
    }
    batch_files = {
        "batch_0.jsonl": json.dumps(
            {
                "prompt_index": 0,
                "conversations": [
                    {"from": "human", "value": "done"},
                    {"role": "assistant", "content": "ok"},
                ],
                "completed": True,
                "tool_stats": {},
            }
        )
        + "\n",
        "batch_1.jsonl": json.dumps(
            {
                "prompt_index": 5,
                "conversations": [
                    {"from": "human", "value": "other-old-prompt"},
                    {"role": "assistant", "content": "ok"},
                ],
                "completed": True,
                "tool_stats": {},
            }
        )
        + "\n",
    }
    runner, out_dir = _setup(
        tmp_path,
        monkeypatch,
        rows=[{"prompt": "done"}, {"prompt": "never-done"}],
        batch_files=batch_files,
        checkpoint={
            "run_name": RUN_NAME,
            "completed_prompts": [0],
            "batch_stats": json.loads(json.dumps(old_stats)),
        },
    )
    before = {name: (out_dir / name).read_bytes() for name in batch_files}
    processed = []
    _record_processed(monkeypatch, processed)

    with patch.object(batch_runner, "Pool", return_value=_fake_pool_ctx()):
        runner.run(resume=True)

    assert processed == [1]
    # The previous run's shards must be byte-identical afterwards...
    for name, payload in before.items():
        assert (out_dir / name).read_bytes() == payload, (
            f"resumed run appended its rows into the previous run's {name}"
        )
    # ...and the resumed rows land in a fresh shard past the existing ones.
    new_rows = [
        json.loads(line)
        for line in (out_dir / "batch_2.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["prompt_index"] for row in new_rows] == [1]
    checkpoint = _load_json(out_dir / "checkpoint.json")
    # Old per-shard stats survive untouched; the resumed subset gets its own key.
    assert checkpoint["batch_stats"]["0"] == old_stats["0"]
    assert checkpoint["batch_stats"]["1"] == old_stats["1"]
    assert checkpoint["batch_stats"]["2"]["processed"] == 1


# ─────────────────────────────────────────────────────────────────────
# 3. Non-string prompt values must not abort the resume scan
# ─────────────────────────────────────────────────────────────────────

def test_resume_content_filter_tolerates_non_string_prompts(tmp_path):
    runner = BatchRunner.__new__(BatchRunner)
    runner.output_dir = tmp_path
    runner.batch_size = 1
    # _load_dataset accepts any JSON type for 'prompt' (key presence only).
    runner.dataset = [
        {"prompt": 123},
        {"prompt": {"nested": "object"}},
        {"prompt": None},
        {"prompt": "done already"},
        {"prompt": "fresh"},
    ]
    (tmp_path / "batch_0.jsonl").write_text(
        json.dumps(
            {"prompt_index": 3, "discarded": "no_reasoning", "prompt": "done already"}
        )
        + "\n",
        encoding="utf-8",
    )

    # Must not raise AttributeError ('int' object has no attribute 'strip').
    assert runner._apply_resume() is True

    kept_indices = [idx for batch in runner.batches for idx, _ in batch]
    assert kept_indices == [0, 1, 2, 4], "content filter mis-handled a non-string prompt"
