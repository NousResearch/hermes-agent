"""Regression tests for issue #95322 — batch_runner --resume correctness.

Three defects in the resume machinery:

1. Stale index-based checkpoint filter re-applied to re-indexed resume
   batches: ``run()`` builds ``completed_prompts_set`` from the checkpoint
   unconditionally and every worker drops ``(idx, data)`` pairs whose index
   is in it. On resume, though, ``self.batches`` was rebuilt from
   content-filtered entries carrying *current-file* indices; after a
   dataset edit the stale indices point at different rows, so a
   never-completed prompt whose new index happens to collide is silently
   skipped — exactly the index-drift bug the content scan exists to fix.

2. Resumed runs renumber shards from 0 and append into the previous run's
   files: workers derive ``batch_<n>.jsonl`` filenames from the batch
   number and open in append mode, and the parent overwrites per-shard
   ``batch_stats`` keyed by the same number.

3. Non-string ``prompt`` values crash ``_filter_dataset_by_completed``
   (``entry.get("prompt", "").strip()`` raises AttributeError on int/None),
   aborting the resume; the nearby ``_entry_prompt_text`` helper is
   defensive and the two paths disagreed about the contract.
"""

import json
import sys
from pathlib import Path

import pytest

# batch_runner uses relative imports, ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import batch_runner
from batch_runner import BatchRunner, _process_batch_worker


def _prompt_result(prompt_data):
    """Canned successful _process_single_prompt result."""
    return {
        "success": True,
        "trajectory": [
            {"from": "human", "value": _entry_text(prompt_data)},
            {"role": "assistant", "content": "reply"},
        ],
        "reasoning_stats": {"has_any_reasoning": True},
        "tool_stats": {},
        "metadata": {},
        "completed": True,
        "api_calls": 1,
        "toolsets_used": [],
    }


def _entry_text(entry):
    return batch_runner._entry_prompt_text(entry)


class _RecordingPool:
    """Stands in for multiprocessing.Pool: records the exact task tuples
    run() dispatches, while executing the real worker inline so file and
    checkpoint side effects happen."""

    instances = []

    def __init__(self, processes=None):
        self.processes = processes
        self.dispatched_tasks = []
        _RecordingPool.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, fn, tasks):
        # Snapshot mutable payloads (run() keeps mutating the completed-set
        # as batches finish); we want what workers saw AT DISPATCH TIME.
        self.dispatched_tasks = [
            tuple(set(v) if isinstance(v, set) else v for v in t)
            for t in tasks
        ]
        for t in tasks:
            yield fn(t)

    def terminate(self):
        pass

    def join(self):
        pass


@pytest.fixture
def fake_pool(monkeypatch):
    _RecordingPool.instances = []
    monkeypatch.setattr(batch_runner, "Pool", _RecordingPool)
    return _RecordingPool


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(
        "batch_runner._process_single_prompt",
        lambda idx, data, num, cfg: _prompt_result(data),
    )


def _make_runner(tmp_path, dataset_entries, batch_size=2):
    """Build a BatchRunner wired to tmp_path without running __init__."""
    r = BatchRunner.__new__(BatchRunner)
    r.run_name = "resume_regression"
    r.dataset = list(dataset_entries)
    r.batch_size = batch_size
    r.batches = r._create_batches()
    r.output_dir = tmp_path / "out"
    r.output_dir.mkdir(parents=True, exist_ok=True)
    r.checkpoint_file = r.output_dir / "checkpoint.json"
    r.stats_file = r.output_dir / "statistics.json"
    r.num_workers = 2
    # main's _worker_config forwards every _AGENT_PASSTHROUGH key, which now
    # includes "platform" (set by __init__, which this fixture bypasses).
    r.platform = "batch"
    r.distribution = "default"
    r.model = "test-model"
    r.max_iterations = 1
    r.base_url = None
    r.api_key = "test-key"
    r.verbose = False
    r.ephemeral_system_prompt = None
    r.log_prefix_chars = 10
    r.providers_allowed = None
    r.providers_ignored = None
    r.providers_order = None
    r.provider_sort = None
    r.openrouter_min_coding_score = None
    r.max_tokens = None
    r.reasoning_config = None
    r.prefill_messages = None
    return r


def _seed_first_run(output_dir, prompts):
    """Simulate an interrupted first run: workers wrote trajectory rows for
    ``prompts`` (a list of prompt strings) across shards, and the checkpoint
    recorded their ORIGINAL indices as completed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for shard_num, text in enumerate(prompts):
        entry = {
            "prompt_index": shard_num,
            "conversations": [
                {"from": "human", "value": text},
                {"role": "assistant", "content": "reply"},
            ],
            "metadata": {},
            "completed": True,
            "api_calls": 1,
            "toolsets_used": [],
            "tool_stats": {},
            "tool_error_counts": {},
        }
        with open(output_dir / f"batch_{shard_num}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        completed.append(shard_num)
    checkpoint = {
        "run_name": "resume_regression",
        "completed_prompts": completed,
        "batch_stats": {
            str(n): {"processed": 1, "skipped": 0, "discarded_no_reasoning": 0}
            for n in range(len(prompts))
        },
        "last_updated": None,
    }
    (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2))


# ---------------------------------------------------------------------------
# Defect 3: non-string prompt values crash the resume filter
# ---------------------------------------------------------------------------


class TestFilterTolerantOfNonStringPrompt:
    def _runner_with(self, tmp_path, entries):
        r = BatchRunner.__new__(BatchRunner)
        r.dataset = entries
        return r

    def test_non_string_prompt_does_not_crash_filter(self, tmp_path):
        """One malformed row {"prompt": 123} must not AttributeError-abort
        the whole resume after the content scan already succeeded."""
        r = self._runner_with(tmp_path, [
            {"prompt": 123},
            {"prompt": None},
            {"prompt": "plain"},
        ])
        kept, skipped = r._filter_dataset_by_completed({"plain"})
        assert skipped == [2]
        assert [e for _, e in kept] == [{"prompt": 123}, {"prompt": None}]

    def test_filter_extraction_matches_content_scan(self, tmp_path):
        """The filter must recognize exactly what the content scan recorded,
        including chat-style messages entries the old inline logic ignored."""
        entry_msgs = {"messages": [{"role": "user", "content": "chat-q"},
                                   {"role": "assistant", "content": "a"}]}
        r = self._runner_with(tmp_path, [entry_msgs, {"prompt": "other"}])

        # The scan extracts "chat-q" from this entry shape...
        scanned = batch_runner.BatchRunner._scan_completed_prompts_by_content
        assert _entry_text(entry_msgs) == "chat-q"

        # ...so filtering against the scanned set must drop it too.
        kept, skipped = r._filter_dataset_by_completed({"chat-q"})
        assert skipped == [0]
        assert [e for _, e in kept] == [{"prompt": "other"}]


# ---------------------------------------------------------------------------
# Defect 1: stale index filter silently skips re-indexed rows on resume
# ---------------------------------------------------------------------------


class TestResumeDoesNotApplyStaleIndexFilter:
    def test_resume_workers_get_no_stale_indices(self, tmp_path, fake_pool, fake_llm):
        """After a dataset edit, resume must not hand workers the previous
        run's completed INDICES: the rebuilt batches carry current-file
        indices, and a collision silently skips a never-run prompt."""
        # Run 1: alpha(0) and beta(1) completed, gamma(2) never ran.
        _seed_first_run(tmp_path / "out", ["alpha", "beta"])

        # Dataset edited between runs: new row "zed" inserted at the front.
        # Its new index 0 collides with the stale completed set {0, 1}.
        dataset_v2 = [
            {"prompt": "zed"},     # new idx 0  <-- stale-filter collision
            {"prompt": "alpha"},   # idx 1 (done)
            {"prompt": "beta"},    # idx 2 (done)
            {"prompt": "gamma"},   # idx 3 (never ran)
        ]
        r = _make_runner(tmp_path, dataset_v2)

        r.run(resume=True)

        tasks = fake_pool.instances[0].dispatched_tasks
        assert len(tasks) == 1
        worker_completed_set = tasks[0][3]
        # Content filtering already excluded alpha/beta; the index set the
        # worker receives must NOT reintroduce stale checkpoints indices.
        assert worker_completed_set == set(), (
            f"workers received stale checkpoint indices {worker_completed_set}; "
            "re-indexed resume batches must not be filtered by the "
            "interrupted run's indices (#95322)"
        )

    def test_resumed_run_processes_collision_prompt(self, tmp_path, fake_pool, fake_llm):
        """End-to-end: the never-completed prompt whose new index collides
        with the stale set must actually be executed on resume."""
        _seed_first_run(tmp_path / "out", ["alpha", "beta"])

        dataset_v2 = [
            {"prompt": "zed"},
            {"prompt": "alpha"},
            {"prompt": "beta"},
            {"prompt": "gamma"},
        ]
        r = _make_runner(tmp_path, dataset_v2)
        r.run(resume=True)

        # Scan outputs the way the next resume would: both "zed" and
        # "gamma" must have produced trajectory rows.
        scanner = BatchRunner.__new__(BatchRunner)
        scanner.output_dir = r.output_dir
        completed_texts = scanner._scan_completed_prompts_by_content()
        assert "zed" in completed_texts, (
            "'zed' (new index 0 collided with stale completed index 0) was "
            "silently skipped instead of being re-run"
        )
        assert "gamma" in completed_texts


# ---------------------------------------------------------------------------
# Defect 2: resumed runs renumber shards from 0 and append into old files
# ---------------------------------------------------------------------------


class TestResumeShardsContinueNumbering:
    def test_resume_writes_to_fresh_shard_numbers(self, tmp_path, fake_pool, fake_llm):
        _seed_first_run(tmp_path / "out", ["alpha", "beta"])
        old_shards = {
            name: (tmp_path / "out" / name).read_bytes()
            for name in ("batch_0.jsonl", "batch_1.jsonl")
        }

        dataset_v2 = [{"prompt": "zed"}, {"prompt": "alpha"},
                      {"prompt": "beta"}, {"prompt": "gamma"}]
        r = _make_runner(tmp_path, dataset_v2)
        r.run(resume=True)

        # New shards continue after the highest existing number...
        new_files = sorted(p.name for p in r.output_dir.glob("batch_*.jsonl"))
        assert "batch_2.jsonl" in new_files, f"resume did not create a fresh shard: {new_files}"

        # ...and never append into the previous run's files.
        for name, blob in old_shards.items():
            assert (tmp_path / "out" / name).read_bytes() == blob, (
                f"resume appended new rows into the previous run's {name}"
            )

    def test_resume_preserves_previous_shard_stats(self, tmp_path, fake_pool, fake_llm):
        _seed_first_run(tmp_path / "out", ["alpha", "beta"])

        dataset_v2 = [{"prompt": "zed"}, {"prompt": "alpha"},
                      {"prompt": "beta"}, {"prompt": "gamma"}]
        r = _make_runner(tmp_path, dataset_v2)
        r.run(resume=True)

        checkpoint = json.loads(r.checkpoint_file.read_text())
        stats = checkpoint.get("batch_stats", {})
        # The previous run's per-shard stats survive untouched...
        assert stats.get("0", {}).get("processed") == 1
        assert stats.get("1", {}).get("processed") == 1
        # ...and the resumed shard lands under its own continued number.
        assert stats.get("2", {}).get("processed") == 2, (
            f"resumed shard stats missing/mis-keyed: {stats}"
        )


# ---------------------------------------------------------------------------
# Fresh runs keep the backward-compatible index filter
# ---------------------------------------------------------------------------


class TestFreshRunKeepsIndexFilter:
    def test_fresh_run_passes_checkpoint_indices(self, tmp_path, fake_pool, fake_llm):
        """Non-resume runs keep the legacy behavior: checkpoint indices are
        passed through to workers verbatim, so a same-name rerun over an
        UNCHANGED dataset still skips the already-done row."""
        _seed_first_run(tmp_path / "out", ["alpha"])

        r = _make_runner(tmp_path, [{"prompt": "alpha"}, {"prompt": "bravo"}])
        r.run(resume=False)

        tasks = fake_pool.instances[0].dispatched_tasks
        assert tasks, "fresh run dispatched no tasks"
        assert tasks[0][3] == {0}
        # alpha was skipped by index, bravo ran.
        scanner = BatchRunner.__new__(BatchRunner)
        scanner.output_dir = r.output_dir
        assert scanner._scan_completed_prompts_by_content() >= {"alpha", "bravo"}
