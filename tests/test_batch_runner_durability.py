"""Tests for batch_runner trajectory durability and pool cleanup.

Verifies:
  1. Trajectory entries are fsync'd to disk before the checkpoint marks
     them as completed (crash-between-write-and-sync safety).
  2. BatchRunner.run() calls pool.terminate() + pool.join() on
     KeyboardInterrupt and Exception during batch execution (responsive
     worker shutdown).  CPython's Pool.join() takes no timeout parameter —
     join(timeout=10) raises TypeError — so the tests also assert join()
     is invoked with no arguments.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# batch_runner is a root-level module (not part of an installed package),
# so make the repo root importable when tests run from elsewhere.
sys.path.insert(0, str(Path(__file__).parent.parent))

import batch_runner
from batch_runner import BatchRunner, _process_batch_worker


# =========================================================================
# Trajectory write durability (fsync)
# =========================================================================

class TestTrajectoryWriteDurability:
    """Verify that trajectory entries are flushed and fsync'd to disk.

    Without fsync, a crash between the write and the disk sync could leave
    the checkpoint claiming completion with no trajectory data on disk.
    """

    def test_trajectory_entry_is_synced_to_disk(self, tmp_path, monkeypatch):
        """_process_batch_worker should flush+fsync the trajectory file."""
        prompt_result = {
            "success": True,
            "trajectory": [{"role": "assistant", "content": "x"}],
            "reasoning_stats": {"has_any_reasoning": True},
            "tool_stats": {},
            "metadata": {},
            "completed": True,
            "api_calls": 1,
            "toolsets_used": [],
        }

        monkeypatch.setattr(
            "batch_runner._process_single_prompt", lambda *a, **kw: prompt_result
        )

        # Intercept os.fsync to record calls
        fsync_calls = []
        monkeypatch.setattr("os.fsync", lambda fd: fsync_calls.append(fd))

        _process_batch_worker(
            (
                1,
                [(0, {"prompt": "hi"})],
                tmp_path,
                set(),
                {"verbose": False},
            )
        )

        # Verify fsync was called at least once during trajectory write
        assert len(fsync_calls) >= 1, (
            "os.fsync was not called — trajectory writes are not durable"
        )

        # Verify the trajectory file exists and is valid
        output_files = list(tmp_path.glob("*.jsonl"))
        assert len(output_files) >= 1
        for f in output_files:
            lines = f.read_text().strip().split("\n")
            for line in lines:
                if line:
                    entry = json.loads(line)
                    assert "conversations" in entry
                    assert "completed" in entry


# =========================================================================
# Pool cleanup on interruption / exception — drives the REAL run()
# =========================================================================

def _make_runner(tmp_path, monkeypatch):
    """Build a minimal real BatchRunner against a 1-line tmp dataset."""
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"prompt": "hi"}) + "\n", encoding="utf-8")
    # BatchRunner writes to Path("data")/run_name relative to cwd.
    monkeypatch.chdir(tmp_path)
    return BatchRunner(
        dataset_file=str(dataset),
        batch_size=1,
        run_name="pool-cleanup-test",
        num_workers=1,
    )


def _make_failing_pool(exc):
    """Context-manager mock whose pool raises `exc` from imap_unordered."""
    pool = MagicMock()
    pool.imap_unordered.side_effect = exc
    pool_cm = MagicMock()
    pool_cm.__enter__ = MagicMock(return_value=pool)
    pool_cm.__exit__ = MagicMock(return_value=False)
    return pool, pool_cm


class TestPoolCleanupOnInterruption:
    """Drive the real BatchRunner.run() with a patched Pool and verify the
    cleanup contract: terminate() + join() (join with NO timeout argument —
    CPython's Pool.join signature is (self), so join(timeout=10) would
    raise TypeError).
    """

    @pytest.mark.parametrize("exc_type", [KeyboardInterrupt, RuntimeError])
    def test_run_terminates_and_joins_pool(self, tmp_path, monkeypatch, exc_type):
        runner = _make_runner(tmp_path, monkeypatch)
        pool, pool_cm = _make_failing_pool(exc_type("boom"))

        with patch.object(batch_runner, "Pool", return_value=pool_cm):
            with pytest.raises(exc_type):
                runner.run()

        pool.terminate.assert_called_once()
        # join() must be called with no positional/keyword arguments.
        assert pool.join.call_args_list == [call()], (
            f"pool.join() called with unexpected args: {pool.join.call_args_list}"
        )
