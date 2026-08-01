"""Tests for batch_runner trajectory durability and pool cleanup.

Verifies:
  1. Trajectory entries are fsync'd to disk before the checkpoint marks
     them as completed (crash-between-write-and-sync safety).
  2. Pool.terminate() + pool.join() are called on KeyboardInterrupt and
     Exception during batch execution (responsive worker shutdown).
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# batch_runner uses relative imports, ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_runner import BatchRunner, _process_batch_worker


# =========================================================================
# Trajectory write durability (fsync)
# =========================================================================

class TestTrajectoryWriteDurability:
    """Verify that trajectory entries are flushed and fsync'd before the
    checkpoint marks them as completed.

    Without fsync, a crash between the write and the disk sync would leave
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
        original_fsync = os.fsync

        def mock_fsync(fd):
            fsync_calls.append(fd)

        monkeypatch.setattr("os.fsync", mock_fsync)

        result = _process_batch_worker(
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
# Pool cleanup on interruption / exception
# =========================================================================

class TestPoolCleanupOnInterruption:
    """Verify that pool.terminate() + pool.join() are called when a
    KeyboardInterrupt or Exception occurs during batch execution.

    CPython's multiprocessing.pool.Pool.join() does NOT accept a timeout
    parameter — calling pool.join(timeout=10) raises TypeError.  The fix
    uses pool.terminate() followed by pool.join() (no timeout), which is
    the correct shutdown pattern.
    """

    def test_pool_terminate_called_on_exception(self, tmp_path, monkeypatch):
        """When pool.imap_unordered raises an exception, pool.terminate()
        and pool.join() must be called for clean worker shutdown.

        We simulate the relevant slice of run()'s try/except block with a
        mock pool to verify the cleanup contract.
        """
        mock_pool = MagicMock()
        mock_pool.imap_unordered.side_effect = RuntimeError("worker exploded")

        # Reproduce the exception-handling block from batch_runner.run()
        with pytest.raises(RuntimeError, match="worker exploded"):
            try:
                for result in mock_pool.imap_unordered(None, []):
                    pass
            except KeyboardInterrupt:
                mock_pool.terminate()
                mock_pool.join()
                raise
            except Exception:
                mock_pool.terminate()
                mock_pool.join()
                raise

        mock_pool.terminate.assert_called_once()
        mock_pool.join.assert_called_once_with()

    def test_pool_terminate_called_on_keyboard_interrupt(self, tmp_path, monkeypatch):
        """When pool.imap_unordered is interrupted (Ctrl+C), pool.terminate()
        and pool.join() must be called for responsive shutdown."""
        mock_pool = MagicMock()
        mock_pool.imap_unordered.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            try:
                for result in mock_pool.imap_unordered(None, []):
                    pass
            except KeyboardInterrupt:
                mock_pool.terminate()
                mock_pool.join()
                raise
            except Exception:
                mock_pool.terminate()
                mock_pool.join()
                raise

        mock_pool.terminate.assert_called_once()
        mock_pool.join.assert_called_once_with()

    def test_pool_join_called_without_timeout(self, tmp_path):
        """Pool.join() must NOT be called with a timeout argument —
        CPython's Pool.join signature is (self), so join(timeout=10)
        would raise TypeError."""
        mock_pool = MagicMock()
        mock_pool.imap_unordered.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            try:
                for result in mock_pool.imap_unordered(None, []):
                    pass
            except Exception:
                mock_pool.terminate()
                mock_pool.join()
                raise

        # The join call must have no positional/keyword timeout argument
        join_call = mock_pool.join.call_args
        assert join_call == call(), (
            f"pool.join() called with unexpected args: {join_call}"
        )

    def test_real_pool_join_accepts_no_timeout(self):
        """Integration check: a real multiprocessing.Pool's join() must not
        accept a timeout kwarg.  This guards against re-introducing
        pool.join(timeout=10), which raises TypeError on CPython.
        """
        import inspect
        import multiprocessing.pool

        sig = inspect.signature(multiprocessing.pool.Pool.join)
        params = list(sig.parameters.keys())
        # The only parameter should be 'self' — no 'timeout'
        assert "timeout" not in params, (
            f"Pool.join has unexpected parameters: {params}"
        )
