"""Regression tests: a failed compression split must arm the failure cooldown.

Extracted from PR #98137's test file (author: vsd2807); two sibling unit
tests were dropped as duplicates of existing coverage in
test_compression_rotation_state.py / test_context_compressor.py, and the
timeout-reconciliation tests were not carried because that production path
was not salvaged (blocking review on #98137).

Issue #97948 symptom B: without the cooldown, the turn after a
session_split_failed abort immediately re-runs the identical doomed
compression.
"""

import time
from unittest.mock import MagicMock


def test_manual_compress_bypasses_cooldown():
    """Manual /compress (force=True) bypasses cooldown — existing behavior preserved."""
    from agent.conversation_compression import compress_context

    compressor_mock = MagicMock()
    compressor_mock._summary_failure_cooldown_until = time.monotonic() + 60.0
    compressor_mock._last_summary_error = "session_split_failed"
    compressor_mock._cooldown_persist_failed = False
    compressor_mock._anti_thrash_recovery_deadline = 0.0
    compressor_mock._consecutive_ineffective_compressions = 0
    compressor_mock._summary_failure_streak = 0
    compressor_mock._last_compaction_boundary_tokens = None
    compressor_mock._verify_compaction_cleared_threshold = False
    compressor_mock._proactive_prune_rearm_tokens = None
    compressor_mock.compression_count = 0
    compressor_mock.threshold_tokens = 10000
    compressor_mock.context_length = 128000
    compressor_mock.tail_token_budget = 25000
    compressor_mock.summary_target_ratio = 0.1
    compressor_mock._last_cooldown_refresh_was_authoritative = None

    agent_mock = MagicMock()
    agent_mock.context_compressor = compressor_mock
    agent_mock.compression_enabled = True
    agent_mock.session_id = "test-session"
    agent_mock._session_db = MagicMock()
    agent_mock._session_db.try_acquire_compression_lock.return_value = False

    messages = [{"role": "user", "content": "hello"}]

    result = compress_context(
        agent_mock,
        messages,
        None,
        force=True,
        task_id="test",
    )

    assert result is not None
