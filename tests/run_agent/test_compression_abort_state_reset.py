"""Regression tests for #58630: every compression abort path must reset
per-attempt in-place compaction state.

After a successful in-place compaction sets ``_last_compaction_in_place=True``
(run-level gateway signal), a later attempt that aborts or skips through ANY
early-return path in ``compress_context`` must NOT reuse that stale flag as
the flush baseline: ``conversation_history_after_compression()`` would then
treat all current messages (including unflushed new turns) as persisted, and a
restart would lose them.

The fix records a per-attempt outcome (``_last_compression_attempt_recorded``/
``_last_compression_attempt_in_place``) at the very top of
``compress_context`` — before the codex-app-server route, breaker gates, lock
acquisition, rotated-parent skips, compressor-abort, no-progress and
empty-transcript returns — so every abort path leaves the attempt outcome
``None`` and callers retain the previous flush baseline.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch


def _make_agent(session_db):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id="abort-state-session",
            skip_context_files=True,
            skip_memory=True,
        )
    return agent


class _InPlaceSuccessCompressor:
    _last_compress_aborted = False
    _last_summary_error = None
    compression_count = 1
    _last_compression_made_progress = True
    _last_summary_fallback_used = False
    last_compression_rough_tokens = 0
    last_prompt_tokens = 0
    last_completion_tokens = 0
    awaiting_real_usage_after_compression = False

    def compress(self, _messages, **_kwargs):
        return [
            {"role": "user", "content": "[summary] earlier state"},
            {"role": "assistant", "content": "retained tail"},
        ]


class _BreakerBlockedCompressor(_InPlaceSuccessCompressor):
    """Trips the pre-lock automatic-compression breaker gate."""

    def _automatic_compression_blocked(self):
        return True

    def compress(self, messages, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("compress() must not be reached when blocked")


class _NoProgressCompressor(_InPlaceSuccessCompressor):
    """Returns a semantically-equal transcript (no-op attempt)."""

    _last_compression_made_progress = False

    def compress(self, messages, **_kwargs):
        return [dict(m) for m in messages]


class TestAbortPathsResetPerAttemptState:
    def _in_place_success(self, agent, messages):
        from agent.conversation_compression import (
            compress_context,
            conversation_history_after_compression,
        )
        agent.context_compressor = _InPlaceSuccessCompressor()
        compacted, _ = compress_context(
            agent, messages, "system", approx_tokens=100_000
        )
        assert agent._last_compaction_in_place is True
        assert agent._last_compression_attempt_in_place is True
        return compacted, conversation_history_after_compression(
            agent, compacted, None
        )

    def test_breaker_blocked_skip_retains_previous_baseline(self):
        """Pre-lock breaker skip after an in-place success must keep baseline."""
        from agent.conversation_compression import (
            compress_context,
            conversation_history_after_compression,
        )
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = _make_agent(db)
            agent.compression_in_place = True
            original = [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
            ]
            agent._flush_messages_to_session_db(original, [])
            compacted, history = self._in_place_success(agent, original)

            messages = compacted + [
                {"role": "user", "content": "new request"},
                {"role": "assistant", "content": "new answer"},
            ]
            agent.context_compressor = _BreakerBlockedCompressor()
            returned, _ = compress_context(
                agent, messages, "system", approx_tokens=100_000
            )
            assert returned is messages
            # Per-attempt outcome must be reset even though this attempt
            # returned before acquiring the lock.
            assert agent._last_compression_attempt_recorded is True
            assert agent._last_compression_attempt_in_place is None
            new_history = conversation_history_after_compression(
                agent, returned, history
            )
            # Skip = previous baseline stays authoritative: not all-persisted
            # (would drop the new pair on restart), not None (would re-append
            # the compacted rows).
            assert new_history is history
            db.close()

    def test_no_progress_attempt_retains_previous_baseline(self):
        """A semantic no-op attempt must not reuse the stale in-place flag."""
        from agent.conversation_compression import (
            compress_context,
            conversation_history_after_compression,
        )
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = _make_agent(db)
            agent.compression_in_place = True
            original = [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
            ]
            agent._flush_messages_to_session_db(original, [])
            compacted, history = self._in_place_success(agent, original)

            messages = compacted + [
                {"role": "user", "content": "new request"},
                {"role": "assistant", "content": "new answer"},
            ]
            agent.context_compressor = _NoProgressCompressor()
            returned, _ = compress_context(
                agent, messages, "system", approx_tokens=100_000
            )
            assert agent._last_compression_attempt_in_place is None
            new_history = conversation_history_after_compression(
                agent, returned, history
            )
            assert new_history is history
            # Run-level gateway signal is untouched by the aborted attempt.
            assert agent._last_compaction_in_place is True
            db.close()

    def test_rotation_boundary_still_clears_baseline(self):
        """A completed rotation attempt must still return None (full rewrite)."""
        from agent.conversation_compression import (
            compress_context,
            conversation_history_after_compression,
        )
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = _make_agent(db)
            agent.compression_in_place = False
            original = [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
            ]
            agent._flush_messages_to_session_db(original, [])
            agent.context_compressor = _InPlaceSuccessCompressor()
            compacted, _ = compress_context(
                agent, original, "system", approx_tokens=100_000
            )
            assert agent._last_compression_attempt_in_place is False
            assert conversation_history_after_compression(
                agent, compacted, list(original)
            ) is None
            db.close()
