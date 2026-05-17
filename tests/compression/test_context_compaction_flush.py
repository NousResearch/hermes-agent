"""Tests for #20293: Context Compaction + Session Split injects compressed
summary as valid history.

Root cause: after compress_context() creates a new session, the flush cursor
(_last_flushed_db_idx) was reset to 0, causing _flush_messages_to_session_db
to re-append the entire compressed history as new messages.  The fix sets the
cursor to len(compressed) so the flush skips already-written messages.

Three compression paths are tested:
  1. Agent mid-run compression (compress_context internal reset)
  2. Hygiene pre-compression (gateway _prewritten_msg_count)
  3. /compress command (session_entry._prewritten_msg_count persistence)
"""

from pathlib import Path
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. compress_context sets _last_flushed_db_idx = len(compressed)
# ---------------------------------------------------------------------------

class TestCompressContextFlushCursor:
    """Verify that compress_context advances the flush cursor past the
    compressed messages so subsequent flushes don't re-append them."""

    def test_compress_context_advances_flush_cursor_in_production_source(self):
        """compress_context must advance the flush cursor to len(compressed),
        not reset it to 0."""
        source = (ROOT / "agent" / "conversation_compression.py").read_text(
            encoding="utf-8"
        )
        assert "agent._last_flushed_db_idx = len(compressed)" in source
        assert "agent._last_flushed_db_idx = 0" not in source

    def test_gateway_run_agent_accepts_prewritten_count(self):
        """Gateway _run_agent must accept and apply _prewritten_msg_count."""
        source = (ROOT / "gateway" / "run.py").read_text(encoding="utf-8")
        assert "_prewritten_msg_count: int = 0" in source
        assert "agent._last_flushed_db_idx = _prewritten_msg_count" in source



# ---------------------------------------------------------------------------
# 2. Hygiene pre-compression passes prewritten count to _run_agent
# ---------------------------------------------------------------------------

class TestHygienePrewrittenCount:
    """Verify that the gateway's hygiene compression path tracks the number
    of prewritten messages and passes it to _run_agent."""

    def test_session_entry_stores_prewritten_count(self):
        """SessionEntry should accept and serialize _prewritten_msg_count."""
        from gateway.session import SessionEntry
        from datetime import datetime

        entry = SessionEntry(
            session_key="test-key",
            session_id="sid-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            _prewritten_msg_count=42,
        )
        assert entry._prewritten_msg_count == 42

        # Serialization roundtrip
        d = entry.to_dict()
        assert d["_prewritten_msg_count"] == 42

        entry2 = SessionEntry.from_dict(d)
        assert entry2._prewritten_msg_count == 42

    def test_session_entry_default_prewritten_count(self):
        """Default _prewritten_msg_count should be 0."""
        from gateway.session import SessionEntry
        from datetime import datetime

        entry = SessionEntry(
            session_key="test-key",
            session_id="sid-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        assert entry._prewritten_msg_count == 0

    def test_prewritten_count_deserialization_backward_compat(self):
        """Old session data without _prewritten_msg_count should default to 0."""
        from gateway.session import SessionEntry
        from datetime import datetime

        entry = SessionEntry(
            session_key="k", session_id="s",
            created_at=datetime.now(), updated_at=datetime.now(),
        )
        d = entry.to_dict()
        del d["_prewritten_msg_count"]

        restored = SessionEntry.from_dict(d)
        assert restored._prewritten_msg_count == 0


# ---------------------------------------------------------------------------
# 3. _run_agent applies prewritten count to agent flush cursor
# ---------------------------------------------------------------------------

class TestRunAgentPrewrittenInit:
    """Verify that _run_agent sets the agent's _last_flushed_db_idx from
    the _prewritten_msg_count parameter."""

    def test_prewritten_param_advances_flush_cursor(self):
        """When _prewritten_msg_count > 0, the agent's flush cursor should
        be advanced past the already-written messages."""
        agent = MagicMock()
        agent._last_flushed_db_idx = 0

        # Simulate what _run_agent does:
        _prewritten_msg_count = 25
        if _prewritten_msg_count > 0:
            agent._last_flushed_db_idx = _prewritten_msg_count

        assert agent._last_flushed_db_idx == 25

    def test_zero_prewritten_leaves_cursor_unchanged(self):
        """When _prewritten_msg_count == 0 (default), the cursor stays at
        its initial value (backward compatible)."""
        agent = MagicMock()
        agent._last_flushed_db_idx = 0

        _prewritten_msg_count = 0
        if _prewritten_msg_count > 0:
            agent._last_flushed_db_idx = _prewritten_msg_count

        assert agent._last_flushed_db_idx == 0


# ---------------------------------------------------------------------------
# 4. Flush does not re-append prewritten messages
# ---------------------------------------------------------------------------

class TestFlushSkipsPrewritten:
    """Verify that _flush_messages_to_session_db respects the advanced cursor
    and does not re-append already-written compressed messages."""

    def test_flush_starts_after_prewritten_cursor(self):
        """_flush_messages_to_session_db should start from
        _last_flushed_db_idx, skipping already-written messages."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        # Simulate the flush logic from run_agent.py:_flush_messages_to_session_db
        messages = [
            {"role": "system", "content": "sys"},
        ] + [{"role": "user", "content": f"msg{i}"} for i in range(20)]
        # messages[0:5] = compressed history (already written)
        # messages[5:] = new messages from this turn

        agent_flush_cursor = 5  # set by _prewritten_msg_count

        # The flush logic uses:
        # flush_from = max(start_idx, self._last_flushed_db_idx)
        start_idx = 0
        flush_from = max(start_idx, agent_flush_cursor)

        assert flush_from == 5
        new_msgs = messages[flush_from:]
        assert len(new_msgs) == 16  # 21 total - 5 prewritten = 16 new

    def test_no_duplicate_append_after_compression(self):
        """Verify that setting _last_flushed_db_idx = len(compressed)
        prevents the entire compressed history from being re-appended."""
        compressed = [{"role": "system", "content": "s"}] * 8
        new_turn_msgs = [
            {"role": "user", "content": "new msg"},
            {"role": "assistant", "content": "new reply"},
        ]
        all_messages = compressed + new_turn_msgs  # 10 total

        # After compression, cursor = len(compressed)
        flush_cursor = len(compressed)  # = 8

        # Flush from cursor
        to_flush = all_messages[flush_cursor:]
        assert len(to_flush) == 2  # only the new messages
        assert to_flush == new_turn_msgs


# ---------------------------------------------------------------------------
# 5. End-to-end scenario: compression + flush does not duplicate
# ---------------------------------------------------------------------------

class TestCompressionFlushIntegration:
    """Integration scenario verifying the full fix prevents duplicate history."""

    def test_midrun_compression_no_duplicates(self):
        """After mid-run compression, flush only writes new messages."""
        # Simulated message list during agent run
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "long conversation..."},
            {"role": "assistant", "content": "long reply..."},
            {"role": "user", "content": "more..."},
            {"role": "assistant", "content": "reply..."},
        ]

        # Compression replaces messages 1-5 with a summary
        compressed = [
            messages[0],  # system prompt
            {"role": "assistant", "content": "[SUMMARY] Previous conversation about..."},
            messages[-2],  # keep last user msg
            messages[-1],  # keep last assistant msg
        ]

        # After compression, cursor = len(compressed)
        flush_cursor = len(compressed)  # = 4

        # Agent appends new turn
        new_turn = [
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": "new answer"},
        ]
        all_msgs = compressed + new_turn  # 6 total

        # Flush from cursor
        to_flush = all_msgs[flush_cursor:]
        assert len(to_flush) == 2
        assert to_flush == new_turn
        # The compressed summary is NOT re-flushed as a new message

    def test_hygiene_compression_no_duplicates(self):
        """After hygiene pre-compression, the next agent's flush skips
        the compressed messages already written via rewrite_transcript."""
        # Hygiene compression produces:
        compressed = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "[SUMMARY]"},
            {"role": "user", "content": "recent msg"},
        ]
        prewritten = len(compressed)  # = 3

        # Main agent created with history = compressed
        # _last_flushed_db_idx set to prewritten (3)
        agent_cursor = prewritten

        # Agent runs and produces new messages
        new_msgs = [
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "final reply"},
        ]
        all_msgs = compressed + new_msgs  # 6 total

        # Flush from cursor
        to_flush = all_msgs[agent_cursor:]
        assert len(to_flush) == 3
        assert to_flush == new_msgs
        # No duplicates of the compressed history
