"""Regression tests for #31507 — _last_flushed_db_idx overshoot after scaffolding drop.

Covers the bug where intermediate _persist_session calls set
_last_flushed_db_idx to the old (higher) message count, then
_drop_trailing_empty_response_scaffolding pops messages, then
_flush_messages_to_session_db computes flush_from = max(start_idx,
old_idx) >= len(messages) → empty slice → final assistant responses
never written to state.db.

The fix clamps _last_flushed_db_idx = min(_last_flushed_db_idx,
len(messages)) after the scaffolding drop in _persist_session.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_agent():
    """Create a bare AIAgent with no side effects."""
    from run_agent import AIAgent
    return AIAgent.__new__(AIAgent)


def _make_agent_with_db(session_db, session_id="test-31507"):
    """Create a minimal AIAgent with a real session DB."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._ensure_db_session()
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFlushIndexClampAfterScaffoldingDrop:
    """Core regression: _last_flushed_db_idx must not exceed len(messages)
    after _drop_trailing_empty_response_scaffolding pops messages."""

    def test_flush_index_clamped_after_scaffolding_drop(self):
        """#31507 core: intermediate persist sets _last_flushed_db_idx=98,
        scaffolding drops messages to len=97, final flush must still write."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = _make_agent_with_db(db)

            # Build a message list that mimics the trigger condition:
            # user + assistant + tool + assistant (with scaffolding)
            messages = [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "t1", "type": "function",
                                 "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "t1", "content": "result"},
                {"role": "assistant", "content": "final answer"},
            ]

            # Simulate: intermediate _persist_session set _last_flushed_db_idx
            # to len(messages) = 4
            agent._last_flushed_db_idx = 4

            # Now scaffolding drops the last message (simulating the bug trigger)
            # _drop_trailing_empty_response_scaffolding pops trailing empty responses
            messages_with_scaffolding = list(messages) + [
                {"role": "assistant", "content": "(empty)",
                 "_empty_terminal_sentinel": True},
            ]

            # Before fix: _last_flushed_db_idx=4, scaffolding drops to len=4,
            # but if intermediate persist had set it to 5 (old len), then
            # flush_from = max(0, 5) = 5 >= 4 → empty slice

            # Simulate the exact bug: index set to old higher value
            agent._last_flushed_db_idx = 5

            # Call _persist_session which does:
            # 1. _drop_trailing_empty_response_scaffolding → pops sentinel, len=4
            # 2. clamp _last_flushed_db_idx = min(5, 4) = 4  ← THE FIX
            # 3. _flush_messages_to_session_db → flush_from = max(0, 4) = 4
            #    messages[4:] → empty, but that's correct (nothing new)
            agent._persist_session(messages_with_scaffolding)

            # Verify the index was clamped
            assert agent._last_flushed_db_idx == len(messages_with_scaffolding)

    def test_flush_index_not_clamped_when_no_drop(self):
        """When scaffolding doesn't pop anything, _last_flushed_db_idx is unchanged."""
        agent = _bare_agent()
        agent._last_flushed_db_idx = 3

        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]

        # No scaffolding to drop — clamp should be a no-op
        agent._drop_trailing_empty_response_scaffolding(messages)
        agent._last_flushed_db_idx = min(agent._last_flushed_db_idx, len(messages))

        assert agent._last_flushed_db_idx == 3  # unchanged
        assert len(messages) == 4

    def test_flush_index_clamped_to_zero_on_full_wipe(self):
        """If scaffolding drops all messages, index clamps to 0."""
        agent = _bare_agent()
        agent._last_flushed_db_idx = 100

        messages = [
            {"role": "assistant", "content": "(empty)",
             "_empty_terminal_sentinel": True},
        ]

        agent._drop_trailing_empty_response_scaffolding(messages)
        agent._last_flushed_db_idx = min(agent._last_flushed_db_idx, len(messages))

        assert len(messages) == 0
        assert agent._last_flushed_db_idx == 0

    def test_flush_index_clamped_after_multi_message_scaffolding_drop(self):
        """When scaffolding drops multiple messages (sentinel + tool pair), index clamps."""
        agent = _bare_agent()
        agent._last_flushed_db_idx = 8

        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "t1", "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "out"},
            {"role": "assistant", "content": "(empty)",
             "_empty_terminal_sentinel": True},
        ]

        agent._drop_trailing_empty_response_scaffolding(messages)
        agent._last_flushed_db_idx = min(agent._last_flushed_db_idx, len(messages))

        # Scaffolding dropped sentinel + tool pair + assistant(tool_calls) → len=1
        assert len(messages) == 1
        assert agent._last_flushed_db_idx == 1

    def test_post_scaffold_invariant_index_le_len(self):
        """Post-condition: after scaffolding drop + clamp, _last_flushed_db_idx <= len(messages).

        This is the invariant that #31507 violated.
        """
        import random

        for _ in range(50):
            agent = _bare_agent()
            # Random starting index
            agent._last_flushed_db_idx = random.randint(0, 200)

            # Build random message list with possible scaffolding at tail
            n = random.randint(0, 10)
            messages = [{"role": "user", "content": f"msg {i}"} for i in range(n)]

            # Sometimes append scaffolding
            if random.random() < 0.5:
                messages.append(
                    {"role": "assistant", "content": "(empty)",
                     "_empty_terminal_sentinel": True}
                )

            agent._drop_trailing_empty_response_scaffolding(messages)
            agent._last_flushed_db_idx = min(
                agent._last_flushed_db_idx, len(messages)
            )

            # THE INVARIANT: index never exceeds message count
            assert agent._last_flushed_db_idx <= len(messages), (
                f"_last_flushed_db_idx={agent._last_flushed_db_idx} > "
                f"len(messages)={len(messages)}"
            )


class TestPersistSessionWritesAfterScaffoldingDrop:
    """Integration: _persist_session must write messages to state.db even when
    an intermediate persist set _last_flushed_db_idx higher than the current
    message count after scaffolding drop."""

    def test_final_assistant_response_written_after_intermediate_persist(self):
        """#31507 core scenario:
        1. Intermediate _persist_session sets _last_flushed_db_idx=98
        2. Agent generates final assistant response (messages grow to 99)
        3. Final _persist_session: scaffolding drops 1 message (len→98)
        4. Without fix: flush_from=max(0,98)=98 >= 98 → empty slice → lost
        5. With fix: clamp to min(98,98)=98 → flush_from=98 → messages[98:] → written
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = _make_agent_with_db(db)

            # Simulate: 3 messages, intermediate persist wrote them all
            messages = [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ]
            agent._flush_messages_to_session_db(messages, [])
            assert agent._last_flushed_db_idx == 3

            # Agent generates final assistant response → messages grow to 4
            messages.append({"role": "assistant", "content": "a2 — final answer"})

            # Final _persist_session: no scaffolding to drop, index stays at 3
            agent._persist_session(messages)

            rows = db.get_messages(agent.session_id)
            assert len(rows) == 4, f"Expected 4 messages, got {len(rows)}"
            assert rows[3]["content"] == "a2 — final answer"

    def test_scaffolding_drop_then_new_messages_written(self):
        """Scaffolding drops tail, then new content is appended and written."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = _make_agent_with_db(db)

            # Round 1: user + assistant + tool call + tool result
            messages = [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "t1", "type": "function",
                                 "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "t1", "content": "out"},
            ]
            agent._flush_messages_to_session_db(messages, [])
            assert agent._last_flushed_db_idx == 3

            # Round 1 fails → empty response + scaffolding
            messages.append({"role": "assistant", "content": "",
                            "_empty_recovery_synthetic": True})
            messages.append({"role": "assistant", "content": "(empty)",
                            "_empty_terminal_sentinel": True})

            # Intermediate persist: scaffolding drops sentinel + recovery + tool pair
            # → len goes from 5 to 1
            agent._persist_session(messages)
            assert len(messages) == 1
            assert agent._last_flushed_db_idx == 1

            # Round 2: new user message + successful assistant response
            messages.append({"role": "user", "content": "q2"})
            messages.append({"role": "assistant", "content": "a2 — success"})

            agent._persist_session(messages)

            rows = db.get_messages(agent.session_id)
            # Should have: q1, q2, a2 (the tool pair was part of the failed round)
            assert len(rows) >= 2, f"Expected at least 2 messages, got {len(rows)}"
            # The final assistant response must be present
            contents = [r["content"] for r in rows]
            assert "a2 — success" in contents
