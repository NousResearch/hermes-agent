"""Tests for context compression persistence in the gateway.

Verifies that when context compression fires during run_conversation(),
the compressed messages are properly persisted to both SQLite (via the
agent) and JSONL (via the gateway).

Bug scenario (pre-fix):
  1. Gateway loads 200-message history, passes to agent
  2. Agent's run_conversation() compresses to ~30 messages mid-run
  3. _compress_context() resets _last_flushed_db_idx = 0
  4. On exit, _flush_messages_to_session_db() calculates:
     flush_from = max(len(conversation_history=200), _last_flushed_db_idx=0) = 200
  5. messages[200:] is empty (only ~30 messages after compression)
  6. Nothing written to new session's SQLite — compressed context lost
  7. Gateway's history_offset was still 200, producing empty new_messages
  8. Fallback wrote only user/assistant pair — summary lost
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch



# ---------------------------------------------------------------------------
# Part 1: Agent-side — _flush_messages_to_session_db after compression
# ---------------------------------------------------------------------------

class TestFlushAfterCompression:
    """Verify that compressed messages are flushed to the new session's SQLite
    even when conversation_history (from the original session) is longer than
    the compressed messages list."""

    def _make_agent(self, session_db):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )
        return agent

    def test_flush_after_compression_with_long_history(self):
        """The actual bug: conversation_history longer than compressed messages.

        Before the fix, flush_from = max(len(conversation_history), 0) = 200,
        but messages only has ~30 entries, so messages[200:] is empty.
        After the fix, conversation_history is cleared to None after compression,
        so flush_from = max(0, 0) = 0, and ALL compressed messages are written.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)

            # Simulate the original long history (200 messages)
            original_history = [
                {"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"message {i}"}
                for i in range(200)
            ]

            # First, flush original messages to the original session
            agent._flush_messages_to_session_db(original_history, [])
            original_rows = db.get_messages("original-session")
            assert len(original_rows) == 200

            # Now simulate compression: new session, reset idx, shorter messages
            agent.session_id = "compressed-session"
            db.create_session(session_id="compressed-session", source="test")
            agent._last_flushed_db_idx = 0

            # The compressed messages (summary + tail + new turn)
            compressed_messages = [
                {"role": "user", "content": "[CONTEXT COMPACTION] Summary of work..."},
                {"role": "user", "content": "What should we do next?"},
                {"role": "assistant", "content": "Let me check..."},
                {"role": "user", "content": "new question"},
                {"role": "assistant", "content": "new answer"},
            ]

            # THE BUG: passing the original history as conversation_history
            # causes flush_from = max(200, 0) = 200, skipping everything.
            # After the fix, conversation_history should be None.
            agent._flush_messages_to_session_db(compressed_messages, None)

            new_rows = db.get_messages("compressed-session")
            assert len(new_rows) == 5, (
                f"Expected 5 compressed messages in new session, got {len(new_rows)}. "
                f"Compression persistence bug: messages not written to SQLite."
            )

    def test_flush_with_stale_history_loses_messages(self):
        """Stale conversation_history no longer causes data loss."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)

            # Simulate compression reset
            agent.session_id = "new-session"
            db.create_session(session_id="new-session", source="test")
            agent._last_flushed_db_idx = 0

            compressed = [
                {"role": "user", "content": "summary"},
                {"role": "assistant", "content": "continuing..."},
            ]

            # Stale history longer than messages: the old positional flush
            # sliced past the end and dropped both messages (#46053).
            stale_history = [{"role": "user", "content": f"msg{i}"} for i in range(100)]
            agent._flush_messages_to_session_db(compressed, stale_history)

            rows = db.get_messages("new-session")
            assert len(rows) == 2
            assert [row["content"] for row in rows] == ["summary", "continuing..."]

    def test_in_place_compression_rebaseline_prevents_duplicate_compacted_rows(self):
        """In-place compaction already persisted the compacted transcript.

        Regression for the 2026-06-26 SRE compression loop: archive_and_compact()
        inserted a compacted active block, then the same turn continued with
        conversation_history=None and _flush_messages_to_session_db() appended
        the compacted dicts again, doubling live context.
        """
        from agent.conversation_compression import conversation_history_after_compression
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)
            agent._ensure_db_session()

            original_history = [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
            ]
            agent._flush_messages_to_session_db(original_history, [])
            assert [row["content"] for row in db.get_messages("original-session")] == [
                "old question",
                "old answer",
            ]

            compacted = [
                {"role": "assistant", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "user", "content": "recent question"},
                {"role": "assistant", "content": "recent answer"},
            ]
            db.archive_and_compact("original-session", compacted)
            setattr(agent, "_last_compaction_in_place", True)
            agent._last_flushed_db_idx = 0

            # Same agent turn continues after compaction. The compacted dicts
            # must be treated as already-persisted history; only later appends
            # should be flushed.
            post_compaction_history = conversation_history_after_compression(
                agent, compacted
            )
            assert post_compaction_history is not None
            assert post_compaction_history is not compacted
            assert post_compaction_history == compacted

            messages = compacted + [
                {"role": "tool", "content": "tool result"},
                {"role": "assistant", "content": "final answer"},
            ]
            agent._flush_messages_to_session_db(messages, post_compaction_history)

            rows = db.get_messages("original-session")
            assert [row["content"] for row in rows] == [
                "[CONTEXT COMPACTION] summary",
                "recent question",
                "recent answer",
                "tool result",
                "final answer",
            ]

    def test_rotation_child_session_flushes_full_compressed_transcript_with_markers(self):
        """Regression for #57491: live cached-agent markers must not block child flush."""
        from agent.conversation_compression import compress_context
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)
            parent_sid = "20260701_152840_parent"
            db.create_session(parent_sid, "gateway", model="test/model")

            agent = self._make_agent(db)
            agent.session_id = parent_sid
            agent.compression_in_place = False
            agent._ensure_db_session()

            # Plain marked messages only: the exact-equality assertion below
            # relies on `compressed` containing no message that _flush filters
            # for a reason INDEPENDENT of _db_persisted (ephemeral scaffolding,
            # synthetic recovery turns). Keep this fixture free of such messages
            # or the row count would legitimately differ from len(compressed).
            messages = [
                {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"message {i}",
                    "_db_persisted": True,
                }
                for i in range(12)
            ]

            with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("no provider")):
                compressed, _ = compress_context(
                    agent, messages, approx_tokens=100_000, system_message="sys"
                )

            assert agent.session_id != parent_sid
            child_sid = agent.session_id

            agent._flush_messages_to_session_db(compressed, None)

            child_rows = db.get_messages(child_sid)
            assert len(child_rows) == len(compressed), (
                f"Expected {len(compressed)} rows in child session, got {len(child_rows)}. "
                f"_db_persisted marker propagation bug (#57491)."
            )
            db.close()


# ---------------------------------------------------------------------------
# Part 2: Gateway-side — history_offset after session split
# ---------------------------------------------------------------------------

class TestGatewayHistoryOffsetAfterSplit:
    """Verify that when the agent creates a new session during compression,
    the gateway uses history_offset=0 so all compressed messages are written
    to the JSONL transcript."""

    def test_history_offset_zero_on_session_split(self):
        """When agent.session_id differs from the original, history_offset must be 0."""
        # This tests the logic in gateway/run.py run_sync():
        # _session_was_split = agent.session_id != session_id
        # _effective_history_offset = 0 if _session_was_split else len(agent_history)

        original_session_id = "session-abc"
        agent_session_id = "session-compressed-xyz"  # Different = compression happened
        agent_history_len = 200

        # Simulate the gateway's offset calculation (post-fix)
        _session_was_split = (agent_session_id != original_session_id)
        _effective_history_offset = 0 if _session_was_split else agent_history_len

        assert _session_was_split is True
        assert _effective_history_offset == 0

    def test_history_offset_preserved_without_split(self):
        """When no compression happened, history_offset is the original length."""
        session_id = "session-abc"
        agent_session_id = "session-abc"  # Same = no compression
        agent_history_len = 200

        _session_was_split = (agent_session_id != session_id)
        _effective_history_offset = 0 if _session_was_split else agent_history_len

        assert _session_was_split is False
        assert _effective_history_offset == 200

    def test_new_messages_extraction_after_split(self):
        """After compression with offset=0, new_messages should be ALL agent messages."""
        # Simulates the gateway's new_messages calculation
        agent_messages = [
            {"role": "user", "content": "[CONTEXT COMPACTION] Summary..."},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": "new answer"},
        ]
        history_offset = 0  # After fix: 0 on session split

        new_messages = agent_messages[history_offset:] if len(agent_messages) > history_offset else []
        assert len(new_messages) == 5, (
            f"Expected all 5 messages with offset=0, got {len(new_messages)}"
        )

    def test_new_messages_empty_with_stale_offset(self):
        """Demonstrates the bug: stale offset produces empty new_messages."""
        agent_messages = [
            {"role": "user", "content": "summary"},
            {"role": "assistant", "content": "answer"},
        ]
        # Bug: offset is the pre-compression history length
        history_offset = 200

        new_messages = agent_messages[history_offset:] if len(agent_messages) > history_offset else []
        assert len(new_messages) == 0, (
            "Expected 0 messages with stale offset=200 (demonstrates the bug)"
        )


# ---------------------------------------------------------------------------
# Part 3: Background-candidate apply — shared atomic commit
# ---------------------------------------------------------------------------

class TestApplyPreparedCandidatePersistence:
    """apply_prepared_candidate() must reuse the exact persistence contract of
    the synchronous path: same SQLite lock, same in-place archive_and_compact
    (soft-archive + rebaseline) vs legacy rotation (child session), and no
    write of any kind when validation fails under the lock."""

    def _make_agent(self, session_db, session_id):
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
        agent._compression_feasibility_checked = True
        return agent

    def _make_messages(self, n=12):
        return [
            {"role": "user" if i % 2 == 0 else "assistant",
             "content": f"message {i}"}
            for i in range(n)
        ]

    def _make_candidate(self, messages, session_id, *, generation=1,
                        prefix_count=None):
        from agent.async_context_compression import (
            PreparedCompressionCandidate,
            canonical_prefix_digest,
        )
        if prefix_count is None:
            prefix_count = len(messages) - 2
        prepared = (
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            dict(messages[prefix_count - 1]),
        )
        return PreparedCompressionCandidate(
            session_id=session_id,
            generation=generation,
            prefix_message_count=prefix_count,
            prefix_digest=canonical_prefix_digest(messages, prefix_count),
            prepared_messages=prepared,
            source_prompt_tokens=180_000,
            created_at_monotonic=0.0,
            created_at_turn=1,
            used_fallback=False,
            summary_error=None,
        )

    def test_apply_in_place_archives_history_and_preserves_suffix(self):
        import tempfile as _tempfile

        from agent.conversation_compression import apply_prepared_candidate
        from hermes_state import SessionDB

        with _tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "apply-in-place"
            db.create_session(sid, "gateway", model="test/model")
            agent = self._make_agent(db, sid)
            agent.compression_in_place = True

            messages = self._make_messages()
            agent._flush_messages_to_session_db(messages, [])
            pre_rows = db.get_messages(sid)
            assert len(pre_rows) == len(messages)

            candidate = self._make_candidate(messages, sid)
            result = apply_prepared_candidate(agent, candidate, messages, "sys")

            assert result is not None
            new_messages, new_prompt = result
            assert isinstance(new_prompt, str) and new_prompt
            # Suffix after the frozen prefix survives as the SAME objects.
            suffix = new_messages[len(candidate.prepared_messages):]
            assert [m.get("content") for m in suffix if not str(m.get("content", "")).startswith("[")] \
                == [m["content"] for m in messages[candidate.prefix_message_count:]]
            assert all(
                a is b for a, b in zip(
                    new_messages[len(candidate.prepared_messages):
                                 len(candidate.prepared_messages) + 2],
                    messages[candidate.prefix_message_count:],
                )
            )

            # Same session id, in-place semantics.
            assert agent.session_id == sid
            assert agent._last_compaction_in_place is True
            assert agent._flushed_db_message_ids == set()

            # Non-destructive: history soft-archived, compacted set active.
            assert db.has_archived_messages(sid) is True
            active = db.get_messages(sid)
            assert len(active) == len(new_messages)
            assert any(
                "[CONTEXT COMPACTION]" in str(r.get("content", "")) for r in active
            )

            # Lock released after commit.
            assert db.get_compression_lock_holder(sid) is None
            db.close()

    def test_apply_rotation_creates_child_session(self):
        import tempfile as _tempfile

        from agent.conversation_compression import apply_prepared_candidate
        from hermes_state import SessionDB

        with _tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "apply-rotation"
            db.create_session(sid, "gateway", model="test/model")
            agent = self._make_agent(db, sid)
            agent.compression_in_place = False

            messages = self._make_messages()
            candidate = self._make_candidate(messages, sid)
            result = apply_prepared_candidate(agent, candidate, messages, "sys")

            assert result is not None
            assert agent.session_id != sid
            assert agent._last_compaction_in_place is False
            child = db.get_session(agent.session_id)
            assert child is not None
            assert child.get("parent_session_id") == sid
            # Lock keyed on the OLD session id is released.
            assert db.get_compression_lock_holder(sid) is None
            db.close()

    def test_apply_digest_mismatch_makes_no_writes(self):
        import tempfile as _tempfile

        from agent.conversation_compression import apply_prepared_candidate
        from hermes_state import SessionDB

        with _tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "apply-stale"
            db.create_session(sid, "gateway", model="test/model")
            agent = self._make_agent(db, sid)
            agent.compression_in_place = True

            messages = self._make_messages()
            agent._flush_messages_to_session_db(messages, [])
            candidate = self._make_candidate(messages, sid)
            # Prefix diverges after the candidate was frozen.
            messages[1]["content"] += " EDITED"

            result = apply_prepared_candidate(agent, candidate, messages, "sys")
            assert result is None
            assert agent.session_id == sid
            assert db.has_archived_messages(sid) is False
            rows = db.get_messages(sid)
            assert len(rows) == len(messages)
            assert db.get_compression_lock_holder(sid) is None
            db.close()

    def test_apply_steps_aside_when_lock_is_held(self):
        import tempfile as _tempfile

        from agent.conversation_compression import apply_prepared_candidate
        from hermes_state import SessionDB

        with _tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "apply-contended"
            db.create_session(sid, "gateway", model="test/model")
            agent = self._make_agent(db, sid)
            agent.compression_in_place = True

            messages = self._make_messages()
            candidate = self._make_candidate(messages, sid)

            assert db.try_acquire_compression_lock(sid, "other-holder") is True
            try:
                result = apply_prepared_candidate(
                    agent, candidate, messages, "sys"
                )
                assert result is None
                assert agent.session_id == sid
                assert db.has_archived_messages(sid) is False
                # The competing holder still owns the lock.
                assert db.get_compression_lock_holder(sid) == "other-holder"
            finally:
                db.release_compression_lock(sid, "other-holder")
            db.close()

    def test_sync_and_background_paths_commit_equivalently(self):
        """Equivalence gate: for a deterministic compressor output, the
        refactored synchronous path and the background apply path must leave
        the same final message list, the same session:compress event shape
        and the same active DB rows (in-place mode)."""
        import tempfile as _tempfile
        from unittest.mock import MagicMock

        from agent.async_context_compression import canonical_prefix_digest
        from agent.conversation_compression import (
            apply_prepared_candidate,
            compress_context,
        )
        from hermes_state import SessionDB

        base_messages = self._make_messages()
        prefix_count = len(base_messages) - 2
        prepared = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            dict(base_messages[prefix_count - 1]),
        ]

        def _run_sync(tmpdir):
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "equiv-sync"
            db.create_session(sid, "gateway", model="test/model")
            agent = self._make_agent(db, sid)
            agent.compression_in_place = True
            messages = [dict(m) for m in base_messages]
            agent._flush_messages_to_session_db(messages, [])
            events = []
            agent.event_callback = lambda name, payload: events.append(
                (name, payload)
            )

            compressor = MagicMock()
            compressor.compress.side_effect = lambda msgs, **kw: (
                [dict(m) for m in prepared] + msgs[prefix_count:]
            )
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            compressor._last_compression_made_progress = True
            compressor._last_summary_fallback_used = False
            compressor._last_aux_model_failure_model = None
            compressor._last_aux_model_failure_error = None
            agent.context_compressor = compressor

            result, _prompt = compress_context(
                agent, messages, "sys", approx_tokens=120_000
            )
            rows = db.get_messages(sid)
            db.close()
            return result, events, rows

        def _run_background(tmpdir):
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "equiv-sync"  # same id so event payloads match exactly
            db.create_session(sid, "gateway", model="test/model")
            agent = self._make_agent(db, sid)
            agent.compression_in_place = True
            messages = [dict(m) for m in base_messages]
            agent._flush_messages_to_session_db(messages, [])
            events = []
            agent.event_callback = lambda name, payload: events.append(
                (name, payload)
            )

            compressor = MagicMock()
            compressor.compression_count = 1
            compressor.adopt_prepared_state = lambda candidate: None
            agent.context_compressor = compressor

            from agent.async_context_compression import (
                PreparedCompressionCandidate,
            )
            candidate = PreparedCompressionCandidate(
                session_id=sid,
                generation=1,
                prefix_message_count=prefix_count,
                prefix_digest=canonical_prefix_digest(messages, prefix_count),
                prepared_messages=tuple(dict(m) for m in prepared),
                source_prompt_tokens=120_000,
                created_at_monotonic=0.0,
                created_at_turn=1,
                used_fallback=False,
                summary_error=None,
            )
            result = apply_prepared_candidate(agent, candidate, messages, "sys")
            assert result is not None
            rows = db.get_messages(sid)
            db.close()
            return result[0], events, rows

        with _tempfile.TemporaryDirectory() as d1:
            sync_list, sync_events, sync_rows = _run_sync(d1)
        with _tempfile.TemporaryDirectory() as d2:
            bg_list, bg_events, bg_rows = _run_background(d2)

        def _semantic(msgs):
            return [
                {k: v for k, v in m.items() if not k.startswith("_")}
                for m in msgs
            ]

        assert _semantic(sync_list) == _semantic(bg_list)

        sync_compress_events = [e for e in sync_events if e[0] == "session:compress"]
        bg_compress_events = [e for e in bg_events if e[0] == "session:compress"]
        assert len(sync_compress_events) == len(bg_compress_events) == 1
        assert sync_compress_events[0][1] == bg_compress_events[0][1]

        def _row_view(rows):
            return [(r.get("role"), r.get("content")) for r in rows]

        assert _row_view(sync_rows) == _row_view(bg_rows)
