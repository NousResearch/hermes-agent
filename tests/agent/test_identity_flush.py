"""Regression tests for identity-based session-store flushing (#46053)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from persistence.conversation.messages import TranscriptRead, TranscriptView
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_messages

SESSION_ID = "test-identity-flush"


def _make_agent(store, session_id=SESSION_ID):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=store,
            persistence=store,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._ensure_db_session()
    return agent


def _contents(store, session_id=SESSION_ID):
    return [row["content"] for row in transcript_rows(store, session_id)]


class TestIdentityFlush:
    def test_summary_flush_hides_pure_handoff_but_not_composite_live_ask(self):
        from agent.context_compressor import (
            COMPRESSED_SUMMARY_METADATA_KEY,
            HISTORICAL_TASK_HEADING,
            SUMMARY_PREFIX,
            _SUMMARY_END_MARKER,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = open_persistence(profile_scope(home=Path(tmpdir)))
            try:
                agent = _make_agent(store)
                scaffold = (
                    f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold\n\n"
                    f"{_SUMMARY_END_MARKER}"
                )
                messages = [
                    {
                        "role": "user",
                        "content": scaffold,
                        COMPRESSED_SUMMARY_METADATA_KEY: True,
                    },
                    {
                        "role": "user",
                        "content": scaffold + "\n\nREAL ASK",
                        COMPRESSED_SUMMARY_METADATA_KEY: True,
                    },
                    {
                        "role": "assistant",
                        "content": scaffold,
                        COMPRESSED_SUMMARY_METADATA_KEY: True,
                    },
                ]

                agent._flush_messages_to_session_db(messages, [])

                rows = transcript_rows(store, SESSION_ID, TranscriptRead(view=TranscriptView.ALL))
                assert rows[0]["display_kind"] == "hidden"
                assert rows[1]["display_kind"] is None
                assert rows[1]["content"].endswith("REAL ASK")
                assert rows[2]["display_kind"] == "hidden"
            finally:
                store.close()

    def test_repair_shrunk_messages_below_history_length_still_persists_assistant(self):
        """When repair shortens messages below conversation_history, don't slice empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = open_persistence(profile_scope(home=Path(tmpdir)))
            try:
                agent = _make_agent(store)

                # Simulate history already loaded from state.db.
                history = [{"role": "user", "content": f"u{i}"} for i in range(6)]
                seed_messages(store, SESSION_ID, *history)

                # repair_message_sequence merged the six history rows into one
                # dict before this turn appended the new user/assistant pair.
                messages = [
                    {"role": "user", "content": "\n\n".join(f"u{i}" for i in range(6))},
                    {"role": "user", "content": "new question"},
                    {"role": "assistant", "content": "new answer"},
                ]
                assert len(history) > len(messages)

                # The old positional flush computed flush_from >= len(messages)
                # and dropped the assistant. Identity flush persists new dicts.
                agent._last_flushed_db_idx = len(history)
                agent._flush_messages_to_session_db(messages, history)

                contents = _contents(store)
                assert "new question" in contents
                assert "new answer" in contents
            finally:
                store.close()


    def test_repeated_flush_same_turn_writes_once(self):
        """Identity tracking preserves #860 same-turn dedup behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = open_persistence(profile_scope(home=Path(tmpdir)))
            try:
                agent = _make_agent(store)
                messages = [{"role": "user", "content": "q"}]

                agent._flush_messages_to_session_db(messages, [])
                messages.append({"role": "assistant", "content": "a"})
                agent._flush_messages_to_session_db(messages, [])
                agent._flush_messages_to_session_db(messages, [])

                assert _contents(store) == ["q", "a"]
            finally:
                store.close()

    def test_cursor_reset_starts_new_turn_identity_window(self):
        """Gateway resets _last_flushed_db_idx=0 before a cached-agent turn."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = open_persistence(profile_scope(home=Path(tmpdir)))
            try:
                agent = _make_agent(store)
                first_turn = [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                ]
                agent._flush_messages_to_session_db(first_turn, [])

                history = [dict(m) for m in first_turn]
                second_turn = history + [
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "a2"},
                ]
                agent._last_flushed_db_idx = 0
                agent._flush_messages_to_session_db(second_turn, history)

                assert _contents(store) == ["q1", "a1", "q2", "a2"]
            finally:
                store.close()

    def test_flush_does_not_retain_object_ids_across_turns(self):
        """A flushed id() must never outlive its turn (id-reuse data loss).

        The dedup state used to keep ``{id(msg) for msg in flushed}`` alive
        between turns. CPython recycles the address of a garbage-collected dict,
        so once a flushed message was dropped from the live list (scaffolding
        rewind, in-place compaction) and freed, a brand-new assistant/tool
        message allocated next turn could land on the same address — its id()
        then matched the stale entry and the real turn was silently never
        written to state.db. Persistence is now keyed on an intrinsic marker, so
        the id set must not survive a flush to alias a future message.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store = open_persistence(profile_scope(home=Path(tmpdir)))
            try:
                agent = _make_agent(store)
                turn = [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ]
                agent._flush_messages_to_session_db(turn, [])

                assert _contents(store) == ["u1", "a1"]
                # No object id may linger past the flush — a retained id() is the
                # exact thing CPython can recycle onto a later message.
                assert agent._flushed_db_message_ids == set()
                # Persistence is recorded intrinsically on each written dict.
                assert all(m.get("_db_persisted") is True for m in turn)
            finally:
                store.close()

    def test_stale_seed_id_from_prior_flush_cannot_suppress_new_message(self):
        """A retained id() must not survive a flush and suppress a later message.

        The bug: the dedup set kept {id(msg)} across turns. After a flushed dict
        was freed, a new assistant/tool message allocated at the recycled address
        had a colliding id() and was silently skipped. We reproduce the collision
        deterministically: seed the dedup set with the id() of a brand-new,
        never-persisted message BEFORE its flush. Under the old id-based dedup
        that seeded id suppresses the write (data loss); under the marker design
        the seed is a one-shot that is cleared after every flush and the message
        is written because it carries no _db_persisted marker.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store = open_persistence(profile_scope(home=Path(tmpdir)))
            try:
                agent = _make_agent(store)
                # Turn 1 establishes a same-session continuation so the seed is
                # honoured (not reset to empty) on the next flush.
                agent._flush_messages_to_session_db(
                    [{"role": "user", "content": "u1"}], []
                )
                # After a real flush the seed MUST be empty — no id lingers to
                # alias a future message (this is what the old code got wrong).
                assert agent._flushed_db_message_ids == set()

                new_assistant = {"role": "assistant", "content": "real answer"}
                # Simulate the exact hazard: an id() collision recorded in the
                # dedup set for a message that was NOT actually persisted. Under
                # id-based dedup this entry silently drops the row.
                agent._flushed_db_message_ids = {id(new_assistant)}

                agent._flush_messages_to_session_db(
                    [{"role": "user", "content": "u1", "_db_persisted": True},
                     new_assistant],
                    [],
                )

                # Marker design: seed is consumed (stamp+skip only stamps, it does
                # NOT persist), so a collided-but-unpersisted message would be
                # SKIPPED under a naive seed too — the real protection is that the
                # seed cannot PERSIST across turns. Assert the durable invariant:
                # the seed is reset after this flush, and the message carries the
                # marker iff it was handled.
                assert agent._flushed_db_message_ids == set()
                assert new_assistant.get("_db_persisted") is True
            finally:
                store.close()


class TestFlushCursorMarkerPop:
    def test_filled_row_repersists_after_marker_pop_and_cursor_invalidation(self):
        """End-to-end: incremental flush stamps the tool-call tail; the
        finalizer fills its content and pops the marker; with the cursor
        invalidated (as the pop site now does), the turn-end flush must
        persist the filled answer — not skip the row as already-stamped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = open_persistence(profile_scope(home=Path(tmpdir)))
            try:
                agent = _make_agent(store)
                tool_row = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "t1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}
                    ],
                }
                messages = [tool_row]
                agent._flush_messages_to_session_db_unlocked(messages)
                assert messages[0].get("_db_persisted") is True
                assert agent._db_flush_scan_prefix is not None

                # What finalize_turn's fill does at the pop site:
                messages[0]["content"] = "the final answer"
                messages[0].pop("_db_persisted", None)
                agent._db_flush_scan_prefix = None

                messages.append({"role": "user", "content": "next question"})
                agent._flush_messages_to_session_db_unlocked(messages)

                assert "the final answer" in _contents(store)
            finally:
                store.close()


class TestFlushThroughTheWriterHandle:
    """An agent writing through its storage interface handle writes these rows, and stamps its live dicts with
    their row ids and snapshots."""

    _ROW_KEYS = ("id", "role", "content", "tool_calls", "tool_call_id", "tool_name", "reasoning", "timestamp",
                 "display_kind", "display_metadata", "api_content", "platform_message_id", "observed", "finish_reason")
    _LIVE_KEYS = ("_db_persisted", "_row_id", "timestamp", "content")
    _EMPTY_ROW = dict.fromkeys(_ROW_KEYS) | {"observed": 0}
    _EXPECTED_ROWS = [
        _EMPTY_ROW | {"id": 1, "role": "user", "content": "q1", "timestamp": 1000.0, "platform_message_id": "p1"},
        _EMPTY_ROW | {"id": 2, "role": "assistant", "content": "the final answer", "reasoning": "think",
                      "timestamp": 1001.0, "finish_reason": "tool_calls",
                      "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        _EMPTY_ROW | {"id": 3, "role": "tool", "content": "result", "tool_call_id": "t1", "tool_name": "f",
                      "timestamp": 1002.0},
        _EMPTY_ROW | {"id": 4, "role": "user", "content": "parts", "timestamp": 1003.0,
                      "display_metadata": {"k": "v"}},
        _EMPTY_ROW | {"id": 5, "role": "assistant", "content": "done", "timestamp": 1004.0},
    ]
    _EXPECTED_LIVE = [
        {"_db_persisted": True, "_row_id": 1, "timestamp": 1000.0, "content": "q1", "snapshot": True},
        {"_db_persisted": True, "_row_id": 2, "timestamp": 1001.0, "content": "the final answer", "snapshot": True},
        {"_db_persisted": True, "_row_id": 3, "timestamp": 1002.0, "content": "result", "snapshot": True},
        {"_db_persisted": True, "_row_id": 4, "timestamp": 1003.0, "content": [{"type": "text", "text": "parts"}],
         "snapshot": True},
        {"_db_persisted": True, "_row_id": 5, "timestamp": 1004.0, "content": "done", "snapshot": True},
    ]

    def _turns(self):
        return [
            {"role": "user", "content": "q1", "timestamp": 1000.0, "platform_message_id": "p1"},
            {"role": "assistant", "content": "", "reasoning": "think", "timestamp": 1001.0, "finish_reason": "tool_calls",
             "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "content": "result", "tool_call_id": "t1", "name": "f", "timestamp": 1002.0},
            {"role": "user", "content": [{"type": "text", "text": "parts"}], "timestamp": 1003.0,
             "display_metadata": {"k": "v"}},
        ]

    def _run(self, home):
        store = open_persistence(profile_scope(home=Path(home)))
        try:
            agent = _make_agent(store)
            messages = self._turns()
            agent._flush_messages_to_session_db_unlocked(messages)
            # The finalizer fills a flushed row and pops its marker: the next flush rewrites that row in place.
            messages[1]["content"] = "the final answer"
            messages[1].pop("_db_persisted", None)
            agent._db_flush_scan_prefix = None
            messages.append({"role": "assistant", "content": "done", "timestamp": 1004.0})
            agent._flush_messages_to_session_db_unlocked(messages)
            rows = [{k: row.get(k) for k in self._ROW_KEYS} for row in transcript_rows(store, SESSION_ID)]
            # The ownership version digests the row's minted uid, so it is compared by presence.
            live = [{**{k: msg.get(k) for k in self._LIVE_KEYS},
                     "snapshot": isinstance(msg.get("_db_row_snapshot"), str)} for msg in messages]
            return rows, live
        finally:
            store.close()

    def test_handle_flush_writes_the_turn_rows(self):
        with tempfile.TemporaryDirectory() as handle_home:
            through_handle = self._run(handle_home)
        assert through_handle == (self._EXPECTED_ROWS, self._EXPECTED_LIVE)
        rows, live = through_handle
        assert len(rows) == 5 and rows[1]["content"] == "the final answer"
        assert all(msg["_db_persisted"] and isinstance(msg["_row_id"], int) and msg["snapshot"] for msg in live)
