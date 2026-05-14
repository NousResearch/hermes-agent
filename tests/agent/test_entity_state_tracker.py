"""Tests for entity_state_tracker.py and its integration in context_compressor.py.

Covers:
  - EntityStateTracker unit tests
  - ContextCompressor._record_entities_from_messages() integration tests
  - Full compress() flow with entity conflict detection
"""

import json
import pytest
from unittest.mock import patch

from agent.entity_state_tracker import (
    EntityStateTracker,
    StateSnapshot,
    ConflictRecord,
)


# ============================================================================
# EntityStateTracker unit tests
# ============================================================================

class TestEntityStateTrackerRecord:
    """Tests for record() and basic state tracking."""

    def test_record_simple_value(self):
        tracker = EntityStateTracker()
        tracker.clear()
        tracker.record("var_x", 42)
        assert tracker.get_current_state("var_x") == 42

    def test_record_deep_copies_value(self):
        """record() must deep-copy so mutations don't affect tracked state."""
        tracker = EntityStateTracker()
        tracker.clear()
        mutable = {"a": [1, 2, 3]}
        tracker.record("key", mutable)
        mutable["a"].append(4)  # mutate after recording
        stored = tracker.get_current_state("key")
        assert stored["a"] == [1, 2, 3]  # unchanged

    def test_record_multiple_entities(self):
        tracker = EntityStateTracker()
        tracker.clear()
        tracker.record("file_path", "/tmp/out.txt")
        tracker.record("command", "pip install pandas")
        tracker.record("config", {"debug": True})
        assert tracker.get_current_state("file_path") == "/tmp/out.txt"
        assert tracker.get_current_state("command") == "pip install pandas"
        assert tracker.get_current_state("config") == {"debug": True}

    def test_record_appends_to_history(self):
        tracker = EntityStateTracker()
        tracker.clear()
        tracker.record("x", 1)
        tracker.record("x", 2)
        history = tracker.get_history("x")
        assert len(history) == 2
        assert history[0].value == 1
        assert history[1].value == 2

    def test_record_overwrites_current_state(self):
        tracker = EntityStateTracker()
        tracker.clear()
        tracker.record("x", 1)
        tracker.record("x", 99)
        assert tracker.get_current_state("x") == 99


class TestEntityStateTrackerSnapshotAndConflicts:
    """Tests for snapshot_all(), check_conflicts(), has_conflicts()."""

    def test_snapshot_all_freezes_current_state(self):
        tracker = EntityStateTracker()
        tracker.clear()
        tracker.record("a", 1)
        tracker.record("b", 2)
        tracker.snapshot_all()
        # Snapshot should contain what was recorded
        assert tracker._pre_compression_snapshot == {"a": 1, "b": 2}
        # Changing tracked state after snapshot should NOT affect snapshot
        tracker.record("a", 999)
        assert tracker._pre_compression_snapshot["a"] == 1

    def test_no_conflicts_when_nothing_changed(self):
        tracker = EntityStateTracker()
        tracker.clear()
        tracker.record("a", 1)
        tracker.record("b", "hello")
        tracker.snapshot_all()
        # Re-record same values (simulating post-compression recording)
        tracker._entity_state.clear()
        tracker.record("a", 1)
        tracker.record("b", "hello")
        conflicts = tracker.check_conflicts()
        assert len(conflicts) == 0
        assert tracker.has_conflicts() is False

    def test_conflict_when_entity_lost(self):
        """Entity recorded pre-compression but not re-recorded post-compression."""
        tracker = EntityStateTracker()
        tracker.clear()
        tracker.record("file.write:/x.py", {"path": "/x.py", "content_len": 50})
        tracker.record("tool_result:call_abc", {"call_id": "call_abc", "content_hash": "deadbeef"})
        tracker.snapshot_all()
        tracker._entity_state.clear()
        # Only re-record one entity — the other is lost
        tracker.record("file.write:/x.py", {"path": "/x.py", "content_len": 50})
        conflicts = tracker.check_conflicts()
        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.entity_key == "tool_result:call_abc"
        assert "-> None" in conflict.warning

    def test_conflict_when_value_changed(self):
        tracker = EntityStateTracker()
        tracker.clear()
        tracker.record("file.write:/x.py", {"path": "/x.py", "content_len": 100})
        tracker.snapshot_all()
        tracker._entity_state.clear()
        # Re-record with different value (content truncated)
        tracker.record("file.write:/x.py", {"path": "/x.py", "content_len": 10})
        conflicts = tracker.check_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].entity_key == "file.write:/x.py"

    def test_multiple_conflicts(self):
        tracker = EntityStateTracker()
        tracker.clear()
        for i in range(5):
            tracker.record(f"entity_{i}", i)
        tracker.snapshot_all()
        tracker._entity_state.clear()
        # Only keep entities 0 and 2
        tracker.record("entity_0", 0)
        tracker.record("entity_2", 2)
        conflicts = tracker.check_conflicts()
        assert len(conflicts) == 3  # 1, 3, 4 lost
        lost_keys = {c.entity_key for c in conflicts}
        assert lost_keys == {"entity_1", "entity_3", "entity_4"}

    def test_empty_pre_snapshot_no_conflicts(self):
        """If nothing was recorded before snapshot, no conflicts."""
        tracker = EntityStateTracker()
        tracker.clear()
        tracker.snapshot_all()
        tracker.record("late", "value")  # recorded after snapshot
        conflicts = tracker.check_conflicts()
        assert len(conflicts) == 0


class TestEntityStateTrackerWarnings:
    """Tests for get_warnings() and inject_warnings_into_summary()."""

    def test_get_warnings_empty(self):
        tracker = EntityStateTracker()
        tracker.clear()
        assert tracker.get_warnings() == []

    def test_get_warnings_after_conflicts(self):
        tracker = EntityStateTracker()
        tracker.clear()
        tracker.record("key1", "val1")
        tracker.record("key2", "val2")
        tracker.snapshot_all()
        tracker._entity_state.clear()
        # key1 survived, key2 lost
        tracker.record("key1", "val1")
        tracker.check_conflicts()
        warnings = tracker.get_warnings()
        assert len(warnings) == 1
        assert "key2" in warnings[0]

    def test_inject_warnings_into_summary_nonempty(self):
        tracker = EntityStateTracker()
        tracker.clear()
        tracker.record("x", 1)
        tracker.snapshot_all()
        tracker._entity_state.clear()
        tracker.check_conflicts()
        summary = tracker.inject_warnings_into_summary()
        assert "[Entity State Conflict Detected]" in summary
        assert "x" in summary
        assert "-> None" in summary

    def test_inject_warnings_into_summary_empty(self):
        tracker = EntityStateTracker()
        tracker.clear()
        assert tracker.inject_warnings_into_summary() == ""


class TestEntityStateTrackerHistory:
    """Tests for get_history() and get_current_state()."""

    def test_get_history_all(self):
        tracker = EntityStateTracker()
        tracker.clear()
        tracker.record("a", 1)
        tracker.record("b", 2)
        tracker.record("a", 3)
        history = tracker.get_history()
        assert len(history) == 3

    def test_get_history_filtered(self):
        tracker = EntityStateTracker()
        tracker.clear()
        tracker.record("a", 1)
        tracker.record("b", 2)
        history = tracker.get_history("a")
        assert len(history) == 1
        assert history[0].value == 1

    def test_get_current_state_missing_returns_none(self):
        tracker = EntityStateTracker()
        tracker.clear()
        assert tracker.get_current_state("nonexistent") is None


class TestEntityStateTrackerClear:
    """Tests for clear() method."""

    def test_clear_resets_all_state(self):
        tracker = EntityStateTracker()
        tracker.record("x", 42)
        tracker.snapshot_all()
        tracker.check_conflicts()
        tracker.clear()
        assert tracker._entity_state == {}
        assert tracker._entity_history == []
        assert tracker._pre_compression_snapshot == {}
        assert tracker._conflict_log == []


# ============================================================================
# ContextCompressor._record_entities_from_messages integration tests
# ============================================================================

class TestRecordEntitiesFromMessages:
    """Tests for _record_entities_from_messages() on ContextCompressor."""

    @pytest.fixture()
    def compressor(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            from agent.context_compressor import ContextCompressor
            c = ContextCompressor(
                model="test/model",
                quiet_mode=True,
            )
            return c

    def test_records_write_file_path(self, compressor):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "write_file", "arguments": json.dumps({
                    "path": "/home/user/output.py", "content": "print('hello world')"
                })}}
            ]},
        ]
        compressor._entity_tracker.clear()
        compressor._record_entities_from_messages(msgs)
        state = compressor._entity_tracker.get_current_state("file.write:/home/user/output.py")
        assert state is not None
        assert state["path"] == "/home/user/output.py"
        assert state["content_len"] > 0

    def test_records_read_file_path(self, compressor):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": json.dumps({
                    "path": "/etc/config.yaml", "offset": 10
                })}}
            ]},
        ]
        compressor._entity_tracker.clear()
        compressor._record_entities_from_messages(msgs)
        state = compressor._entity_tracker.get_current_state("file.read:/etc/config.yaml")
        assert state is not None
        assert state["offset"] == 10

    def test_records_patch_path(self, compressor):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "patch", "arguments": json.dumps({
                    "path": "/src/main.py", "mode": "replace"
                })}}
            ]},
        ]
        compressor._entity_tracker.clear()
        compressor._record_entities_from_messages(msgs)
        state = compressor._entity_tracker.get_current_state("file.patch:/src/main.py")
        assert state is not None
        assert state["mode"] == "replace"

    def test_records_terminal_command(self, compressor):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "terminal", "arguments": json.dumps({
                    "command": "find / -name '*.py' | head -20"
                })}}
            ]},
        ]
        compressor._entity_tracker.clear()
        compressor._record_entities_from_messages(msgs)
        # Find the terminal entity by checking all keys
        terminal_keys = [k for k in compressor._entity_tracker._entity_state
                         if k.startswith("terminal:")]
        assert len(terminal_keys) == 1
        state = compressor._entity_tracker.get_current_state(terminal_keys[0])
        assert "find" in state["command"]

    def test_records_search_files(self, compressor):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "search_files", "arguments": json.dumps({
                    "pattern": "entity_state", "path": "/home/agent"
                })}}
            ]},
        ]
        compressor._entity_tracker.clear()
        compressor._record_entities_from_messages(msgs)
        state = compressor._entity_tracker.get_current_state("search:entity_state@/home/agent")
        assert state is not None
        assert state["pattern"] == "entity_state"

    def test_records_web_search(self, compressor):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "web_search", "arguments": json.dumps({
                    "query": "python context compression"
                })}}
            ]},
        ]
        compressor._entity_tracker.clear()
        compressor._record_entities_from_messages(msgs)
        state = compressor._entity_tracker.get_current_state("web_search:python context compression")
        assert state is not None

    def test_records_browser_navigate(self, compressor):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "browser_navigate", "arguments": json.dumps({
                    "url": "https://example.com/docs"
                })}}
            ]},
        ]
        compressor._entity_tracker.clear()
        compressor._record_entities_from_messages(msgs)
        state = compressor._entity_tracker.get_current_state("browser:https://example.com/docs")
        assert state is not None

    def test_records_delegate_task(self, compressor):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "delegate_task", "arguments": json.dumps({
                    "goal": "Research Python memory optimization"
                })}}
            ]},
        ]
        compressor._entity_tracker.clear()
        compressor._record_entities_from_messages(msgs)
        goal_keys = [k for k in compressor._entity_tracker._entity_state
                     if k.startswith("goal:")]
        assert len(goal_keys) == 1

    def test_records_tool_result(self, compressor):
        msgs = [
            {"role": "tool", "tool_call_id": "call_test123", "content": '{"exit_code": 0, "output": "success"}'},
        ]
        compressor._entity_tracker.clear()
        compressor._record_entities_from_messages(msgs)
        state = compressor._entity_tracker.get_current_state("tool_result:call_test123")
        assert state is not None
        assert state["call_id"] == "call_test123"
        assert state["content_len"] > 0
        assert state["is_pruned"] is False

    def test_detects_pruned_tool_result(self, compressor):
        msgs = [
            {"role": "tool", "tool_call_id": "call_pruned",
             "content": "[Old tool output cleared to save context space]"},
        ]
        compressor._entity_tracker.clear()
        compressor._record_entities_from_messages(msgs)
        state = compressor._entity_tracker.get_current_state("tool_result:call_pruned")
        assert state["is_pruned"] is True

    def test_ignores_non_dict_tool_calls(self, compressor):
        """Should not crash on non-dict tool call entries."""
        msgs = [
            {"role": "assistant", "tool_calls": ["not_a_dict"]},
        ]
        compressor._entity_tracker.clear()
        compressor._record_entities_from_messages(msgs)
        # No entities should be recorded
        assert compressor._entity_tracker._entity_state == {}

    def test_handles_invalid_json_args(self, compressor):
        """Should skip tool calls with unparseable JSON args."""
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "write_file", "arguments": "not valid json!!!"}}
            ]},
        ]
        compressor._entity_tracker.clear()
        compressor._record_entities_from_messages(msgs)
        assert compressor._entity_tracker._entity_state == {}

    def test_multiple_tool_calls_in_one_message(self, compressor):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "write_file", "arguments": json.dumps({
                    "path": "/tmp/a.py", "content": "1"
                })}},
                {"id": "c2", "function": {"name": "read_file", "arguments": json.dumps({
                    "path": "/tmp/b.py"
                })}},
            ]},
        ]
        compressor._entity_tracker.clear()
        compressor._record_entities_from_messages(msgs)
        state_a = compressor._entity_tracker.get_current_state("file.write:/tmp/a.py")
        state_b = compressor._entity_tracker.get_current_state("file.read:/tmp/b.py")
        assert state_a is not None
        assert state_b is not None


# ============================================================================
# Full compress() flow with entity conflict detection
# ============================================================================

class TestCompressEntityConflictDetection:
    """End-to-end tests verifying entity tracking during compress()."""

    @pytest.fixture()
    def compressor(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            from agent.context_compressor import ContextCompressor
            c = ContextCompressor(
                model="test/model",
                threshold_percent=0.85,
                protect_first_n=2,
                protect_last_n=3,
                quiet_mode=True,
            )
            # Must set token count above threshold for compression to trigger
            c.last_prompt_tokens = 90000
            return c

    FAKE_SUMMARY = "[CONTEXT COMPACTION] Summary of earlier turns."

    def test_entities_in_middle_lost(self, compressor):
        """Tool calls and results in the middle region should be tracked
        and flagged as lost after compression."""
        msgs = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Write script."},
            {"role": "assistant", "tool_calls": [
                {"id": "ca", "function": {"name": "write_file", "arguments": json.dumps({
                    "path": "/tmp/s.py", "content": "p"})}}
            ]},
            {"role": "tool", "tool_call_id": "ca", "content": '{"ok":true}'},
            {"role": "user", "content": "Now run it."},
            {"role": "assistant", "tool_calls": [
                {"id": "cb", "function": {"name": "terminal", "arguments": json.dumps({
                    "command": "python /tmp/s.py"})}}
            ]},
            {"role": "tool", "tool_call_id": "cb",
             "content": '{"exit_code":0,"output":"hello"}'},
            {"role": "assistant", "content": "Script ran."},
            {"role": "user", "content": "Tail q."},
            {"role": "assistant", "content": "Tail a."},
            {"role": "user", "content": "Tail q2."},
        ]
        with patch.object(compressor, "_generate_summary", return_value=self.FAKE_SUMMARY):
            result = compressor.compress(msgs)

        # Compression must actually reduce the message count
        assert compressor.compression_count >= 1
        assert len(result) < len(msgs)

        conflicts = compressor._entity_tracker._conflict_log
        assert len(conflicts) >= 1, f"Expected conflicts, got {len(conflicts)}"
        lost_keys = {c.entity_key for c in conflicts}
        # tool_result in the middle should be flagged as lost
        assert "tool_result:cb" in lost_keys, f"tool_result:cb not in {lost_keys}"

    def test_entities_in_tail_survive(self, compressor):
        """Entities in the protected tail should survive compression."""
        msgs = [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "Start."},
            {"role": "assistant", "tool_calls": [
                {"id": "cm", "function": {"name": "write_file", "arguments": json.dumps({
                    "path": "/tmp/mid.py", "content": "x=1"})}}
            ]},
            {"role": "tool", "tool_call_id": "cm", "content": "ok"},
            {"role": "user", "content": "Mid user."},
            {"role": "assistant", "content": "Mid assistant."},
            # Tail: these should survive
            {"role": "assistant", "tool_calls": [
                {"id": "ct", "function": {"name": "read_file", "arguments": json.dumps({
                    "path": "/tmp/tail.py"})}}
            ]},
            {"role": "tool", "tool_call_id": "ct", "content": "x=1"},
            {"role": "user", "content": "Tail 1."},
            {"role": "assistant", "content": "Tail 2."},
        ]
        with patch.object(compressor, "_generate_summary", return_value=self.FAKE_SUMMARY):
            compressor.compress(msgs)

        # read_file in tail should still be recorded
        state = compressor._entity_tracker.get_current_state("file.read:/tmp/tail.py")
        assert state is not None, "Tail entity should survive compression"

    def test_no_false_conflicts_text_only(self, compressor):
        """Text-only conversations should produce no false conflicts."""
        # Reduce protect boundaries so compression triggers with fewer messages
        compressor.protect_first_n = 2
        compressor.protect_last_n = 2
        msgs = [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "How?"},
            {"role": "assistant", "content": "Good."},
            {"role": "user", "content": "Ok."},
            {"role": "assistant", "content": "Bye."},
        ]
        with patch.object(compressor, "_generate_summary", return_value=self.FAKE_SUMMARY):
            compressor.compress(msgs)

        assert len(compressor._entity_tracker._conflict_log) == 0, (
            "Text-only messages should not produce conflicts"
        )

    def test_entity_tracker_reset_between_compressions(self, compressor):
        """Each compress() call starts fresh with entity tracking."""
        msgs_a = [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "Run."},
            {"role": "assistant", "content": "Working."},
            {"role": "user", "content": "Do this."},
            {"role": "assistant", "tool_calls": [
                {"id": "ca1", "function": {"name": "terminal", "arguments": json.dumps({
                    "command": "ls /tmp"})}}
            ]},
            {"role": "tool", "tool_call_id": "ca1", "content": "a.txt"},
            {"role": "user", "content": "Got it."},
            {"role": "assistant", "content": "Good."},
            {"role": "user", "content": "Tail 1."},
            {"role": "assistant", "content": "Tail 2."},
            {"role": "user", "content": "Tail 3."},
        ]
        with patch.object(compressor, "_generate_summary", return_value=self.FAKE_SUMMARY):
            compressor.compress(msgs_a)
        first_conflicts = len(compressor._entity_tracker._conflict_log)
        assert first_conflicts >= 1, f"First compression should have conflicts, got {first_conflicts}"

        # Second compression — different tool calls
        msgs_b = [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "Run2."},
            {"role": "assistant", "content": "Working."},
            {"role": "user", "content": "Do this."},
            {"role": "assistant", "tool_calls": [
                {"id": "cb1", "function": {"name": "write_file", "arguments": json.dumps({
                    "path": "/z.py", "content": "p"})}}
            ]},
            {"role": "tool", "tool_call_id": "cb1", "content": "ok"},
            {"role": "user", "content": "Got it."},
            {"role": "assistant", "content": "Good."},
            {"role": "user", "content": "Tail 1."},
            {"role": "assistant", "content": "Tail 2."},
            {"role": "user", "content": "Tail 3."},
        ]
        with patch.object(compressor, "_generate_summary", return_value=self.FAKE_SUMMARY):
            compressor.compress(msgs_b)
        second_conflicts = len(compressor._entity_tracker._conflict_log)
        # Each compress() call resets the tracker via clear(), so conflicts
        # are detected independently per call, not accumulated.
        assert second_conflicts >= 1, (
            f"Each compression should independently detect conflicts. "
            f"First={first_conflicts}, second={second_conflicts}"
        )

    def test_multiple_entity_types_detected(self, compressor):
        """Multiple tool call types (write, read, patch) in the middle should
        all be tracked and flagged."""
        msgs = [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "Run."},
            {"role": "assistant", "content": "I'll do that."},
            {"role": "user", "content": "Yes please."},
            {"role": "assistant", "tool_calls": [
                {"id": "c_w", "function": {"name": "write_file", "arguments": json.dumps({
                    "path": "/f1.py", "content": "x"})}},
                {"id": "c_r", "function": {"name": "read_file", "arguments": json.dumps({
                    "path": "/f2.py"})}},
                {"id": "c_p", "function": {"name": "patch", "arguments": json.dumps({
                    "path": "/f3.py"})}},
            ]},
            {"role": "tool", "tool_call_id": "c_w", "content": "ok"},
            {"role": "tool", "tool_call_id": "c_r", "content": "content"},
            {"role": "tool", "tool_call_id": "c_p", "content": "diff"},
            {"role": "user", "content": "Mid done."},
            {"role": "assistant", "content": "All done."},
            {"role": "user", "content": "T1."},
            {"role": "assistant", "content": "T2."},
            {"role": "user", "content": "T3."},
        ]
        with patch.object(compressor, "_generate_summary", return_value=self.FAKE_SUMMARY):
            compressor.compress(msgs)

        conflicts = compressor._entity_tracker._conflict_log
        assert len(conflicts) >= 3, (
            f"Expected >=3 conflicts (3 tool_results lost), got {len(conflicts)}"
        )
        lost_keys = {c.entity_key for c in conflicts}
        assert "tool_result:c_w" in lost_keys
        assert "tool_result:c_r" in lost_keys
        assert "tool_result:c_p" in lost_keys
