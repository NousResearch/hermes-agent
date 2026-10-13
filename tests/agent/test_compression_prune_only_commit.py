"""Pruning that empties the summarizer window must survive the compression boundary."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX
from agent.conversation_compression import compress_context
from agent.model_metadata import estimate_messages_tokens_rough


class _Agent:
    def __init__(self, compressor):
        self.context_compressor = compressor
        self.session_id = "prune-only-test"
        self.platform = "cli"
        self.model = "test/model"
        self.provider = "test-provider"
        self.tools = []
        self._compression_feasibility_checked = True
        self.compression_in_place = False
        self._memory_manager = self._session_db = self._cached_system_prompt = None
        self._todo_store = SimpleNamespace(format_for_injection=lambda: "")

    def _emit_status(self, message):
        pass

    _emit_warning = _emit_status
    commit_memory_session = _emit_status

    def _invalidate_system_prompt(self):
        self._cached_system_prompt = None

    def _build_system_prompt(self, message):
        return message


def _messages():
    messages = [
        {"role": "user", "content": SUMMARY_PREFIX + "\nPreviously verified facts.", "_compressed_summary": True},
        {"role": "assistant", "content": "Discovery is complete."},
        {"role": "user", "content": "Proceed, and verify all results at the end."},
    ]
    for i in range(20):
        messages.extend([
            {"role": "assistant", "content": f"Checking file {i}.", "tool_calls": [{
                "id": f"call-{i}", "type": "function", "function": {
                    "name": "read_file", "arguments": '{"path":"audit.txt"}',
                },
            }]},
            {"role": "tool", "tool_call_id": f"call-{i}",
             "content": "Archived audit output.\n" * 1500 if i < 8 else f"Verified file {i}."},
        ])
    return messages


def test_pruning_only_handoff_window_commits_without_losing_live_work():
    compressor = ContextCompressor(
        model="test/model", config_context_length=65_536, threshold_tokens_cap=32_768, quiet_mode=True,
    )
    agent = _Agent(compressor)
    messages = _messages()
    before = copy.deepcopy(messages)
    with patch.object(compressor, "_generate_summary") as summarize:
        result, _ = compress_context(agent, messages, "system", approx_tokens=64_562)

    summarize.assert_not_called()
    assert estimate_messages_tokens_rough(result) < estimate_messages_tokens_rough(before) / 2
    assert messages == before
    assert all(row in result for row in before[:3])
    assert result[-2:] == before[-2:]
    assert [row.get("tool_calls") for row in result if row.get("tool_calls")] == [
        row["tool_calls"] for row in before if row.get("tool_calls")
    ]
    assert compressor.awaiting_real_usage_after_compression
    assert compressor._verify_compaction_cleared_threshold
    assert compressor._structural_no_op_backoff_until == 0


@pytest.mark.parametrize("in_place,fail_commit", [(True, False), (False, False), (True, True)])
def test_prune_candidate_survives_database_reload(tmp_path, in_place, fail_commit):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.create_session("prune-db", source="cli")
    agent = AIAgent(
        api_key="test-key", base_url="https://example.invalid/v1", model="test/model",
        quiet_mode=True, session_db=db, session_id="prune-db",
        skip_context_files=True, skip_memory=True,
    )
    agent._compression_feasibility_checked = True
    agent.compression_in_place = in_place
    agent.context_compressor = ContextCompressor(
        model="test/model", config_context_length=65_536, threshold_tokens_cap=32_768, quiet_mode=True,
    )
    messages = _messages()
    agent._flush_messages_to_session_db(messages)
    persisted_before = db.get_messages_as_conversation(agent.session_id)
    assert len(persisted_before) == len(messages)
    commit = patch.object(db, "archive_and_compact", side_effect=RuntimeError("database is locked")) if fail_commit else nullcontext()
    with commit, patch.object(agent.context_compressor, "_generate_summary") as summarize:
        result, _ = compress_context(agent, messages, "system", approx_tokens=74_000)
    summarize.assert_not_called()
    reopened = SessionDB(db_path=path)
    try:
        restored = reopened.get_messages_as_conversation(agent.session_id)
        assert [m.get("content") for m in restored] == [m.get("content") for m in result]
        if fail_commit:
            assert [m.get("content") for m in restored] == [m.get("content") for m in persisted_before]
            assert not agent.context_compressor._verify_compaction_cleared_threshold
        else:
            assert estimate_messages_tokens_rough(restored) < estimate_messages_tokens_rough(messages) / 2
        assert len(restored) == len(messages)
    finally:
        reopened.close()
        db.close()


@pytest.mark.parametrize("has_summary", [True, False])
def test_structural_exits_keep_pruning_and_second_pass_is_noop(has_summary):
    compressor = ContextCompressor(model="test/model", config_context_length=65_536, quiet_mode=True)
    messages = _messages()
    if not has_summary:
        messages[0] = {"role": "user", "content": "Earlier request."}
    original = copy.deepcopy(messages)
    boundary = nullcontext() if has_summary else patch.object(compressor, "_find_tail_cut_by_tokens", return_value=0)
    with boundary, patch.object(compressor, "_generate_summary", side_effect=AssertionError("Unexpected summary call")) as summarize:
        result = compressor.compress(messages, current_tokens=74_000)
        second = compressor.compress(result, current_tokens=10_000)
    summarize.assert_not_called()
    assert messages == original
    assert estimate_messages_tokens_rough(result) < estimate_messages_tokens_rough(messages) / 2
    assert second == result
    assert not compressor._last_compression_made_progress
    assert compressor._structural_no_op_backoff_until > 0
    assert compressor._last_compression_telemetry["failure_class"] == (
        "empty_post_handoff_window" if has_summary else "no_compressible_window"
    )
