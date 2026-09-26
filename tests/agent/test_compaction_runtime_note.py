"""Runtime-only CLI prefixes must not become renewed requests at compaction."""
import copy
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor, _SUMMARY_END_MARKER
from agent.conversation_compression import compress_context
from hermes_state import SessionDB

NOTE = "[Note: model was just switched from old to new via test. Adjust your self-identification accordingly.]"
TASK = "Continue auditing the local fixture and report the result."


@pytest.fixture
def case(tmp_path):
    from run_agent import AIAgent
    db = SessionDB(tmp_path / "state.db")
    db.create_session("note-test", source="cli", model="test/model")
    agent = AIAgent(api_key="test-key", base_url="http://127.0.0.1:1/v1", model="test/model",
                    quiet_mode=True, session_db=db, session_id="note-test",
                    skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    agent._compression_feasibility_checked = True
    agent._cached_system_prompt = "Stable fixture system."
    agent.compression_in_place = True
    agent.context_compressor = ContextCompressor("test/model", config_context_length=100000,
                                                protect_first_n=2, protect_last_n=2, quiet_mode=True)
    agent.context_compressor.tail_token_budget = 500
    agent._persist_user_message_idx = 1
    agent._persist_user_message_override = TASK
    messages = [{"role": "system", "content": agent._cached_system_prompt},
                {"role": "user", "content": NOTE + "\n\n" + TASK,
                 "api_content": NOTE + "\n\n" + TASK}]
    for i in range(30):
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function",
             "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "fixture output " * 200},
        ])
    yield agent, messages, db
    db.close()


def test_compaction_does_not_renew_model_switch_note(case):
    agent, messages, db = case
    response = MagicMock()
    response.choices[0].message.content = "## Summary\nFixture work is ongoing."
    with patch("agent.context_compressor.call_llm", return_value=response) as llm:
        result, _ = compress_context(agent, messages, agent._cached_system_prompt,
                                     approx_tokens=200000, force=True)
    assert llm.called
    boundary = next(i for i, m in enumerate(result) if _SUMMARY_END_MARKER in str(m.get("content")))
    replay = str(result[boundary]["content"]).split(_SUMMARY_END_MARKER)[-1] + str(result[boundary+1:])
    assert TASK in replay
    assert NOTE not in replay
    assert NOTE not in str(llm.call_args), "runtime metadata must not enter summary input"
    # Continue the same task through a second real boundary, as in the report.
    result.extend(copy.deepcopy(messages[2:]))
    with patch("agent.context_compressor.call_llm", return_value=response) as second:
        again, _ = compress_context(agent, result, agent._cached_system_prompt,
                                    approx_tokens=200000, force=True)
    assert second.called
    assert NOTE not in str(again)
    assert NOTE not in str(second.call_args)
    rows = db.get_messages(agent.session_id)
    assert NOTE not in str(rows), "active durable handoff must not restore the API-only note"


@pytest.mark.parametrize("outcome", ["same", "copy", "raise", "empty", "aborted", "markers"])
def test_noncompaction_preserves_original_wire_bytes(case, outcome):
    agent, messages, db = case
    original = copy.deepcopy(messages)
    def run(rows, **kwargs):
        if outcome == "raise":
            raise RuntimeError("fixture failure")
        if outcome == "empty":
            return []
        if outcome == "aborted":
            agent.context_compressor._last_compress_aborted = True
        if outcome == "markers":
            return [dict(m, _db_persisted=True) for m in rows]
        return rows if outcome == "same" else copy.deepcopy(rows)
    with patch.object(agent.context_compressor, "compress", side_effect=run):
        if outcome == "raise":
            with pytest.raises(RuntimeError, match="fixture failure"):
                compress_context(agent, messages, agent._cached_system_prompt, force=True)
        else:
            result, _ = compress_context(agent, messages, agent._cached_system_prompt, force=True)
            assert result == original
    assert messages == original


def test_summary_exception_fallback_keeps_clean_request(case):
    agent, messages, db = case
    with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("fixture summary failure")) as llm:
        result, _ = compress_context(agent, messages, agent._cached_system_prompt,
                                     approx_tokens=200000, force=True)
    assert llm.called
    assert NOTE not in str(llm.call_args)
    if result == messages:
        assert NOTE in result[1]["content"]  # aborted summary preserves the live prefix
    else:
        assert NOTE not in str(result)
        assert TASK in str(result)


def test_commit_failure_restores_original_note_and_sidecar(case):
    agent, messages, db = case
    original = copy.deepcopy(messages)
    response = MagicMock()
    response.choices[0].message.content = "## Summary\nWork remains."
    with patch("agent.context_compressor.call_llm", return_value=response), patch.object(
        db, "archive_and_compact", side_effect=RuntimeError("fixture commit failure")
    ) as commit:
        result, _ = compress_context(agent, messages, agent._cached_system_prompt,
                                     approx_tokens=200000, force=True)
    assert commit.called
    assert result == original
    assert messages == original


def test_clean_finalized_content_drops_runtime_sidecar_at_boundary():
    from types import SimpleNamespace
    from agent.conversation_compression import _summary_user_content
    from agent.turn_context import substitute_api_content
    agent = SimpleNamespace(_persist_user_message_idx=0, _persist_user_message_override=TASK)
    rows = [{"role": "user", "content": TASK, "api_content": NOTE + TASK}]
    projected = _summary_user_content(agent, rows)
    wire = dict(projected[0])
    substitute_api_content(wire)
    assert wire["content"] == TASK
    assert rows[0]["api_content"] == NOTE + TASK


@pytest.mark.parametrize("override", [None, "", NOTE + " quoted deliberately", [{"type": "text", "text": TASK}]])
def test_projection_uses_caller_override_not_note_syntax(override):
    from types import SimpleNamespace
    from agent.conversation_compression import _summary_user_content
    agent = SimpleNamespace(_persist_user_message_idx=1, _persist_user_message_override=override)
    rows = [{"role": "system", "content": "fixed"}, {"role": "user", "content": NOTE + TASK, "api_content": NOTE + TASK}]
    original = copy.deepcopy(rows)
    projected = _summary_user_content(agent, rows)
    assert projected[1]["content"] == (rows[1]["content"] if override is None else override)
    assert rows == original
    if override is not None:
        assert "api_content" not in projected[1]
    if isinstance(override, list):
        projected[1]["content"].append({"type": "text", "text": "changed"})
        assert len(override) == 1
