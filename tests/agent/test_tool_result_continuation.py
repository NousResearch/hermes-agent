"""Durable approval recovery continues from a tool result, not a new user turn."""

from unittest.mock import MagicMock

import pytest

import agent.turn_context as turn_context
from hermes_state import SessionDB
from run_agent import AIAgent


@pytest.fixture
def recovered_agent(tmp_path):
    db = SessionDB(tmp_path / "sessions.db")
    session_id = "recovered-tool"
    db.create_session(session_id, source="api_server", system_prompt="SYSTEM")
    db.append_message(session_id, "user", "Run the approved tool.")
    db.append_message(session_id, "assistant", tool_calls=[{
        "id": "approved-call", "type": "function",
        "function": {"name": "terminal", "arguments": '{"command":"probe"}'},
    }])
    db.append_message(session_id, "tool", "tool settled", "terminal", None, "approved-call")
    agent = AIAgent(
        api_key="test-key", base_url="https://openrouter.ai/api/v1",
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        enabled_toolsets=[], session_db=db, session_id=session_id,
    )
    agent._session_db_created = True
    agent._cached_system_prompt = "SYSTEM"
    agent._skip_mcp_refresh = True
    try:
        yield agent, db, session_id
    finally:
        agent.close()
        db.close()


def test_continuation_preserves_durable_and_model_transcript(recovered_agent, monkeypatch):
    agent, db, session_id = recovered_agent
    history = db.get_messages_as_conversation(session_id)
    forbidden = MagicMock(side_effect=AssertionError("continuation started a user turn"))
    for name in ("_stage_turn_user_message", "_tick_memory_nudge", "_emit_reaction",
                 "_collect_pre_llm_call_context", "_memory_turn_start_and_prefetch",
                 "_maybe_title_session_at_turn_start"):
        monkeypatch.setattr(turn_context, name, forbidden)
    from agent.message_sanitization import _sanitize_surrogates
    from agent.process_bootstrap import _install_safe_stdio
    from agent.codex_responses_adapter import _summarize_user_message_for_log

    context = turn_context.build_turn_context(
        agent, None, None, history, session_id, None, None,
        restore_or_build_system_prompt=lambda *args: None,
        install_safe_stdio=_install_safe_stdio, sanitize_surrogates=_sanitize_surrogates,
        summarize_user_message_for_log=_summarize_user_message_for_log,
        set_session_context=lambda value: None, set_current_write_origin=lambda value: None,
        ra=lambda: __import__("run_agent"),
    )
    assert context.messages == history
    assert db.get_messages_as_conversation(session_id) == history
    assert context.current_turn_user_idx == -1
    assert turn_context.reanchor_current_turn_user_idx(history, None) == -1
    assert agent._user_turn_count == sum(message["role"] == "user" for message in history)
    assert agent._is_user_initiated_turn is False
    wire, system = turn_context.build_api_messages(
        agent, context.messages, current_turn_user_idx=context.current_turn_user_idx,
        ext_prefetch_cache=context.ext_prefetch_cache, plugin_user_context=context.plugin_user_context,
        moa_config=None, active_system_prompt=context.active_system_prompt,
    )
    assert system == "SYSTEM"
    assert [message["role"] for message in wire] == ["system", "user", "assistant", "tool"]
    assert wire[-1]["content"] == "tool settled"
    forbidden.assert_not_called()


@pytest.mark.parametrize("history", [[], [{"role": "user", "content": "real input"}],
                                     [{"role": "assistant", "content": "already finished"}]])
def test_continuation_requires_completed_tool_history(recovered_agent, history):
    agent, _, _ = recovered_agent
    with pytest.raises(ValueError, match="tool result"):
        agent.run_conversation(None, conversation_history=history)
