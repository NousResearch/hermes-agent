from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.turn_preflight import compress_after_tool_results
from run_agent import AIAgent


def test_post_tool_compression_reanchors_the_active_user_boundary(monkeypatch):
    compressed = [
        {"role": "user", "content": "compressed history"},
        {"role": "user", "content": "current ask"},
        {"role": "assistant", "tool_calls": [{"id": "2"}]},
        {"role": "tool", "content": "fresh result", "tool_call_id": "2"},
    ]

    class Compressor:
        last_prompt_tokens = 100
        threshold_tokens = 50

        @staticmethod
        def should_compress(_tokens):
            return True

    agent = SimpleNamespace(
        context_compressor=Compressor(),
        compression_enabled=True,
        _clear_context_overflow_warn=lambda: None,
        _safe_print=lambda *_args: None,
        _compress_context=lambda *_args, **_kwargs: (compressed, "system"),
        _persist_user_message_idx=4,
    )
    monkeypatch.setattr(
        "agent.turn_preflight.conversation_history_after_compression",
        lambda _agent, _messages, _history: [],
    )
    monkeypatch.setattr(
        "agent.conversation_loop._should_skip_model_call_for_reference_handoff",
        lambda _messages, _user_message: False,
    )

    verdict = compress_after_tool_results(
        agent,
        messages=[{"role": "user", "content": "current ask"}],
        system_message="system",
        user_message="current ask",
        active_system_prompt="system",
        conversation_history=[],
        compression_attempts=0,
        max_compression_attempts=1,
        effective_task_id="task",
        final_response="",
        turn_exit_reason=None,
        current_turn_user_idx=0,
    )

    assert verdict.messages is compressed
    assert verdict.current_turn_user_idx == 1
    assert agent._persist_user_message_idx == 1


def _response(*, tool: bool):
    tool_calls = [SimpleNamespace(
        id="call_1", type="function",
        function=SimpleNamespace(name="web_search", arguments='{"query": "x"}'),
    )] if tool else None
    message = SimpleNamespace(
        content=None if tool else "done", reasoning_content=None, reasoning=None, tool_calls=tool_calls,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if tool else "stop")],
        model="test/model", usage=None,
    )


def test_pre_api_compression_mid_turn_keeps_this_turns_tool_pair_on_the_wire():
    """Pre-API compression after a tool round shrinks ``messages`` from the front; the next request
    must still carry this turn's assistant tool_call and its result (state.db holds both)."""
    tool_def = {"type": "function", "function": {
        "name": "web_search", "description": "s",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }}
    with (
        patch("model_tools.get_tool_definitions", return_value=[tool_def]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=256_000),
        patch("agent.context_compressor.get_model_context_length", return_value=256_000),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=6,
        )
    agent.client = MagicMock()
    # No usage: the post-tool gate has no real count, so the pre-API gate owns the compaction.
    agent.client.chat.completions.create.side_effect = [_response(tool=True), _response(tool=False)]
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False

    compressor = MagicMock()
    compressor.protect_first_n = 3
    compressor.protect_last_n = 20
    compressor.threshold_tokens = 100
    compressor.context_length = 1_000
    compressor.last_prompt_tokens = -1
    compressor._verify_compaction_cleared_threshold = False
    compressor.awaiting_real_usage_after_compression = False
    compressor.should_compress.side_effect = lambda tokens: tokens >= 100
    compressor.should_compress_info.return_value = (False, None)
    compressor.should_compress_preflight.return_value = False
    compressor.should_defer_preflight_to_real_usage.return_value = False
    compressor.get_active_compression_failure_cooldown.return_value = None
    compressor.select_context.return_value = None
    compressor.get_automatic_compaction_status_message.return_value = ""
    agent.compression_enabled = True
    agent.context_compressor = compressor

    compactions = []

    def _estimate(messages=None, *_args, **_kwargs):
        # The tool result tips the request over threshold until one compaction ran.
        return 200 if not compactions and any(m.get("role") == "tool" for m in messages or []) else 10

    def _compress(messages, _system_message, **_kwargs):
        compactions.append(len(messages))
        return list(messages[2:]), "compressed prompt"  # two historical rows summarized away

    def _execute(_assistant_message, messages, *_args):
        messages.append({"role": "tool", "name": "web_search", "tool_call_id": "call_1", "content": "RESULT-1"})

    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(30)]
    with (
        patch("agent.turn_context.estimate_request_tokens_rough", return_value=10),
        patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=_estimate),
        patch("agent.conversation_loop._estimate_tools_tokens_rough", return_value=0),
        patch.object(agent, "_compress_context", side_effect=_compress),
        patch.object(agent, "_execute_tool_calls", side_effect=_execute),
        patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do tool work", conversation_history=history)

    assert result["final_response"] == "done"
    assert len(compactions) == 1
    sent = agent.client.chat.completions.create.call_args_list[-1].kwargs["messages"]
    assert [(m["role"], m.get("tool_call_id")) for m in sent[-3:]] == [
        ("user", None), ("assistant", None), ("tool", "call_1"),
    ]
    assert [t["id"] for t in sent[-2]["tool_calls"]] == ["call_1"]
    assert sent[-1]["content"] == "RESULT-1"
