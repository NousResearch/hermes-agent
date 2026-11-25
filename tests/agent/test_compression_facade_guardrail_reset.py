"""compress_context facade resets guardrail no-progress memory only after a committed compaction."""
import json
from unittest.mock import MagicMock, patch

from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController

ARGS = {"name": "some-skill"}
RESULT = json.dumps({"success": True, "name": "some-skill", "content": "x" * 600})


def _agent_at_block_threshold():
    guard = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))
    for _ in range(5):
        guard.before_call("skill_view", ARGS)
        guard.after_call("skill_view", ARGS, RESULT, failed=False)
    assert not guard.before_call("skill_view", ARGS).allows_execution
    with patch("agent.process_bootstrap.OpenAI"):
        from run_agent import AIAgent

        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._tool_guardrails = guard
    compressor = MagicMock()
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    agent.context_compressor = compressor
    return agent


def _compress(agent, messages, rewrite):
    agent.context_compressor.compress.side_effect = rewrite
    return agent._compress_context(messages, "sys", approx_tokens=10000, force=True)


def test_aborted_compaction_keeps_no_progress_memory():
    agent = _agent_at_block_threshold()
    messages = [{"role": "user", "content": "a" * 400}, {"role": "assistant", "content": "b"}]
    _compress(agent, messages, lambda current, **_kwargs: current)
    assert not agent._tool_guardrails.before_call("skill_view", ARGS).allows_execution


def test_committed_compaction_resets_no_progress_memory():
    agent = _agent_at_block_threshold()
    messages = [{"role": "user", "content": "a" * 400}, {"role": "assistant", "content": "b"}]
    _compress(agent, messages, lambda current, **_kwargs: [current[0]])
    assert agent._tool_guardrails.before_call("skill_view", ARGS).allows_execution


def test_in_place_compaction_resets_no_progress_memory():
    agent = _agent_at_block_threshold()
    messages = [{"role": "user", "content": "a" * 400}, {"role": "assistant", "content": "b"}]
    def rewrite(current, **_kwargs):
        del current[1]
        return current

    _compress(agent, messages, rewrite)
    assert agent._tool_guardrails.before_call("skill_view", ARGS).allows_execution


def test_agent_without_guardrails_is_ignored():
    messages = [{"role": "user", "content": "a" * 400}]
    agent = _agent_at_block_threshold()
    agent._tool_guardrails = None
    _compress(agent, messages, lambda current, **_kwargs: [{"role": "user", "content": "s"}])
