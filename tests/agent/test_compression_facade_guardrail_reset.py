"""compress_context facade resets guardrail no-progress memory only after a committed compaction."""
import json
from types import SimpleNamespace

from agent.compression_facade import _forget_guardrail_results_if_compacted
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController

ARGS = {"name": "some-skill"}
RESULT = json.dumps({"success": True, "name": "some-skill", "content": "x" * 600})


def _agent_at_block_threshold():
    guard = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))
    for _ in range(5):
        guard.before_call("skill_view", ARGS)
        guard.after_call("skill_view", ARGS, RESULT, failed=False)
    assert not guard.before_call("skill_view", ARGS).allows_execution
    return SimpleNamespace(_tool_guardrails=guard)


def test_aborted_compaction_keeps_no_progress_memory():
    agent = _agent_at_block_threshold()
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    _forget_guardrail_results_if_compacted(agent, (messages, "sys"), messages, len(messages))
    assert not agent._tool_guardrails.before_call("skill_view", ARGS).allows_execution


def test_committed_compaction_resets_no_progress_memory():
    agent = _agent_at_block_threshold()
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    _forget_guardrail_results_if_compacted(agent, ([messages[0]], "sys"), messages, len(messages))
    assert agent._tool_guardrails.before_call("skill_view", ARGS).allows_execution


def test_in_place_compaction_resets_no_progress_memory():
    agent = _agent_at_block_threshold()
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    before = len(messages)
    del messages[1]  # in-place commit: same list object, new length
    _forget_guardrail_results_if_compacted(agent, (messages, "sys"), messages, before)
    assert agent._tool_guardrails.before_call("skill_view", ARGS).allows_execution


def test_agent_without_guardrails_is_ignored():
    messages = [{"role": "user", "content": "a"}]
    _forget_guardrail_results_if_compacted(SimpleNamespace(), ([], "sys"), messages, 1)
