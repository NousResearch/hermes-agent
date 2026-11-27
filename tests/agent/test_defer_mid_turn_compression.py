"""compression.defer_mid_turn: hold automatic compaction while a turn is running."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.agent_init import _parse_defer_hard_limit_ratio
from agent.turn_preflight import _defer_midturn_compression, compress_after_tool_results

CONTEXT = 200_000


def _agent(enabled=True, ratio=0.85):
    agent = MagicMock()
    agent.compression_enabled = True
    agent.compression_defer_mid_turn = enabled
    agent.compression_defer_hard_limit_ratio = ratio
    agent.tools = None
    agent._usage_anchor = None
    return agent


def _compressor(tokens):
    comp = MagicMock()
    comp.context_length = CONTEXT
    comp.threshold_tokens = 80_000
    comp.awaiting_real_usage_after_compression = False
    comp.last_prompt_tokens = tokens
    comp.should_compress.return_value = tokens >= 80_000
    return comp


def test_helper_off_by_default():
    assert _defer_midturn_compression(SimpleNamespace(), _compressor(90_000), 90_000) is False
    assert _defer_midturn_compression(_agent(enabled=False), _compressor(90_000), 90_000) is False


def test_helper_defers_below_hard_limit_only():
    comp = _compressor(0)
    agent = _agent()
    assert _defer_midturn_compression(agent, comp, 90_000) is True
    assert _defer_midturn_compression(agent, comp, int(CONTEXT * 0.85) - 1) is True
    assert _defer_midturn_compression(agent, comp, int(CONTEXT * 0.85)) is False


def _run(agent, comp, attempts=0):
    agent.context_compressor = comp
    agent._compress_context.return_value = ([], "sys")
    return compress_after_tool_results(
        agent, messages=[{"role": "user", "content": "hi"}], system_message="s", user_message="hi",
        active_system_prompt="s", conversation_history=[], compression_attempts=attempts,
        max_compression_attempts=3, effective_task_id="t", final_response=None,
        turn_exit_reason=None, current_turn_user_idx=0,
    )


def test_post_tool_compaction_runs_when_not_deferred():
    agent, comp = _agent(enabled=False), _compressor(90_000)
    _run(agent, comp)
    agent._compress_context.assert_called_once()


def test_post_tool_compaction_deferred_over_threshold():
    agent, comp = _agent(), _compressor(90_000)
    verdict = _run(agent, comp)
    agent._compress_context.assert_not_called()
    agent._warn_context_overflow_blocked.assert_not_called()
    assert verdict.end_turn is False
    assert verdict.compression_attempts == 0


def test_post_tool_compaction_runs_above_hard_limit():
    agent, comp = _agent(), _compressor(int(CONTEXT * 0.9))
    _run(agent, comp)
    agent._compress_context.assert_called_once()


def test_ratio_parsing_clamps_and_falls_back():
    assert _parse_defer_hard_limit_ratio(None) == 0.85
    assert _parse_defer_hard_limit_ratio("junk") == 0.85
    assert _parse_defer_hard_limit_ratio(0.1) == 0.5
    assert _parse_defer_hard_limit_ratio(5) == 0.98
    assert _parse_defer_hard_limit_ratio("0.7") == 0.7
