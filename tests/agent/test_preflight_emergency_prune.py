"""Emergency prune-to-fit before the preflight summarize loop (#114594 part 2).

When estimated tokens exceed the MODEL context window (not just the threshold)
the summarize-first path can never fit in one pass. The deterministic LLM-free
``prune_tool_results_only`` path must be tried first; the normal summarize loop
then continues for any remaining over-threshold pressure.
"""

from types import SimpleNamespace
from unittest.mock import Mock

from agent.turn_context_compaction import CompactionOutcome, _run_preflight_passes


def _make_agent(compress_result):
    agent = SimpleNamespace(
        model="test-model",
        session_id="test-session",
        max_compression_attempts=3,
        _compress_context=Mock(return_value=compress_result),
        _emit_status=Mock(),
    )
    return agent


def _make_compressor(prune_result, context_length=1_000_000):
    return SimpleNamespace(
        threshold_tokens=500_000,
        context_length=context_length,
        should_compress=Mock(return_value=False),
        prune_tool_results_only=Mock(return_value=prune_result),
        emit_automatic_compaction_status=False,
    )


def test_over_window_prunes_before_summarize():
    """Over-window sessions call prune first, adopt its output, then summarize."""
    orig = [{"role": "user", "content": "a"} for _ in range(5)]
    pruned = [{"role": "user", "content": "a"} for _ in range(4)]
    shrunk = [{"role": "user", "content": "a"} for _ in range(3)]
    order = []

    compressor = _make_compressor((pruned, 2))
    orig_prune = compressor.prune_tool_results_only
    compressor.prune_tool_results_only = Mock(
        side_effect=lambda msgs, current_tokens=None: (order.append("prune"), orig_prune(msgs, current_tokens=current_tokens))[1]
    )
    agent = _make_agent((shrunk, "sys"))
    orig_compress = agent._compress_context
    agent._compress_context = Mock(
        side_effect=lambda msgs, *a, **k: (order.append("summarize"), orig_compress(msgs, *a, **k))[1]
    )
    out = CompactionOutcome(
        messages=orig, active_system_prompt="sys",
        conversation_history=[], current_turn_user_idx=0,
    )
    _run_preflight_passes(agent, out, compressor, 1_310_000, "sys", "task-1")

    assert order == ["prune", "summarize"]
    compressor.prune_tool_results_only.assert_called_once_with(orig, current_tokens=1_310_000)
    assert out.messages is shrunk
    assert out.compressed is True


def test_under_window_skips_prune_but_still_summarizes():
    """Below the window the prune branch is skipped; the summarize loop is unchanged."""
    orig = [{"role": "user", "content": "a"} for _ in range(5)]
    shrunk = [{"role": "user", "content": "a"} for _ in range(3)]
    compressor = _make_compressor(((orig, 0)))
    agent = _make_agent((shrunk, "sys"))
    out = CompactionOutcome(
        messages=orig, active_system_prompt="sys",
        conversation_history=[], current_turn_user_idx=0,
    )
    _run_preflight_passes(agent, out, compressor, 600_000, "sys", "task-1")

    compressor.prune_tool_results_only.assert_not_called()
    assert out.messages is shrunk
    assert out.compressed is True
