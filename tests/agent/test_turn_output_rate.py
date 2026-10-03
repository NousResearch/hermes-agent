"""Per-turn generation accounting behind the gateway footer's ``tps`` field."""

from __future__ import annotations

from types import SimpleNamespace

from agent.context_compressor import ContextCompressor
from agent.turn_context import _PER_TURN_RESET_STATE
from agent.turn_usage import record_response_usage


def _agent():
    compressor = ContextCompressor("test-model", base_url="", provider="openai", quiet_mode=True)
    agent = SimpleNamespace(
        model="test-model", provider="openai", api_mode="chat_completions", base_url="",
        context_compressor=compressor, _buffer_vprint=lambda *a: None, _safe_print=lambda *a: None,
        log_prefix="", client=None, _session_db=None, verbose_logging=False, quiet_mode=True,
        session_api_calls=0, session_estimated_cost_usd=0,
    )
    for name in ("prompt", "completion", "total", "input", "output", "cache_read", "cache_write", "reasoning"):
        setattr(agent, f"session_{name}_tokens", 0)
    for name, value in _PER_TURN_RESET_STATE:
        setattr(agent, name, value)
    return agent


def _call(agent, output_tokens, seconds):
    response = SimpleNamespace(usage={"input_tokens": 100, "output_tokens": output_tokens})
    record_response_usage(agent, response, messages=[{"role": "user", "content": "hi"}], api_call_count=1,
                          api_duration=seconds, compression_attempts=0, max_compression_attempts=3)


def test_turn_counters_sum_every_api_call_of_the_turn():
    agent = _agent()
    _call(agent, 300, 2.0)
    _call(agent, 500, 3.0)
    assert agent._turn_output_tokens == 800
    assert agent._turn_api_seconds == 5.0
    assert agent._turn_output_tokens == agent.session_output_tokens


def test_turn_counters_restart_from_zero_while_session_totals_keep_growing():
    agent = _agent()
    _call(agent, 300, 2.0)
    for name, value in _PER_TURN_RESET_STATE:  # what turn start applies
        setattr(agent, name, value)
    _call(agent, 50, 1.0)
    assert (agent._turn_output_tokens, agent._turn_api_seconds) == (50, 1.0)
    assert agent.session_output_tokens == 350
