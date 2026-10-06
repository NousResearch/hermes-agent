"""Focused unit tests for the context-overflow compression fixes (A/B/C).

C: a summary stream that never ends (repetition loop) is aborted by the aux stream
   accumulator's generous runaway-output bound — the wire request still carries NO
   max_tokens cap, so the rule comment stays honoured.
B: an aborted/failed compression followed by growing context re-enters compression
   (wire-proven overflow bypasses the summary-failure cooldown) instead of giving
   up for the rest of the run.
A: a prompt one token over the window gets its oldest tool outputs cleared with
   _PRUNED_TOOL_PLACEHOLDER until it fits; system prompt and latest user message
   are never touched.
"""

import pytest
from types import SimpleNamespace

# Import the lazy chain at collection time so the home-I/O guard (installed at test
# session start) never sees the bootstrap manifest read during a test body.
import agent.conversation_loop  # noqa
import agent.process_bootstrap  # noqa

from agent.auxiliary_client import (
    _ChatStreamAccumulator,
    _aggregate_chat_stream,
    _summary_runaway_output_bound_chars,
)
from agent.chat_completion_helpers import estimate_request_context_tokens
from agent.context_compressor import _PRUNED_TOOL_PLACEHOLDER
from agent.turn_request_assembly import _ensure_request_fits_window
from agent.turn_preflight import compress_after_tool_results


# ---------------------------------------------------------------------------
# C: runaway summary stream (never ends)
# ---------------------------------------------------------------------------


def _chunk(text: str):
    delta = SimpleNamespace(content=text, reasoning=None, tool_calls=None,
                            model_extra=None, reasoning_details=None)
    return SimpleNamespace(id="c", model="m", usage=None,
                           choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)])


def test_summary_runaway_bound_is_generous_and_compression_only():
    assert _summary_runaway_output_bound_chars("compression") == 4 * 10_000 * 4
    assert _summary_runaway_output_bound_chars(None) is None
    assert _summary_runaway_output_bound_chars("other_task") is None


def test_runaway_summary_stream_is_aborted():
    """A summary stream that keeps producing past the generous bound raises TimeoutError
    phrased 'timed out' so the existing failed-attempt fallback + cooldown machinery routes it."""
    bound = _summary_runaway_output_bound_chars("compression")
    acc = _ChatStreamAccumulator(model="m", output_bound_chars=bound)
    piece = "x" * 4000
    produced = 0
    with pytest.raises(TimeoutError) as exc_info:
        while produced <= bound:
            acc.feed(_chunk(piece))
            produced += len(piece)
    assert "timed out" in str(exc_info.value)
    # The stream is bounded, not the wire request: no max_tokens involved.
    assert acc._output_bound_chars == bound


def test_normal_summary_stream_unaffected():
    """A healthy summary under the bound aggregates normally."""
    bound = _summary_runaway_output_bound_chars("compression")
    acc = _ChatStreamAccumulator(model="m", output_bound_chars=bound)
    acc.feed(_chunk("a short healthy summary"))
    resp = acc.finish()
    assert resp.choices[0].message.content == "a short healthy summary"


def test_non_compression_tasks_keep_no_bound():
    """The bound is scoped to the compression task; other aux streams are untouched."""
    chunks = iter([_chunk("y" * 50_000)] * 10)
    resp = _aggregate_chat_stream(chunks, model="m", total_ceiling=None,
                                  output_bound_chars=_summary_runaway_output_bound_chars(None))
    assert len(resp.choices[0].message.content) == 500_000


# ---------------------------------------------------------------------------
# B: aborted compression then growing context → re-entry, not give-up
# ---------------------------------------------------------------------------


class _FakeCompressor:
    def __init__(self, window: int, cooldown_blocked: bool):
        self.context_length = window
        self.last_prompt_tokens = 90_000
        self.threshold_tokens = 60_000
        self._cooldown_blocked = cooldown_blocked

    def should_compress(self, prompt_tokens=None):
        # Cooldown active after the aborted attempt: auto-compaction says "no".
        return False if self._cooldown_blocked else True


def _fake_agent_for_reentry(window: int, cooldown_blocked: bool, calls: list):
    class Agent:
        def _compress_context(self, messages, system_message, **kwargs):
            calls.append(kwargs)
            return messages, system_message

        def _safe_print(self, *a, **k):
            pass

    agent = Agent()
    agent.compression_enabled = True
    agent.tools = None
    agent.context_compressor = _FakeCompressor(window, cooldown_blocked)
    agent._compression_feasibility_checked = True  # skip the aux feasibility probe (no home I/O)
    return agent


def test_aborted_compression_retries_when_wire_cannot_fit():
    """After an aborted compression (cooldown blocks should_compress), a session whose
    wire payload cannot fit the window must re-enter compression with bypass_cooldown=True."""
    calls: list = []
    agent = _fake_agent_for_reentry(window=1000, cooldown_blocked=True, calls=calls)
    # A wire payload far over the 1000-token window.
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "a" * 20_000},
        {"role": "tool", "tool_call_id": "1", "content": "b" * 20_000},
    ]
    verdict = compress_after_tool_results(
        agent, messages=messages, system_message="sys", user_message="hello",
        active_system_prompt="sys", conversation_history=[], compression_attempts=0,
        max_compression_attempts=3, effective_task_id="t", final_response=None,
        turn_exit_reason=None, current_turn_user_idx=1,
    )
    assert len(calls) == 1, "compression must re-enter despite the cooldown"
    assert calls[0]["bypass_cooldown"] is True
    assert verdict.compression_attempts == 1


def test_fitting_request_under_cooldown_still_defers():
    """A request that FITS the window keeps the cooldown defer behaviour (no forced re-entry)."""
    calls: list = []
    agent = _fake_agent_for_reentry(window=100_000, cooldown_blocked=True, calls=calls)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    compress_after_tool_results(
        agent, messages=messages, system_message="sys", user_message="hello",
        active_system_prompt="sys", conversation_history=[], compression_attempts=0,
        max_compression_attempts=3, effective_task_id="t", final_response=None,
        turn_exit_reason=None, current_turn_user_idx=1,
    )
    assert calls == [], "a fitting request under cooldown must not force re-entry"


# ---------------------------------------------------------------------------
# A: prompt one token over the window → deterministic placeholder fallback
# ---------------------------------------------------------------------------


def _mk_agent(window: int, max_tokens):
    class A:
        pass

    a = A()
    a.context_compressor = SimpleNamespace(context_length=window)
    a.max_tokens = max_tokens
    return a


def test_prompt_one_token_over_window_gets_oldest_tool_cleared():
    """Estimate is budget+1 (one token over). The oldest tool output is replaced with the
    placeholder until the payload fits; system prompt and latest user message untouched."""
    window, reserve = 1000, 100
    agent = _mk_agent(window, reserve)
    budget = window - reserve  # _effective_input_window with the output reservation
    system = {"role": "system", "content": "s" * 400}
    user = {"role": "user", "content": "u" * 400}
    tool_old = {"role": "tool", "tool_call_id": "1", "content": "o" * 2000}
    tool_new = {"role": "tool", "tool_call_id": "2", "content": "n" * 404}
    msgs = [system, user, tool_old, tool_new]
    # Pad the newest tool row until the wire-aware estimate is EXACTLY one token over budget.
    while estimate_request_context_tokens(msgs) < budget + 1:
        tool_new["content"] += "n"
    assert estimate_request_context_tokens(msgs) == budget + 1

    out = _ensure_request_fits_window(agent, msgs, tools_for_api=None)
    assert out is msgs
    assert out[0]["content"] == "s" * 400, "system prompt must be untouched"
    assert out[1]["content"] == "u" * 400, "latest user message must be untouched"
    assert out[2]["content"] == _PRUNED_TOOL_PLACEHOLDER, "oldest tool output cleared first"
    assert out[3]["content"] != _PRUNED_TOOL_PLACEHOLDER, "newer tool output kept while it fits"
    assert estimate_request_context_tokens(out) <= budget


def test_fitting_request_untouched():
    agent = _mk_agent(1000, 100)
    msgs = [
        {"role": "system", "content": "s" * 40},
        {"role": "user", "content": "u" * 40},
        {"role": "tool", "tool_call_id": "1", "content": "o" * 40},
    ]
    assert _ensure_request_fits_window(agent, msgs, tools_for_api=None) is msgs
    assert all(m["content"] for m in msgs)


# ---------------------------------------------------------------------------
# Regression: reasoning-heavy 15-65K conversation must NOT look "un-fittable"
# ---------------------------------------------------------------------------


def _realistic_reasoning_session(n_rounds: int = 30):
    """A custom-provider session as persisted in memory: every assistant row carries
    ``reasoning`` AND ``reasoning_content`` (~7.6K chars each, as in run_1f2d6ad9…),
    plus api_content/display bookkeeping. None of that stale thinking reaches the wire."""
    msgs = [{"role": "user", "content": "TASK: review the Bluetooth trading code " + "t" * 2000,
             "api_content": "TASK: review the Bluetooth trading code " + "t" * 2000,
             "timestamp": 1.0}]
    for i in range(n_rounds):
        think = f"thinking {i} " + "r" * 7600
        msgs.append({
            "role": "assistant", "content": "", "reasoning": think, "reasoning_content": think,
            "finish_reason": "tool_calls", "timestamp": 2.0 + i,
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "a.kt"}'}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "k" * 5000,
                     "timestamp": 2.5 + i})
    return msgs


def _realistic_tools(n: int = 30):
    return [{"type": "function", "function": {
        "name": f"tool_{i}", "description": "d" * 400,
        "parameters": {"type": "object", "properties": {"path": {"type": "string",
                                                                  "description": "p" * 200}}},
    }} for i in range(n)]


def _custom_provider_agent(calls: list):
    agent = _fake_agent_for_reentry(window=131_072, cooldown_blocked=True, calls=calls)
    agent.max_tokens = None  # no max_tokens in the config
    agent.tools = _realistic_tools()
    agent.api_mode, agent.provider = "chat_completions", "custom"
    agent.model, agent.base_url = "qwen3.8-flash-next-iq2_xs", "http://127.0.0.1:8082/v1"
    return agent


def test_reasoning_heavy_session_under_window_does_not_force_compression():
    """False positive seen live: a ~25K-token wire request (131K window, no max_tokens) was
    judged 'cannot fit' because stale reasoning/bookkeeping fields were priced, forcing
    cooldown-bypassing compressions over and over. It must NOT force re-entry."""
    from agent.conversation_compression import wire_request_cannot_fit
    from agent.model_metadata import estimate_request_tokens_rough

    calls: list = []
    agent = _custom_provider_agent(calls)
    messages = _realistic_reasoning_session()
    wire_like = estimate_request_tokens_rough(
        messages, tools=agent.tools, charge_stale_thinking=False)
    assert 15_000 <= wire_like <= 65_000, wire_like  # realistic size class
    assert wire_request_cannot_fit(agent, messages, agent.tools, system_prompt="s" * 8000) is False
    compress_after_tool_results(
        agent, messages=messages, system_message="s" * 8000, user_message="go",
        active_system_prompt="s" * 8000, conversation_history=[], compression_attempts=0,
        max_compression_attempts=3, effective_task_id="t", final_response=None,
        turn_exit_reason=None, current_turn_user_idx=0,
    )
    assert calls == [], "a request that fits must not force cooldown-bypassing compression"
    # Same verdict from the assembly-time fallback: nothing is cleared.
    tool_contents = [m["content"] for m in messages if m["role"] == "tool"]
    _ensure_request_fits_window(agent, messages, agent.tools)
    assert [m["content"] for m in messages if m["role"] == "tool"] == tool_contents


def test_reasoning_heavy_session_truly_over_window_still_forces_compression():
    """The original intent holds: once the WIRE payload really exceeds the window, an aborted
    compression is retried (bypass_cooldown) despite the cooldown."""
    calls: list = []
    agent = _custom_provider_agent(calls)
    messages = _realistic_reasoning_session()
    for m in messages:
        if m["role"] == "tool":
            m["content"] = "k" * 20_000  # 30 x 20K chars ~ 150K wire tokens > 131K
    compress_after_tool_results(
        agent, messages=messages, system_message="sys", user_message="go",
        active_system_prompt="sys", conversation_history=[], compression_attempts=0,
        max_compression_attempts=3, effective_task_id="t", final_response=None,
        turn_exit_reason=None, current_turn_user_idx=0,
    )
    assert len(calls) == 1 and calls[0]["bypass_cooldown"] is True
