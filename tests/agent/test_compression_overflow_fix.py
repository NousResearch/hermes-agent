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

import math

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
    # Hard bound ~100K tokens: the old 4 x _SUMMARY_TOKENS_CEILING (160K chars) aborted
    # legitimate summaries (run_29371f18); loops are caught earlier by the tail check.
    assert _summary_runaway_output_bound_chars("compression") == 100_000 * 4
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
    assert out is not msgs and tool_old["content"] == "o" * 2000, "transcript rows are not mutated"
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


# ---------------------------------------------------------------------------
# Regression run_29371f18: no max_tokens, 130,902-token request in a 131,072 window
# ---------------------------------------------------------------------------


def _near_full_window_request(target_tokens: int = 130_902):
    """Assembled api_messages (system + user + many tool rows) whose wire estimate is
    ``target_tokens`` — the request run_29371f18 sent into its 131,072 window."""
    msgs = [{"role": "system", "content": "s" * 8000}, {"role": "user", "content": "review the app"}]
    i = 0
    while estimate_request_context_tokens(msgs) < target_tokens - 3000:
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "a.swift"}'}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "k" * 12_000})
        i += 1
    while estimate_request_context_tokens(msgs) < target_tokens:
        msgs[-1]["content"] += "k"
    assert estimate_request_context_tokens(msgs) == target_tokens
    return msgs


def test_budget_leaves_room_to_answer_without_max_tokens():
    from agent.conversation_compression import (
        _MIN_ANSWER_RESERVE_TOKENS, request_input_budget)

    agent = _mk_agent(131_072, None)
    budget = request_input_budget(agent)
    assert budget <= 131_072 - _MIN_ANSWER_RESERVE_TOKENS
    assert budget > 100_000, "the margin must stay modest"
    # A configured reservation larger than the minimum is honoured.
    assert request_input_budget(_mk_agent(131_072, 32_768)) <= 131_072 - 32_768
    # A reservation that swallows the window cannot make every request un-fittable.
    assert request_input_budget(_mk_agent(131_072, 200_000)) > 100_000


def test_no_max_tokens_near_full_window_request_is_never_sent_as_is():
    """run_29371f18: two aborted compressions, then a 130,902-token request went into the
    131,072 window with no max_tokens; the fit check said it fit (budget == window) and the
    provider stopped with 'Context window full ... leaving no room to answer'."""
    from agent.conversation_compression import request_input_budget, wire_request_cannot_fit

    agent = _mk_agent(131_072, None)
    msgs = _near_full_window_request()
    assert wire_request_cannot_fit(agent, msgs, None) is True
    tool_before = sum(1 for m in msgs if m["role"] == "tool")
    out = _ensure_request_fits_window(agent, msgs, tools_for_api=None)
    budget = request_input_budget(agent)
    assert estimate_request_context_tokens(out) <= budget
    assert 131_072 - estimate_request_context_tokens(out) >= 8_192, "room to answer"
    assert out[0]["content"] == "s" * 8000 and out[1]["content"] == "review the app"
    cleared = [m for m in out if m["role"] == "tool" and m["content"] == _PRUNED_TOOL_PLACEHOLDER]
    assert 0 < len(cleared) < tool_before
    assert out[2 + 1]["content"] == _PRUNED_TOOL_PLACEHOLDER, "oldest tool output first"


def test_no_max_tokens_near_full_window_forces_compression_after_abort():
    """Right after an aborted compression (cooldown armed, should_compress False) the same
    request must re-enter compression with bypass_cooldown instead of being sent."""
    calls: list = []
    agent = _fake_agent_for_reentry(window=131_072, cooldown_blocked=True, calls=calls)
    agent.max_tokens = None
    agent.api_mode, agent.provider = "chat_completions", "custom"
    msgs = _near_full_window_request()
    system = msgs[0]["content"]
    compress_after_tool_results(
        agent, messages=msgs[1:], system_message=system, user_message="review the app",
        active_system_prompt=system, conversation_history=[], compression_attempts=2,
        max_compression_attempts=3, effective_task_id="t", final_response=None,
        turn_exit_reason=None, current_turn_user_idx=0,
    )
    assert len(calls) == 1 and calls[0]["bypass_cooldown"] is True


def test_send_time_fit_check_runs_on_every_send():
    """The loop calls the fit check after the preflight gate falls through and before the
    API call is announced/sent, on every iteration (incl. after an aborted compression)."""
    import inspect

    from agent import conversation_loop

    from agent import turn_api_request

    src = inspect.getsource(conversation_loop._run_conversation_turn)
    gate = src.index("_pg = _run_phase(run_preflight_gate")
    send = src.index("_run_api_retry_loop(agent, s)")
    assert gate < send
    assert src.count("_run_api_retry_loop(agent, s)") == 1
    # The guard lives in build_api_request, which runs for every attempt (incl. retries)
    # right before the provider call, after every message mutation of the attempt.
    build = inspect.getsource(turn_api_request.build_api_request)
    fit = build.index("_ensure_request_fits_window(agent, api_messages, tools_for_api)")
    assert fit < build.index("agent._build_api_kwargs(api_messages")
    retry = inspect.getsource(conversation_loop._run_api_retry_loop)
    assert retry.index("build_api_request") < retry.index("perform_api_call")


# ---------------------------------------------------------------------------
# Regression run_29371f18: a long legitimate summary must not be aborted
# ---------------------------------------------------------------------------


def _delta_chunk(content=None, reasoning=None):
    delta = SimpleNamespace(content=content, reasoning=reasoning, tool_calls=None,
                            model_extra=None, reasoning_details=None)
    return SimpleNamespace(id="c", model="m", usage=None,
                           choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)])


def _varied_text(n_chars: int, seed: int) -> str:
    """Non-repeating prose/code-review text: the shape of a real thinking summary."""
    import random

    rnd = random.Random(seed)
    words = ("the scanner view model card trade session bluetooth peer state queue error "
             "retry handler swift actor main thread race decode json cache image price "
             "collection deck inventory sync conflict update test coverage missing guard "
             "optional unwrap crash leak closure weak self delegate notification").split()
    lines, size, i = [], 0, 0
    while size < n_chars:
        line = f"{i}. In File{rnd.randint(1, 400)}.swift:{rnd.randint(1, 900)} " + " ".join(
            rnd.choice(words) for _ in range(rnd.randint(8, 20))) + "."
        lines.append(line)
        size += len(line) + 1
        i += 1
    return "\n".join(lines)[:n_chars]


def _stream(acc, text, field, step=400):
    for k in range(0, len(text), step):
        piece = text[k:k + step]
        acc.feed(_delta_chunk(**{field: piece}))


def test_long_legitimate_summary_is_not_aborted():
    """run_29371f18: both summaries of a ~100K conversation were aborted at exactly 160,000
    chars (reasoning + content). A 240K-char reasoning stream plus a 40K-char summary with no
    loop (~70K tokens, beyond the old bound) must complete."""
    acc = _ChatStreamAccumulator(
        model="m", output_bound_chars=_summary_runaway_output_bound_chars("compression"))
    _stream(acc, _varied_text(240_000, 1), "reasoning")
    _stream(acc, _varied_text(40_000, 2), "content")
    resp = acc.finish()
    assert len(resp.choices[0].message.content) == 40_000
    assert len(resp.choices[0].message.reasoning) == 240_000


def test_looping_summary_is_stopped_well_before_hard_bound():
    """A stream that degenerates into a verbatim loop after a legitimate start is stopped
    shortly after the soft size, long before the ~100K-token hard bound."""
    acc = _ChatStreamAccumulator(
        model="m", output_bound_chars=_summary_runaway_output_bound_chars("compression"))
    _stream(acc, _varied_text(150_000, 3), "reasoning")
    unit = "Wait, let me re-check the trade session state handling once more before writing.\n"
    with pytest.raises(TimeoutError) as exc_info:
        _stream(acc, unit * 5000, "reasoning")
    assert "timed out" in str(exc_info.value) and "repetition loop" in str(exc_info.value)
    assert acc._content_chars < 250_000


def test_true_runaway_without_verbatim_loop_hits_hard_bound():
    """Even a non-verbatim runaway (>100K tokens) is stopped by the hard bound."""
    bound = _summary_runaway_output_bound_chars("compression")
    acc = _ChatStreamAccumulator(model="m", output_bound_chars=bound)
    with pytest.raises(TimeoutError) as exc_info:
        _stream(acc, _varied_text(bound + 10_000, 4), "content", step=4000)
    assert "hard" in str(exc_info.value) and "timed out" in str(exc_info.value)


def test_summary_abort_logs_check_and_stream_tail_at_info(caplog):
    """run_9e549b82: every abort was only visible as summary_generation_aborted; the log must
    say which check fired and show what the model was emitting."""
    import logging

    acc = _ChatStreamAccumulator(
        model="m", output_bound_chars=_summary_runaway_output_bound_chars("compression"))
    unit = "Wait, let me re-check the CardLists.swift findings before writing the summary.\n"
    with caplog.at_level(logging.INFO, logger="agent.auxiliary_client"):
        with pytest.raises(TimeoutError):
            _stream(acc, unit * 3000, "reasoning", step=4000)
    rec = [r for r in caplog.records if "Compression summary stream aborted" in r.getMessage()]
    assert len(rec) == 1 and rec[0].levelno == logging.INFO
    msg = rec[0].getMessage()
    assert "check=repetition_loop:reasoning" in msg
    assert "re-check the CardLists.swift findings" in msg, "tail of the looping stream logged"


def test_summary_abort_logging_is_compression_only(caplog):
    import logging

    acc = _ChatStreamAccumulator(model="m", total_ceiling=0.0)
    with caplog.at_level(logging.INFO, logger="agent.auxiliary_client"):
        with pytest.raises(TimeoutError):
            acc.feed(_delta_chunk(content="x"))
    assert not [r for r in caplog.records if "summary stream aborted" in r.getMessage()]


# ---------------------------------------------------------------------------
# Regression run_9e549b82: the char/4 wire estimate under-counts dense code ~1.17x
# ---------------------------------------------------------------------------


def _calibrated_agent(window=131_072):
    a = _mk_agent(window, None)
    a.provider, a.model, a.api_mode = "custom", "qwen3.8-flash-next-iq2_xs", "chat_completions"
    return a


def test_provider_usage_calibrates_the_fit_check():
    """Request #33: estimate 100,376, provider 117,523 (budget 116,327). Uncalibrated the next,
    equally dense request 'fits'; after the provider priced the previous one it does not."""
    from agent.conversation_compression import (
        calibrate_wire_estimate_from_usage, note_wire_estimate_sent, request_input_budget,
        wire_estimate_calibration, wire_request_cannot_fit, wire_request_tokens,
    )

    agent = _calibrated_agent()
    msgs = _near_full_window_request(target_tokens=100_400)
    est = wire_request_tokens(agent, msgs)
    assert est < request_input_budget(agent) == 116_327
    assert wire_request_cannot_fit(agent, msgs) is False
    # The previous request (#32) was estimated 94,515 and priced 110,337 (ratio 1.167).
    note_wire_estimate_sent(agent, 94_515)
    assert abs(calibrate_wire_estimate_from_usage(agent, 110_337) - 1.1674) < 1e-3
    assert wire_request_cannot_fit(agent, msgs) is True
    out = _ensure_request_fits_window(agent, msgs, tools_for_api=None)
    assert math.ceil(wire_request_tokens(agent, out) * wire_estimate_calibration(agent)) <= 116_327
    assert out[0] is msgs[0] and out[1] is msgs[1], "system prompt and user message untouched"
    # The guard recorded what it would send; one usage reading per sent request.
    assert agent._wire_estimate_sent == wire_request_tokens(agent, out)
    assert calibrate_wire_estimate_from_usage(agent, int(agent._wire_estimate_sent * 1.17))
    assert calibrate_wire_estimate_from_usage(agent, 50_000) is None
    # A ratio below 1 never relaxes the char/4 floor.
    note_wire_estimate_sent(agent, 100_000)
    calibrate_wire_estimate_from_usage(agent, 80_000)
    assert wire_estimate_calibration(agent) == 1.0
    # Absurd readings are ignored; the factor is capped.
    note_wire_estimate_sent(agent, 1_000)
    assert calibrate_wire_estimate_from_usage(agent, 50_000) is None
    note_wire_estimate_sent(agent, 1_000)
    calibrate_wire_estimate_from_usage(agent, 2_900)
    assert wire_estimate_calibration(agent) == 2.0


def test_calibration_does_not_carry_over_to_another_model():
    from agent.conversation_compression import (
        calibrate_wire_estimate_from_usage, note_wire_estimate_sent, wire_estimate_calibration,
    )

    agent = _calibrated_agent()
    note_wire_estimate_sent(agent, 100_000)
    calibrate_wire_estimate_from_usage(agent, 117_000)
    assert wire_estimate_calibration(agent) == pytest.approx(1.17)
    agent.model = "fallback-model"
    assert wire_estimate_calibration(agent) == 1.0


def test_guard_records_sent_estimate_and_never_sends_unfittable():
    from agent.conversation_compression import wire_request_tokens
    from agent.turn_request_assembly import RequestExceedsContextWindowError

    agent = _calibrated_agent(window=10_000)
    fitting = [{"role": "system", "content": "s" * 400}, {"role": "user", "content": "u" * 400}]
    assert _ensure_request_fits_window(agent, fitting, None) is fitting
    assert agent._wire_estimate_sent == wire_request_tokens(agent, fitting)
    unfittable = [
        {"role": "system", "content": "s" * 400},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "o" * 4000},
        {"role": "user", "content": "u" * 60_000},
    ]
    with pytest.raises(RequestExceedsContextWindowError) as exc_info:
        _ensure_request_fits_window(agent, unfittable, None)
    assert "does not fit the model context window" in str(exc_info.value)
    assert unfittable[2]["content"] == "o" * 4000, "transcript rows are not mutated"
