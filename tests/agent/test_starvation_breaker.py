"""Server-side truncation starvation: ``finish_reason`` length/max_tokens with empty
content must short-circuit the empty-response ladder.

Failure shape (observed on a hosted OpenAI-compatible gateway): a reasoning model
with thinking enabled gets its *entire* output allocation consumed by reasoning
tokens against the endpoint's internal output cap (applied when the request omits
``max_tokens``). The response ends ``finish_reason=length``/``max_tokens`` with zero
visible content. The ladder previously treated this identically to a benign empty:
nudge (mutates the prompt → breaks the cached prefix) → prefill ×2 → retries ×3 →
fallback — re-billing the full input on every attempt with no chance of success,
because the cap will truncate the retry identically.

The starvation guard fails open: ambiguous evidence (any visible content, tool
calls, unknown finish reason) keeps legacy ladder behaviour.
"""

from types import SimpleNamespace

from agent import empty_response_guard as guard


def _agent(**overrides):
    base = dict(
        model="gemini-3.8-flash",
        provider="custom:gateway",
        api_mode="chat_completions",
        base_url="https://gateway.example/v1",
        api_key=None,
        _empty_content_retries=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _response(prompt_tokens=34_000, completion_tokens=1_000, reasoning_tokens=0, usage_present=True):
    if not usage_present:
        return SimpleNamespace(usage=None)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )
    return SimpleNamespace(usage=usage)


class TestStarvationDetection:
    def test_length_finish_with_reasoning_only_output_is_starvation(self):
        agent = _agent()
        assert guard.is_output_starvation(
            agent, finish_reason="max_tokens",
            response=_response(reasoning_tokens=996),
            has_visible_content=False, has_tool_calls=False,
        ) is True

    def test_openai_length_finish_is_starvation(self):
        agent = _agent()
        assert guard.is_output_starvation(
            agent, finish_reason="length",
            response=_response(reasoning_tokens=956),
            has_visible_content=False, has_tool_calls=False,
        ) is True

    def test_stop_finish_is_never_starvation(self):
        """finish_reason=stop + empty content is a model-behaviour problem (refusal,
        decode blip) — the existing ladder owns it, not the truncation guard."""
        agent = _agent()
        assert guard.is_output_starvation(
            agent, finish_reason="stop",
            response=_response(reasoning_tokens=0),
            has_visible_content=False, has_tool_calls=False,
        ) is False

    def test_tool_calls_present_fails_open(self):
        """A truncated response that still produced tool calls is mid-task work;
        the tool round must proceed, not the breaker."""
        agent = _agent()
        assert guard.is_output_starvation(
            agent, finish_reason="max_tokens",
            response=_response(reasoning_tokens=956),
            has_visible_content=False, has_tool_calls=True,
        ) is False

    def test_visible_content_fails_open(self):
        """Content present (think-block stripping leaves text) is not starvation."""
        agent = _agent()
        assert guard.is_output_starvation(
            agent, finish_reason="max_tokens",
            response=_response(reasoning_tokens=956),
            has_visible_content=True, has_tool_calls=False,
        ) is False

    def test_missing_usage_fails_open(self):
        """No usage object → no proof of truncation shape → legacy behaviour."""
        agent = _agent()
        assert guard.is_output_starvation(
            agent, finish_reason="max_tokens",
            response=_response(usage_present=False),
            has_visible_content=False, has_tool_calls=False,
        ) is False

    def test_guard_disabled_fails_open(self):
        agent = _agent(_empty_guard_enabled=False)
        assert guard.is_output_starvation(
            agent, finish_reason="max_tokens",
            response=_response(reasoning_tokens=956),
            has_visible_content=False, has_tool_calls=False,
        ) is False

    def test_unclosed_think_block_only_content_counts_as_empty(self):
        """DeepSeek-style endpoints inline <think> tags in content; a response truncated
        mid-thought yields 'content' that is only an unclosed think tag. Callers
        strip think blocks before the has_visible_content check, so this shape
        classifies as starvation (Flash round-1 BLOCKER 2)."""
        agent = _agent()
        assert guard.is_output_starvation(
            agent, finish_reason="length",
            response=_response(reasoning_tokens=0, completion_tokens=1_000),
            has_visible_content=False,  # after _strip_think_blocks("<think>...partial")
            has_tool_calls=False,
        ) is True

    def test_truncating_finish_with_unreported_reasoning_is_starvation(self):
        """Gateway omits the reasoning breakdown (reported 0) but completion>0 on a
        truncating finish reason with no content: the cap consumed the output
        (GPT-OSS round-1 requested documentation of this exact shape)."""
        agent = _agent()
        assert guard.is_output_starvation(
            agent, finish_reason="length",
            response=_response(reasoning_tokens=0, completion_tokens=1_000),
            has_visible_content=False, has_tool_calls=False,
        ) is True


class TestStarvationBudget:
    def test_starved_streak_gets_single_retry(self):
        """Budget is queried with _empty_content_retries at its PRE-increment value
        (the loop records the attempt, picks the budget, then increments)."""
        agent = _agent()
        _record(agent, finish_reason="max_tokens", response=_response(reasoning_tokens=956))
        agent._empty_content_retries = 0  # first starved attempt: budget check time
        assert guard.starvation_retry_budget(agent, finish_reason="max_tokens") == 1

    def test_starvation_budget_zero_after_one_starvation_retry(self):
        agent = _agent()
        _record(agent, finish_reason="max_tokens", response=_response(reasoning_tokens=956))
        agent._empty_content_retries = 1
        assert guard.starvation_retry_budget(agent, finish_reason="max_tokens") == 0

    def test_non_starvation_streak_keeps_default_budget(self):
        agent = _agent()
        _record(agent, finish_reason="stop", response=_response(usage_present=False))
        assert guard.starvation_retry_budget(agent, finish_reason="stop") == guard.DEFAULT_EMPTY_RETRY_BUDGET

    def test_guard_disabled_keeps_default_budget(self):
        agent = _agent(_empty_guard_enabled=False)
        _record(agent, finish_reason="max_tokens", response=_response(reasoning_tokens=956))
        assert guard.starvation_retry_budget(agent, finish_reason="max_tokens") == guard.DEFAULT_EMPTY_RETRY_BUDGET


def _record(agent, *, finish_reason, response, observed_generation=False):
    guard.record_empty_attempt(
        agent, finish_reason=finish_reason, response=response,
        observed_generation=observed_generation,
    )
    agent._empty_content_retries += 1
