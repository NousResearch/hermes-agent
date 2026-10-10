"""Claude Haiku 5.5 is a 4.6+-generation model, not another Haiku 4.5.

Unlike every earlier Haiku it uses adaptive thinking with the effort parameter
(``budget_tokens`` 400s), keeps prior turns' thinking in context, has a 1M
window, and bills a whole request at a higher tier above 100K prompt tokens.
https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from agent.anthropic_adapter import build_anthropic_kwargs
from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, _longest_key_match
from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

HAIKU_5_5_IDS = ["claude-haiku-5-5", "anthropic/claude-haiku-5.5"]


def _kwargs(model: str, reasoning_config: dict | None):
    return build_anthropic_kwargs(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        max_tokens=4096,
        reasoning_config=reasoning_config,
    )


@pytest.mark.parametrize("model", HAIKU_5_5_IDS)
def test_effort_reaches_the_wire_as_adaptive_thinking(model):
    kwargs = _kwargs(model, {"enabled": True, "effort": "high"})
    assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kwargs["output_config"] == {"effort": "high"}
    assert "temperature" not in kwargs  # sampling params 400 on Haiku 5.5


@pytest.mark.parametrize("model", HAIKU_5_5_IDS)
def test_thinking_off_sends_the_explicit_disable(model):
    assert _kwargs(model, {"enabled": False})["thinking"] == {"type": "disabled"}


def test_pre_5_haiku_still_gets_no_thinking():
    kwargs = _kwargs("claude-haiku-4-5", {"enabled": True, "effort": "high"})
    assert "thinking" not in kwargs and "output_config" not in kwargs


@pytest.mark.parametrize("model", HAIKU_5_5_IDS)
def test_context_window_is_one_million_not_the_claude_catch_all(model):
    bare = model.split("/")[-1].replace(".", "-")
    assert _longest_key_match(DEFAULT_CONTEXT_LENGTHS, bare)[1] == 1_000_000
    assert _longest_key_match(DEFAULT_CONTEXT_LENGTHS, "claude-haiku-4-5")[1] == 200_000


def test_prompts_up_to_100k_bill_at_the_base_rates():
    result = estimate_usage_cost(
        "claude-haiku-5-5",
        CanonicalUsage(input_tokens=90_000, output_tokens=10_000, cache_read_tokens=10_000),
        provider="anthropic",
    )
    # 90k * $0.10/M + 10k * $0.50/M + 10k * $0.01/M
    assert result.amount_usd == Decimal("0.0141")


def test_prompts_over_100k_reprice_the_whole_request():
    result = estimate_usage_cost(
        "claude-haiku-5-5",
        CanonicalUsage(input_tokens=90_000, output_tokens=10_000, cache_read_tokens=20_000, cache_write_tokens=1_000),
        provider="anthropic",
    )
    # prompt = 111k > 100k: 90k * $0.50/M + 10k * $2.50/M + 20k * $0.05/M + 1k * $0.625/M
    assert result.amount_usd == Decimal("0.071625")
