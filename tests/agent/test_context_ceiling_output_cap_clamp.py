"""Tests for the local output-cap clamp at the terminal gate.

The terminal gate (``enforce_final_context_budget``) classifies overflow:

* **Unshrinkable** — input alone is at/over the effective limit →
  :class:`ContextCeilingExceeded` raised (zero provider I/O).
* **Shrinkable** — input fits but input + output_reservation > limit →
  the outgoing output cap is clamped in-place to the maximum legal
  allowance; no raise; the provider call proceeds with the reduced cap.

The existing ``test_output_cap_retry_with_large_api_only_content`` (in
``tests/run_agent/test_run_agent.py``) covers the provider-error-recovery
integration path — a provider 400 that the clamp did NOT prevent (because
the provider's own tokenizer counted the input larger than Hermes's rough
estimate).  These tests cover the preflight-clamp contract at the shared
boundary, which is the single correct clamp owner for every dispatch path
(OpenAI / Anthropic / Bedrock / Codex / streaming / auxiliary / Relay).
"""
from __future__ import annotations

import pytest

from agent.model_metadata import (
    ContextCeilingExceeded,
    FinalContextBudget,
    _final_request_output_cap,
    _set_final_request_output_cap,
    build_final_context_budget,
    enforce_final_context_budget,
)


def _budget(input_tokens: int, output: int) -> FinalContextBudget:
    return FinalContextBudget(
        input_tokens_estimate=input_tokens,
        output_reservation=output,
    )


# ── 1. Input overflow → hard refusal (unshrinkable) ─────────────────────


class TestInputOverflowRefusal:
    def test_input_alone_exceeds_limit_raises_even_with_api_kwargs(self):
        # input=200K, limit=200K → unshrinkable (input AT the limit).
        budget = _budget(input_tokens=200_000, output=65_536)
        api_kwargs = {"max_tokens": 65_536, "messages": []}
        with pytest.raises(ContextCeilingExceeded):
            enforce_final_context_budget(
                budget, pre_cap=200_000, api_kwargs=api_kwargs
            )

    def test_input_alone_exceeds_limit_raises(self):
        budget = _budget(input_tokens=200_001, output=4_096)
        with pytest.raises(ContextCeilingExceeded):
            enforce_final_context_budget(budget, pre_cap=200_000)

    def test_input_overflow_is_unshrinkable_no_clamp(self):
        # Even with api_kwargs supplied, input overflow must raise — the
        # clamp must NOT lower the cap to hide a real input overflow.
        budget = _budget(input_tokens=250_000, output=4_096)
        api_kwargs = {"max_tokens": 4_096, "messages": []}
        with pytest.raises(ContextCeilingExceeded):
            enforce_final_context_budget(
                budget, pre_cap=200_000, api_kwargs=api_kwargs
            )


# ── 2. Output-only overflow → local clamp (shrinkable) ──────────────────


class TestOutputOnlyClamp:
    def test_output_overflow_clamps_cap_and_proceeds(self):
        # input=199K fits, but input(199K) + output(65536) = 264K > 200K.
        # Shrinkable: clamp cap to 200K - 199K = 1K, no raise.
        budget = _budget(input_tokens=199_000, output=65_536)
        api_kwargs = {"max_tokens": 65_536, "messages": []}
        # Must NOT raise.
        enforce_final_context_budget(
            budget, pre_cap=200_000, api_kwargs=api_kwargs
        )
        # The cap was clamped downward to the max legal allowance.
        assert api_kwargs["max_tokens"] == 200_000 - 199_000

    def test_output_overflow_clamps_to_exact_remaining_budget(self):
        budget = _budget(input_tokens=150_000, output=65_536)
        api_kwargs = {"max_tokens": 65_536, "messages": []}
        enforce_final_context_budget(budget, pre_cap=200_000, api_kwargs=api_kwargs)
        # clamp = 200K - 150K = 50K
        assert api_kwargs["max_tokens"] == 50_000

    def test_no_overflow_cap_unchanged(self):
        # total <= limit → no clamp, cap stays as requested.
        budget = _budget(input_tokens=100_000, output=65_536)
        api_kwargs = {"max_tokens": 65_536, "messages": []}
        enforce_final_context_budget(budget, pre_cap=200_000, api_kwargs=api_kwargs)
        assert api_kwargs["max_tokens"] == 65_536


# ── 5. Explicit cap is never INCREASED by the clamp ─────────────────────


class TestClampNeverIncreases:
    def test_small_explicit_cap_stays_when_under_allowed(self):
        # input=100K, requested cap=3000. allowed = 200K-100K = 100K.
        # 3000 < 100K → cap is NOT raised to 100K; it stays 3000.
        budget = _budget(input_tokens=100_000, output=3_000)
        api_kwargs = {"max_tokens": 3_000, "messages": []}
        enforce_final_context_budget(budget, pre_cap=200_000, api_kwargs=api_kwargs)
        assert api_kwargs["max_tokens"] == 3_000


# ── 6. Clamp respects the effective hard ceiling, not native context ────


class TestClampRespectsCeiling:
    def test_ceiling_lower_than_native_clamps_to_ceiling(self):
        # native pre_cap=256K, but ceiling=128K (profile max).
        # input=100K, cap=65536 → total=165K > 128K (ceiling).
        # Clamp to ceiling: 128K - 100K = 28K.
        budget = _budget(input_tokens=100_000, output=65_536)
        api_kwargs = {"max_tokens": 65_536, "messages": []}
        enforce_final_context_budget(
            budget, pre_cap=256_000, ceiling=128_000, api_kwargs=api_kwargs
        )
        # clamp = min(256K, 128K) - 100K = 28K — ceiling, not native.
        assert api_kwargs["max_tokens"] == 28_000

    def test_ceiling_is_the_binding_limit(self):
        # ceiling=128K, input=120K → allowed=8K (ceiling-bound).
        budget = _budget(input_tokens=120_000, output=65_536)
        api_kwargs = {"max_tokens": 65_536, "messages": []}
        enforce_final_context_budget(
            budget, pre_cap=256_000, ceiling=128_000, api_kwargs=api_kwargs
        )
        assert api_kwargs["max_tokens"] == 8_000


# ── 7. Transport-specific cap fields are clamped correctly ──────────────


class TestTransportFieldClamp:
    def test_openai_max_tokens_clamped(self):
        budget = _budget(input_tokens=199_000, output=65_536)
        api_kwargs = {"max_tokens": 65_536, "messages": []}
        enforce_final_context_budget(budget, pre_cap=200_000, api_kwargs=api_kwargs)
        assert api_kwargs["max_tokens"] == 1_000

    def test_openai_completion_tokens_clamped(self):
        budget = _budget(input_tokens=199_000, output=65_536)
        api_kwargs = {"max_completion_tokens": 65_536, "messages": []}
        enforce_final_context_budget(budget, pre_cap=200_000, api_kwargs=api_kwargs)
        assert api_kwargs["max_completion_tokens"] == 1_000

    def test_anthropic_max_tokens_clamped(self):
        budget = _budget(input_tokens=199_000, output=65_536)
        api_kwargs = {"max_tokens": 65_536, "messages": []}
        enforce_final_context_budget(budget, pre_cap=200_000, api_kwargs=api_kwargs)
        assert api_kwargs["max_tokens"] == 1_000

    def test_bedrock_nested_maxtokens_clamped(self):
        budget = _budget(input_tokens=199_000, output=65_536)
        api_kwargs = {
            "inferenceConfig": {"maxTokens": 65_536},
            "messages": [],
        }
        enforce_final_context_budget(budget, pre_cap=200_000, api_kwargs=api_kwargs)
        assert api_kwargs["inferenceConfig"]["maxTokens"] == 1_000


# ── Boundary unit tests: _set_final_request_output_cap ──────────────────


class TestSetFinalRequestOutputCap:
    def test_sets_max_tokens(self):
        kw = {"max_tokens": 65_536}
        _set_final_request_output_cap(kw, 1_000)
        assert kw["max_tokens"] == 1_000

    def test_sets_max_completion_tokens(self):
        kw = {"max_completion_tokens": 65_536}
        _set_final_request_output_cap(kw, 2_000)
        assert kw["max_completion_tokens"] == 2_000

    def test_sets_bedrock_nested(self):
        kw = {"inferenceConfig": {"maxTokens": 65_536}}
        _set_final_request_output_cap(kw, 3_000)
        assert kw["inferenceConfig"]["maxTokens"] == 3_000

    def test_no_explicit_cap_noop(self):
        # No cap field present → no-op (the resolver's Level-3/4 reservation
        # is not on the wire to clamp).
        kw = {"messages": []}
        _set_final_request_output_cap(kw, 1_000)
        assert "max_tokens" not in kw


# ── Boundary unit tests: _final_request_output_cap (read path) ──────────


class TestFinalRequestOutputCapRead:
    def test_reads_max_tokens(self):
        assert _final_request_output_cap({"max_tokens": 65_536}) == 65_536

    def test_reads_bedrock_nested(self):
        assert (
            _final_request_output_cap(
                {"inferenceConfig": {"maxTokens": 8_192}}
            )
            == 8_192
        )

    def test_no_cap_returns_none(self):
        assert _final_request_output_cap({"messages": []}) is None


# ── Integration: the failing test's scenario through the boundary ───────


class TestIntegrationClamp:
    def test_large_api_only_content_clamps_and_proceeds(self):
        """The exact scenario from
        ``test_output_cap_retry_with_large_api_only_content``: a 796K-char
        system prompt makes the API payload huge (~199K input tokens) while
        the persisted messages are tiny.  The gate must clamp the output cap
        to the remaining budget (not refuse) so the provider call proceeds
        with a legal cap.
        """
        sys_prompt = "S" * 796_000
        api_kwargs = {
            "model": "some/model",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "hello"},
            ],
            "max_tokens": 65_536,
        }
        budget = build_final_context_budget(
            api_kwargs, provider="openrouter", model="some/model"
        )
        # input ~199K, output 65536, total ~264K > 200K (effective window).
        # Shrinkable: clamp cap to ~1K, no raise.
        enforce_final_context_budget(
            budget, pre_cap=200_000, api_kwargs=api_kwargs
        )
        # Cap was reduced from 65536 to the remaining budget.
        assert api_kwargs["max_tokens"] < 65_536
        assert api_kwargs["max_tokens"] == 200_000 - budget.input_tokens_estimate
        # The post-clamp budget fits: input + clamped_cap == limit.
        assert budget.input_tokens_estimate + api_kwargs["max_tokens"] == 200_000
