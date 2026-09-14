"""Context-aware output default: metadata-derived max_tokens for custom providers.

Contract under test (PR 1 of the starvation fix set):
- No explicit max_tokens/ephemeral/profile cap + known context length → explicit
  ``max_tokens`` on the wire: min(65535, headroom), quantized to a 4,096 bucket.
- Unknown context (metadata lookup fails / 0) → None: legacy no-cap behavior.
- Prompt nearly fills the window → None (no safe default; starvation breaker owns
  the aftermath).
- Explicit max_tokens / ephemeral boost / profile cap → default never consulted.
- Native provider bases (openai/anthropic/google/openrouter) → untouched.
"""

from unittest.mock import patch

from agent.output_token_default import (
    OUTPUT_DEFAULT_FLOOR,
    OUTPUT_DEFAULT_QUANTUM,
    OUTPUT_DEFAULT_TARGET_CAP,
    context_aware_output_default,
    context_aware_output_default_for_params,
    estimate_prompt_tokens,
)


def _params(**overrides):
    base = dict(
        base_url="https://gateway.example/v1",
        provider="custom:gateway",
        api_key="k",
        custom_providers=[],
        config_context_length=None,
        messages=[{"role": "user", "content": "x" * 4_000}],  # ~1,000 tokens
        max_tokens=None,
        ephemeral_max_output_tokens=None,
    )
    base.update(overrides)
    return base


def _ctx(value):
    return patch(
        "agent.model_metadata.get_model_context_length", return_value=value
    )


class TestContextAwareDefault:
    def test_large_context_hosted_gateway_gets_target_cap(self):
        """270K-context model, 1K prompt → quantized target cap (65,535 quantizes
        down to the 61,440 bucket — within one bucket of the target)."""
        with _ctx(270_000):
            got = context_aware_output_default(
                model="m", base_url="https://gateway.example/v1", provider="custom:gateway",
                api_key="k", custom_providers=[], config_context_length=None,
                prompt_token_estimate=1_000,
            )
        assert got == 61_440
        assert got >= OUTPUT_DEFAULT_TARGET_CAP - OUTPUT_DEFAULT_QUANTUM

    def test_tight_local_context_clamps_to_headroom(self):
        """16K model, 12K prompt → headroom 3,872 → quantizes to the floor
        (1,024): never 65,535, never a 400 from vLLM."""
        with _ctx(16_384):
            got = context_aware_output_default(
                model="m", base_url="http://127.0.0.1:8080/v1", provider="custom:local",
                api_key=None, custom_providers=[], config_context_length=None,
                prompt_token_estimate=12_000,
            )
        assert got == OUTPUT_DEFAULT_FLOOR
        assert got < OUTPUT_DEFAULT_TARGET_CAP

    def test_unknown_context_returns_none(self):
        with _ctx(0):
            got = context_aware_output_default(
                model="m", base_url="https://gateway.example/v1", provider="custom",
                api_key=None, custom_providers=None, config_context_length=None,
                prompt_token_estimate=1_000,
            )
        assert got is None

    def test_metadata_failure_returns_none(self):
        with patch(
            "agent.model_metadata.get_model_context_length", side_effect=RuntimeError("boom")
        ):
            got = context_aware_output_default(
                model="m", base_url="https://gateway.example/v1", provider="custom",
                api_key=None, custom_providers=None, config_context_length=None,
                prompt_token_estimate=1_000,
            )
        assert got is None

    def test_prompt_filling_window_returns_none(self):
        with _ctx(16_384):
            got = context_aware_output_default(
                model="m", base_url="http://127.0.0.1:8080/v1", provider="custom",
                api_key=None, custom_providers=None, config_context_length=None,
                prompt_token_estimate=16_300,
            )
        assert got is None

    def test_quantization_buckets_stable(self):
        """Headroom drift within one 4,096 bucket produces the same wire value.
        20K context keeps headroom BELOW the target cap so quantization (not the
        target clamp) is what makes the two estimates agree."""
        with _ctx(20_000):
            a = context_aware_output_default(
                model="m", base_url="https://g.example/v1", provider="custom",
                api_key=None, custom_providers=None, config_context_length=None,
                prompt_token_estimate=10_000,
            )
        with _ctx(20_000):
            b = context_aware_output_default(
                model="m", base_url="https://g.example/v1", provider="custom",
                api_key=None, custom_providers=None, config_context_length=None,
                prompt_token_estimate=10_500,
            )
        assert a == b == 8_192


class TestApplyGate:
    def test_apply_populates_when_unset(self):
        with _ctx(270_000):
            got = context_aware_output_default_for_params("m", _params())
        assert got == 61_440

    def test_native_base_urls_untouched(self):
        for host in (
            "https://api.openai.com/v1", "https://api.openai.azure.com/v1",
            "https://api.anthropic.com/v1",
            "https://generativelanguage.googleapis.com/openai",
            "https://openrouter.ai/api/v1", "https://api.nous.ai/v1",
        ):
            with _ctx(270_000):
                got = context_aware_output_default_for_params("m", _params(base_url=host))
            assert got is None, host

    def test_no_base_url_untouched(self):
        """Legacy/unknown transports without a base URL keep legacy behavior."""
        assert context_aware_output_default_for_params("m", _params(base_url="")) is None


def test_estimate_prompt_tokens_counts_tool_call_args():
    """JSON-serialization estimate includes tool_calls arguments and image parts
    (Pro round-1: the old text-only walker missed them -> under-estimate)."""
    msgs = [
        {"role": "assistant", "tool_calls": [{"function": {"arguments": "{\"x\": \"" + "a" * 4_000 + "\"}"}}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 4_000}}]},
    ]
    est = estimate_prompt_tokens(msgs)
    # args(~4,010 chars) + image blob(~4,024 chars) serialized: well over 1,000 tokens
    assert est > 1_500


def test_estimate_prompt_tokens_multimodal_parts():
    """JSON serialization counts BOTH shapes (string and multipart text); the
    exact value includes JSON scaffolding, so assert a lower bound."""
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "x" * 400}]},
        {"role": "assistant", "content": "y" * 400},
    ]
    assert estimate_prompt_tokens(msgs) >= 200
