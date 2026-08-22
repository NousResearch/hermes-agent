"""Output caps must fit the context window at request-build time (2026-08-19).

Live failure mode: a provider whose default output cap equals the full context
window (CustomProfile.default_max_tokens=65536 against a 65536-token vLLM
server) makes EVERY request fail with a deterministic 400 — strict servers
reject input + max_tokens > window.  The post-error remediation
(_ephemeral_max_output_tokens) is one-shot, so each subsequent tool-loop call
re-failed, re-compressed (1-3 min each), and after max_compression_attempts
the turn died.

The fix: build_api_kwargs computes an output-token ceiling
(context_length - estimated input - margin) per request and the transport
clamps whatever cap was resolved (ephemeral / user / profile default) to it.
No session state is mutated, so fallback switches, /model, continuation
boosts, and compressor accounting are unaffected.
"""

from unittest.mock import MagicMock

from agent.model_metadata import compute_output_token_ceiling


def _make_agent(max_tokens=None, context_length=65536):
    """Minimal chat_completions AIAgent aimed at a vLLM-style custom provider."""
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    attrs = dict(
        api_mode="chat_completions",
        provider="custom",
        base_url="http://127.0.0.1:8000/v1",
        model="qwen3.8-27b",
        tools=[],
        max_tokens=max_tokens,
        reasoning_config=None,
        request_overrides={},
        session_id="t",
        _ephemeral_max_output_tokens=None,
        _ollama_num_ctx=None,
        openrouter_min_coding_score=None,
        providers_allowed=None,
        providers_denied=None,
        providers_ignored=None,
        providers_order=None,
        providers_quantizations=None,
        provider_sort=None,
        provider_require_parameters=None,
        provider_data_collection=None,
        api_key="x",
    )
    for name, value in attrs.items():
        setattr(agent, name, value)
    compressor = MagicMock()
    compressor.context_length = context_length
    agent.context_compressor = compressor
    agent._prepare_messages_for_non_vision_model = lambda m: m
    agent._resolved_api_call_timeout = lambda: 60
    agent._supports_reasoning_extra_body = lambda: False
    return agent


def _built_cap(agent, messages):
    kwargs = agent._build_api_kwargs(messages)
    return kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")


SMALL = [{"role": "user", "content": "hi"}]
# ~120K chars — the shape of the real outage: input alone leaves far less
# than the provider-default 65536 output cap inside a 65536-token window.
BIG = [{"role": "user", "content": "x" * 120_000}]


class TestComputeOutputTokenCeiling:
    def test_basic_headroom(self):
        assert compute_output_token_ceiling(65536, 30000) == 65536 - 30000 - 256

    def test_unknown_window_returns_none(self):
        assert compute_output_token_ceiling(None, 1000) is None
        assert compute_output_token_ceiling(0, 1000) is None

    def test_near_window_prompt_still_gets_a_clamp(self):
        # A prompt that fits but leaves less than the margin must NOT escape
        # unclamped — the full-window default would deterministically 400.
        assert compute_output_token_ceiling(65536, 65535) == 1
        assert compute_output_token_ceiling(65536, 65400) == 1
        assert compute_output_token_ceiling(65536, 65280) == 1
        assert compute_output_token_ceiling(65536, 65279) == 1

    def test_input_at_or_over_window_returns_none(self):
        # Input overflow is compression's job — no clamp, let the provider
        # report the real error.
        assert compute_output_token_ceiling(65536, 65536) is None
        assert compute_output_token_ceiling(65536, 80000) is None


class TestRequestBuildClamping:
    def test_full_window_provider_default_is_clamped(self):
        """The regression: the CustomProfile 65536 default must shrink so the
        request fits, on EVERY build — no error round-trip required."""
        agent = _make_agent(max_tokens=None)
        cap = _built_cap(agent, BIG)
        est_input_upper = 65536  # any clamp beats the old full-window cap
        assert cap < est_input_upper
        # And the request actually fits the window per the same estimator.
        from agent.model_metadata import estimate_request_tokens_rough
        assert cap + estimate_request_tokens_rough(BIG) <= 65536

    def test_clamp_repeats_on_subsequent_builds(self):
        agent = _make_agent(max_tokens=None)
        first = _built_cap(agent, BIG)
        assert _built_cap(agent, BIG) == first

    def test_small_prompt_keeps_user_cap(self):
        agent = _make_agent(max_tokens=16384)
        assert _built_cap(agent, SMALL) == 16384

    def test_ephemeral_override_is_clamped_too(self):
        agent = _make_agent(max_tokens=None)
        agent._ephemeral_max_output_tokens = 65536
        cap = _built_cap(agent, BIG)
        assert cap < 65536

    def test_large_window_provider_is_unaffected(self):
        agent = _make_agent(max_tokens=16384, context_length=200_000)
        assert _built_cap(agent, BIG) == 16384
