# tests/agent/benchmarks/test_tier1_token_efficiency.py
from agent.model_metadata import estimate_messages_tokens_rough
from tests.agent.benchmarks._report import record
from tests.agent.benchmarks.fixture_builders import make_loop_session


def test_1_1_dedup_friendly_loop_saves_tokens(compressor_pair, stub_summarizer):
    """30 iterations of read+patch on 3 paths = 90 redundant tool results.
    Pass 1.5 should supersede the older reads/patches; net token count
    after compaction should drop materially."""
    baseline, with_flags = compressor_pair
    msgs = make_loop_session(n_iterations=30, chars_per_read=4_000)
    pre_tokens = estimate_messages_tokens_rough(msgs)

    out_b = baseline.compress(msgs.copy(), current_tokens=pre_tokens)
    out_c = with_flags.compress(msgs.copy(), current_tokens=pre_tokens)

    tk_b = estimate_messages_tokens_rough(out_b)
    tk_c = estimate_messages_tokens_rough(out_c)
    record("1.1", "post_compact_tokens", tk_b, tk_c, "tok")
    record("1.1", "op_deduped_count",
           getattr(baseline, "_last_op_deduped", 0),
           getattr(with_flags, "_last_op_deduped", 0), "ops")

    # Acceptance: candidate ≤ 0.85 × baseline (≥ 15% improvement)
    assert tk_c <= 0.85 * tk_b, (
        f"Expected ≥15% token reduction; got {tk_c}/{tk_b} = {tk_c/tk_b:.2%}"
    )
    # And the dedup pass actually fired
    assert with_flags._last_op_deduped > 0


def test_1_2_neutral_session_does_not_lose_information(stub_summarizer):
    """No resource reuse → dedup should not fire. Token counts within ±5%.
    A larger drop would indicate dedup deleted non-redundant information.

    NOTE: This test isolates ``dedup_operations`` from the other qwen_aware
    flags. Using the integrated ``compressor_pair`` here would confound the
    measurement: ``threshold_absolute_max`` shrinks ``tail_token_budget``
    (since ``tail_token_budget = threshold_tokens * summary_target_ratio``)
    so the with_flags compressor compacts more aggressively for reasons
    unrelated to dedup. We construct a custom pair that differs ONLY in
    ``dedup_operations`` so the assertion measures dedup's effect alone.
    """
    from unittest.mock import patch
    from tests.agent.benchmarks.fixture_builders import make_neutral_session
    from agent.context_compressor import ContextCompressor

    def _make(**kw):
        defaults = dict(
            model="bench/qwen-instruct",
            threshold_percent=0.50,
            protect_first_n=3,
            protect_last_n=20,
            summary_target_ratio=0.20,
            quiet_mode=True,
            base_url="",
            api_key="",
            config_context_length=262_144,
            provider="bench",
            api_mode="chat_completions",
        )
        defaults.update(kw)
        with patch(
            "agent.context_compressor.get_model_context_length",
            return_value=262_144,
        ):
            return ContextCompressor(**defaults)

    # Identical configs except for dedup_operations.
    baseline = _make()
    dedup_only = _make(dedup_operations=True)

    msgs = make_neutral_session(n_turns=40, chars_per_turn=2_000)
    pre = estimate_messages_tokens_rough(msgs)

    out_b = baseline.compress(msgs.copy(), current_tokens=pre)
    out_c = dedup_only.compress(msgs.copy(), current_tokens=pre)
    tk_b = estimate_messages_tokens_rough(out_b)
    tk_c = estimate_messages_tokens_rough(out_c)
    record("1.2", "post_compact_tokens", tk_b, tk_c, "tok")

    # ratio in [0.95, 1.05] — dedup is a no-op on neutral content, so
    # output sizes must match closely. Anything outside this band means
    # dedup mistakenly collapsed unrelated tool calls.
    ratio = tk_c / tk_b if tk_b else 1.0
    assert 0.95 <= ratio <= 1.05, (
        f"Neutral session ratio {ratio:.3f} outside [0.95, 1.05] — "
        f"dedup_operations may be deleting non-redundant content"
    )
