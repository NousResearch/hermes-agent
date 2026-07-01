"""Tests for the infinite compaction loop fix (issue #40803).

When summary_target_ratio is large enough that the entire transcript fits
within soft_ceiling, the backward walk in _find_tail_cut_by_tokens never
breaks early.  Without the fix this produces either a no-op compression
(compress_start >= compress_end) or a single-message compression whose
summary-of-one overhead saves 0 tokens — both of which cause the
compressor to fire on every subsequent turn with no progress.

The fix adds two safeguards:
1. _find_tail_cut_by_tokens: when the whole transcript fits in soft_ceiling,
   re-walk with the raw (non-inflated) budget to find a meaningful cut.
2. compress(): when compress_start >= compress_end, record the no-op as
   an ineffective compression so should_compress() anti-thrashing fires.
"""

from unittest.mock import patch, MagicMock

import time

from agent.context_compressor import ContextCompressor, _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_compressor(**kwargs) -> ContextCompressor:
    defaults = dict(
        model="test-model",
        threshold_percent=0.65,
        protect_first_n=2,
        protect_last_n=3,
        quiet_mode=True,
    )
    defaults.update(kwargs)
    with patch("agent.context_compressor.get_model_context_length", return_value=96000):
        return ContextCompressor(**defaults)


def _build_session(n_turns: int, words_per_turn: int = 20) -> list:
    """Build a multi-turn conversation with a system prompt."""
    base_text = " ".join(["a"] * words_per_turn)
    messages = [{"role": "system", "content": "You are a helpful agent."}]
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"{base_text} (user turn {i})"})
        messages.append({"role": "assistant", "content": f"{base_text} (assistant turn {i})"})
    return messages


# ---------------------------------------------------------------------------
# Test: compress_start >= compress_end registers as ineffective
# ---------------------------------------------------------------------------

class TestCompressNoOpRegistersIneffective:
    """When compress_start >= compress_end, the fix records this as
    an ineffective compression so the anti-thrashing guard fires.

    We trigger this path by having _find_tail_cut_by_tokens return
    head_end (which makes compress_end = head_end + 1, same as
    compress_start after alignment)."""

    def test_no_op_increments_counter(self):
        """compress_start >= compress_end -> _ineffective_compression_count += 1"""
        comp = _make_compressor(
            summary_target_ratio=0.45,
            config_context_length=96000,
        )
        # A large session that passes the min_for_compress check
        messages = _build_session(10, words_per_turn=10)
        comp.last_prompt_tokens = 65_000

        # Mock _find_tail_cut_by_tokens to return head_end,
        # causing compress_start >= compress_end
        original = comp._find_tail_cut_by_tokens
        comp._find_tail_cut_by_tokens = lambda msgs, he: he  # force no-op

        result = comp.compress(messages, current_tokens=65_000)

        assert comp._ineffective_compression_count >= 1, (
            f"Expected ineffective_compression_count >= 1, got {comp._ineffective_compression_count}"
        )

    def test_no_op_sets_savings_to_zero(self):
        """compress_start >= compress_end -> _last_compression_savings_pct = 0"""
        comp = _make_compressor(
            summary_target_ratio=0.45,
            config_context_length=96000,
        )
        messages = _build_session(10, words_per_turn=10)
        comp.last_prompt_tokens = 65_000
        comp._find_tail_cut_by_tokens = lambda msgs, he: he  # force no-op

        comp.compress(messages, current_tokens=65_000)

        assert comp._last_compression_savings_pct == 0.0

    def test_two_no_ops_block_should_compress(self):
        """After 2 no-op compressions, should_compress returns False."""
        comp = _make_compressor(
            summary_target_ratio=0.45,
            config_context_length=96000,
        )
        messages = _build_session(10, words_per_turn=10)
        comp.last_prompt_tokens = 65_000
        comp._find_tail_cut_by_tokens = lambda msgs, he: he  # force no-op

        comp.compress(messages, current_tokens=65_000)
        comp.compress(messages, current_tokens=65_000)

        assert comp._ineffective_compression_count >= 2
        assert not comp.should_compress(65_000), (
            "should_compress should return False after 2+ ineffective compressions"
        )

    def test_no_op_returns_unchanged_messages(self):
        """compress_start >= compress_end -> messages returned unchanged"""
        comp = _make_compressor(
            summary_target_ratio=0.45,
            config_context_length=96000,
        )
        messages = _build_session(10, words_per_turn=10)
        comp.last_prompt_tokens = 65_000
        original_cut = comp._find_tail_cut_by_tokens
        comp._find_tail_cut_by_tokens = lambda msgs, he: he  # force no-op

        result = comp.compress(messages, current_tokens=65_000)

        assert len(result) == len(messages), (
            f"Expected unchanged message count {len(messages)}, got {len(result)}"
        )
        comp._find_tail_cut_by_tokens = original_cut


# ---------------------------------------------------------------------------
# Test: _find_tail_cut_by_tokens raw-budget fallback
# ---------------------------------------------------------------------------

class TestTailCutRawBudgetFallback:
    """When the entire transcript fits within soft_ceiling, the fix
    re-walks with the raw budget to find a meaningful cut point."""

    def test_meaningful_cut_with_large_ratio(self):
        """With summary_target_ratio=0.45, _find_tail_cut_by_tokens still
        leaves a meaningful compressable region."""
        comp = _make_compressor(
            summary_target_ratio=0.45,
            config_context_length=96000,
        )
        messages = _build_session(20, words_per_turn=20)
        head_end = comp._protect_head_size(messages)
        head_end = comp._align_boundary_forward(messages, head_end)

        cut = comp._find_tail_cut_by_tokens(messages, head_end)

        n = len(messages)
        middle_size = cut - head_end
        assert middle_size >= 3, (
            f"Expected at least 3 messages in compressable region, got {middle_size} "
            f"(cut={cut}, head_end={head_end}, n={n})"
        )

    def test_default_ratio_still_works(self):
        """Default ratio (0.20) should not be affected by the fix."""
        comp = _make_compressor(
            summary_target_ratio=0.20,
            config_context_length=96000,
        )
        messages = _build_session(20, words_per_turn=50)
        head_end = comp._protect_head_size(messages)
        head_end = comp._align_boundary_forward(messages, head_end)

        cut = comp._find_tail_cut_by_tokens(messages, head_end)

        n = len(messages)
        assert head_end < cut < n, (
            f"Expected head_end ({head_end}) < cut ({cut}) < n ({n})"
        )

    def test_proactive_fix_prevents_no_op_window(self):
        """The raw-budget fallback in _find_tail_cut_by_tokens should prevent
        compress_start >= compress_end for the exact issue scenario:
        context_length=96000, summary_target_ratio=0.45."""
        comp = _make_compressor(
            summary_target_ratio=0.45,
            config_context_length=96000,
        )
        # Simulate the issue scenario: 16 messages, all fitting in soft_ceiling
        messages = _build_session(8, words_per_turn=30)  # 17 messages
        head_end = comp._protect_head_size(messages)
        head_end = comp._align_boundary_forward(messages, head_end)

        cut = comp._find_tail_cut_by_tokens(messages, head_end)

        # With the fix, cut should be well past head_end
        assert cut > head_end + 1, (
            f"Expected cut ({cut}) > head_end ({head_end}) + 1, "
            f"meaning the compressable window is non-trivial"
        )


# ---------------------------------------------------------------------------
# Test: Effective compression resets counter
# ---------------------------------------------------------------------------

class TestEffectiveCompressionResetsCounter:
    """When compression actually saves tokens, the ineffective counter resets."""

    def test_effective_compression_resets_counter(self):
        """After an effective compression, _ineffective_compression_count = 0."""
        comp = _make_compressor(
            summary_target_ratio=0.20,
            config_context_length=96000,
        )
        messages = _build_session(30, words_per_turn=100)
        comp._generate_summary = MagicMock(return_value="Compacted summary of earlier turns.")
        comp.last_prompt_tokens = 65_000

        comp.compress(messages, current_tokens=65_000)

        assert comp._ineffective_compression_count == 0, (
            f"Expected 0 ineffective compressions with effective compression, "
            f"got {comp._ineffective_compression_count}"
        )


# ---------------------------------------------------------------------------
# Test: anti-thrashing in should_compress
# ---------------------------------------------------------------------------

class TestAntiThrashing:
    """Directly test the should_compress anti-thrashing guard."""

    def test_ineffective_count_2_blocks(self):
        """_ineffective_compression_count >= 2 -> should_compress returns False."""
        comp = _make_compressor(config_context_length=96000)
        comp.last_prompt_tokens = 65_000
        comp._ineffective_compression_count = 2
        assert not comp.should_compress(65_000)

    def test_ineffective_count_1_allows(self):
        """_ineffective_compression_count = 1 -> should_compress still True."""
        comp = _make_compressor(config_context_length=96000)
        comp.last_prompt_tokens = 65_000
        comp._ineffective_compression_count = 1
        assert comp.should_compress(65_000)

    def test_below_threshold_allows(self):
        """Tokens below threshold -> should_compress returns False regardless."""
        comp = _make_compressor(config_context_length=96000)
        comp.last_prompt_tokens = 10_000
        assert not comp.should_compress(10_000)


# ---------------------------------------------------------------------------
# Test: summary-LLM cooldown guard in should_compress (#11529)
# ---------------------------------------------------------------------------

class TestCooldownGuard:
    """should_compress() must skip compression while the summary LLM is in
    cooldown, otherwise a 429/transient failure re-fires _compress_context()
    every turn (inserting a fallback marker repeatedly) and freezes the CLI.
    """

    def test_active_cooldown_blocks(self):
        """A future cooldown deadline -> should_compress returns False even
        when tokens are over threshold."""
        comp = _make_compressor(config_context_length=96000)
        comp.last_prompt_tokens = 65_000
        comp._summary_failure_cooldown_until = time.monotonic() + 60
        assert not comp.should_compress(65_000)

    def test_expired_cooldown_allows(self):
        """A past cooldown deadline -> compression resumes normally."""
        comp = _make_compressor(config_context_length=96000)
        comp.last_prompt_tokens = 65_000
        comp._summary_failure_cooldown_until = time.monotonic() - 1
        assert comp.should_compress(65_000)

    def test_no_cooldown_allows(self):
        """The default (no cooldown set) does not block compression."""
        comp = _make_compressor(config_context_length=96000)
        comp.last_prompt_tokens = 65_000
        assert comp._summary_failure_cooldown_until == 0.0
        assert comp.should_compress(65_000)
