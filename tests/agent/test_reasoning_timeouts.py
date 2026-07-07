"""Test get_reasoning_stale_timeout_floor for DeepSeek V4 reasoning models."""

from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor


def test_deepseek_v4_reasoning_models_have_timeout_floor():
    """DeepSeek V4 reasoning models should have 600s timeout floor.

    DeepSeek V4 models (deepseek-v4-flash, deepseek-v4-pro) emit
    reasoning_content in a separate delta field before final content,
    requiring the same extended stale timeout floor as R1.
    See #60338.
    """
    # Direct model names (no aggregator prefix)
    assert get_reasoning_stale_timeout_floor("deepseek-v4-flash") == 600.0
    assert get_reasoning_stale_timeout_floor("deepseek-v4-pro") == 600.0

    # With aggregator prefixes (common usage)
    assert get_reasoning_stale_timeout_floor("deepseek/deepseek-v4-flash") == 600.0
    assert get_reasoning_stale_timeout_floor("deepseek/deepseek-v4-pro") == 600.0

    # With custom provider prefixes
    assert get_reasoning_stale_timeout_floor("opencode/deepseek-v4-flash") == 600.0
    assert get_reasoning_stale_timeout_floor("custom/deepseek-v4-pro") == 600.0


def test_deepseek_v4_timeout_floor_matches_r1():
    """V4 models should have the same floor as R1 (600s)."""
    v4_flash_floor = get_reasoning_stale_timeout_floor("deepseek-v4-flash")
    v4_pro_floor = get_reasoning_stale_timeout_floor("deepseek-v4-pro")
    r1_floor = get_reasoning_stale_timeout_floor("deepseek-r1")
    reasoner_floor = get_reasoning_stale_timeout_floor("deepseek-reasoner")

    assert v4_flash_floor == v4_pro_floor == r1_floor == reasoner_floor == 600.0


def test_deepseek_v4_does_not_match_non_reasoning_deepseek():
    """Non-reasoning deepseek variants should not match V4 patterns."""
    # deepseek-chat is not a reasoning model and should not match
    # the deepseek-v4 patterns (requires start-of-slug anchor).
    assert get_reasoning_stale_timeout_floor("deepseek-chat") is None
    assert get_reasoning_stale_timeout_floor("deepseek/deepseek-chat") is None

    # But V4 patterns should still match V4 models even with non-V4 prefixes
    assert get_reasoning_stale_timeout_floor("some-deepseek-v4-flash") is None  # wrong prefix
    assert get_reasoning_stale_timeout_floor("deepseek-v4-flash-model") == 600.0  # suffix OK
