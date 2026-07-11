"""
Regression test for issue #62353 - MiniMax M2.x missing from
reasoning_timeouts.py stale-stream floor.

Before the fix: get_reasoning_stale_timeout_floor("minimax/minimax-m2.7")
returned None, so the model was treated as a chat model and the default
180s stale-stream timeout killed it mid-think.

After the fix: returns 300 (matching the Grok reasoning / QwQ tier).
"""
import sys
sys.path.insert(0, "/tmp/hermes-pr-work-60859/hermes-agent")


def test_minimax_m2_7_has_300s_floor():
    """The exact model from the issue (minimax/minimax-m2.7) gets 300s floor."""
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    floor = get_reasoning_stale_timeout_floor("minimax/minimax-m2.7")
    assert floor == 300, (
        f"Issue #62353 regression: minimax/minimax-m2.7 should get 300s floor "
        f"but got {floor}"
    )


def test_minimax_m2_family_all_match():
    """All MiniMax M2.x variants should match the floor (m2, m2.5, m2.7)."""
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    for variant in ["minimax/minimax-m2", "minimax/minimax-m2.5", "minimax/minimax-m2.7"]:
        floor = get_reasoning_stale_timeout_floor(variant)
        assert floor == 300, f"{variant} should get 300s floor but got {floor}"


def test_minimax_m3_does_not_match():
    """MiniMax M3 (different family) should NOT match - no floor entry."""
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    floor = get_reasoning_stale_timeout_floor("minimax/minimax-m3")
    assert floor is None, (
        f"minimax-m3 should NOT match (it's a different family) but got {floor}"
    )


def test_existing_floors_unchanged():
    """Make sure adding MiniMax didn't break existing floors."""
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    assert get_reasoning_stale_timeout_floor("claude-opus-4-5") == 240
    assert get_reasoning_stale_timeout_floor("qwq-32b") == 300
    assert get_reasoning_stale_timeout_floor("o3-mini") == 300
    assert get_reasoning_stale_timeout_floor("nemotron-3-super") == 600