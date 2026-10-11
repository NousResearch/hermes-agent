"""Dead `_candidate_pool_exhausted` helper must stay deleted; live path is `_pool_exhaustion_detail`."""

from __future__ import annotations


def test_candidate_pool_exhausted_helper_removed_live_path_intact():
    import agent.chat_completion_helpers as helpers
    from agent.chat_completion_helpers_pool import _pool_exhaustion_detail

    assert not hasattr(helpers, "_candidate_pool_exhausted")
    assert callable(_pool_exhaustion_detail)
