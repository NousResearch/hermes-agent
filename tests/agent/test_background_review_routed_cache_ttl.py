"""A routed background-review / ``/btw`` fork never keeps the 1h prompt-cache tier.

The fork inherits the parent's ``cli``/``telegram`` platform, so under the ``auto`` default it would
resolve to 1h. Routed to a different model its cache is cold and it calls back-to-back until done:
the 2x write price buys no retention, the same reasoning as the delegate child clamp (#104168).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import run_agent
import agent.background_review as bg_review
from agent.background_review import build_cache_parity_fork

from tests.agent.test_background_review_cache_parity import _make_agent_stub, _make_recorder_class
from tests.agent.test_background_review_routed_effort import ROUTED_RUNTIME


@pytest.mark.parametrize("resolved, expected", [("1h", "5m"), ("5m", "5m"), (None, None)])
def test_routed_fork_drops_the_1h_tier(resolved, expected):
    base = _make_recorder_class({})

    class _Fork(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._cache_ttl = resolved  # what _init_prompt_cache_config resolved for the parent's platform

    with patch.object(run_agent, "AIAgent", _Fork), \
            patch.object(bg_review, "_resolve_review_runtime", return_value=ROUTED_RUNTIME):
        fork, _rt, routed = build_cache_parity_fork(_make_agent_stub(run_agent.AIAgent), {}, max_iterations=5)
    assert routed
    assert fork._cache_ttl == expected
