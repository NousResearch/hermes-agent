"""Bounded hosts keep retry boosts below their preset without disabling recovery."""
from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import _consume_ephemeral_max_output


@pytest.mark.parametrize("requested,ceiling,expected", [(256, 128, 128), (64, 128, 64), (256, None, 256)])
def test_retry_output_cap_preserves_host_ceiling_and_smaller_overflow_limits(requested, ceiling, expected):
    agent = SimpleNamespace(_ephemeral_max_output_tokens=requested, _max_output_tokens_ceiling=ceiling)
    assert _consume_ephemeral_max_output(agent) == expected
    assert agent._ephemeral_max_output_tokens is None
    assert _consume_ephemeral_max_output(agent) is None
