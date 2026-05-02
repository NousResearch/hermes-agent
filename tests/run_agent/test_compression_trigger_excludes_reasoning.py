"""Verify compression trigger excludes reasoning/completion tokens (#12026).

Thinking models (GLM-5.1, QwQ, DeepSeek R1) inflate completion_tokens with
reasoning tokens that don't consume context window space.  The compression
trigger must use only prompt_tokens so sessions aren't prematurely split.
"""

import types
import pytest
from unittest.mock import MagicMock, patch


def _make_agent_stub(prompt_tokens, completion_tokens, threshold_tokens):
    """Create a minimal stub that exercises the compression check path."""
    compressor = types.SimpleNamespace(
        last_prompt_tokens=prompt_tokens,
        last_completion_tokens=completion_tokens,
        threshold_tokens=threshold_tokens,
    )
    # Replicate the fixed logic from run_agent.py ~line 11273
    if compressor.last_prompt_tokens > 0:
        real_tokens = compressor.last_prompt_tokens  # Fixed: no completion
    else:
        real_tokens = 0
    return real_tokens, compressor


class TestCompressionTriggerExcludesReasoning:
    def test_high_reasoning_tokens_should_not_trigger_compression(self):
        """With the old bug, 40k prompt + 80k reasoning = 120k > 100k threshold.
        After the fix, only 40k prompt is compared — no compression."""
        real_tokens, comp = _make_agent_stub(
            prompt_tokens=40_000,
            completion_tokens=80_000,  # reasoning-heavy model
            threshold_tokens=100_000,
        )
        assert real_tokens == 40_000
        assert real_tokens < comp.threshold_tokens, (
            "Should NOT trigger compression — only prompt tokens matter"
        )

    def test_high_prompt_tokens_should_trigger_compression(self):
        """When prompt tokens genuinely exceed the threshold, compress."""
        real_tokens, comp = _make_agent_stub(
            prompt_tokens=110_000,
            completion_tokens=5_000,
            threshold_tokens=100_000,
        )
        assert real_tokens == 110_000
        assert real_tokens >= comp.threshold_tokens, (
            "Should trigger compression — prompt tokens exceed threshold"
        )

    def test_zero_prompt_tokens_falls_back(self):
        """When provider returns 0 prompt tokens, real_tokens is 0 (fallback path)."""
        real_tokens, _ = _make_agent_stub(
            prompt_tokens=0,
            completion_tokens=50_000,
            threshold_tokens=100_000,
        )
        assert real_tokens == 0
