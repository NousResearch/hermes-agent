"""Tests for session reset completeness (fixes #2635).

/clear and /new must not carry stale state into the next session.
Two fields were added after reset_session_state() was written and were
therefore never cleared:
  - ContextCompressor._previous_summary
  - AIAgent._user_turn_count
"""
import sys
import types
from pathlib import Path

import pytest

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Stub out optional heavy dependencies not installed in the test environment
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor


def _make_minimal_agent() -> AIAgent:
    """Return an AIAgent constructed with the absolute minimum args.

    We pass dummy values that bypass network calls and filesystem access.
    The object is never used to make API calls — only its attributes and
    reset_session_state() are exercised.
    """
    agent = AIAgent.__new__(AIAgent)  # skip __init__ entirely

    # Seed the exact attributes that reset_session_state() writes
    agent.session_total_tokens = 0
    agent.session_input_tokens = 0
    agent.session_output_tokens = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_reasoning_tokens = 0
    agent.session_api_calls = 0
    agent.session_estimated_cost_usd = 0.0
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "none"

    # The two fields under test
    agent._user_turn_count = 0
    agent.context_compressor = None  # will be set per-test as needed

    return agent


class TestResetSessionState:
    """reset_session_state() must clear ALL session-scoped state."""

    def test_previous_summary_cleared_on_reset(self):
        """Compression summary from old session must not leak into new session."""
        agent = _make_minimal_agent()
        compressor = ContextCompressor.__new__(ContextCompressor)
        compressor._previous_summary = "Old session summary about unrelated topic"
        # Seed counter attributes that reset_session_state touches
        compressor.last_prompt_tokens = 100
        compressor.last_completion_tokens = 50
        compressor.last_total_tokens = 150
        compressor.compression_count = 3
        compressor._context_probed = True

        agent.context_compressor = compressor

        agent.reset_session_state()

        assert compressor._previous_summary is None, (
            "_previous_summary must be None after reset; got: "
            f"{compressor._previous_summary!r}"
        )

    def test_user_turn_count_cleared_on_reset(self):
        """Turn counter must reset to 0 on new session."""
        agent = _make_minimal_agent()
        agent._user_turn_count = 7  # simulates turns accumulated in previous session
        agent.context_compressor = None

        agent.reset_session_state()

        assert agent._user_turn_count == 0, (
            f"_user_turn_count must be 0 after reset; got: {agent._user_turn_count}"
        )

    def test_both_fields_cleared_together(self):
        """Both stale fields are cleared in a single reset_session_state() call."""
        agent = _make_minimal_agent()
        agent._user_turn_count = 3

        compressor = ContextCompressor.__new__(ContextCompressor)
        compressor._previous_summary = "Stale summary"
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor.last_total_tokens = 0
        compressor.compression_count = 0
        compressor._context_probed = False
        agent.context_compressor = compressor

        agent.reset_session_state()

        assert agent._user_turn_count == 0
        assert compressor._previous_summary is None

    def test_reset_without_compressor_does_not_raise(self):
        """reset_session_state() must not raise when context_compressor is None."""
        agent = _make_minimal_agent()
        agent._user_turn_count = 2
        agent.context_compressor = None

        # Must not raise
        agent.reset_session_state()

        assert agent._user_turn_count == 0
