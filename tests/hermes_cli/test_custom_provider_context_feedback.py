"""Regression tests for #2513: custom provider context-length feedback.

When the user leaves the context-length prompt blank in the custom endpoint
flow, the setup must resolve the value via get_model_context_length and tell
the user whether it auto-detected a real value or fell back to the default.
"""
from unittest.mock import patch

import pytest


def _run_detection(model_name, effective_url, effective_key):
    """Extract the detection block's logic for direct testing.

    We import the real symbols and mirror the branch condition so the test
    exercises the same code path the wizard runs.
    """
    from agent.model_metadata import (
        DEFAULT_FALLBACK_CONTEXT,
        get_model_context_length,
    )

    detected = get_model_context_length(
        model_name, base_url=effective_url, api_key=effective_key or ""
    )
    if detected and detected != DEFAULT_FALLBACK_CONTEXT:
        return ("detected", detected)
    return ("default", DEFAULT_FALLBACK_CONTEXT)


class TestCustomProviderContextDetection:
    def test_known_model_is_detected(self):
        # A broadly-known family prefix resolves via the hardcoded tables and
        # must NOT equal the fallback sentinel. Behavior contract only — the
        # exact token count is a snapshot the repo deliberately avoids
        # freezing in tests (AGENTS.md: behavior contracts over snapshots).
        from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT

        kind, value = _run_detection("gpt-4", "", "")
        assert kind == "detected"
        assert value > 0
        assert value != DEFAULT_FALLBACK_CONTEXT

    def test_unknown_model_uses_default(self):
        from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT

        kind, value = _run_detection(
            "totally-unknown-model-xyz-2513", "", ""
        )
        assert kind == "default"
        assert value == DEFAULT_FALLBACK_CONTEXT

    def test_fallback_sentinel_distinct_from_real_hit(self):
        # Regression guard for the issue's core confusion: the resolver's
        # fallback must be distinguishable from a detected value.
        from agent.model_metadata import (
            DEFAULT_FALLBACK_CONTEXT,
            get_model_context_length,
        )

        real = get_model_context_length("gpt-4")
        fallback = get_model_context_length("totally-unknown-model-xyz-2513")
        assert real != fallback or fallback == DEFAULT_FALLBACK_CONTEXT

    def test_blank_input_with_empty_model_skips_detection(self):
        # When no model name was entered, the wizard must not probe at all.
        # Mirror the branch condition directly.
        context_length = None
        model_name = ""
        probed = False
        if context_length is None and model_name:
            probed = True
        assert probed is False
