"""Test coverage for agent/thinking_timeout_guidance.py — zero prior coverage.

Tests the detection logic (is_thinking_timeout) and the guidance message
builder (build_thinking_timeout_guidance) without driving the full retry
loop. All inputs are constructed inline — no network, no API calls.
"""

import pytest

from agent.thinking_timeout_guidance import (
    build_thinking_timeout_guidance,
    is_thinking_timeout,
)


class _FakeClassified:
    def __init__(self, reason_value="timeout"):
        self.reason = type("R", (), {"value": reason_value})()


REASONING_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
NON_REASONING_MODEL = "meta-llama/llama-3.1-8b-instruct"


class TestIsThinkingTimeout:
    def test_reasoning_model_transport_kill(self):
        assert is_thinking_timeout(
            _FakeClassified("timeout"), REASONING_MODEL, "broken pipe"
        ) is True

    def test_non_reasoning_model_always_false(self):
        assert is_thinking_timeout(
            _FakeClassified("timeout"), NON_REASONING_MODEL, "broken pipe"
        ) is False

    def test_non_timeout_reason_always_false(self):
        assert is_thinking_timeout(
            _FakeClassified("rate_limit"), REASONING_MODEL, "broken pipe"
        ) is False

    def test_no_transport_substring_false(self):
        assert is_thinking_timeout(
            _FakeClassified("timeout"), REASONING_MODEL, "invalid api key"
        ) is False

    @pytest.mark.parametrize("substr", [
        "broken pipe", "errno 32", "remote protocol",
        "connection reset", "connection lost", "peer closed",
        "server disconnected",
    ])
    def test_all_transport_substrings_detected(self, substr):
        assert is_thinking_timeout(
            _FakeClassified("timeout"), REASONING_MODEL, substr
        ) is True


class TestBuildGuidance:
    def test_contains_config_snippet(self):
        text = build_thinking_timeout_guidance("nvidia", REASONING_MODEL)
        assert "stale_timeout_seconds: 900" in text
        assert "providers.nvidia" in text

    def test_custom_label(self):
        text = build_thinking_timeout_guidance(
            "nvidia", REASONING_MODEL, model_label="Nemotron 3 Ultra"
        )
        assert "Nemotron 3 Ultra" in text

    def test_fallback_label_is_slug(self):
        text = build_thinking_timeout_guidance("openai", "o3")
        assert "o3" in text
