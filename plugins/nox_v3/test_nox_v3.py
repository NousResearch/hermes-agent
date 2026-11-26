"""
NOX V3 Plugin Tests
"""

import pytest
from nox_v3 import (
    get_config,
    get_state,
    update_state,
    is_enabled,
    get_mode,
    get_latency_budget,
    get_fast_path_threshold,
    can_use_tokens,
    record_token_usage,
    reset_daily_usage,
    record_verification_result,
)
from nox_v3.hooks import (
    get_nox_system_prompt,
    estimate_tokens,
    should_use_fast_path,
    parse_nox_reasoning,
    verify_reasoning,
    optimize_reasoning,
)


class TestNOXV3Basics:
    """Test basic NOX V3 functionality."""

    def test_default_config(self):
        """Test default configuration."""
        config = get_config()
        assert config["enabled"] == False
        assert config["mode"] == "balanced"
        assert config["max_daily_tokens"] == 10000
        assert config["latency_budget_ms"] == 50
        assert config["fast_path_threshold"] == 100

    def test_default_state(self):
        """Test default state."""
        state = get_state()
        assert state["daily_token_usage"] == 0
        assert state["session_count"] == 0
        assert state["verification_success_rate"] == 1.0

    def test_is_enabled(self):
        """Test enabled status."""
        assert is_enabled() == False

    def test_get_mode(self):
        """Test mode retrieval."""
        assert get_mode() == "balanced"

    def test_get_latency_budget(self):
        """Test latency budget retrieval."""
        assert get_latency_budget() == 50

    def test_get_fast_path_threshold(self):
        """Test fast path threshold retrieval."""
        assert get_fast_path_threshold() == 100


class TestTokenManagement:
    """Test token management."""

    def test_can_use_tokens(self):
        """Test token budget checking."""
        assert can_use_tokens(100) == True
        assert can_use_tokens(100000) == False

    def test_record_token_usage(self):
        """Test token usage recording."""
        initial = get_state()["daily_token_usage"]
        record_token_usage(100)
        assert get_state()["daily_token_usage"] == initial + 100

    def test_reset_daily_usage(self):
        """Test daily usage reset."""
        record_token_usage(1000)
        reset_daily_usage()
        assert get_state()["daily_token_usage"] == 0


class TestVerification:
    """Test verification functionality."""

    def test_record_verification_result(self):
        """Test verification result recording."""
        record_verification_result(True)
        record_verification_result(False)
        state = get_state()
        assert state["verification_count"] >= 2
        assert 0 <= state["verification_success_rate"] <= 1


class TestNOXSystemPrompt:
    """Test NOX system prompt generation."""

    def test_conservative_prompt(self):
        """Test conservative mode prompt."""
        prompt = get_nox_system_prompt("conservative")
        assert "Conservative Mode" in prompt
        assert "keep terms readable" in prompt.lower()

    def test_balanced_prompt(self):
        """Test balanced mode prompt."""
        prompt = get_nox_system_prompt("balanced")
        assert "Balanced Mode" in prompt
        assert "symbolic logic" in prompt.lower()

    def test_aggressive_prompt(self):
        """Test aggressive mode prompt."""
        prompt = get_nox_system_prompt("aggressive")
        assert "Aggressive Mode" in prompt
        assert "maximum shorthand" in prompt.lower()


class TestTokenEstimation:
    """Test token estimation."""

    def test_estimate_tokens(self):
        """Test token estimation."""
        # Rough approximation: ~4 characters per token
        text = "This is a test sentence."
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens < len(text)


class TestFastPath:
    """Test fast path logic."""

    def test_should_use_fast_path(self):
        """Test fast path decision."""
        short_response = "Yes"
        long_response = "This is a very long response that exceeds the fast path threshold and should trigger verification processing."

        assert should_use_fast_path(short_response) == True
        assert should_use_fast_path(long_response) == False


class TestNOXParsing:
    """Test NOX reasoning parsing."""

    def test_parse_nox_reasoning(self):
        """Test parsing NOX reasoning from response."""
        response = """FACT[cats→mammals]
FACT[Fluffy→cat]
INFER[Fluffy→mammal]

Yes, Fluffy is a mammal."""

        nox_reasoning, final_answer = parse_nox_reasoning(response)

        assert nox_reasoning is not None
        assert "FACT[" in nox_reasoning
        assert final_answer is not None
        assert "Fluffy is a mammal" in final_answer

    def test_parse_no_nox_reasoning(self):
        """Test parsing response without NOX reasoning."""
        response = "Yes, Fluffy is a mammal."

        nox_reasoning, final_answer = parse_nox_reasoning(response)

        assert nox_reasoning is None
        assert final_answer == response


class TestVerification:
    """Test verification logic."""

    def test_verify_valid_reasoning(self):
        """Test verification of valid reasoning."""
        reasoning = "FACT[cats→mammals]\nFACT[Fluffy→cat]\nINFER[Fluffy→mammal]"
        query = "Is Fluffy a mammal?"

        is_valid, error = verify_reasoning(reasoning, query)

        assert is_valid == True
        assert error == ""

    def test_verify_empty_reasoning(self):
        """Test verification of empty reasoning."""
        reasoning = ""
        query = "Test query"

        is_valid, error = verify_reasoning(reasoning, query)

        assert is_valid == False
        assert "Empty" in error

    def test_verify_no_structure(self):
        """Test verification of reasoning without structure."""
        reasoning = "This is just plain text without NOX structure."
        query = "Test query"

        is_valid, error = verify_reasoning(reasoning, query)

        assert is_valid == False
        assert "structure" in error.lower()

    def test_verify_review_tag(self):
        """Test verification with REVIEW tag."""
        reasoning = "FACT[test]\nREVIEW: uncertain"
        query = "Test query"

        is_valid, error = verify_reasoning(reasoning, query)

        assert is_valid == False
        assert "REVIEW" in error


class TestOptimization:
    """Test optimization logic."""

    def test_optimize_conservative(self):
        """Test conservative optimization."""
        reasoning = "FACT[cats are mammals]\nFACT[Fluffy is a cat]"
        optimized = optimize_reasoning(reasoning, "conservative")

        assert "FACT[" in optimized
        # Conservative should just clean whitespace
        assert len(optimized) <= len(reasoning)

    def test_optimize_balanced(self):
        """Test balanced optimization."""
        reasoning = "FACT[cats are mammals]\nFACT[Fluffy is a cat]"
        optimized = optimize_reasoning(reasoning, "balanced")

        assert "FACT[" in optimized
        # Balanced should remove filler words
        assert " are " not in optimized

    def test_optimize_aggressive(self):
        """Test aggressive optimization."""
        reasoning = "FACT[cats implies mammals]\nFACT[Fluffy implies cat]"
        optimized = optimize_reasoning(reasoning, "aggressive")

        assert "FACT[" in optimized
        # Aggressive should use symbols
        assert "->" in optimized or "→" in optimized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
