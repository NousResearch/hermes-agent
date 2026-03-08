#!/usr/bin/env python3
"""Tests for model validation module."""

import pytest
from hermes_cli.model_validation import (
    validate_model_name,
    format_validation_result,
    interactive_model_validation,
    KNOWN_MODELS,
    MODEL_PATTERNS,
)


class TestValidateModelName:
    """Tests for validate_model_name function."""
    
    def test_valid_openrouter_format(self):
        """OpenRouter provider/model format should be valid."""
        valid_models = [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "meta-llama/llama-3.3-70b-instruct",
            "google/gemini-2.0-flash-001",
            "mistralai/mistral-large",
            "deepseek/deepseek-r1",
        ]
        for model in valid_models:
            is_valid, warning, suggestions = validate_model_name(model)
            assert is_valid, f"Model {model} should be valid"
            assert warning is None
            assert suggestions == []
    
    def test_valid_with_variant(self):
        """Models with :variant suffix should be valid."""
        is_valid, warning, _ = validate_model_name("anthropic/claude-3.5-sonnet:beta")
        assert is_valid
        assert warning is None
    
    def test_valid_direct_provider_models(self):
        """Direct provider model names should be valid."""
        valid_models = [
            "gpt-4o",
            "claude-3.5-sonnet",
            "gemini-pro",
            "llama-3.3-70b",
            "mistral-large",
            "deepseek-chat",
        ]
        for model in valid_models:
            is_valid, warning, _ = validate_model_name(model)
            assert is_valid, f"Model {model} should be valid"
    
    def test_invalid_empty_model(self):
        """Empty model name should be invalid."""
        is_valid, warning, _ = validate_model_name("")
        assert not is_valid
        assert "cannot be empty" in warning
        
        is_valid, warning, _ = validate_model_name("   ")
        assert not is_valid
        assert "cannot be empty" in warning
    
    def test_invalid_random_string(self):
        """Random strings should be invalid with suggestions."""
        is_valid, warning, suggestions = validate_model_name("asdf1234")
        assert not is_valid
        assert "doesn't match any known model pattern" in warning
    
    def test_invalid_model_gets_suggestions(self):
        """Invalid model similar to known models should get suggestions."""
        # Typo of "anthropic/claude-3.5-sonnet"
        is_valid, warning, suggestions = validate_model_name("anthropic/claude-3.5-sonet")
        assert not is_valid
        # Should suggest the correct model
        assert any("claude" in s.lower() for s in suggestions)
    
    def test_case_insensitive_validation(self):
        """Validation should be case-insensitive for patterns."""
        is_valid, _, _ = validate_model_name("OpenAI/GPT-4o")
        assert is_valid
        
        is_valid, _, _ = validate_model_name("ANTHROPIC/CLAUDE-3.5-SONNET")
        assert is_valid


class TestFormatValidationResult:
    """Tests for format_validation_result function."""
    
    def test_valid_returns_none(self):
        """Valid model should return None."""
        result = format_validation_result("anthropic/claude-3.5-sonnet", True, None, [])
        assert result is None
    
    def test_invalid_returns_message(self):
        """Invalid model should return warning message."""
        result = format_validation_result(
            "invalid-model",
            False,
            "Model doesn't match patterns",
            ["anthropic/claude-3.5-sonnet"]
        )
        assert result is not None
        assert "⚠️" in result
        assert "Model doesn't match patterns" in result
        assert "anthropic/claude-3.5-sonnet" in result
    
    def test_includes_did_you_mean(self):
        """Should include 'Did you mean' when suggestions exist."""
        result = format_validation_result(
            "invalid",
            False,
            "Invalid",
            ["suggestion1", "suggestion2"]
        )
        assert "Did you mean" in result
        assert "suggestion1" in result
        assert "suggestion2" in result


class TestInteractiveModelValidation:
    """Tests for interactive_model_validation function."""
    
    def test_valid_model_proceeds(self):
        """Valid model should return proceed=True with empty message."""
        proceed, message = interactive_model_validation("anthropic/claude-3.5-sonnet")
        assert proceed is True
        assert message == ""
    
    def test_invalid_model_still_proceeds_with_warning(self):
        """Invalid model should still proceed but with warning."""
        proceed, message = interactive_model_validation("invalid-model")
        assert proceed is True  # Still allow setting
        assert len(message) > 0  # But show warning


class TestKnownModels:
    """Tests for KNOWN_MODELS list."""
    
    def test_known_models_all_valid(self):
        """All models in KNOWN_MODELS should pass validation."""
        for model in KNOWN_MODELS:
            is_valid, warning, _ = validate_model_name(model)
            assert is_valid, f"Known model {model} should be valid but got warning: {warning}"


class TestModelPatterns:
    """Tests for MODEL_PATTERNS regex patterns."""
    
    def test_openrouter_pattern_basic(self):
        """OpenRouter pattern should match provider/model format."""
        pattern = MODEL_PATTERNS["openrouter"]
        assert pattern.match("provider/model")
        assert pattern.match("anthropic/claude-3.5-sonnet")
        assert pattern.match("meta-llama/llama-3.3-70b-instruct")
        assert pattern.match("openai/gpt-4o:variant")
        assert not pattern.match("just-a-model")
        assert not pattern.match("/model")
        assert not pattern.match("provider/")
    
    def test_openai_pattern(self):
        """OpenAI pattern should match GPT models."""
        pattern = MODEL_PATTERNS["openai"]
        assert pattern.match("gpt-4o")
        assert pattern.match("gpt-3.5-turbo")
        assert pattern.match("o1-mini")
        assert not pattern.match("claude-3.5")
    
    def test_anthropic_pattern(self):
        """Anthropic pattern should match Claude models."""
        pattern = MODEL_PATTERNS["anthropic"]
        assert pattern.match("claude-3.5-sonnet")
        assert pattern.match("claude-opus-4")
        assert not pattern.match("gpt-4o")
