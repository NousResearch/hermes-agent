"""Tests for agent.model_capabilities — pre-flight validation."""

from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import pytest

from agent.model_capabilities import (
    ModelCapabilities,
    KNOWN_CAPABILITIES,
    get_model_capabilities,
    validate_preflight,
    check_model_deprecation,
    validate_api_key_format,
    _KEY_PREFIX_PATTERNS,
    _messages_contain_images,
    _normalize_model_name,
)


# ---------------------------------------------------------------------------
# ModelCapabilities dataclass
# ---------------------------------------------------------------------------


class TestModelCapabilities:
    def test_defaults(self):
        caps = ModelCapabilities()
        assert caps.supports_tools is True
        assert caps.supports_vision is False
        assert caps.supports_reasoning is False
        assert caps.supports_streaming is True
        assert caps.max_output_tokens is None
        assert caps.context_window is None
        assert caps.source == "default"

    def test_custom_values(self):
        caps = ModelCapabilities(
            supports_tools=False,
            supports_vision=True,
            supports_reasoning=True,
            source="catalog",
        )
        assert caps.supports_tools is False
        assert caps.supports_vision is True
        assert caps.supports_reasoning is True
        assert caps.source == "catalog"


# ---------------------------------------------------------------------------
# Pattern matching / KNOWN_CAPABILITIES
# ---------------------------------------------------------------------------


class TestKnownCapabilities:
    """Verify that KNOWN_CAPABILITIES patterns match expected model names."""

    def test_has_entries(self):
        assert len(KNOWN_CAPABILITIES) > 0

    def test_all_entries_are_tuples(self):
        for entry in KNOWN_CAPABILITIES:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            pattern, caps = entry
            assert isinstance(pattern, str)
            assert isinstance(caps, ModelCapabilities)


# ---------------------------------------------------------------------------
# _normalize_model_name
# ---------------------------------------------------------------------------


class TestNormalizeModelName:
    def test_no_prefix(self):
        assert _normalize_model_name("claude-sonnet-4") == "claude-sonnet-4"

    def test_provider_prefix(self):
        assert _normalize_model_name("anthropic/claude-sonnet-4") == "claude-sonnet-4"

    def test_openrouter_prefix(self):
        assert _normalize_model_name("openai/gpt-4o") == "gpt-4o"

    def test_deep_path(self):
        assert _normalize_model_name("Qwen/Qwen3.5-397B-A17B") == "Qwen3.5-397B-A17B"


# ---------------------------------------------------------------------------
# get_model_capabilities
# ---------------------------------------------------------------------------


class TestGetModelCapabilities:
    """Test capability lookups for specific models."""

    # --- Claude models ---

    def test_claude_sonnet_4(self):
        caps = get_model_capabilities("claude-sonnet-4")
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_reasoning is True
        assert caps.source == "catalog"

    def test_claude_opus_4_6(self):
        caps = get_model_capabilities("claude-opus-4-6")
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_reasoning is True

    def test_claude_sonnet_4_6_openrouter(self):
        caps = get_model_capabilities("anthropic/claude-sonnet-4.6")
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_reasoning is True

    def test_claude_haiku_4_5(self):
        caps = get_model_capabilities("claude-haiku-4-5")
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_reasoning is True

    def test_claude_3_5_sonnet(self):
        caps = get_model_capabilities("claude-3-5-sonnet-20241022")
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_reasoning is True

    def test_claude_3_opus(self):
        caps = get_model_capabilities("claude-3-opus-20240229")
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_reasoning is False

    # --- OpenAI models ---

    def test_gpt_4o(self):
        caps = get_model_capabilities("gpt-4o")
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_reasoning is False

    def test_gpt_4o_mini(self):
        caps = get_model_capabilities("gpt-4o-mini")
        assert caps.supports_tools is True
        assert caps.supports_vision is True

    def test_gpt_4_1(self):
        caps = get_model_capabilities("gpt-4.1")
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_reasoning is True

    def test_gpt_3_5_turbo(self):
        caps = get_model_capabilities("gpt-3.5-turbo")
        assert caps.supports_tools is True
        assert caps.supports_vision is False
        assert caps.supports_reasoning is False

    def test_gpt_5_4(self):
        caps = get_model_capabilities("gpt-5.4")
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_reasoning is True

    def test_openai_gpt_5_4_openrouter(self):
        caps = get_model_capabilities("openai/gpt-5.4")
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_reasoning is True

    def test_o1_reasoning(self):
        caps = get_model_capabilities("o1")
        assert caps.supports_reasoning is True

    def test_o3_reasoning(self):
        caps = get_model_capabilities("o3")
        assert caps.supports_reasoning is True

    def test_o4_mini_reasoning(self):
        caps = get_model_capabilities("o4-mini")
        assert caps.supports_reasoning is True

    # --- Google Gemini ---

    def test_gemini_2_5_pro(self):
        caps = get_model_capabilities("gemini-2.5-pro")
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_reasoning is True

    def test_gemini_3_pro_preview(self):
        caps = get_model_capabilities("gemini-3-pro-preview")
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_reasoning is True

    def test_gemini_2_0_flash(self):
        caps = get_model_capabilities("gemini-2.0-flash")
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_reasoning is False

    def test_gemini_openrouter(self):
        caps = get_model_capabilities("google/gemini-3-pro-preview")
        assert caps.supports_tools is True
        assert caps.supports_vision is True

    # --- DeepSeek ---

    def test_deepseek_chat(self):
        caps = get_model_capabilities("deepseek-chat")
        assert caps.supports_tools is True
        assert caps.supports_vision is False
        assert caps.supports_reasoning is False

    def test_deepseek_reasoner(self):
        caps = get_model_capabilities("deepseek-reasoner")
        assert caps.supports_tools is True
        assert caps.supports_vision is False
        assert caps.supports_reasoning is True

    # --- Meta Llama ---

    def test_llama_3_1_70b(self):
        caps = get_model_capabilities("llama-3.1-70b")
        assert caps.supports_tools is True
        assert caps.supports_vision is False
        assert caps.supports_reasoning is False

    def test_llama_3_2_3b(self):
        caps = get_model_capabilities("llama-3.2-3b")
        assert caps.supports_tools is True
        assert caps.supports_vision is False

    # --- Qwen ---

    def test_qwen_2_5_72b(self):
        caps = get_model_capabilities("qwen-2.5-72b")
        assert caps.supports_tools is True
        assert caps.supports_vision is False
        assert caps.supports_reasoning is False

    def test_qwen_vl(self):
        caps = get_model_capabilities("qwen-vl-plus")
        assert caps.supports_tools is True
        assert caps.supports_vision is True

    def test_qwq(self):
        caps = get_model_capabilities("qwq-32b")
        assert caps.supports_reasoning is True

    # --- Mistral ---

    def test_mistral_large(self):
        caps = get_model_capabilities("mistral-large-latest")
        assert caps.supports_tools is True
        assert caps.supports_vision is False
        assert caps.supports_reasoning is False

    def test_pixtral(self):
        caps = get_model_capabilities("pixtral-large-latest")
        assert caps.supports_tools is True
        assert caps.supports_vision is True

    # --- Grok ---

    def test_grok(self):
        caps = get_model_capabilities("grok-4.20-beta")
        assert caps.supports_tools is True
        assert caps.supports_vision is True

    # --- Unknown model ---

    def test_unknown_model(self):
        caps = get_model_capabilities("totally-unknown-model-xyz")
        assert caps.supports_tools is True  # Safe default
        assert caps.supports_vision is False  # Conservative
        assert caps.supports_reasoning is False  # Conservative
        assert caps.source == "default"

    def test_empty_model(self):
        caps = get_model_capabilities("")
        assert caps.source == "default"

    def test_none_provider(self):
        caps = get_model_capabilities("claude-sonnet-4", provider=None)
        assert caps.supports_tools is True


# ---------------------------------------------------------------------------
# _messages_contain_images
# ---------------------------------------------------------------------------


class TestMessagesContainImages:
    def test_no_images(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert _messages_contain_images(messages) is False

    def test_text_list_content_no_images(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Hello"},
            ]},
        ]
        assert _messages_contain_images(messages) is False

    def test_image_url_content(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ]},
        ]
        assert _messages_contain_images(messages) is True

    def test_image_type_content(self):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "source": {"data": "base64data"}},
            ]},
        ]
        assert _messages_contain_images(messages) is True

    def test_empty_messages(self):
        assert _messages_contain_images([]) is False


# ---------------------------------------------------------------------------
# validate_preflight
# ---------------------------------------------------------------------------


class TestValidatePreflight:
    """Test pre-flight validation generates correct warnings."""

    def test_capable_model_no_warnings(self):
        """A fully capable model should produce no warnings."""
        warnings = validate_preflight(
            model="claude-sonnet-4",
            provider="anthropic",
            tools=[{"type": "function", "function": {"name": "test"}}],
            messages=[
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ]},
            ],
            reasoning_config={"effort": "medium"},
        )
        assert warnings == []

    def test_tools_warning_for_old_llama(self):
        """Old llama model should warn about tools."""
        warnings = validate_preflight(
            model="llama-2-70b",
            provider="openrouter",
            tools=[{"type": "function", "function": {"name": "test"}}],
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_config=None,
        )
        assert len(warnings) == 1
        assert "tool" in warnings[0].lower()

    def test_vision_warning_for_gpt35(self):
        """GPT-3.5 should warn about image content."""
        warnings = validate_preflight(
            model="gpt-3.5-turbo",
            provider="openai",
            tools=None,
            messages=[
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ]},
            ],
            reasoning_config=None,
        )
        assert len(warnings) == 1
        assert "vision" in warnings[0].lower()

    def test_reasoning_warning_for_gpt4o(self):
        """GPT-4o should warn about reasoning config."""
        warnings = validate_preflight(
            model="gpt-4o",
            provider="openai",
            tools=None,
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_config={"effort": "high"},
        )
        assert len(warnings) == 1
        assert "reasoning" in warnings[0].lower()

    def test_reasoning_effort_none_no_warning(self):
        """reasoning effort=none should NOT produce a warning."""
        warnings = validate_preflight(
            model="gpt-4o",
            provider="openai",
            tools=None,
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_config={"effort": "none"},
        )
        assert warnings == []

    def test_reasoning_effort_none_value_no_warning(self):
        """reasoning effort=None (Python None) should NOT produce a warning."""
        warnings = validate_preflight(
            model="gpt-4o",
            provider="openai",
            tools=None,
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_config={"effort": None},
        )
        assert warnings == []

    def test_no_warnings_for_unknown_model(self):
        """Unknown models should produce NO warnings (we don't know enough)."""
        warnings = validate_preflight(
            model="totally-unknown-model",
            provider="custom",
            tools=[{"type": "function", "function": {"name": "test"}}],
            messages=[
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ]},
            ],
            reasoning_config={"effort": "high"},
        )
        assert warnings == []

    def test_multiple_warnings(self):
        """A model with multiple incompatibilities should produce multiple warnings."""
        warnings = validate_preflight(
            model="llama-2-70b",
            provider="openrouter",
            tools=[{"type": "function", "function": {"name": "test"}}],
            messages=[
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ]},
            ],
            reasoning_config={"effort": "high"},
        )
        # llama-2 doesn't support tools, vision, or reasoning
        assert len(warnings) == 3

    def test_empty_tools_no_warning(self):
        """Empty tools list should not trigger a tools warning."""
        warnings = validate_preflight(
            model="llama-2-70b",
            provider="openrouter",
            tools=[],
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_config=None,
        )
        assert warnings == []

    def test_none_tools_no_warning(self):
        """None tools should not trigger a tools warning."""
        warnings = validate_preflight(
            model="llama-2-70b",
            provider="openrouter",
            tools=None,
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_config=None,
        )
        assert warnings == []

    def test_no_messages_no_vision_warning(self):
        """No messages should not trigger a vision warning."""
        warnings = validate_preflight(
            model="gpt-3.5-turbo",
            provider="openai",
            tools=None,
            messages=None,
            reasoning_config=None,
        )
        assert warnings == []

    def test_deepseek_chat_with_images(self):
        """DeepSeek chat should warn about images."""
        warnings = validate_preflight(
            model="deepseek-chat",
            provider="deepseek",
            tools=None,
            messages=[
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ]},
            ],
            reasoning_config=None,
        )
        assert len(warnings) == 1
        assert "vision" in warnings[0].lower()

    def test_mistral_with_reasoning(self):
        """Mistral should warn about reasoning config."""
        warnings = validate_preflight(
            model="mistral-large-latest",
            provider="mistral",
            tools=None,
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_config={"effort": "medium"},
        )
        assert len(warnings) == 1
        assert "reasoning" in warnings[0].lower()


# ---------------------------------------------------------------------------
# Default capabilities for unknown models
# ---------------------------------------------------------------------------


class TestDefaultCapabilities:
    def test_safe_defaults(self):
        """Unknown models get safe defaults — tools=True, vision=False, reasoning=False."""
        caps = get_model_capabilities("my-custom-finetune-v1")
        assert caps.supports_tools is True
        assert caps.supports_vision is False
        assert caps.supports_reasoning is False
        assert caps.supports_streaming is True
        assert caps.source == "default"

    def test_default_with_provider(self):
        """Provider hint doesn't change defaults for unknown models."""
        caps = get_model_capabilities("my-custom-model", provider="custom")
        assert caps.source == "default"


# ---------------------------------------------------------------------------
# check_model_deprecation
# ---------------------------------------------------------------------------


class TestCheckModelDeprecation:
    """Test model deprecation checking via models.dev."""

    def test_returns_none_for_non_deprecated_model(self):
        """Non-deprecated models should return None."""
        mock_models_dev = MagicMock()
        mock_models_dev.get_model_info.return_value = None
        mock_info_active = MagicMock()
        mock_info_active.status = "active"
        mock_models_dev.get_model_info_any_provider.return_value = mock_info_active
        with patch.dict("sys.modules", {"agent.models_dev": mock_models_dev}):
            result = check_model_deprecation("gpt-4o")
        assert result is None

    def test_returns_warning_for_deprecated_model(self):
        """Deprecated models should return a warning string."""
        mock_models_dev = MagicMock()
        mock_info_deprecated = MagicMock()
        mock_info_deprecated.status = "deprecated"
        mock_models_dev.get_model_info.return_value = mock_info_deprecated
        mock_models_dev.get_model_info_any_provider.return_value = mock_info_deprecated
        with patch.dict("sys.modules", {"agent.models_dev": mock_models_dev}):
            result = check_model_deprecation("gpt-3.5-turbo", provider="openai")
        assert result is not None
        assert "deprecated" in result.lower()
        assert "gpt-3.5-turbo" in result

    def test_returns_warning_via_any_provider(self):
        """When provider lookup returns None, falls back to any-provider search."""
        mock_models_dev = MagicMock()
        mock_info_deprecated = MagicMock()
        mock_info_deprecated.status = "deprecated"
        mock_models_dev.get_model_info.return_value = None
        mock_models_dev.get_model_info_any_provider.return_value = mock_info_deprecated
        with patch.dict("sys.modules", {"agent.models_dev": mock_models_dev}):
            result = check_model_deprecation("old-model", provider="openai")
        assert result is not None
        assert "deprecated" in result.lower()

    def test_returns_none_on_import_error(self):
        """Should return None if models_dev is unavailable."""
        # Temporarily break the import by patching
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = check_model_deprecation("gpt-4o", provider="openai")
        assert result is None

    def test_returns_none_when_model_not_found(self):
        """Should return None if model not found in models.dev."""
        mock_models_dev = MagicMock()
        mock_models_dev.get_model_info.return_value = None
        mock_models_dev.get_model_info_any_provider.return_value = None
        with patch.dict("sys.modules", {"agent.models_dev": mock_models_dev}):
            result = check_model_deprecation("nonexistent-model-xyz")
        assert result is None

    def test_strips_provider_prefix_from_model(self):
        """Should strip provider prefix when looking up with provider."""
        mock_models_dev = MagicMock()
        mock_info = MagicMock()
        mock_info.status = "deprecated"
        mock_models_dev.get_model_info.return_value = mock_info
        mock_models_dev.get_model_info_any_provider.return_value = None
        with patch.dict("sys.modules", {"agent.models_dev": mock_models_dev}):
            result = check_model_deprecation("openai/gpt-3.5-turbo", provider="openai")
        # Should have called get_model_info with bare model name
        mock_models_dev.get_model_info.assert_called_with("openai", "gpt-3.5-turbo")
        assert result is not None


# ---------------------------------------------------------------------------
# validate_api_key_format
# ---------------------------------------------------------------------------


class TestValidateApiKeyFormat:
    """Test API key format validation."""

    def test_returns_none_for_empty_key(self):
        assert validate_api_key_format("openai", "") is None

    def test_returns_none_for_none_key(self):
        assert validate_api_key_format("openai", None) is None

    def test_returns_none_for_empty_provider(self):
        assert validate_api_key_format("", "sk-abc123") is None

    def test_returns_none_for_none_provider(self):
        assert validate_api_key_format(None, "sk-abc123") is None

    def test_returns_none_for_unknown_provider(self):
        assert validate_api_key_format("custom-provider", "some-key-123") is None

    def test_returns_none_for_mistral_any_key(self):
        """Mistral has no known prefix pattern — always returns None."""
        assert validate_api_key_format("mistral", "any-key-format") is None

    def test_openai_correct_prefix(self):
        assert validate_api_key_format("openai", "sk-abc123def456") is None

    def test_openai_wrong_prefix(self):
        result = validate_api_key_format("openai", "wrong-prefix-key")
        assert result is not None
        assert "openai" in result.lower()
        assert "sk-" in result

    def test_anthropic_correct_prefix(self):
        assert validate_api_key_format("anthropic", "sk-ant-abc123def456") is None

    def test_anthropic_wrong_prefix(self):
        result = validate_api_key_format("anthropic", "sk-abc123def456")
        assert result is not None
        assert "anthropic" in result.lower()
        assert "sk-ant-" in result

    def test_openrouter_correct_prefix(self):
        assert validate_api_key_format("openrouter", "sk-or-abc123def456") is None

    def test_openrouter_wrong_prefix(self):
        result = validate_api_key_format("openrouter", "sk-abc123def456")
        assert result is not None
        assert "openrouter" in result.lower()

    def test_deepseek_correct_prefix(self):
        assert validate_api_key_format("deepseek", "sk-abc123def456") is None

    def test_deepseek_wrong_prefix(self):
        result = validate_api_key_format("deepseek", "wrong-key")
        assert result is not None
        assert "deepseek" in result.lower()

    def test_groq_correct_prefix(self):
        assert validate_api_key_format("groq", "gsk_abc123def456") is None

    def test_groq_wrong_prefix(self):
        result = validate_api_key_format("groq", "sk-abc123def456")
        assert result is not None
        assert "groq" in result.lower()
        assert "gsk_" in result

    def test_case_insensitive_provider(self):
        """Provider matching should be case-insensitive."""
        assert validate_api_key_format("OpenAI", "sk-abc123") is None
        assert validate_api_key_format("ANTHROPIC", "sk-ant-abc123") is None
        result = validate_api_key_format("OpenAI", "wrong-key")
        assert result is not None

    def test_all_known_providers_have_entries(self):
        """Verify all known providers in the pattern dict are tested."""
        for provider in _KEY_PREFIX_PATTERNS:
            # Should not raise
            validate_api_key_format(provider, "test-key")
