"""Tests for per-model thinking field override resolution.

Covers get_custom_provider_thinking_field(), which lets users configure
model-specific field names for the thinking-off toggle (e.g.
chat_template_kwargs.enable_thinking instead of the default think=False).
"""
from __future__ import annotations

from hermes_cli.config import get_custom_provider_thinking_field


class TestGetCustomProviderThinkingField:
    def test_returns_field_and_subkey(self):
        custom = [
            {
                "base_url": "http://127.0.0.1:8000/v1",
                "models": {
                    "Qwen3.6-35B-A3B-bf16": {
                        "thinking_field": "chat_template_kwargs",
                        "thinking_subkey": "enable_thinking",
                    }
                },
            }
        ]
        result = get_custom_provider_thinking_field(
            "Qwen3.6-35B-A3B-bf16", "http://127.0.0.1:8000/v1", custom
        )
        assert result == {"field": "chat_template_kwargs", "subkey": "enable_thinking"}

    def test_returns_field_without_subkey(self):
        custom = [
            {
                "base_url": "http://localhost:11434/v1",
                "models": {
                    "my-model": {"thinking_field": "disable_thinking"}
                },
            }
        ]
        result = get_custom_provider_thinking_field(
            "my-model", "http://localhost:11434/v1", custom
        )
        assert result == {"field": "disable_thinking"}

    def test_trailing_slash_insensitive(self):
        custom = [
            {
                "base_url": "http://127.0.0.1:8000/v1/",
                "models": {
                    "m": {"thinking_field": "chat_template_kwargs", "thinking_subkey": "enable_thinking"}
                },
            }
        ]
        assert (
            get_custom_provider_thinking_field("m", "http://127.0.0.1:8000/v1", custom)
            == {"field": "chat_template_kwargs", "subkey": "enable_thinking"}
        )

    def test_returns_none_when_no_thinking_field(self):
        custom = [
            {
                "base_url": "http://127.0.0.1:8000/v1",
                "models": {"m": {"context_length": 32768}},
            }
        ]
        assert get_custom_provider_thinking_field(
            "m", "http://127.0.0.1:8000/v1", custom
        ) is None

    def test_returns_none_when_url_mismatch(self):
        custom = [
            {
                "base_url": "http://127.0.0.1:8000/v1",
                "models": {"m": {"thinking_field": "think"}},
            }
        ]
        assert get_custom_provider_thinking_field(
            "m", "http://other-host:8000/v1", custom
        ) is None

    def test_returns_none_when_model_mismatch(self):
        custom = [
            {
                "base_url": "http://127.0.0.1:8000/v1",
                "models": {"m": {"thinking_field": "think"}},
            }
        ]
        assert get_custom_provider_thinking_field(
            "other-model", "http://127.0.0.1:8000/v1", custom
        ) is None

    def test_empty_inputs_return_none(self):
        assert get_custom_provider_thinking_field("", "http://x", []) is None
        assert get_custom_provider_thinking_field("m", "", []) is None
        assert get_custom_provider_thinking_field("m", "http://x", None) is None

    def test_ignores_malformed_entries(self):
        custom = [
            "not a dict",
            None,
            {"base_url": "http://x/v1", "models": "not a dict"},
            {"base_url": "http://x/v1", "models": {"m": "not a dict"}},
            {"base_url": "http://x/v1", "models": {"m": {"thinking_field": ""}}},
            {
                "base_url": "http://x/v1",
                "models": {"m": {"thinking_field": "chat_template_kwargs"}},
            },
        ]
        result = get_custom_provider_thinking_field("m", "http://x/v1", custom)
        assert result == {"field": "chat_template_kwargs"}

    def test_ignores_non_string_thinking_field(self):
        custom = [
            {
                "base_url": "http://x/v1",
                "models": {"m": {"thinking_field": 123}},
            }
        ]
        assert get_custom_provider_thinking_field("m", "http://x/v1", custom) is None
