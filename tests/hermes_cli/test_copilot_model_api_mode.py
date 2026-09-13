"""Tests for Copilot model API-mode routing."""

from __future__ import annotations


def test_copilot_claude_prefers_messages_when_catalog_lists_messages():
    from hermes_cli.models import copilot_model_api_mode

    catalog = [
        {
            "id": "claude-opus-4.8",
            "supported_endpoints": ["/v1/messages"],
        }
    ]

    assert copilot_model_api_mode("claude-opus-4.8", catalog=catalog) == "anthropic_messages"

def test_copilot_grok_uses_responses_via_catalog():
    """Grok models listing /responses in catalog get codex_responses."""
    from hermes_cli.models import copilot_model_api_mode

    catalog = [
        {
            "id": "grok-4.5",
            "supported_endpoints": ["/responses"],
        }
    ]
    assert copilot_model_api_mode("grok-4.5", catalog=catalog) == "codex_responses"

def test_copilot_gpt5_still_uses_responses_api():
    from hermes_cli.models import copilot_model_api_mode

    assert copilot_model_api_mode("gpt-5.5", catalog=[]) == "codex_responses"
    assert copilot_model_api_mode("gpt-5-mini", catalog=[]) == "chat_completions"
