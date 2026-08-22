"""Tests for Copilot model API-mode routing."""

from __future__ import annotations


def test_copilot_claude_stays_on_chat_completions_even_if_catalog_lists_messages():
    from hermes_cli.models import copilot_model_api_mode

    catalog = [
        {
            "id": "claude-opus-4.8",
            "supported_endpoints": ["/v1/messages"],
        }
    ]

    assert copilot_model_api_mode("claude-opus-4.8", catalog=catalog) == "chat_completions"


def test_copilot_gpt5_still_uses_responses_api():
    from hermes_cli.models import copilot_model_api_mode

    assert copilot_model_api_mode("gpt-5.5", catalog=[]) == "codex_responses"
    assert copilot_model_api_mode("gpt-5-mini", catalog=[]) == "chat_completions"


def test_copilot_responses_only_non_gpt_model_uses_responses_api():
    """The catalog, not the vendor prefix, owns endpoint selection."""
    from hermes_cli.models import copilot_model_api_mode

    catalog = [
        {
            "id": "grok-4.6",
            "supported_endpoints": ["/responses"],
        }
    ]

    assert copilot_model_api_mode("grok-4.6", catalog=catalog) == "codex_responses"


def test_copilot_dual_endpoint_non_gpt_model_stays_on_chat_completions():
    """Both endpoints advertised: keep the cheaper existing chat path."""
    from hermes_cli.models import copilot_model_api_mode

    catalog = [
        {
            "id": "grok-4.6",
            "supported_endpoints": ["/responses", "/chat/completions"],
        }
    ]

    assert copilot_model_api_mode("grok-4.6", catalog=catalog) == "chat_completions"


def test_copilot_claude_stays_on_chat_when_catalog_is_responses_only():
    """Copilot Claude always uses its OpenAI-compatible chat transport."""
    from hermes_cli.models import copilot_model_api_mode

    catalog = [
        {
            "id": "claude-opus-5",
            "supported_endpoints": ["/responses", "/v1/messages"],
        }
    ]

    assert copilot_model_api_mode("claude-opus-5", catalog=catalog) == "chat_completions"
