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


def test_copilot_non_gpt_responses_only_model_uses_responses_api():
    """A /responses-only model from a non-GPT vendor must not fall back to chat.

    Regression: the name-pattern heuristic only matched ``^gpt-(\\d+)``, so
    ``grok-4.5`` (advertised by Copilot as ``/responses``-only) was routed to
    ``/chat/completions`` and the API rejected every call with
    400 ``unsupported_api_for_model``.
    """
    from hermes_cli.models import copilot_model_api_mode

    catalog = [{"id": "grok-4.5", "supported_endpoints": ["/responses"]}]

    assert copilot_model_api_mode("grok-4.5", catalog=catalog) == "codex_responses"


def test_copilot_chat_only_model_stays_on_chat_completions():
    from hermes_cli.models import copilot_model_api_mode

    catalog = [{"id": "gemini-3.6-flash", "supported_endpoints": ["/chat/completions"]}]

    assert copilot_model_api_mode("gemini-3.6-flash", catalog=catalog) == "chat_completions"


def test_copilot_dual_endpoint_model_falls_back_to_name_heuristic():
    """Models advertising both endpoints keep the previous GPT-name behaviour."""
    from hermes_cli.models import copilot_model_api_mode

    catalog = [
        {"id": "gpt-5.4", "supported_endpoints": ["/responses", "/chat/completions"]},
        {"id": "claude-opus-5", "supported_endpoints": ["/v1/messages", "/chat/completions"]},
    ]

    assert copilot_model_api_mode("gpt-5.4", catalog=catalog) == "codex_responses"
    assert copilot_model_api_mode("claude-opus-5", catalog=catalog) == "chat_completions"


def test_copilot_ws_responses_endpoint_is_recognised():
    """``ws:/responses`` variants must not be mistaken for a chat endpoint."""
    from hermes_cli.models import copilot_model_api_mode

    catalog = [{"id": "gpt-5.3-codex", "supported_endpoints": ["/responses", "ws:/responses"]}]

    assert copilot_model_api_mode("gpt-5.3-codex", catalog=catalog) == "codex_responses"
