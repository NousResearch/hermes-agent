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
    """grok-4.x is /responses-only on Copilot; the id heuristic alone sent it to chat (HTTP 400)."""
    from hermes_cli.models import copilot_model_api_mode

    catalog = [{"id": "grok-4.7", "supported_endpoints": ["/responses"]}]

    assert copilot_model_api_mode("grok-4.7", catalog=catalog) == "codex_responses"


def test_copilot_ws_responses_endpoint_is_recognised():
    from hermes_cli.models import copilot_model_api_mode

    catalog = [{"id": "grok-4.7", "supported_endpoints": ["ws:/responses", "/responses"]}]

    assert copilot_model_api_mode("grok-4.7", catalog=catalog) == "codex_responses"


def test_copilot_chat_only_and_dual_endpoint_models_keep_heuristic():
    from hermes_cli.models import copilot_model_api_mode

    catalog = [
        {"id": "gemini-4-argon", "supported_endpoints": ["/chat/completions"]},
        {"id": "gpt-5.5", "supported_endpoints": ["/responses", "/chat/completions"]},
        {"id": "claude-opus-5.5", "supported_endpoints": ["/chat/completions", "/v1/messages"]},
        {"id": "gpt-5-mini", "supported_endpoints": ["/chat/completions"]},
    ]

    assert copilot_model_api_mode("gemini-4-argon", catalog=catalog) == "chat_completions"
    assert copilot_model_api_mode("gpt-5.5", catalog=catalog) == "codex_responses"
    assert copilot_model_api_mode("claude-opus-5.5", catalog=catalog) == "chat_completions"
    assert copilot_model_api_mode("gpt-5-mini", catalog=catalog) == "chat_completions"


def test_should_use_responses_api_consults_catalog_cache(monkeypatch):
    """Helpers that only see a model id (aux client, run_agent) must honour the cached catalog."""
    from hermes_cli import models

    monkeypatch.setattr(models, "_github_model_catalog_cache",
                        [{"id": "grok-4.7", "supported_endpoints": ["/responses"]}])

    assert models._should_use_copilot_responses_api("grok-4.7") is True
    assert models._should_use_copilot_responses_api("claude-sonnet-5.5") is False
