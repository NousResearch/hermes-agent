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


def test_responses_only_model_routes_to_responses():
    """Copilot serves every Grok model on /responses only; chat 400s."""
    from hermes_cli.models import copilot_model_api_mode

    catalog = [{"id": "grok-4.7", "supported_endpoints": ["/responses"]}]

    assert copilot_model_api_mode("grok-4.7", catalog=catalog) == "codex_responses"


def test_grok_routes_to_responses_with_no_catalog_signal():
    """Regression: cron/gateway resolve credentials without a live api_key,
    so the catalog fetch never happens and ``catalog=[]``/``None`` reaches
    here with zero supported_endpoints signal. Grok must still resolve to
    codex_responses from the model-ID pattern alone, or cron jobs pinned to
    grok-*/copilot get routed to /chat/completions and hit HTTP 400."""
    from hermes_cli.models import copilot_model_api_mode

    assert copilot_model_api_mode("grok-4.7", catalog=[]) == "codex_responses"
    assert copilot_model_api_mode("grok-4.7", catalog=None, api_key=None) == "codex_responses"
    assert copilot_model_api_mode("grok-code-fast-1", catalog=[]) == "codex_responses"


def test_dual_endpoint_model_keeps_pattern_derived_mode():
    """A model offering both endpoints must not be forced onto /responses."""
    from hermes_cli.models import copilot_model_api_mode

    catalog = [
        {
            "id": "gpt-5-mini",
            "supported_endpoints": ["/chat/completions", "/responses"],
        },
        {"id": "kimi-k3", "supported_endpoints": ["/chat/completions"]},
    ]

    assert copilot_model_api_mode("gpt-5-mini", catalog=catalog) == "chat_completions"
    assert copilot_model_api_mode("kimi-k3", catalog=catalog) == "chat_completions"


def test_missing_catalog_entry_is_no_signal_not_chat_unsupported():
    from hermes_cli.models import copilot_model_api_mode

    assert copilot_model_api_mode("some-unknown-model", catalog=[]) == "chat_completions"
