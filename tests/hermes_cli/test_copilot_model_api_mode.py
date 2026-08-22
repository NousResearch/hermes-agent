"""Tests for Copilot model API-mode routing."""

from __future__ import annotations

import pytest


def test_copilot_claude_routes_to_anthropic_messages_on_default_host():
    """Claude on Copilot's default host must use the Anthropic Messages
    endpoint (/v1/messages) so we get the real ~1M input window instead of
    /chat/completions' ~168k clamp.
    """
    from hermes_cli.models import copilot_model_api_mode

    catalog = [
        {
            "id": "claude-opus-4.8",
            "supported_endpoints": ["/v1/messages"],
        }
    ]

    assert (
        copilot_model_api_mode(
            "claude-opus-4.8",
            catalog=catalog,
            base_url="https://api.githubcopilot.com",
        )
        == "anthropic_messages"
    )


def test_copilot_claude_defaults_to_anthropic_messages_without_base_url():
    """When no base_url is supplied the routing defaults to the canonical
    Copilot host, so provider+model routing works with no extra config.
    """
    from hermes_cli.models import copilot_model_api_mode

    assert (
        copilot_model_api_mode("claude-opus-4.8", catalog=[])
        == "anthropic_messages"
    )


@pytest.mark.parametrize(
    "host",
    [
        "https://api.githubcopilot.com",
        "https://api.business.githubcopilot.com",
        "https://api.enterprise.githubcopilot.com",
    ],
)
def test_copilot_claude_routes_to_messages_on_default_and_plan_scoped_hosts(host):
    """Suffix-safe host match: enterprise/business Copilot hosts must ALSO
    route Claude to /v1/messages. An exact 'host == api.githubcopilot.com'
    check would wrongly exclude these plan-scoped endpoints (PR #51437 review).
    """
    from hermes_cli.models import copilot_model_api_mode

    assert (
        copilot_model_api_mode("claude-sonnet-4.6", catalog=[], base_url=host)
        == "anthropic_messages"
    )


def test_copilot_claude_on_non_copilot_host_stays_chat_completions():
    """A lookalike or non-Copilot host must NOT trigger the messages route."""
    from hermes_cli.models import copilot_model_api_mode

    assert (
        copilot_model_api_mode(
            "claude-opus-4.8",
            catalog=[],
            base_url="https://evil.com/githubcopilot.com/v1",
        )
        == "chat_completions"
    )


def test_copilot_gpt5_still_uses_responses_api():
    from hermes_cli.models import copilot_model_api_mode

    assert copilot_model_api_mode("gpt-5.5", catalog=[]) == "codex_responses"
    assert copilot_model_api_mode("gpt-5-mini", catalog=[]) == "chat_completions"
