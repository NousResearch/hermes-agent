"""Fallback activation must honor host_mandated_api_mode.

Regression (2026-08-05): _try_activate_fallback defaulted kimi-coding to
chat_completions → POST https://api.kimi.com/coding/chat/completions → 404,
even though host_mandated_api_mode(api.kimi.com/coding) returns
anthropic_messages and the correct path is /coding/v1/messages.
"""

from __future__ import annotations

from hermes_cli.providers import host_mandated_api_mode


def test_kimi_coding_host_mandates_anthropic_messages():
    assert host_mandated_api_mode("https://api.kimi.com/coding") == "anthropic_messages"
    assert host_mandated_api_mode("https://api.kimi.com/coding/") == "anthropic_messages"
    assert host_mandated_api_mode("https://api.kimi.com/coding/v1") == "anthropic_messages"


def test_anthropic_and_openai_hosts_still_mandated():
    assert host_mandated_api_mode("https://api.anthropic.com") == "anthropic_messages"
    assert host_mandated_api_mode("https://api.openai.com/v1") == "codex_responses"


def test_fallback_mode_resolution_prefers_host_mandate():
    """Mirror the fixed branch order in chat_completion_helpers._try_activate_fallback."""
    fb_base_url = "https://api.kimi.com/coding"
    fb_provider = "kimi-coding"
    mandated = host_mandated_api_mode(fb_base_url)
    fb_api_mode = mandated or "chat_completions"
    if fb_api_mode == "chat_completions":
        if fb_provider == "openai-codex":
            fb_api_mode = "codex_responses"
        elif fb_provider == "anthropic":
            fb_api_mode = "anthropic_messages"
    assert fb_api_mode == "anthropic_messages"
    # Without host mandate this would stay chat_completions and 404.
    assert (mandated or "chat_completions") != "chat_completions" or fb_provider == "anthropic"
