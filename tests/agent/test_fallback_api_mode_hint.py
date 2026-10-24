from __future__ import annotations

from agent.chat_completion_helpers import _fallback_api_mode_hint
from providers import get_provider_profile


def test_plugin_provider_profile_declares_the_fallback_wire():
    profile = get_provider_profile("commandcode-anthropic")
    assert profile is not None and profile.api_mode == "anthropic_messages"
    assert _fallback_api_mode_hint(
        {"provider": profile.name, "model": "claude-sonnet-5-5"},
        profile.name,
        profile.base_url,
    ) == (True, "anthropic_messages")


def test_default_chat_completions_profile_keeps_re_detection():
    profile = get_provider_profile("commandcode")
    assert profile is not None and profile.api_mode == "chat_completions"
    assert _fallback_api_mode_hint(
        {"provider": profile.name, "model": "gpt-5"},
        profile.name,
        profile.base_url,
    ) == (False, "chat_completions")


def test_entry_api_mode_still_wins_over_the_profile():
    profile = get_provider_profile("commandcode-anthropic")
    assert profile is not None and profile.api_mode == "anthropic_messages"
    assert _fallback_api_mode_hint(
        {"provider": profile.name, "model": "m", "api_mode": "chat_completions"},
        profile.name,
        profile.base_url,
    ) == (True, "chat_completions")


def test_unregistered_provider_is_unchanged():
    assert _fallback_api_mode_hint(
        {"provider": "custom", "model": "m"}, "custom", None
    ) == (False, "chat_completions")
