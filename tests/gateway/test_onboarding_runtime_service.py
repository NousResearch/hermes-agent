"""DEAD path: not imported by gateway/run.py — contract-only unit tests.

See gateway/RUNTIME_SERVICES.md. Marked dead_runtime_service so suites can
optionally filter with ``-m "not dead_runtime_service"``; default still runs.
"""
import pytest

pytestmark = pytest.mark.dead_runtime_service

from gateway.config import Platform
from gateway.onboarding_runtime_service import (
    FIRST_MESSAGE_ONBOARDING_NOTE,
    append_first_message_onboarding_note,
    build_home_channel_prompt,
    home_channel_env_var_name,
    should_prompt_for_home_channel,
)


def test_append_first_message_onboarding_note_only_for_first_turn_ever():
    result = append_first_message_onboarding_note(
        "base",
        history=[],
        has_any_sessions=False,
    )

    assert result == f"base{FIRST_MESSAGE_ONBOARDING_NOTE}"


def test_append_first_message_onboarding_note_skips_when_history_exists():
    result = append_first_message_onboarding_note(
        "base",
        history=[{"role": "user", "content": "hi"}],
        has_any_sessions=False,
    )

    assert result == "base"


def test_should_prompt_for_home_channel_skips_local_and_webhook():
    assert should_prompt_for_home_channel(
        history=[],
        platform=Platform.LOCAL,
        home_channel_configured=False,
    ) is False
    assert should_prompt_for_home_channel(
        history=[],
        platform=Platform.WEBHOOK,
        home_channel_configured=False,
    ) is False


def test_home_channel_prompt_helpers_for_remote_platform():
    assert home_channel_env_var_name(Platform.QQ_NAPCAT) == "QQ_NAPCAT_HOME_CHANNEL"
    prompt = build_home_channel_prompt(Platform.QQ_NAPCAT)
    assert "No home channel is set for Qq_Napcat." in prompt
    assert "/sethome" in prompt
