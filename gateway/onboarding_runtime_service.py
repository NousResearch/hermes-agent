"""Shared runtime helpers for gateway onboarding and home-channel prompts."""

from __future__ import annotations

from gateway.config import Platform

FIRST_MESSAGE_ONBOARDING_NOTE = (
    "\n\n[System note: This is the user's very first message ever. "
    "Briefly introduce yourself and mention that /help shows available commands. "
    "Keep the introduction concise -- one or two sentences max.]"
)


def append_first_message_onboarding_note(
    context_prompt: str,
    *,
    history: list[dict] | None,
    has_any_sessions: bool,
) -> str:
    """Append the first-message onboarding note when this is the first turn ever."""

    if history or has_any_sessions:
        return context_prompt
    return f"{context_prompt}{FIRST_MESSAGE_ONBOARDING_NOTE}"


def home_channel_env_var_name(platform: Platform | None) -> str:
    """Return the environment variable that stores the platform home channel."""

    if platform is None:
        return ""
    return f"{platform.value.upper()}_HOME_CHANNEL"


def should_prompt_for_home_channel(
    *,
    history: list[dict] | None,
    platform: Platform | None,
    home_channel_configured: bool,
) -> bool:
    """Return True when the one-time home-channel reminder should be shown."""

    if history:
        return False
    if platform in (None, Platform.LOCAL, Platform.WEBHOOK):
        return False
    return not home_channel_configured


def build_home_channel_prompt(platform: Platform) -> str:
    """Return the one-time user-facing prompt for configuring a home channel."""

    platform_name = platform.value.title()
    return (
        f"📬 No home channel is set for {platform_name}. "
        f"A home channel is where Hermes delivers cron job results "
        f"and cross-platform messages.\n\n"
        f"Type /sethome to make this chat your home channel, "
        f"or ignore to skip."
    )
