"""Time (Т‑Банк) platform adapter.

Time is Slack-API-compatible, so this adapter is a thin subclass of
``SlackAdapter`` that points the Slack Web API / Socket Mode transport at
Time's endpoints and reads Time-specific env vars.

Env vars:
    TIME_BOT_TOKEN       — bot token (Slack xoxb-equivalent)
    TIME_APP_TOKEN       — Socket Mode app token (Slack xapp-equivalent)
    TIME_API_BASE_URL    — Web API base URL (e.g. https://time.tbank.ru/api/)
    TIME_ALLOWED_USERS   — comma-separated allowed user IDs
    TIME_ALLOW_ALL_USERS — truthy = allow everyone (dev only)
    TIME_HOME_CHANNEL    — channel for cron/notification delivery
"""

import os
import logging
from typing import Optional

from gateway.platforms.slack import SlackAdapter

logger = logging.getLogger(__name__)


class TimeAdapter(SlackAdapter):
    """Slack-compatible adapter for T-Bank's Time messenger."""

    def _app_token_env(self) -> str:
        return "TIME_APP_TOKEN"

    def _api_base_url(self) -> Optional[str]:
        return os.getenv("TIME_API_BASE_URL") or None


def _check_requirements() -> bool:
    try:
        import slack_bolt  # noqa: F401
        return True
    except Exception:
        return False


def _validate_config(cfg) -> bool:
    return bool(os.getenv("TIME_BOT_TOKEN")) and bool(os.getenv("TIME_API_BASE_URL"))


def _env_enablement():
    if not os.getenv("TIME_BOT_TOKEN"):
        return None
    extra = {"api_base_url": os.getenv("TIME_API_BASE_URL", "")}
    home = os.getenv("TIME_HOME_CHANNEL")
    if home:
        return {"extra": extra, "home_channel": {"chat_id": home}}
    return {"extra": extra}


def register(ctx):
    """Plugin entry point."""
    ctx.register_platform(
        name="time",
        label="Time",
        adapter_factory=lambda cfg: TimeAdapter(cfg),
        check_fn=_check_requirements,
        validate_config=_validate_config,
        required_env=["TIME_BOT_TOKEN", "TIME_APP_TOKEN", "TIME_API_BASE_URL"],
        install_hint="pip install slack-bolt",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="TIME_HOME_CHANNEL",
        allowed_users_env="TIME_ALLOWED_USERS",
        allow_all_env="TIME_ALLOW_ALL_USERS",
        max_message_length=39000,
        emoji="🕐",
        platform_hint=(
            "You are chatting via Time (Т‑Банк corporate messenger). "
            "Time renders Slack-style mrkdwn. Keep replies professional."
        ),
    )
