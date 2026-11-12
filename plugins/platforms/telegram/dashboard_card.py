"""Telegram's Channels-page card (Desktop + web dashboard): the PlatformEntry ``dashboard_*`` /
``docs_url`` fields register() spreads in. Kept apart from adapter.py, which is over its size cap."""

import re

# Write checks (kept in sync with ChannelsPage.tsx): reject credentials
# pasted into the wrong field. Allowlists split like the adapter's parser (comma, strip, drop
# empties) so a trailing comma the runtime accepts is not refused here.
_BOT_TOKEN_RE = re.compile(r"\d+:[A-Za-z0-9_-]{30,}")
_USER_ID_RE = re.compile(r"\d+")
_ENV_VALUE_RULES = {
    "TELEGRAM_BOT_TOKEN": (
        _BOT_TOKEN_RE.fullmatch,
        "Telegram bot token must be the complete token from @BotFather, such as 123456789:ABC…"),
    "TELEGRAM_ALLOWED_USERS": (
        lambda v: all(_USER_ID_RE.fullmatch(u.strip()) for u in v.split(",") if u.strip()),
        "Telegram allowed users must be comma-separated numeric user IDs."),
}

DASHBOARD_CARD = dict(
    dashboard_description="Run Hermes from Telegram DMs, groups, and topics.",
    docs_url="https://core.telegram.org/bots/features#botfather",
    dashboard_env_vars=("TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USERS", "TELEGRAM_PROXY"),
    env_value_rules=_ENV_VALUE_RULES,
)
