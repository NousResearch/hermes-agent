"""Slack's Channels-page card (Desktop + web dashboard): the PlatformEntry ``dashboard_*`` /
``docs_url`` fields register() spreads in. Kept apart from adapter.py, which is over its size cap."""

import re

# Write checks (kept in sync with ChannelsPage.tsx): reject tokens pasted into the wrong field.
# "*" is the allow-all wildcard; allowlists split like the adapter's parser.
_MEMBER_ID_RE = re.compile(r"[UW][A-Z0-9]{2,}")
_ENV_VALUE_RULES = {
    "SLACK_BOT_TOKEN": (
        lambda v: v.startswith("xoxb-"),
        "Slack Bot Token must start with xoxb-. Paste the bot token from OAuth & Permissions."),
    "SLACK_APP_TOKEN": (
        lambda v: v.startswith("xapp-"),
        "Slack App Token must start with xapp-. Paste the app-level token from Basic Information > App-Level Tokens."),
    "SLACK_ALLOWED_USERS": (
        lambda v: all(u == "*" or _MEMBER_ID_RE.fullmatch(u)
                      for u in (p.strip() for p in v.split(",")) if u),
        "Slack allowed user IDs must be comma-separated member IDs like U01ABC2DEF3."),
}

DASHBOARD_CARD = dict(
    dashboard_description="Use Hermes from Slack via Socket Mode. Add allowed Slack member IDs so connected bots can respond.",
    docs_url="https://api.slack.com/apps",
    dashboard_env_vars=("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_ALLOWED_USERS"),
    env_value_rules=_ENV_VALUE_RULES,
)
