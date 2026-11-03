"""Discord's Channels-page card (Desktop + web dashboard): the PlatformEntry ``dashboard_*`` /
``docs_url`` fields register() spreads in. Kept apart from adapter.py, which is over its size cap."""

DASHBOARD_CARD = dict(
    dashboard_description="Connect Hermes to Discord DMs, channels, and threads.",
    docs_url="https://discord.com/developers/applications",
    dashboard_env_vars=("DISCORD_BOT_TOKEN", "DISCORD_ALLOWED_USERS"),
)
