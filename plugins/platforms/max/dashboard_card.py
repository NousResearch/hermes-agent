"""MAX's Channels-page card (Desktop + web dashboard): the PlatformEntry ``dashboard_*`` /
``docs_url`` fields register() spreads in. Kept apart from adapter.py, matching the other
platform adapters' convention (see slack/matrix/telegram ``dashboard_card.py``)."""

DASHBOARD_CARD = dict(
    dashboard_description=(
        "Use Hermes from the MAX messenger (max.ru, RU). Minimal 1:1 text dialogs — "
        "no groups/channels/media/buttons yet."
    ),
    docs_url="https://business.max.ru/self",
    dashboard_env_vars=("MAX_BOT_TOKEN", "MAX_ALLOWED_USERS"),
    dashboard_required_env=("MAX_BOT_TOKEN",),
)
