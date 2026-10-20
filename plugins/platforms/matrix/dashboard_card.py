"""Matrix's Channels-page card (Desktop + web dashboard): the PlatformEntry ``dashboard_*`` /
``docs_url`` fields register() spreads in. Kept apart from adapter.py, which is over its size cap."""

DASHBOARD_CARD = dict(
    dashboard_description="Use Hermes in Matrix rooms and direct messages.",
    docs_url="https://matrix.org/ecosystem/servers/",
    dashboard_env_vars=(
        "MATRIX_HOMESERVER", "MATRIX_ACCESS_TOKEN", "MATRIX_USER_ID", "MATRIX_ALLOWED_USERS",
    ),
    dashboard_required_env=("MATRIX_HOMESERVER", "MATRIX_ACCESS_TOKEN", "MATRIX_USER_ID"),
)
