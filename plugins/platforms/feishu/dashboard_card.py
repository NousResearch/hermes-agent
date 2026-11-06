"""Feishu / Lark's Channels-page card (Desktop + web dashboard): the PlatformEntry ``dashboard_*`` /
``docs_url`` fields register() spreads in. Kept apart from adapter.py, which is over its size cap."""

DASHBOARD_CARD = dict(
    dashboard_description="Use Hermes inside Feishu / Lark.",
    docs_url="https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/intro",
    dashboard_env_vars=(
        "FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_ENCRYPT_KEY", "FEISHU_VERIFICATION_TOKEN",
    ),
)
