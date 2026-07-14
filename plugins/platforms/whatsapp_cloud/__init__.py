"""WhatsApp Cloud platform plugin — registers the adapter via the plugin registry."""
from __future__ import annotations
import os
from gateway.platforms.whatsapp_cloud import (
    WhatsAppCloudAdapter,
    check_whatsapp_cloud_requirements,
)


def _is_connected(config) -> bool:
    return bool(
        (
            os.getenv("WHATSAPP_CLOUD_PHONE_NUMBER_ID")
            and os.getenv("WHATSAPP_CLOUD_ACCESS_TOKEN")
        )
        or (
            config.extra.get("phone_number_id")
            and config.extra.get("access_token")
        )
    )


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="whatsapp_cloud",
        label="WhatsApp Cloud",
        adapter_factory=lambda cfg: WhatsAppCloudAdapter(cfg),
        check_fn=check_whatsapp_cloud_requirements,
        validate_config=_is_connected,
        is_connected=_is_connected,
        required_env=["WHATSAPP_CLOUD_PHONE_NUMBER_ID", "WHATSAPP_CLOUD_ACCESS_TOKEN"],
        install_hint="pip install aiohttp httpx",
        allowed_users_env="WHATSAPP_CLOUD_ALLOWED_USERS",
        allow_all_env="WHATSAPP_CLOUD_ALLOW_ALL_USERS",
        cron_deliver_env_var="WHATSAPP_CLOUD_HOME_CHANNEL",
        emoji="📱",
        platform_hint="You are on WhatsApp. Keep responses concise. Avoid heavy markdown.",
    )
