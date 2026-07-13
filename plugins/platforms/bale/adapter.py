"""Bale platform adapter.

Bale exposes a Telegram-compatible Bot API at ``https://tapi.bale.ai.
This adapter reuses the Telegram implementation while swapping the default
Bot API root and Bale-specific environment variable names.
"""

import os
import logging
from typing import Any, Optional

from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    check_telegram_requirements as _check_telegram_requirements
)
from gateway.config import Platform

logger = logging.getLogger(__name__)

def check_bale_requirements() -> bool:
    """Check if Bale (Telegram-compatible) dependencies are available."""
    return _check_telegram_requirements()

class BaleAdapter(TelegramAdapter):
    """
    Bale bot adapter.
    Inherits from TelegramAdapter because Bale is API-compatible.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'bot') and self.bot:
            self.bot.base_url = "https://tapi.bale.ai/bot"

    @property
    def platform_name(self) -> str:
        return "bale"

def _build_adapter(config: Any) -> BaleAdapter:
    """Factory to create a Bale adapter instance."""
    return BaleAdapter(config)

def _standalone_send(token: str, chat_id: str, text: str, **kwargs) -> Any:
    """Standalone sender for Bale using their Bot API."""
    import requests
    url = f"https://tapi.bale.ai/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, **kwargs}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

def _apply_yaml_config(telegram_cfg: dict, extras: dict) -> Optional[dict]:
    """
    Map Bale-specific YAML config to environment variables.
    """
    mappings = {
        "bot_token": "BALE_BOT_TOKEN",
        "home_channel": "BALE_HOME_CHANNEL",
        "allowed_users": "BALE_ALLOWED_USERS",
        "require_mention": "BALE_REQUIRE_MENTION",
    }
    for key, env_var in mappings.items():
        val = telegram_cfg.get(key)
        if val is not None:
            os.environ[env_var] = str(val)
    return extras

def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="bale",
        label="Bale",
        adapter_factory=_build_adapter,
        check_fn=check_bale_requirements,
        is_connected=lambda: os.getenv("BALE_BOT_TOKEN") is not None,
        required_env=["BALE_BOT_TOKEN"],
        install_hint="pip install 'hermes-agent[telegram]'",
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="BALE_ALLOWED_USERS",
        cron_deliver_env_var="BALE_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=4096,
        emoji="🔵",
    )