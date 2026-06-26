import os
from typing import Dict, Any, Optional

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    check_telegram_requirements,
)

def env_enablement_fn() -> Optional[Dict[str, Any]]:
    """Seed PlatformConfig from environment variables before init."""
    token = os.getenv("BALE_BOT_TOKEN")
    if not token:
        return None
    
    return {
        "extra": {
            "base_url": "https://tapi.bale.ai/bot",
            "base_file_url": "https://tapi.bale.ai/file/bot"
        },
        "home_channel": {
            "id": os.getenv("BALE_HOME_CHANNEL"),
            "name": os.getenv("BALE_HOME_CHANNEL_NAME", "Bale Home")
        } if os.getenv("BALE_HOME_CHANNEL") else None
    }

def apply_yaml_config_fn(yaml_cfg: Dict[str, Any], platform_cfg: PlatformConfig) -> Optional[Dict[str, Any]]:
    """Translate config.yaml keys into env vars and extra dict."""
    extras = {}
    bale_cfg = yaml_cfg.get("bale", {})
    if not isinstance(bale_cfg, dict):
        return {
            "extra": {
                "base_url": "https://tapi.bale.ai/bot",
                "base_file_url": "https://tapi.bale.ai/file/bot"
            }
        }

    allowed_users = bale_cfg.get("allow_from")
    if allowed_users is not None and not os.getenv("BALE_ALLOWED_USERS"):
        if isinstance(allowed_users, list):
            allowed_users = ",".join(str(v) for v in allowed_users)
        os.environ["BALE_ALLOWED_USERS"] = str(allowed_users)

    extras["base_url"] = "https://tapi.bale.ai/bot"
    extras["base_file_url"] = "https://tapi.bale.ai/file/bot"
    return {"extra": extras}

class BaleAdapter(TelegramAdapter):
    """
    Bale bot adapter.
    Inherits from TelegramAdapter but points to the Bale API servers via base_url.
    """
    def __init__(self, config: PlatformConfig):
        super().__init__(config)
        self.name = "bale"

def _build_adapter(config: PlatformConfig):
    return BaleAdapter(config)

def _is_connected(adapter) -> bool:
    if not adapter:
        return False
    return adapter._running

async def _standalone_send(chat_id: str, message: str, **kwargs) -> Dict[str, Any]:
    """Standalone sender for cron jobs and external tool delivery."""
    import aiohttp
    
    token = os.getenv("BALE_BOT_TOKEN")
    if not token:
        raise ValueError("BALE_BOT_TOKEN not set")
    
    async with aiohttp.ClientSession() as session:
        url = f"https://tapi.bale.ai/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return {"result": "ok", "message_id": data.get("result", {}).get("message_id")}

def register(ctx) -> None:
    """Plugin entry point."""
    ctx.register_platform(
        name="bale",
        label="Bale",
        adapter_factory=_build_adapter,
        check_fn=check_telegram_requirements,
        is_connected=_is_connected,
        required_env=["BALE_BOT_TOKEN"],
        install_hint="pip install 'hermes-agent[telegram]'",
        setup_fn=None,
        apply_yaml_config_fn=apply_yaml_config_fn,
        allowed_users_env="BALE_ALLOWED_USERS",
        allow_all_env="BALE_ALLOW_ALL_USERS",
        cron_deliver_env_var="BALE_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=4096,
        emoji="💬",
        allow_update_command=True,
    )
