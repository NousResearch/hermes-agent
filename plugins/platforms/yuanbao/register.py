"""Yuanbao plugin registration: setup row, configured check, standalone sender, ``register``."""

import logging

from gateway.platforms._shared import send_error

# Absolute import (see __init__.py): one ``YuanbaoAdapter`` class object for the registry,
# tools/yuanbao_tools.py and the standalone sender.
from plugins.platforms.yuanbao.adapter import WEBSOCKETS_AVAILABLE, YuanbaoAdapter

logger = logging.getLogger(__name__)


_INSTALL_HINT = "Yuanbao: websockets not installed. Run: pip install websockets"

_SETUP_PLATFORM = {
    "key": "yuanbao", "label": "Yuanbao", "emoji": "💎", "token_var": "YUANBAO_APP_ID",
    "setup_instructions": [
        "1. Download the Yuanbao app from https://yuanbao.tencent.com/",
        "2. In the app, go to PAI → My Bot and create a new bot",
        "3. After the bot is created, copy the App ID and App Secret",
        "4. Enter them below and Hermes will connect automatically over WebSocket",
    ],
    "vars": [
        {"name": "YUANBAO_APP_ID", "prompt": "App ID", "password": False,
         "help": "The App ID from your Yuanbao IM Bot credentials."},
        {"name": "YUANBAO_APP_SECRET", "prompt": "App Secret", "password": True,
         "help": "The App Secret (used for HMAC signing) from your Yuanbao IM Bot."},
    ],
}


def check_yuanbao_requirements() -> bool:
    """Runtime deps present (the websockets client)."""
    return WEBSOCKETS_AVAILABLE


def interactive_setup() -> None:
    """``hermes gateway setup`` flow: the standard App ID / App Secret prompts."""
    from hermes_cli import gateway as gateway_mod
    gateway_mod._setup_standard_platform(dict(_SETUP_PLATFORM))


def _is_connected(config) -> bool:
    """Configured = App ID (or legacy App Key) + App Secret, from config extras or env."""
    extra = getattr(config, "extra", {}) or {}
    import hermes_cli.gateway as gateway_mod
    app_id = (extra.get("app_id") or gateway_mod.get_env_value("YUANBAO_APP_ID")
              or gateway_mod.get_env_value("YUANBAO_APP_KEY") or "")
    secret = extra.get("app_secret") or gateway_mod.get_env_value("YUANBAO_APP_SECRET") or ""
    return bool(str(app_id).strip() and str(secret).strip())


async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, **_kwargs):
    """standalone_sender_fn: send via the running adapter's persistent WebSocket (no throwaway
    client possible — Yuanbao needs the live signed connection)."""
    adapter = YuanbaoAdapter.get_active()
    if adapter is None:
        return send_error("Yuanbao adapter is not running. Start the gateway with yuanbao platform enabled first.")
    try:
        return await adapter._outbound.sender.send_direct(chat_id, message, media_files)
    except Exception as e:
        logger.debug("Yuanbao standalone send failed", exc_info=True)
        return send_error(f"Yuanbao send failed: {e}")


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="yuanbao", label="Yuanbao", adapter_factory=YuanbaoAdapter,
        check_fn=check_yuanbao_requirements, is_connected=_is_connected,
        required_env=["YUANBAO_APP_ID", "YUANBAO_APP_SECRET"], install_hint=_INSTALL_HINT,
        setup_fn=interactive_setup, allowed_users_env="YUANBAO_ALLOWED_USERS",
        allow_all_env="YUANBAO_ALLOW_ALL_USERS", cron_deliver_env_var="YUANBAO_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send, emoji="💎", allow_update_command=False)
