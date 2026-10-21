"""Yuanbao plugin registration: setup row, configured check, standalone sender, ``register``."""

import re
import logging

from gateway.platforms._shared import send_error
from gateway.platforms._shared import config_connected_extra
from gateway.platforms.send_targets import NOT_EXPLICIT, StandaloneSendPolicy

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


# System-prompt platform hint (PlatformEntry.platform_hint); cached-prompt bytes, keep stable.
_PLATFORM_HINT = (
    "You are on Yuanbao (\u817e\u8baf\u5143\u5b9d), a Chinese AI assistant "
    "platform. Markdown renders (code blocks, tables, bold/italic). "
    "You can send files natively: write MEDIA:/absolute/path/to/file in your response. "
    "Images (.jpg, .png, .webp, .gif) send as photos, other files as downloadable documents (max 50 MB); "
    "image URLs via ![alt](url) are downloaded and sent as photos. Never claim you lack file-sending. "
    "Stickers (\u8d34\u7eb8/\u8868\u60c5\u5305): when the user sends one (you see '[emoji: "
    "\u540d\u79f0]') or asks for one, use the sticker tools \u2014 yb_search_sticker with a Chinese "
    "keyword, then yb_send_sticker with the chosen id \u2014 which send a real native sticker. Never "
    "draw sticker-like PNGs and send them as images, and bare Unicode emoji is not a substitute."
)
_SEND_TARGET_RE = re.compile(r"^\s*((?:group|direct):[^:]+)\s*$")


def _parse_send_target(ref):
    # "group:<code>" / "direct:<id>"; a bare number is a group code (never generic rules).
    match = _SEND_TARGET_RE.fullmatch(ref)
    if match:
        return match.group(1), None
    return (f"group:{ref.strip()}", None) if ref.strip().isdigit() else NOT_EXPLICIT


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    from . import gateway_env  # env -> GatewayConfig bridge (gateway.config_env)
    ctx.register_platform(
        target_ref_syntax_fn=_parse_send_target,
        standalone_send=StandaloneSendPolicy(label="Yuanbao", media=True, text_direct=True),
        name="yuanbao", label="Yuanbao", adapter_factory=YuanbaoAdapter,
        dashboard_label="Yuanbao (元宝)",
        dashboard_description="Connect Hermes to Tencent Yuanbao.",
        dashboard_required_env=(),
        check_fn=check_yuanbao_requirements, is_connected=_is_connected,
        config_connected_fn=config_connected_extra("app_id", "app_secret"),
        required_env=["YUANBAO_APP_ID", "YUANBAO_APP_SECRET"], install_hint=_INSTALL_HINT,
        setup_fn=interactive_setup, cron_deliver_env_var="YUANBAO_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send, emoji="💎", allow_update_command=False,
        platform_hint=_PLATFORM_HINT,
        env_overrides=gateway_env.ENV_OVERRIDES,
        env_enable_credentials=gateway_env.ENV_ENABLE_CREDENTIALS, **gateway_env.AUTHZ,
    )
