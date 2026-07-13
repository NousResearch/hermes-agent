"""Bale platform adapter.

Bale exposes a Telegram-compatible Bot API at ``https://tapi.bale.ai``.
This adapter reuses the Telegram implementation while swapping the default
Bot API root and Bale-specific environment variable names.

See https://github.com/NousResearch/hermes-agent/blob/main/website/docs/
developer-guide/adding-platform-adapters.md for the plugin architecture.
"""

import asyncio
import json
import logging
import os
from typing import Optional, Any, Dict

from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry

logger = logging.getLogger(__name__)


def check_bale_requirements() -> bool:
    """Check if Bale adapter dependencies are available."""
    try:
        import telegram  # noqa: F401
        return True
    except ImportError:
        return False


def _build_adapter(config: PlatformConfig):
    """Factory wrapper that constructs BaleAdapter from a PlatformConfig."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    class BaleAdapter(TelegramAdapter):
        """Telegram-compatible adapter for Bale.

        Bale exposes a Telegram-compatible Bot API, so we inherit from
        TelegramAdapter and override environment variable names and the
        default API root.
        """

        PLATFORM = Platform.BALE
        PLATFORM_LABEL = "Bale"
        BOT_TOKEN_LOCK_SCOPE = "bale-bot-token"
        BOT_TOKEN_RESOURCE_DESC = "Bale bot token"
        DEFAULT_BOT_API_ROOT = "https://tapi.bale.ai"
        PROXY_ENV_VAR = "BALE_PROXY"
        ALLOWED_USERS_ENV_VAR = "BALE_ALLOWED_USERS"
        REQUIRE_MENTION_ENV_VAR = "BALE_REQUIRE_MENTION"
        FREE_RESPONSE_CHATS_ENV_VAR = "BALE_FREE_RESPONSE_CHATS"
        IGNORED_THREADS_ENV_VAR = "BALE_IGNORED_THREADS"
        MENTION_PATTERNS_ENV_VAR = "BALE_MENTION_PATTERNS"
        REACTIONS_ENV_VAR = "BALE_REACTIONS"
        WEBHOOK_URL_ENV_VAR = "BALE_WEBHOOK_URL"
        WEBHOOK_PORT_ENV_VAR = "BALE_WEBHOOK_PORT"
        WEBHOOK_SECRET_ENV_VAR = "BALE_WEBHOOK_SECRET"

        def __init__(self, config: PlatformConfig):
            super().__init__(config, platform=Platform.BALE)

    return BaleAdapter(config)


def _is_connected(config: PlatformConfig) -> bool:
    """Check if Bale is configured with a bot token."""
    return bool(config.token or config.extra.get("token"))


def _validate_config(config: PlatformConfig) -> bool:
    """Validate Bale configuration."""
    return _is_connected(config)


def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> Optional[dict]:
    """Bridge YAML bale: section into env vars and PlatformConfig.extra.

    Called during load_gateway_config() after the shared-key loop and before
    _apply_env_overrides. Env vars take precedence over YAML, so we use
    `not os.getenv(...)` guards to preserve that precedence.

    Signature: (yaml_cfg, platform_cfg) -> Optional[dict]
    Returns a dict to merge into PlatformConfig.extra, or None.
    """
    if not isinstance(platform_cfg, dict):
        return None

    seeded = {}

    # require_mention: falls back to top-level require_mention
    _effective_rm = platform_cfg.get("require_mention", yaml_cfg.get("require_mention"))
    if _effective_rm is not None and not os.getenv("BALE_REQUIRE_MENTION"):
        os.environ["BALE_REQUIRE_MENTION"] = str(_effective_rm).lower()

    # mention_patterns
    if "mention_patterns" in platform_cfg and not os.getenv("BALE_MENTION_PATTERNS"):
        os.environ["BALE_MENTION_PATTERNS"] = json.dumps(platform_cfg["mention_patterns"])

    # free_response_chats
    frc = platform_cfg.get("free_response_chats")
    if frc is not None and not os.getenv("BALE_FREE_RESPONSE_CHATS"):
        if isinstance(frc, list):
            frc = ",".join(str(v) for v in frc)
        os.environ["BALE_FREE_RESPONSE_CHATS"] = str(frc)

    # ignored_threads
    ignored_threads = platform_cfg.get("ignored_threads")
    if ignored_threads is not None and not os.getenv("BALE_IGNORED_THREADS"):
        if isinstance(ignored_threads, list):
            ignored_threads = ",".join(str(v) for v in ignored_threads)
        os.environ["BALE_IGNORED_THREADS"] = str(ignored_threads)

    # reactions
    if "reactions" in platform_cfg and not os.getenv("BALE_REACTIONS"):
        os.environ["BALE_REACTIONS"] = str(platform_cfg["reactions"]).lower()

    # proxy_url
    if "proxy_url" in platform_cfg and not os.getenv("BALE_PROXY"):
        os.environ["BALE_PROXY"] = str(platform_cfg["proxy_url"]).strip()

    # allow_from (user allowlist)
    allowed_users = platform_cfg.get("allow_from")
    if allowed_users is not None and not os.getenv("BALE_ALLOWED_USERS"):
        if isinstance(allowed_users, list):
            allowed_users = ",".join(str(v) for v in allowed_users)
        os.environ["BALE_ALLOWED_USERS"] = str(allowed_users)

    # group_allow_from (group user allowlist)
    group_allowed_users = platform_cfg.get("group_allow_from")
    if group_allowed_users is not None and not os.getenv("BALE_GROUP_ALLOWED_USERS"):
        if isinstance(group_allowed_users, list):
            group_allowed_users = ",".join(str(v) for v in group_allowed_users)
        os.environ["BALE_GROUP_ALLOWED_USERS"] = str(group_allowed_users)

    # group_allowed_chats (group chat ID allowlist)
    group_allowed_chats = platform_cfg.get("group_allowed_chats")
    if group_allowed_chats is not None and not os.getenv("BALE_GROUP_ALLOWED_CHATS"):
        if isinstance(group_allowed_chats, list):
            group_allowed_chats = ",".join(str(v) for v in group_allowed_chats)
        os.environ["BALE_GROUP_ALLOWED_CHATS"] = str(group_allowed_chats)

    # disable_link_previews (seed into extra)
    if "disable_link_previews" in platform_cfg:
        seeded["disable_link_previews"] = platform_cfg["disable_link_previews"]

    # base_url (Bale API root; default https://tapi.bale.ai)
    if "base_url" in platform_cfg:
        seeded["base_url"] = platform_cfg["base_url"]

    # base_file_url (Bale file API root)
    if "base_file_url" in platform_cfg:
        seeded["base_file_url"] = platform_cfg["base_file_url"]

    return seeded if seeded else None


async def _standalone_send(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list] = None,
    force_document: bool = False,
) -> dict:
    """Send a message via Bale without a live gateway adapter.

    Called by cron/tools when the gateway is not co-resident.
    Opens an ephemeral connection, sends, and closes.

    Signature:
        async (pconfig, chat_id, message, *, thread_id=None,
               media_files=None, force_document=False) -> dict

    Returns {"success": True, "message_id": ...} on success,
    or {"error": str} on failure.
    """
    try:
        # Delegate to the shared Telegram standalone sender; Bale uses
        # the same Bot API surface with a different base URL.
        from tools.send_message_tool import _send_telegram

        base_url = pconfig.extra.get("base_url") or "https://tapi.bale.ai"
        base_file_url = pconfig.extra.get("base_file_url")

        result = await _send_telegram(
            token=pconfig.token or "",
            chat_id=chat_id,
            message=message,
            media_files=media_files,
            thread_id=thread_id,
            disable_link_previews=bool(
                pconfig.extra.get("disable_link_previews")
            ),
            base_url=base_url,
            base_file_url=base_file_url,
            platform_name="bale",
        )
        return result
    except Exception as e:
        logger.error("Bale standalone send failed: %s", e, exc_info=True)
        return {"error": str(e)}


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="bale",
        label="Bale",
        adapter_factory=_build_adapter,
        check_fn=check_bale_requirements,
        is_connected=_is_connected,
        validate_config=_validate_config,
        required_env=["BALE_BOT_TOKEN"],
        install_hint="pip install python-telegram-bot",
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="BALE_ALLOWED_USERS",
        allow_all_env="BALE_ALLOW_ALL_USERS",
        cron_deliver_env_var="BALE_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        emoji="📱",
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Bale, a Telegram-compatible platform. "
            "Markdown formatting is supported similarly to Telegram — "
            "**bold**, *italic*, `code`, and ```code blocks``` work. "
            "Use markdown for emphasis and code clarity."
        ),
    )
