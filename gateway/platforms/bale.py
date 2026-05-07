"""Bale platform adapter.

Bale exposes a Telegram-compatible Bot API at ``https://tapi.bale.ai``.
This adapter reuses the Telegram implementation while swapping the default
Bot API root and Bale-specific environment variable names.
"""

from gateway.config import Platform
from gateway.platforms.telegram import TelegramAdapter, check_telegram_requirements


class BaleAdapter(TelegramAdapter):
    """Telegram-compatible adapter for Bale."""

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

    def __init__(self, config):
        super().__init__(config, platform=Platform.BALE)


check_bale_requirements = check_telegram_requirements

