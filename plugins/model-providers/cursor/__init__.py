import logging

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


class CursorProfile(ProviderProfile):
    """Cursor subscription via the native Agent API."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 12.0,
    ) -> list[str] | None:
        if not api_key:
            return None
        try:
            from agent.cursor.model_discovery import fetch_cursor_usable_models

            return fetch_cursor_usable_models(
                api_key=api_key,
                base_url=self.base_url,
                timeout=timeout,
            )
        except Exception as exc:
            logger.debug("fetch_models(cursor): %s", exc)
            return None


cursor = CursorProfile(
    name="cursor",
    aliases=("cursor-agent", "cursor_subscription"),
    display_name="Cursor",
    description="Cursor subscription (Claude, GPT, Composer via Cursor Agent API)",
    api_mode="cursor_agent",
    env_vars=("CURSOR_ACCESS_TOKEN",),
    base_url="https://api2.cursor.sh",
    auth_type="oauth_external",
    fallback_models=(
        "default",
        "composer-2.5",
        "claude-4.6-opus-high",
        "claude-4.5-sonnet",
        "gpt-5.4-medium",
    ),
)
register_provider(cursor)
