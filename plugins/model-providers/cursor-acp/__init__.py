"""Cursor Agent ACP provider profile.

cursor-acp uses an external ACP subprocess (`agent acp`) — NOT a REST
transport. api_mode="chat_completions" routes through CursorACPClient.
"""

from providers import register_provider
from providers.base import ProviderProfile


class CursorACPProfile(ProviderProfile):
    """Cursor Agent ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None


cursor_acp = CursorACPProfile(
    name="cursor-acp",
    aliases=("cursor", "cursor-agent", "cursor-cli"),
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://cursor",
    auth_type="external_process",
)

register_provider(cursor_acp)
