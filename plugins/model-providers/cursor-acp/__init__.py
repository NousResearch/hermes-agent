"""Cursor ACP provider profile.

cursor-acp uses an external ACP subprocess (`cursor-agent acp`) — NOT the
HTTP `cur-*` shim on :8317/:8321. That shim correctly returns HTTP 400
`unsupported_tools` for any request carrying a tool schema. This provider
is the tool-using execution path.
"""

from providers import register_provider
from providers.base import ProviderProfile


class CursorACPProfile(ProviderProfile):
    """Cursor ACP — external process, no REST models endpoint."""

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
    aliases=("cursor-acp-agent", "cursor-agent-acp"),
    display_name="Cursor ACP",
    description="Cursor ACP (Spawns cursor-agent acp)",
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://cursor",
    auth_type="external_process",
    fallback_models=("gpt-5.6-sol-high", "gemini-3.7-flash-high"),
)

register_provider(cursor_acp)
