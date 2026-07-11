"""Grok CLI ACP provider profile.

grok-acp uses an external ACP subprocess (``grok --no-auto-update agent stdio``)
— NOT a REST chat-completions endpoint. Routing is handled by GrokACPClient,
same pattern as copilot-acp and devin-acp.
"""

from providers import register_provider
from providers.base import ProviderProfile


class GrokACPProfile(ProviderProfile):
    """Grok CLI ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess or hermes_cli.models."""
        return None


grok_acp = GrokACPProfile(
    name="grok-acp",
    display_name="Grok CLI ACP",
    description="Grok CLI via ACP (grok --no-auto-update agent stdio).",
    aliases=("grok-cli", "grok-build", "xai-grok-cli"),
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://grok",
    auth_type="external_process",
)

register_provider(grok_acp)
