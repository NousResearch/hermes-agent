"""Devin CLI ACP provider profile.

devin-acp uses an external ACP subprocess (``devin acp``) — NOT a REST
chat-completions endpoint. Routing is handled by DevinACPClient, same
pattern as copilot-acp.
"""

from providers import register_provider
from providers.base import ProviderProfile


class DevinACPProfile(ProviderProfile):
    """Devin CLI ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None


devin_acp = DevinACPProfile(
    name="devin-acp",
    aliases=("devin", "devin-cli", "cognition-devin"),
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://devin",
    auth_type="external_process",
)

register_provider(devin_acp)
