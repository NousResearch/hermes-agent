"""Kiro CLI ACP provider profile.

kiro-acp uses an external ACP subprocess. The local Kiro CLI handles auth.
Hermes routes this provider through the existing ACP subprocess client.
"""

from providers import register_provider
from providers.base import ProviderProfile


class KiroACPProfile(ProviderProfile):
    """Kiro CLI ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Kiro does not expose a REST /models catalog through Hermes."""
        return None


kiro_acp = KiroACPProfile(
    name="kiro-acp",
    aliases=("kiro-cli-acp", "kiro-cli", "kiro"),
    api_mode="chat_completions",
    display_name="Kiro CLI (ACP)",
    description="Kiro CLI via ACP subprocess",
    env_vars=(),
    base_url="acp://kiro",
    auth_type="external_process",
    fallback_models=("kiro-cli",),
)

register_provider(kiro_acp)
