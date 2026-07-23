"""JetBrains Junie ACP provider profile.

junie-acp uses an external ACP subprocess — NOT the standard transport.
Hermes spawns ``junie --acp=true`` and speaks the Agent Client Protocol to it.
Unlike the ``copilot-acp`` provider (which hand-rolls JSON-RPC over stdio), the
Junie path drives the official Agent Client Protocol Python SDK
(``agent-client-protocol``, the ``[acp]`` extra). The profile captures auth +
endpoint metadata; the actual subprocess driving lives in
``agent/junie_acp_client.py`` and the routing that selects it lives in
``agent/agent_runtime_helpers.py`` / ``agent/auxiliary_client.py``
(``provider == "junie-acp"``).

This provider is intentionally independent of ``copilot-acp`` so upstream
changes to the Copilot path cannot break the Junie integration.
"""

from providers import register_provider
from providers.base import ProviderProfile


class JunieACPProfile(ProviderProfile):
    """JetBrains Junie ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None


junie_acp = JunieACPProfile(
    name="junie-acp",
    aliases=("jetbrains-junie-acp", "junie-acp-agent", "junie"),
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    env_vars=(),  # Managed by the ACP subprocess (auth via --auth / JUNIE_API_KEY)
    base_url="acp://junie",  # ACP internal scheme
    auth_type="external_process",
)

register_provider(junie_acp)
