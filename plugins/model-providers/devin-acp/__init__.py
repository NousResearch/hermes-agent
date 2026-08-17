"""Devin ACP provider profile.

devin-acp uses an external ACP subprocess (devin acp) — similar to copilot-acp.
This lets Hermes use Devin's models (especially SWE-1.7, Adaptive, etc.)
as the generation backend while still using Hermes tool loop.
"""

from providers import register_provider
from providers.base import ProviderProfile


class DevinACPProfile(ProviderProfile):
    """Devin via ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess + --model flag."""
        # We don't list dynamically; aliases are configured by user.
        return None


devin_acp = DevinACPProfile(
    name="devin-acp",
    aliases=("devin", "devin-acp-agent", "cognition-devin"),
    display_name="Devin (Cognition)",
    description="Devin ACP — SWE-1.7, Adaptive, and frontier models via devin acp",
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    env_vars=(),  # Auth is via devin credentials (devin auth login)
    base_url="acp://devin",
    auth_type="external_process",
    fallback_models=(
        "swe-1-7",
        "swe-1-7-lightning",
        "swe-1-7-medium",
        "swe-1-7-lightning-medium",
        "adaptive",
        "claude-opus-5-high",
        "claude-sonnet-5-high",
        "claude-5-fable-medium",
        "gpt-5-6-sol-medium",
        "gpt-5-6-luna-medium",
        "gpt-5-6-terra-medium",
        "glm-5-2",
        "gemini-3-7-flash-medium",
        "kimi-k3-low",
        "deepseek-v4-flash-high",
        "grok-4-6-medium",
    ),
)

register_provider(devin_acp)
