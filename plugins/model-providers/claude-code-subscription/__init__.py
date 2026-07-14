"""Claude Code subscription-backed external-process provider."""

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeCodeSubscriptionProfile(ProviderProfile):
    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        return None


register_provider(
    ClaudeCodeSubscriptionProfile(
        name="claude-code-subscription",
        aliases=("claude-subscription", "claude-max"),
        display_name="Claude Code Subscription",
        description="Claude through the logged-in Claude Code Pro/Max subscription",
        api_mode="chat_completions",
        auth_type="external_process",
        env_vars=(),
        base_url="claude-code://subscription",
        supports_health_check=False,
        fallback_models=("sonnet", "opus", "haiku"),
        default_aux_model="haiku",
    )
)
