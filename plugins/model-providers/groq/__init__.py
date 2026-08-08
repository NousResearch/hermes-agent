"""Groq provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


class GroqProfile(ProviderProfile):
    """GroqCloud — curated picker catalog.

    Groq's live ``/models`` endpoint includes STT, safeguard, prompt-guard,
    and other non-agent chat entries. Returning ``None`` here makes
    ``provider_model_ids('groq')`` fall back to Hermes' curated chat list
    instead of appending noisy live-only entries.
    """

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        return None


groq = GroqProfile(
    name="groq",
    aliases=("groqcloud", "groq-cloud"),
    api_mode="chat_completions",
    env_vars=("GROQ_API_KEY", "GROQ_BASE_URL"),
    display_name="Groq",
    description="GroqCloud — fast OpenAI-compatible inference",
    signup_url="https://console.groq.com/keys",
    base_url="https://api.groq.com/openai/v1",
    auth_type="api_key",
)

register_provider(groq)
