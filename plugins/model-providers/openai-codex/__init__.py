"""OpenAI Codex (Responses API) provider profile."""

from providers import register_provider
from providers.base import ProviderProfile
from providers.model_normalizers import strip_matching_prefix

class OpenAICodexProfile(ProviderProfile):
    def normalize_model_id(self, model: str, *, known_ids=()) -> str:
        return strip_matching_prefix(
            self,
            model,
            extra_prefixes=("openai",),
            excluded_prefixes=("codex", "openai_codex"),
        )


openai_codex = OpenAICodexProfile(
    name="openai-codex", aliases=("codex", "openai_codex", "chatgpt", "chatgpt-codex"),
    display_name="ChatGPT or Codex Subscription",
    description="ChatGPT or Codex Subscription (Sign in with your ChatGPT account, uses Codex models)",
    api_mode="codex_responses",
    env_vars=(),  # OAuth external — no API key
    base_url="https://chatgpt.com/backend-api/codex", auth_type="oauth_external",
)

register_provider(openai_codex)
