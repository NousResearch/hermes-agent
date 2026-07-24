"""Venice AI provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


class VeniceProfile(ProviderProfile):
    """OpenAI-compatible Venice provider with a text-only live catalog."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        from hermes_cli.models import fetch_api_models

        live = fetch_api_models(
            api_key,
            base_url or self.base_url,
            timeout=timeout,
            query_params={"type": "text"},
        )
        if live is None:
            return None

        # Hermes' standard chat-completions transport does not implement the
        # client-side attestation, encryption, and decryption required by
        # Venice E2EE models. Plain TEE and regular text models remain usable.
        return [model_id for model_id in live if not model_id.startswith("e2ee-")]


venice = VeniceProfile(
    name="venice",
    aliases=("venice-ai",),
    display_name="Venice AI",
    description="Venice AI — OpenAI-compatible private inference",
    signup_url="https://venice.ai/settings/api",
    env_vars=("VENICE_API_KEY", "VENICE_BASE_URL"),
    base_url="https://api.venice.ai/api/v1",
    models_url="https://api.venice.ai/api/v1/models?type=text",
    auth_type="api_key",
    default_aux_model="zai-org-glm-4.7-flash",
    fallback_models=(
        "qwen3-coder-480b-a35b-instruct",
        "zai-org-glm-5",
        "zai-org-glm-4.7",
        "zai-org-glm-4.7-flash",
        "kimi-k2-5",
        "deepseek-v3.2",
        "openai-gpt-54",
    ),
)

register_provider(venice)
