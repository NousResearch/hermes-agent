"""Microsoft Foundry provider profile: OpenAI-compatible, per-resource base URL
supplied by the user at setup."""

from providers import register_provider
from providers.base import ProviderProfile


class AzureFoundryProfile(ProviderProfile):
    """Azure Foundry's model-dependent Responses API policy."""

    def resolve_route_policy(self, model: str, base_url: str = "", *, options=None) -> str | None:
        """Use Responses for Azure deployments that do not accept Chat Completions."""
        del base_url
        normalized = str(model or "").strip().lower().rsplit("/", 1)[-1]
        if normalized.startswith(("codex", "gpt-5", "o1", "o3", "o4")):
            return "codex_responses"
        return None


azure_foundry = AzureFoundryProfile(
    name="azure-foundry", aliases=("azure", "azure-ai-foundry", "azure-ai"), display_name="Azure Foundry",
    description="Microsoft Foundry - OpenAI-compatible endpoint (user-supplied base URL)",
    signup_url="https://ai.azure.com/", env_vars=("AZURE_FOUNDRY_API_KEY",),
    base_url="",  # per-resource; user provides at setup
    base_url_env_var="AZURE_FOUNDRY_BASE_URL",
    auth_type="api_key",
)

register_provider(azure_foundry)
