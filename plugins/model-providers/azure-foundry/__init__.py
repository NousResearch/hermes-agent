"""Azure AI Foundry provider profile.

Azure Foundry exposes an OpenAI-compatible endpoint; users supply their own
base URL at setup since endpoints are per-resource.
"""

from providers import register_provider
from providers.base import ProviderProfile

azure_foundry = ProviderProfile(
    name="azure-foundry",
    aliases=("azure", "azure-ai-foundry", "azure-ai"),
    display_name="Azure Foundry",
    description="Azure AI Foundry — OpenAI-compatible endpoint (user-supplied base URL)",
    signup_url="https://ai.azure.com/",
    env_vars=("AZURE_FOUNDRY_API_KEY", "AZURE_FOUNDRY_BASE_URL"),
    base_url="",  # per-resource; user provides at setup
    auth_type="api_key",
)

register_provider(azure_foundry)
