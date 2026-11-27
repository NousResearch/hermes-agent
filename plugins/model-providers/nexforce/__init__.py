"""Nexforce Router provider profile.

Nexforce Router is an OpenAI-compatible inference gateway: one endpoint and
one API key dispatch to Anthropic, OpenAI, Google, DeepSeek, Moonshot,
Zhipu and Cloudflare Workers AI models, with per-key fallback chains and
cost control. The catalog is served live at ``/v1/models`` (public without
a key). Requests may be attributed per agent via the ``X-Nexforce-Agent``
header; Hermes sets it to its own agent name so usage shows up separately
in the Nexforce console.
"""

from providers import register_provider
from providers.base import ProviderProfile


nexforce = ProviderProfile(
    name="nexforce",
    aliases=("nexforce-router",),
    display_name="Nexforce Router",
    description="Nexforce Router (OpenAI-compatible multi-provider endpoint)",
    signup_url="https://marketplace.nexforce.ai/workspace/ai-gateway/ai-gateway-keys",
    env_vars=("NEXFORCE_API_KEY",),
    base_url="https://router.nexforce.ai/v1",
    models_url="https://router.nexforce.ai/v1/models",
    auth_type="api_key",
    default_headers={"X-Nexforce-Agent": "hermes-agent"},
    default_aux_model="deepseek/deepseek-v4-flash",
    fallback_models=(
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
        "deepseek/deepseek-v4-flash",
        "google/gemini-3.6-flash",
        "z-ai/glm-5",
    ),
)

register_provider(nexforce)
