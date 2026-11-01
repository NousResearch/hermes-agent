"""Tetrate Agent Router provider profile.

Unified OpenAI-compatible gateway (https://router.tetrate.ai) that fronts 25+
upstream providers (Anthropic, OpenAI, Google, xAI, DeepInfra, ...) behind a
single Chat Completions endpoint and API key.
"""

from providers import register_provider
from providers.base import ProviderProfile

tetrate_agent_router = ProviderProfile(
    name="tetrate-agent-router", aliases=("tetrate", "agent-router", "agentrouter"),
    display_name="Tetrate Agent Router",
    description="Tetrate Agent Router — unified OpenAI-compatible gateway across 25+ providers",
    signup_url="https://router.tetrate.ai/",
    env_vars=("TETRATE_AGENT_ROUTER_API_KEY", "AGENTROUTER_API_KEY", "TETRATE_AGENT_ROUTER_BASE_URL"),
    base_url="https://api.router.tetrate.ai/v1", auth_type="api_key",
    default_aux_model="gemini-2.5-flash",
    fallback_models=("claude-sonnet-5", "gpt-5.6-terra", "gemini-2.5-flash"),
)

register_provider(tetrate_agent_router)
