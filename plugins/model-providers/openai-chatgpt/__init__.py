"""ChatGPT plan usage via Sign in with ChatGPT (Responses API on api.openai.com).

Opt-in sibling of the openai-codex profile: same ChatGPT account, but a SEPARATE stored
and refreshed credential (providers.openai-chatgpt, use_direct opt-in) routed ONLY to
https://api.openai.com/v1. The Codex backend profile stays untouched.
"""

from providers import register_provider
from providers.base import ProviderProfile

openai_chatgpt = ProviderProfile(
    name="openai-chatgpt", aliases=("chatgpt-plan", "siwc"), api_mode="codex_responses",
    display_name="ChatGPT Plan (Sign in with ChatGPT)",
    description="ChatGPT Plus/Pro/Team quota on api.openai.com via Sign in with ChatGPT",
    env_vars=(),  # OAuth external — no API key
    base_url="https://api.openai.com/v1", auth_type="oauth_external",
)

register_provider(openai_chatgpt)
