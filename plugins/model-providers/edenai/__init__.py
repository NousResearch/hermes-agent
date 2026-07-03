"""Eden AI provider profile.

Eden AI exposes an OpenAI-compatible ``/v3`` endpoint that fronts many
upstream providers (OpenAI, Anthropic, Google, Meta, Mistral, etc.) behind
a single Eden AI API key. Model IDs follow the ``<upstream>/<model>`` form
that OpenRouter popularized, e.g. ``openai/gpt-4o``,
``anthropic/claude-3-5-sonnet``, ``google/gemini-1.5-pro``.

The base chat-completions transport handles auth, streaming, tool calls
and JSON shape natively, so this profile only needs to declare identity
+ endpoints + a static fallback model list. Override
``prepare_messages`` / ``build_extra_body`` if you later need Eden AI's
provider-specific knobs (their native ``/v2`` API has per-upstream fields
that the OpenAI-compat layer does not expose).
"""

from providers import register_provider
from providers.base import ProviderProfile

edenai = ProviderProfile(
    name="edenai",
    aliases=("eden", "eden.ai"),
    api_mode="chat_completions",
    env_vars=("EDENAI_API_KEY",),
    base_url="https://api.edenai.run/v3",
    auth_type="api_key",
    display_name="Eden AI",
    description="Eden AI — multi-provider AI gateway (OpenAI-compatible)",
    signup_url="https://app.edenai.run/",
    fallback_models=(
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet",
        "google/gemini-1.5-pro",
        "meta/llama-3.1-70b-instruct",
    ),
)

register_provider(edenai)