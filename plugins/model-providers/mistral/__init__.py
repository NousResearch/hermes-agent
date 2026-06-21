"""Mistral AI provider profile.

Mistral's platform speaks the OpenAI-compatible Chat Completions API at
``https://api.mistral.ai/v1``, so the default ``openai_chat`` transport carries
tool calling and streaming without special handling. The endpoint is strict and
rejects unknown message fields, but the transport already strips Hermes-internal
keys (reasoning carriers, codex fields, finish_reason, scaffolding markers)
before the request leaves, so no extra message handling is needed here.

Adjustable reasoning (``reasoning_effort``) is intentionally not wired: Mistral
streams reasoning as structured content blocks (``[{"type": "thinking"}, ...]``)
rather than a plain string, which the streaming path does not flatten yet, so
enabling it truncates responses. That belongs in the streaming layer, not this
profile.
"""

from providers import register_provider
from providers.base import ProviderProfile


mistral = ProviderProfile(
    name="mistral",
    aliases=("mistral-ai", "mistralai"),
    env_vars=("MISTRAL_API_KEY", "MISTRAL_BASE_URL"),
    display_name="Mistral AI",
    description="Mistral AI: Mistral, Codestral, and Devstral models (direct API)",
    signup_url="https://console.mistral.ai/",
    base_url="https://api.mistral.ai/v1",
    auth_type="api_key",
    supports_vision=True,
    default_aux_model="mistral-small-latest",
    fallback_models=(
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "codestral-latest",
        "devstral-medium-latest",
        "ministral-8b-latest",
    ),
)

register_provider(mistral)
