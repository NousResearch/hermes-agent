"""Fireworks AI provider profile.

Fireworks AI serves fast, production-grade inference for open and proprietary
models through an OpenAI-compatible chat-completions endpoint.

Address models directly by their catalog ID, e.g.
``accounts/fireworks/models/kimi-k2p6`` or ``accounts/fireworks/models/glm-5p2``.
Model IDs here track the canonical Fireworks catalog (fw-ai/fireconnect
``setup-cli``).
"""

from providers import register_provider
from providers.base import ProviderProfile


fireworks = ProviderProfile(
    name="fireworks",
    aliases=("fireworks-ai", "fw"),
    display_name="Fireworks AI",
    description="Fireworks AI — OpenAI-compatible direct model API",
    signup_url="https://app.fireworks.ai/settings/users/api-keys",
    env_vars=("FIREWORKS_API_KEY",),
    base_url="https://api.fireworks.ai/inference/v1",
    auth_type="api_key",
    # Auxiliary model for cheap tasks (compaction, title generation, vision).
    # A standard pay-as-you-go catalog ``/models/`` ID.
    default_aux_model="accounts/fireworks/models/glm-5p2",
    # Curated safety net shown in the picker when the live catalog fetch fails.
    fallback_models=(
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/kimi-k2p7-code",
    ),
)

register_provider(fireworks)
