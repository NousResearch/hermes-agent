"""Fireworks AI provider profile.

Fireworks AI serves fast, production-grade inference for open and proprietary
models through an OpenAI-compatible chat-completions endpoint. It is a bundled,
preferred BYOK provider — users paste a Fireworks API key and go.

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
    description="Fireworks AI — fastest inference for production AI, 300+ models, one API key",
    signup_url="https://app.fireworks.ai/settings/users/api-keys",
    env_vars=("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL"),
    base_url="https://api.fireworks.ai/inference/v1",
    auth_type="api_key",
    # Attribution: lets Fireworks identify traffic originating from Hermes Agent
    # (partner/revenue attribution). Matches the canonical Hermes attribution
    # values in agent/auxiliary_client.py. Applied at client construction via the
    # generic profile.default_headers path in run_agent.py.
    default_headers={
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Hermes Agent",
    },
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


# ---------------------------------------------------------------------------
# Future work (intentionally disabled for now — do not ship as customer-facing).
#
# A second Fireworks key tier authenticates against the same endpoint but can
# only address managed router IDs (accounts/fireworks/routers/...), not raw
# /models/ IDs, and cannot list the account catalog. Enabling it means detecting
# the key prefix and swapping the picker catalog + aux model to routers. Kept
# here (commented out) so the wiring is ready when we turn it on. Router IDs and
# behavior track fw-ai/fireconnect setup-cli.
#
# _MANAGED_KEY_PREFIX = "fpk_"
# _MANAGED_ROUTERS = (
#     "accounts/fireworks/routers/glm-latest",       # default
#     "accounts/fireworks/routers/glm-fast-latest",
#     "accounts/fireworks/routers/kimi-latest",
#     "accounts/fireworks/routers/kimi-k2p6-turbo",  # turbo tier — not a default
# )
#
# def _is_managed_key(api_key: str) -> bool:
#     return bool(api_key) and api_key.startswith(_MANAGED_KEY_PREFIX)
# ---------------------------------------------------------------------------
