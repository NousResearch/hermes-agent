"""ZAI / GLM provider profile.

The Z.AI inference base URL ``https://api.z.ai/api/paas/v4`` is paired
with an explicit ``models_url`` so the catalog probe goes to a known
endpoint. ``auth.py`` independently auto-detects the right base URL
(general paas vs. GLM Coding Plan ``/coding/paas/v4``) on first use
via ``detect_zai_endpoint()`` and caches the result keyed on the API
key hash — so this static base_url is just the default for offline /
un-keyed clients.
"""

from providers import register_provider
from providers.base import ProviderProfile

zai = ProviderProfile(
    name="zai",
    aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    display_name="Z.AI (GLM)",
    description="Z.AI / GLM — Zhipu AI models",
    signup_url="https://z.ai/",
    # Fallback catalog used when /v1/models is unreachable (no key, 401,
    # network error). Keep newest flagship first so /model picker default
    # points at the latest reasoning model. 2026-06-13: GLM-5.2 added —
    # 1M ctx, 131K max output, reasoning-capable, text-only.
    fallback_models=(
        "glm-5.2",
        "glm-5.1",
        "glm-5",
        "glm-4.7",
        "glm-4.5-flash",
    ),
    base_url="https://api.z.ai/api/paas/v4",
    models_url="https://api.z.ai/api/paas/v4/models",
    # Auxiliary tasks (compression, etc.) keep using a cheap model — the
    # flagship glm-5.2 is overkill for summarisation.
    default_aux_model="glm-4.5-flash",
)

register_provider(zai)
