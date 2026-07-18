"""Volcengine (Doubao) provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


volcengine = ProviderProfile(
    name="volcengine",
    aliases=('volcengine-ark', 'doubao', 'volcano-engine'),
    display_name="Volcengine (Doubao)",
    description="Volcengine ARK / Doubao — OpenAI-compatible API (China)",
    signup_url="https://console.volcengine.com/ark",
    env_vars=('VOLCANO_ENGINE_API_KEY', 'VOLCANO_ENGINE_BASE_URL'),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    auth_type="api_key",
    supports_vision=True,
    default_aux_model="deepseek-v3-2-251201",
    fallback_models=('deepseek-v3-2-251201', 'doubao-seed-1-8-251228', 'doubao-1-5-pro-32k-250115'),
)

register_provider(volcengine)
