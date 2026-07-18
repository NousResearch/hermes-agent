"""SGLang provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


sglang = ProviderProfile(
    name="sglang",
    aliases=('sglang-local',),
    display_name="SGLang",
    description="SGLang — local/self-hosted OpenAI-compatible server (default :30000)",
    signup_url="https://github.com/sgl-project/sglang",
    env_vars=('SGLANG_API_KEY', 'SGLANG_BASE_URL'),
    base_url="http://127.0.0.1:30000/v1",
    auth_type="api_key",
    supports_vision=False,
    default_aux_model="",
    fallback_models=(),
)

register_provider(sglang)
