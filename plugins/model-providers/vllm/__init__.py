"""vLLM provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


vllm = ProviderProfile(
    name="vllm",
    aliases=('vllm-local',),
    display_name="vLLM",
    description="vLLM — local/self-hosted OpenAI-compatible server (default :8000)",
    signup_url="https://docs.vllm.ai/",
    env_vars=('VLLM_API_KEY', 'VLLM_BASE_URL'),
    base_url="http://127.0.0.1:8000/v1",
    auth_type="api_key",
    supports_vision=False,
    default_aux_model="",
    fallback_models=(),
)

register_provider(vllm)
