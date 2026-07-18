"""Ollama (local) provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


ollama_local = ProviderProfile(
    name="ollama-local",
    aliases=('ollama-local-daemon',),
    display_name="Ollama (local)",
    description="Ollama local daemon via OpenAI-compatible /v1 (default :11434). Prefer tool-capable models.",
    signup_url="https://ollama.com",
    env_vars=('OLLAMA_LOCAL_API_KEY', 'OLLAMA_LOCAL_BASE_URL'),
    base_url="http://127.0.0.1:11434/v1",
    auth_type="api_key",
    supports_vision=True,
    default_aux_model="",
    fallback_models=(),
)

register_provider(ollama_local)
