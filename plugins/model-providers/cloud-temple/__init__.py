"""Cloud Temple provider profile.

OpenAI-compatible API hosted at api.ai.cloud-temple.com.
No special wire-format overrides needed — standard chat_completions mode.

Models:
  - qwen3.6:27b  — 1M context window
  - gemma4:31b   — 256K context window
"""

from providers import register_provider
from providers.base import ProviderProfile


cloud_temple = ProviderProfile(
    name="cloud-temple",
    env_vars=("CLOUD_TEMPLE_API_KEY",),
    display_name="Cloud Temple",
    description="Cloud Temple Secure Inference Providers",
    signup_url="https://api.ai.cloud-temple.com/",
    fallback_models=("qwen3.6:27b", "gemma4:31b"),
    base_url="https://api.ai.cloud-temple.com/v1",
    default_aux_model="qwen3.6:27b",
)


register_provider(cloud_temple)
