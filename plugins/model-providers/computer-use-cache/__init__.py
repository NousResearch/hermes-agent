"""Computer-Use Cache provider profile.

Computer-Use Cache is an OpenAI-compatible local proxy for replaying repeated
computer-use, browser, coding, and tool workflows.  Hermes can use it exactly
like any other chat-completions provider by pointing at the local cache server.
"""

from providers import register_provider
from providers.base import ProviderProfile


computer_use_cache = ProviderProfile(
    name="computer-use-cache",
    aliases=("cuc", "computer_use_cache", "code-model-cache"),
    display_name="Computer-Use Cache",
    description="Local OpenAI-compatible cache proxy for repeated agent workflows",
    signup_url="https://github.com/rohanarun/computer-use-cache",
    env_vars=("COMPUTER_USE_CACHE_API_KEY",),
    base_url="http://127.0.0.1:8000/v1",
    auth_type="api_key",
)


register_provider(computer_use_cache)
