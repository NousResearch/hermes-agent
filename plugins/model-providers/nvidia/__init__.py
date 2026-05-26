"""NVIDIA NIM provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

nvidia = ProviderProfile(
    name="nvidia",
    aliases=("nvidia-nim",),
    env_vars=("NVIDIA_API_KEY",),
    display_name="NVIDIA NIM",
    description="NVIDIA NIM — accelerated inference",
    signup_url="https://build.nvidia.com/",
    fallback_models=(
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia/llama-3.3-70b-instruct",
    ),
    base_url="https://integrate.api.nvidia.com/v1",
    default_max_tokens=16384,
    backoff_base_delay=30.0,
    backoff_max_delay=120.0,
    min_request_interval=2.0,
)

register_provider(nvidia)
