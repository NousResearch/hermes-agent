"""SiliconFlow provider profile.

SiliconFlow (siliconflow.cn) provides an OpenAI-compatible API for
a wide range of open-source and commercial LLMs, including DeepSeek,
Qwen, GLM, and many others.
"""

from providers import register_provider
from providers.base import ProviderProfile

siliconflow = ProviderProfile(
    name="siliconflow",
    aliases=("silicon-flow", "silicon_flow"),
    env_vars=("SILICONFLOW_API_KEY", "SILICONFLOW_BASE_URL"),
    base_url="https://api.siliconflow.cn/v1",
    display_name="SiliconFlow",
    description="SiliconFlow — OpenAI-compatible API for open-source & commercial LLMs",
    signup_url="https://cloud.siliconflow.cn/",
    supports_health_check=True,
    supports_vision=True,
)

register_provider(siliconflow)
