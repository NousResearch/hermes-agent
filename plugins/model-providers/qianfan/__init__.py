"""Baidu Qianfan provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


qianfan = ProviderProfile(
    name="qianfan",
    aliases=('baidu-qianfan', 'baidu'),
    display_name="Baidu Qianfan",
    description="Baidu Qianfan — unified OpenAI-compatible MaaS API",
    signup_url="https://console.bce.baidu.com/qianfan/ais/console/apiKey",
    env_vars=('QIANFAN_API_KEY', 'QIANFAN_BASE_URL'),
    base_url="https://qianfan.baidubce.com/v2",
    auth_type="api_key",
    supports_vision=True,
    default_aux_model="deepseek-v3.2",
    fallback_models=('deepseek-v3.2', 'ernie-5.0-thinking-preview', 'ernie-4.5-turbo-128k'),
)

register_provider(qianfan)
