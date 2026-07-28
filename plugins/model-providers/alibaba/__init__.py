"""Alibaba Cloud DashScope provider profiles.

DashScope has region-split endpoints with the same key type:
  - ``alibaba``    → dashscope-intl.aliyuncs.com (international)
  - ``alibaba-cn`` → dashscope.aliyuncs.com (mainland China)

Profile names match the models.dev catalog keys exactly
(``alibaba`` / ``alibaba-cn``) so model metadata lines up and
``model.provider: alibaba-cn`` resolves at runtime (#73265).
"""

from providers import register_provider
from providers.base import ProviderProfile

alibaba = ProviderProfile(
    name="alibaba",
    aliases=("dashscope", "alibaba-cloud", "qwen-dashscope"),
    env_vars=("DASHSCOPE_API_KEY",),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

alibaba_cn = ProviderProfile(
    name="alibaba-cn",
    aliases=("dashscope-cn", "alibaba-cloud-cn"),
    display_name="Alibaba Cloud DashScope (China)",
    description="Alibaba Cloud DashScope, mainland-China endpoint",
    env_vars=("DASHSCOPE_API_KEY",),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

register_provider(alibaba)
register_provider(alibaba_cn)
