"""Tencent TokenPlan provider profile.

Routes Hermes through Tencent's LKEAP TokenPlan gateway, which exposes the
Hy3 (Hunyuan) model via an Anthropic Messages-compatible endpoint. The base
URL ends with ``/anthropic`` so the Anthropic SDK appends ``/v1/messages``
automatically — do not include the ``/v1/messages`` suffix here.
"""

from providers import register_provider
from providers.base import ProviderProfile


tencent_tokenplan = ProviderProfile(
    name="tencent-tokenplan",
    aliases=("tokenplan", "tencent-lkeap"),
    api_mode="anthropic_messages",
    display_name="Tencent TokenPlan",
    description="Tencent TokenPlan (Hy3 via api.lkeap.cloud.tencent.com)",
    signup_url="https://cloud.tencent.com/product/lkeap",
    env_vars=("TOKENPLAN_API_KEY", "TOKENPLAN_BASE_URL"),
    base_url="https://api.lkeap.cloud.tencent.com/plan/anthropic",
    auth_type="api_key",
    default_aux_model="hy3",
    fallback_models=("hy3",),
)

register_provider(tencent_tokenplan)
