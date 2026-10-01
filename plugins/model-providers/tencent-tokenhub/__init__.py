"""Tencent TokenHub provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

tencent_tokenhub = ProviderProfile(
    name="tencent-tokenhub",
    aliases=("tencent", "tokenhub", "tencent-cloud", "tencentmaas"),
    display_name="Tencent TokenHub",
    description="Tencent TokenHub (Hy4 preview via tokenhub.tencentmaas.com)",
    env_vars=("TOKENHUB_API_KEY",),
    base_url="https://tokenhub.tencentmaas.com/v1",
    base_url_env_var="TOKENHUB_BASE_URL",
    default_aux_model="hy4-preview",
)

register_provider(tencent_tokenhub)
