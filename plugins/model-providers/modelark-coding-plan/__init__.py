"""BytePlus/VolcEngine ModelArk Coding Plan provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

modelark_coding_plan = ProviderProfile(
    name="modelark-coding-plan",
    aliases=("modelark", "byteplus-coding", "byteplus_coding", "volcengine-coding"),
    display_name="BytePlus/VolcEngine ModelArk Coding Plan",
    description="BytePlus/VolcEngine ModelArk Coding Plan — Seed, Kimi, GLM, DeepSeek models",
    signup_url="https://docs.byteplus.com/en/docs/ModelArk/1925114",
    env_vars=("BYTEPLUS_API_KEY",),
    base_url="https://ark.ap-southeast.bytepluses.com/api/coding/v3",
    auth_type="api_key",
)

register_provider(modelark_coding_plan)
