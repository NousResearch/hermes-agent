"""Volcengine Ark Agent Plan provider profile.

Agent Plan (https://www.volcengine.com/docs/82379/2366394) is a subscription
tier served from a dedicated endpoint, ``https://ark.cn-beijing.volces.com/api/plan/v3``,
with its own API-key space: a plan key is rejected with 401 on the pay-as-you-go
``/api/v3`` and vice versa, so this profile must stay distinct from any
pay-as-you-go Volcengine entry.

There is no ``/models`` route on the plan endpoint (it returns 404), so the
model catalog comes from models.dev (``volcengine-agent-plan``) and the
static ``fallback_models`` below is only a floor for when that catalog is
unreachable. The static list is NOT the authority — models.dev tracks the
plan roster, and users can extend it via ``model_overrides``/config.
"""

from providers import register_provider
from providers.base import ProviderProfile

volcengine_agent_plan = ProviderProfile(
    name="volcengine-agent-plan",
    aliases=("ark-agent-plan", "volcengine-plan", "ark-plan"),
    display_name="Volcengine Ark (Agent Plan)",
    description="Volcengine Ark Agent Plan (subscription tier, Doubao/DeepSeek/Kimi/GLM/MiniMax)",
    signup_url="https://www.volcengine.com/docs/82379/2366394",
    env_vars=("ARK_API_KEY", "ARK_AGENT_PLAN_BASE_URL"),
    base_url="https://ark.cn-beijing.volces.com/api/plan/v3",
    auth_type="api_key",
    supports_health_check=False,  # no /models route on the plan endpoint (404)
    # Only an entry floor: the authoritative catalog is models.dev's
    # ``volcengine-agent-plan`` (auto-merged into the picker). These two are
    # the documented general-purpose entry models; both verified HTTP 200.
    fallback_models=("ark-code-latest", "deepseek-v4-flash"),
    default_aux_model="deepseek-v4-flash",
)

register_provider(volcengine_agent_plan)
