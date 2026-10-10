"""StepFun provider profiles: ``stepfun`` / ``stepfun-plan`` (api.stepfun.ai, standard chat and
Step Plan) and their ``-cn`` siblings on api.stepfun.com.

Region is an id rather than a base-url toggle because StepFun accounts are regional: a key issued
on api.stepfun.ai returns 401 on api.stepfun.com and vice versa. Within one region the same key
serves both endpoint families.

``supports_vision`` stays off everywhere — /models reports it false for step-3.5-flash and true
for step-3.7-flash and step-5-preview, so vision is a per-model fact from the catalog.
"""

from providers import register_provider
from providers.base import ProviderProfile

_SIGNUP_INTL = "https://platform.stepfun.ai/"
_SIGNUP_CN = "https://platform.stepfun.com/"

stepfun = ProviderProfile(
    name="stepfun", aliases=("stepfun-ai",),
    display_name="StepFun", description="StepFun standard chat completions (international)",
    signup_url=_SIGNUP_INTL, default_aux_model="step-3.7-flash",
    env_vars=("STEPFUN_API_KEY",), base_url="https://api.stepfun.ai/v1",
)

stepfun_cn = ProviderProfile(
    name="stepfun-cn", aliases=("stepfun-china", "stepfun_cn"),
    display_name="StepFun (China)", description="StepFun standard chat completions (China)",
    signup_url=_SIGNUP_CN, default_aux_model="step-3.7-flash",
    env_vars=("STEPFUN_CN_API_KEY",), base_url="https://api.stepfun.com/v1",
)

stepfun_plan = ProviderProfile(
    name="stepfun-plan", aliases=("step", "stepfun-coding-plan", "stepfun-step-plan"),
    display_name="StepFun Step Plan", description="StepFun Step Plan API (international)",
    signup_url=_SIGNUP_INTL, default_aux_model="step-3.7-flash",
    env_vars=("STEPFUN_API_KEY",), base_url="https://api.stepfun.ai/step_plan/v1",
)

stepfun_plan_cn = ProviderProfile(
    name="stepfun-plan-cn", aliases=("stepfun-step-plan-cn", "stepfun-coding-plan-cn"),
    display_name="StepFun Step Plan (China)", description="StepFun Step Plan API (China)",
    signup_url=_SIGNUP_CN, default_aux_model="step-3.7-flash",
    env_vars=("STEPFUN_CN_API_KEY",), base_url="https://api.stepfun.com/step_plan/v1",
)

for _profile in (stepfun, stepfun_cn, stepfun_plan, stepfun_plan_cn):
    register_provider(_profile)
