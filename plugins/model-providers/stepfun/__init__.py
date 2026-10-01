"""StepFun provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

stepfun = ProviderProfile(
    name="stepfun", aliases=("step", "stepfun-coding-plan"), display_name="StepFun Step Plan",
    description="StepFun Step Plan (Agent / coding models via Step Plan API)",
    signup_url="https://platform.stepfun.com/", default_aux_model="step-3.5-flash", env_vars=("STEPFUN_API_KEY",),
    base_url="https://api.stepfun.ai/step_plan/v1", base_url_env_var="STEPFUN_BASE_URL",
)

register_provider(stepfun)
