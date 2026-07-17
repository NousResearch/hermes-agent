"""StepFun provider profiles.

Two endpoints, both region-toggleable (International `.ai` ↔ China `.com`):
  - `stepfun`      → standard chat completions (api.stepfun.ai/v1)
  - `stepfun-plan` → Step Plan reasoning API   (api.stepfun.ai/step_plan/v1)

Region is switched at setup time via the shared region-toggle flow, which
writes the chosen host to each id's own base-url override env var. Both share
the STEPFUN_API_KEY account key.
"""

from providers import register_provider
from providers.base import ProviderProfile

stepfun = ProviderProfile(
    name="stepfun",
    aliases=("stepfun-ai",),
    display_name="StepFun",
    description="StepFun standard chat completions (Step 3.x models)",
    signup_url="https://platform.stepfun.com/",
    default_aux_model="step-3.5-flash",
    env_vars=("STEPFUN_API_KEY",),
    base_url="https://api.stepfun.ai/v1",
)

stepfun_plan = ProviderProfile(
    name="stepfun-plan",
    aliases=("step", "stepfun-coding-plan"),
    display_name="StepFun Step Plan",
    description="StepFun Step Plan reasoning API (plan-based billing)",
    signup_url="https://platform.stepfun.com/",
    default_aux_model="step-3.5-flash",
    env_vars=("STEPFUN_API_KEY",),
    base_url="https://api.stepfun.ai/step_plan/v1",
)

register_provider(stepfun)
register_provider(stepfun_plan)
