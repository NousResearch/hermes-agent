"""Z.AI Coding Plan provider profile.

Separate from the standard ``zai`` profile because coding-plan subscriptions
authenticate on different endpoints than the standard ``/api/paas/v4`` route
(which rejects coding-plan keys with HTTP 429 ``1113 Insufficient balance or
no resource package``), and the coding plan is the subscription tier most
agent users actually hold.

Default endpoint: ``https://api.z.ai/api/anthropic`` — z.ai's Anthropic
Messages wire. Chosen over ``/api/coding/paas/v4`` for agent workloads
(probed live 2026-08-15, glm-5.3):

* **Preserved thinking works here.** Replayed ``thinking`` blocks reach the
  model (verified via a thinking-only secret-recall probe; the model quoted
  the exact secret back). On the OpenAI-compat routes the same key's
  replayed ``reasoning_content`` is accepted with HTTP 200 but silently
  dropped from model attention — billed bytes the model never sees. For a
  tool-loop agent that means re-deriving its plan every iteration instead
  of continuing its chain of thought.
* ``thinking.budget_tokens`` is honored (output_tokens scaled 1710 → 3848
  for budget 512 → 6000 on identical prompts), so reasoning-effort control
  rides the Anthropic budget path rather than the OpenAI-compat
  ``reasoning_effort`` param.
* Hermes auto-selects the Anthropic Messages adapter for base URLs ending
  in ``/anthropic`` (agent/agent_init.py), so the protocol switch is
  transparent.

Mirrors the ``alibaba-coding-plan`` / ``kimi-coding`` pattern: a dedicated
selectable provider so coding-plan users get a working, agent-optimal
default without hand-editing ``GLM_BASE_URL``. Subclasses ``ZaiProfile``
so the GLM thinking / reasoning wiring is shared with the standard
profile; on this wire the Anthropic adapter builds the request shape
(thinking blocks, tool_use) natively.
"""

# NOTE: this import deliberately targets the BUNDLED zai plugin module.
# Discovery loads bundled plugins under their canonical
# ``plugins.model_providers.*`` names; any user-plugin override of ``zai``
# later registers under a ``_hermes_user_provider_*`` module name instead,
# so this path can never bind a user override. If the plugin loader ever
# changes that namespace contract, update this import (and the
# test_bundled_import_binds_bundled_zai_profile pin) accordingly.
from plugins.model_providers.zai import ZaiProfile
from providers import register_provider

zai_coding_plan = ZaiProfile(
    name="zai-coding-plan",
    aliases=("zai-coding", "glm-coding", "z-ai-coding"),
    display_name="Z.AI / GLM (Coding Plan)",
    description="Z.AI Coding Plan (GLM subscription tier, Anthropic wire with preserved thinking)",
    signup_url="https://z.ai/subscribe",
    env_vars=(
        "ZAI_CODING_PLAN_API_KEY",
        "GLM_CODING_PLAN_API_KEY",
        "ZAI_API_KEY",
    ),
    base_url="https://api.z.ai/api/anthropic",
    default_aux_model="glm-4.5-air",
)

register_provider(zai_coding_plan)
