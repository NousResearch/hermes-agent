"""Upstage Solar provider profile: top-level ``reasoning_effort`` (low|medium|high).

Solar's server default is ``minimal`` (reasoning off) — wrong for agentic work —
so an unset reasoning_config defaults reasoning ON at ``medium``, matching the
"medium (default)" the /reasoning panel shows. Explicit settings always win.
"""

from typing import Any

from agent.reasoning_effort import EFFORT_LADDER, SOLAR_EFFORTS, clamp_effort
from providers import register_provider
from providers.base import ProviderProfile

# Deny-list on purpose: new Solar models are assumed reasoning-capable; only these known
# non-reasoning families ignore reasoning_effort. Substring match covers dated variants.
_NON_REASONING_MODEL_MARKERS = ("solar-mini", "syn-pro")


class UpstageProfile(ProviderProfile):
    """Upstage Solar — top-level ``reasoning_effort`` control (no reasoning_content echo needed)."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        m = (model or "").strip().lower()
        if any(marker in m for marker in _NON_REASONING_MODEL_MARKERS):
            return {}, {}
        if not reasoning_config or not isinstance(reasoning_config, dict):
            return {}, {"reasoning_effort": "medium"}  # unset -> reasoning ON for agents
        if reasoning_config.get("enabled") is False:
            return {}, {}  # explicitly disabled -> Solar's own default (minimal = off)
        # Map Hermes' effort vocabulary onto Solar's accepted set via the shared clamp
        # (agent.reasoning_effort). minimal → omit (Solar's minimal means off); unknown-but-enabled bespoke
        # levels collapse to high rather than silently downgrading (#62650 precedent).
        effort = (reasoning_config.get("effort") or "").strip().lower()
        if not effort:
            return {}, {"reasoning_effort": "medium"}
        if effort == "minimal":
            return {}, {}
        mapped = clamp_effort(effort, SOLAR_EFFORTS)
        if mapped not in SOLAR_EFFORTS:
            # Bespoke level outside the ladder runs at full strength rather than quietly
            # falling to the default; ladder levels that still don't map are omitted.
            mapped = "high" if effort not in EFFORT_LADDER else None
        return {}, {"reasoning_effort": mapped} if mapped else {}


upstage = UpstageProfile(
    name="upstage", aliases=("solar",), display_name="Upstage Solar", description="Upstage (Solar API)",
    signup_url="https://console.upstage.ai/api-keys", env_vars=("UPSTAGE_API_KEY", "UPSTAGE_BASE_URL"),
    base_url="https://api.upstage.ai/v1", auth_type="api_key",
    # No default_aux_model: auxiliary tasks use the main model. [0] is the setup default.
    fallback_models=("solar-pro3",),
)

register_provider(upstage)
