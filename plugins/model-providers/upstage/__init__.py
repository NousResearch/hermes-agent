"""Upstage Solar provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


# Model-name markers for Solar families that do NOT accept ``reasoning_effort``.
# Deny-list on purpose: newly released Solar models are assumed
# reasoning-capable by default, so only the known non-reasoning families are
# listed here. Substring match (not startswith) so dated variants like
# ``solar-mini-250127`` are covered too.
_NON_REASONING_MODEL_MARKERS = ("solar-mini", "syn-pro")

# When the user hasn't picked a reasoning effort, Hermes passes
# reasoning_config=None. Solar's own server default is "minimal" (reasoning
# off), which is the wrong default for an agentic workload. We default reasoning
# ON at this effort — matching the "medium (default)" that Hermes' /reasoning
# panel shows for an unset config, so the displayed default and the real wire
# value agree. An explicit saved setting or a `/reasoning <level>` change is
# always honored over this default; `/reasoning none` disables it.
_DEFAULT_REASONING_EFFORT = "medium"


def _model_supports_reasoning(model: str | None) -> bool:
    """Solar reasoning-capable models — True unless the model is deny-listed.

    The Solar Pro family (``solar-pro``, ``solar-pro2``, ``solar-pro3`` and
    dated variants like ``solar-pro3-250127``) and the Solar Open family
    (``solar-open*``) accept ``reasoning_effort``; only ``solar-mini`` /
    ``syn-pro`` ignore the parameter, so we deny-list those and treat every
    other (incl. future) Solar model as reasoning-capable.

    ``None``/empty model → True: the provider default (``fallback_models[0]``,
    ``solar-pro3``) is reasoning-capable, so an unset model gets the same
    default-on behaviour.
    """
    m = (model or "").strip().lower()
    return not any(marker in m for marker in _NON_REASONING_MODEL_MARKERS)


class UpstageProfile(ProviderProfile):
    """Upstage Solar — top-level ``reasoning_effort`` control.

    Solar Pro/Open expose reasoning through a top-level ``reasoning_effort``
    field (``minimal`` | ``low`` | ``medium`` | ``high``), mirroring OpenAI's
    shape. Unlike DeepSeek/Kimi it does NOT require echoing ``reasoning_content``
    back on later turns, so only the request field needs wiring. We emit at most
    ``low`` | ``medium`` | ``high`` — the explicit values both Solar Pro 2 and
    Pro 3 accept.

    Default-on: Solar's own server default is ``minimal`` (off), but for an
    agentic workload we default reasoning ON (``_DEFAULT_REASONING_EFFORT``)
    when the user hasn't picked an effort. The user can still set any level or
    turn it off with ``/reasoning none``.
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        top_level: dict[str, Any] = {}

        # solar-mini / syn-pro (the deny-list) ignore reasoning_effort — send
        # nothing. Everything else, including future Solar models, gets it.
        if not _model_supports_reasoning(model):
            return {}, top_level

        # Unset (reasoning_config is None) → default reasoning ON for agents.
        if not reasoning_config or not isinstance(reasoning_config, dict):
            return {}, {"reasoning_effort": _DEFAULT_REASONING_EFFORT}

        # Explicitly disabled (`/reasoning none`) → omit the field so Solar
        # applies its own default (minimal = off).
        if reasoning_config.get("enabled") is False:
            return {}, top_level

        # Map Hermes' effort vocabulary onto Solar's accepted set. xhigh/max/
        # ultra collapse to high (Solar's strongest). minimal → off (omit). An
        # enabled request with no recognised effort uses the default effort.
        effort = (reasoning_config.get("effort") or "").strip().lower()
        mapped = {
            "minimal": None,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
            "max": "high",
            "ultra": "high",
        }.get(effort, _DEFAULT_REASONING_EFFORT)

        if mapped:
            top_level["reasoning_effort"] = mapped
        return {}, top_level


upstage = UpstageProfile(
    name="upstage",
    aliases=("solar",),
    display_name="Upstage Solar",
    description="Upstage (Solar API)",
    signup_url="https://console.upstage.ai/api-keys",
    env_vars=("UPSTAGE_API_KEY", "UPSTAGE_BASE_URL"),
    base_url="https://api.upstage.ai/v1",
    auth_type="api_key",
    # default_aux_model left empty → auxiliary side tasks use the main model.
    # entry [0] is the setup default — solar-pro3, the current Solar Pro flagship.
    fallback_models=(
        "solar-pro3",
    ),
)

register_provider(upstage)
