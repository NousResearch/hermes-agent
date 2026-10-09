"""Baseten Model APIs provider profile.

Addresses the shared OpenAI-compatible endpoint at ``inference.baseten.co``,
whose catalog is exclusively ``vendor/Model`` ids. Dedicated deployments live
on per-model hosts (``model-<id>.api.baseten.co/environments/<env>/sync/v1``)
and are reached through a ``BASETEN_BASE_URL`` / ``model.base_url`` override
rather than this profile's catalog.
"""

from typing import Any

from agent.reasoning_effort import BASETEN_EFFORTS, clamp_effort
from hermes_cli.version_info import get_version_info
from providers import register_provider
from providers.base import ProviderProfile

# Reasoning-capable families in the Model APIs catalog. The transport only sets
# ``supports_reasoning`` when model metadata carries the flag, so — as on Nebius —
# a marker list keeps effort reaching models the registry hasn't annotated yet.
_REASONING_MARKERS = (
    "deepseek", "glm-", "kimi-k", "minimax-m", "nemotron", "gpt-oss", "inkling",
)


class BasetenProfile(ProviderProfile):
    """Map Hermes reasoning controls onto Baseten's OpenAI-compatible wire."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None,
        supports_reasoning: bool = False, **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model_name = (model or "").strip().rsplit("/", 1)[-1].lower()
        if not supports_reasoning and not any(m in model_name for m in _REASONING_MARKERS):
            return {}, {}
        rc = reasoning_config if isinstance(reasoning_config, dict) else {}
        effort = str(rc.get("effort") or "").strip().lower()
        # Unset leaves the route's own default in charge — a bare request is accepted everywhere.
        if not effort and rc.get("enabled", True) is not False:
            return {}, {}
        # "Off" is a request for as little thinking as the catalog can portably give. Baseten
        # documents a ``none`` level, but it 400s on zai-org/GLM-5.3-Fast and, where it is
        # accepted, yields the same reasoning-token count as ``minimal`` — so it buys nothing
        # and costs a hard failure on one route. Clamp the whole off-vocabulary onto ``minimal``.
        if rc.get("enabled", True) is False or effort in {"none", "off", "disabled"}:
            return {}, {"reasoning_effort": "minimal"}
        # Canonical clamp: nearest weaker supported level, never escalate.
        clamped = clamp_effort(effort, BASETEN_EFFORTS)
        return {}, ({"reasoning_effort": clamped} if clamped else {})


baseten = BasetenProfile(
    name="baseten", aliases=("baseten-ai", "basetenai", "baseten-model-apis", "bt"),
    display_name="Baseten", description="Baseten Model APIs — OpenAI-compatible direct model API",
    signup_url="https://app.baseten.co/settings/api-keys",
    env_vars=("BASETEN_API_KEY", "BASETEN_BASE_URL"),
    base_url="https://inference.baseten.co/v1", auth_type="api_key",
    # Attribution headers (canonical Hermes set); via default_headers so they
    # survive switch_model and credential rotation.
    default_headers={
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Hermes Agent",
        "User-Agent": f"HermesAgent/{get_version_info().base_version}",
    },
    default_aux_model="zai-org/GLM-5.3-Flash",
    # Picker safety net when the live catalog fetch fails. Every id here was verified reachable
    # against the live /models endpoint — models.dev lists a wider catalog than the API actually
    # serves, and because this curated list leads the picker, an id the route cannot serve would
    # be the first thing a user picks and the first to fail.
    fallback_models=(
        "zai-org/GLM-5.3", "moonshotai/Kimi-K3", "deepseek-ai/DeepSeek-V4-Pro-0813",
        "zai-org/GLM-5.2", "zai-org/GLM-5.3-Fast", "deepseek-ai/DeepSeek-V4.1-Flash",
        "deepseek-ai/DeepSeek-V4-Flash-0731", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
        "openai/gpt-oss-120b", "zai-org/GLM-5.3-Flash",
    ),
)

register_provider(baseten)
