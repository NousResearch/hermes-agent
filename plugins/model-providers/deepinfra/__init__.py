"""DeepInfra provider profile (chat surface; image-gen/TTS/STT are wired via
their own plugin subsystems)."""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# DeepInfra validates ``reasoning_effort`` against one gateway-wide enum (HTTP
# 422 otherwise, even on non-reasoning models), so the set can't drift per model.
# Equals ``hermes_constants.VALID_REASONING_EFFORTS`` minus ``ultra``.
_DEEPINFRA_REASONING_EFFORTS = frozenset(
    {"minimal", "low", "medium", "high", "xhigh", "max"}
)

# ``ultra`` is the only Hermes effort DeepInfra rejects; degrade instead of
# 422-ing the turn. ``xhigh`` is native — don't fold it into ``max``.
_ULTRA_FALLBACK = "max"

# Default thinking mode is per-model (DeepSeek-V4.x off; GLM-4.6 and
# Qwen3-Thinking on), so an explicit off must send "none" — omitting the
# field is not neutral.
_REASONING_OFF = "none"


class _DeepInfraProfile(ProviderProfile):
    """DeepInfra profile with live vision-default discovery, so shared vision
    resolution in ``agent/auxiliary_client.py`` stays provider-agnostic."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Emit top-level ``reasoning_effort`` for DeepInfra chat models.

        Not gated on ``supports_reasoning``: the ``run_agent.py`` allowlist has
        no DeepInfra branch, so the transport always passes False here and
        honouring it would make this a permanent no-op. Not gated on the
        catalog ``reasoning`` tag either: every DeepInfra chat model accepts
        the field, and some untagged ones (DeepSeek-V4-Pro) still reason.
        """
        if not isinstance(reasoning_config, dict):
            # Nothing requested — keep the per-model default.
            return {}, {}

        if reasoning_config.get("enabled") is False:
            return {}, {"reasoning_effort": _REASONING_OFF}

        # ``reasoning_overrides`` can bypass ``parse_reasoning_effort``; normalize here.
        effort = (reasoning_config.get("effort") or "").strip().lower()

        if not effort:
            # Enabled but unspecified — let the model default.
            return {}, {}
        if effort in {_REASONING_OFF, "false", "disabled"}:
            return {}, {"reasoning_effort": _REASONING_OFF}
        if effort == "ultra":
            return {}, {"reasoning_effort": _ULTRA_FALLBACK}
        if effort in _DEEPINFRA_REASONING_EFFORTS:
            return {}, {"reasoning_effort": effort}

        # Unknown value: omit rather than 422 the whole turn.
        return {}, {}

    def default_vision_model(self):  # type: ignore[override]
        """First vision-capable *chat* model from the live catalog, or None. Key-gated so a box
        without DEEPINFRA_API_KEY never pays the round-trip; requires the ``chat`` surface tag so
        an image-gen model carrying a ``vision`` tag can't be picked as a chat vision backend."""
        from agent.secret_scope import get_secret

        if not (get_secret("DEEPINFRA_API_KEY") or "").strip():
            return None
        try:
            from hermes_cli.models import _fetch_deepinfra_models_by_tag
            items = _fetch_deepinfra_models_by_tag("chat")
        except Exception:
            return None
        for item in items or []:
            metadata = item.get("metadata") or {}
            tags = metadata.get("tags") if isinstance(metadata, dict) else None
            if isinstance(tags, list) and "vision" in tags and item.get("id"):
                return item["id"]
        return None


deepinfra = _DeepInfraProfile(
    name="deepinfra", aliases=("deep-infra", "deepinfra-ai"), display_name="DeepInfra",
    description="DeepInfra — 100+ open models, pay-per-use", signup_url="https://deepinfra.com/dash/api_keys",
    env_vars=("DEEPINFRA_API_KEY", "DEEPINFRA_BASE_URL"), base_url="https://api.deepinfra.com/v1/openai",
    auth_type="api_key",
    default_max_tokens=None,  # DeepInfra applies its documented per-model limit
    # The only hardcoded DeepInfra model: aux resolution is synchronous, so it
    # can't wait on a catalog round-trip. Everything else is discovered live.
    default_aux_model="deepseek-ai/DeepSeek-V4-Flash",
    # Empty on purpose: the live catalog is the source of truth; an empty picker
    # beats silently routing to a retired model.
    fallback_models=(),
)

register_provider(deepinfra)
