"""Config-gated model cascade routing for the main agent."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from agent.complexity_classifier import ComplexityClassifier

logger = logging.getLogger(__name__)


_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}


def _is_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_VALUES
    return False


def _tier_value(mapping: Any, tier: str) -> str:
    if not isinstance(mapping, dict):
        return ""
    value = mapping.get(tier) or ""
    return str(value).strip() if value is not None else ""


def classify_for_model_cascade(message: str) -> str:
    """Return the cascade tier for a user message."""
    return ComplexityClassifier.classify(message)


def resolve_model_cascade_target(config: Dict[str, Any], message: str) -> Dict[str, str]:
    """Resolve the configured cascade target for ``message``.

    The config shape is intentionally small and stable:

    model_cascade:
      enabled: true
      models:
        nano: gpt-4.1-nano
        mini: gpt-5.4-mini
        full: gpt-5.4
        frontier: gpt-5.5
      providers:
        frontier: openrouter
    """
    tier = classify_for_model_cascade(message)
    if not isinstance(config, dict) or not _is_enabled(config.get("enabled", False)):
        return {"tier": tier, "model": "", "provider": ""}

    return {
        "tier": tier,
        "model": _tier_value(config.get("models"), tier),
        "provider": _tier_value(config.get("providers"), tier),
    }


def maybe_apply_model_cascade(agent: Any, message: str) -> Optional[Dict[str, Any]]:
    """Apply a configured per-turn model cascade switch.

    Returns a decision dict for observability, or None when disabled or no
    target model is configured for the classified tier.
    """
    config = getattr(agent, "_model_cascade_config", {}) or {}
    decision = resolve_model_cascade_target(config, message or "")
    target_model = decision.get("model") or ""
    if not target_model:
        setattr(agent, "_model_cascade_last_decision", decision)
        return None

    current_model = str(getattr(agent, "model", "") or "").strip()
    explicit_provider = decision.get("provider") or ""
    if target_model == current_model and not explicit_provider:
        decision["applied"] = False
        decision["reason"] = "already_on_target_model"
        setattr(agent, "_model_cascade_last_decision", decision)
        return decision

    try:
        from tools.switch_model_tool import switch_model_for_agent

        raw = switch_model_for_agent(
            agent,
            target_model,
            reason=f"model_cascade:{decision['tier']}",
            provider=explicit_provider or None,
        )
        result = json.loads(raw) if isinstance(raw, str) else {}
    except Exception as exc:
        logger.warning("model cascade switch failed: %s", exc)
        decision.update({"applied": False, "error": str(exc)})
        setattr(agent, "_model_cascade_last_decision", decision)
        return decision

    decision["applied"] = bool(result.get("success"))
    if result.get("error"):
        decision["error"] = str(result.get("error"))
    if result.get("new_model"):
        decision["resolved_model"] = str(result.get("new_model"))
    if result.get("provider"):
        decision["resolved_provider"] = str(result.get("provider"))
    setattr(agent, "_model_cascade_last_decision", decision)
    return decision
