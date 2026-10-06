"""Reasoning effort resolution and notice helpers for model fallback routes."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def reresolve_fallback_reasoning_config(agent: Any, fallback_entry: Optional[Dict[str, Any]] = None) -> None:
    """Resolve reasoning_config for a fallback model.

    Precedence:
    1. Explicit reasoning_effort in fallback_entry or criteria (e.g. "default", "none", "low").
       "default" / "auto" sets reasoning_config to None so the model's native default applies.
    2. Per-model override from config.yaml (agent.reasoning_overrides) for this specific fallback model.
    3. Provider/model native default (None): we do not force a primary model's global reasoning_effort
       (e.g. "high") onto an arbitrary or lightweight fallback model.
    """
    # Reset rejection flags on model swap so the new fallback model starts with a clean slate
    agent._reasoning_effort_rejected = False
    agent._reasoning_disable_rejected = False
    agent._reasoning_floor_required = False

    try:
        from hermes_cli.config import load_config
        from hermes_constants import parse_reasoning_effort, resolve_per_model_reasoning_effort

        fb = fallback_entry or {}
        criteria = fb.get("criteria") if isinstance(fb.get("criteria"), dict) else {}
        entry_effort = (
            fb.get("reasoning_effort")
            or fb.get("thinking_effort")
            or fb.get("thinking")
            or criteria.get("reasoning_effort")
        )
        if entry_effort is not None:
            val_str = str(entry_effort).strip().lower()
            if val_str in ("default", "auto", "native"):
                agent.reasoning_config = None
                logger.info("Fallback %s: reasoning_config set to native default (None)", agent.model)
                return
            parsed = parse_reasoning_effort(entry_effort)
            agent.reasoning_config = parsed
            logger.info("Fallback %s: reasoning_config from fallback entry: %s", agent.model, agent.reasoning_config)
            return

        cfg = load_config() or {}
        agent_cfg = cfg.get("agent") if isinstance(cfg.get("agent"), dict) else {}
        overrides = agent_cfg.get("reasoning_overrides") or {}
        per_model = resolve_per_model_reasoning_effort(agent.model, overrides)
        if per_model is not None:
            agent.reasoning_config = per_model
            logger.info("Fallback %s: reasoning_config from per-model override: %s", agent.model, per_model)
            return

        agent.reasoning_config = None
        logger.info("Fallback %s: reasoning_config defaulted to native default (None)", agent.model)
    except Exception as _reasoning_err:
        logger.debug("Failed to resolve reasoning_config for fallback %s; keeping current: %s", agent.model, _reasoning_err, exc_info=True)


def format_fallback_label(entry: Dict[str, Any]) -> str:
    """Format a fallback chain entry for user-facing startup logs."""
    provider = entry.get("provider") or "?"
    model = entry.get("model") or "?"
    lbl = f"{model} ({provider})"
    matched = entry.get("criteria_matched")
    rank = entry.get("criteria_rank")
    if matched and rank:
        lbl += f" [#{rank} {matched}]"
    elif matched:
        lbl += f" [{matched}]"
    return lbl


def format_fallback_notice(
    old_model: str,
    old_provider: str,
    fb_model: str,
    fb_provider: str,
    reason: Any,
    fb: Dict[str, Any],
    cooldown_seconds: Optional[int] = None,
    remaining_seconds: int = 0,
) -> str:
    """Format user-visible one-shot notice when falling back during a turn."""
    from agent.chat_completion_helpers import _fallback_reason_text
    reason_str = _fallback_reason_text(reason)
    criteria_info = ""
    if fb.get("criteria_matched"):
        rank_str = f" #{fb['criteria_rank']}" if fb.get("criteria_rank") else ""
        criteria_info = f" [matched heuristic: {fb['criteria_matched']}{rank_str}]"
    notice = f"⚠️ Model fallback: {old_model} via {old_provider} unavailable ({reason_str}); using {fb_model} via {fb_provider}{criteria_info}."
    if cooldown_seconds is not None:
        notice += f" Primary retry eligible in ~{remaining_seconds} s; recovery is not guaranteed."
    return notice
