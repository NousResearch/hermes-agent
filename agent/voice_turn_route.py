"""Per-turn model route for voice chat: ``auxiliary.voice_chat``.

A voice turn (CLI voice input, a Desktop/TUI voice conversation, a messaging voice note) can run
on a faster model than the session's main one. The route binds at the end of turn setup, after the
system prompt and session row were built for the main model and after preflight compression
judged the history against the main model's window, and is undone before post-turn work (memory
sync, background review) so nothing outside the turn sees it. A turn whose request would not fit
under the voice model's compression threshold at turn start runs on the main model instead, so
switching to voice never compacts the conversation to suit a smaller model.

Surfaces mark the next turn with ``agent._voice_turn_pending = True``; it is consumed here. GPT-Live
delegations are not voice turns in this sense: the live model already owns the voice layer and
delegates real work to the session's model.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TASK = "voice_chat"

# Agent fields the route swap touches outside the ``_primary_runtime``-shaped snapshot. The
# reasoning-rejection flags are facts about one route: the voice model starts without the main
# model's and leaves its own behind (a learned floor is remembered per route instead).
_REJECTION_FIELDS = ("_reasoning_disable_rejected", "_reasoning_floor_required", "_reasoning_effort_rejected")
_EXTRA_FIELDS = (
    "_fallback_activated", "_fallback_index", "_provider_fallback_active", "_provider_fallback_route",
    "_rate_limit_backoff_count", "_credential_pool", "_credential_pool_entry_id", "_config_context_length",
    *_REJECTION_FIELDS,
)


def _route_target(agent: Any, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The entry to bind, or None when the slot follows the main model (``auto`` + no model)."""
    provider = str(cfg.get("provider") or "").strip().lower()
    model = str(cfg.get("model") or "").strip()
    base_url = str(cfg.get("base_url") or "").strip()
    if provider in ("", "auto", "main"):
        if base_url:
            provider = "custom"
        elif model:
            provider = str(getattr(agent, "provider", "") or "").strip().lower()
        else:
            return None
    if not model:
        # Provider pinned without a model: that provider's fast auxiliary model.
        from agent.auxiliary_client import _get_aux_model_for_provider
        model = _get_aux_model_for_provider(provider, prefer_fast=True) or ""
    if not (provider and model):
        return None
    if not base_url and provider == str(agent.provider or "").lower() and model == agent.model:
        return None
    # ``provider: openai`` means the user's OpenAI endpoint, as on every other auxiliary task.
    from hermes_cli.runtime_provider_custom import expand_direct_api_alias
    provider, base_url = expand_direct_api_alias(provider, base_url or None)
    return {**cfg, "provider": provider, "model": model, "base_url": base_url or ""}


def _fits(agent: Any, messages: List[Dict[str, Any]], system_prompt: str) -> bool:
    from agent.model_metadata import estimate_request_tokens_rough
    limit = getattr(getattr(agent, "context_compressor", None), "threshold_tokens", 0) or 0
    if not limit:
        return True
    tokens = estimate_request_tokens_rough(
        messages, system_prompt=system_prompt or "", tools=getattr(agent, "tools", None) or None)
    return tokens < limit


def _lowest_effort(agent: Any) -> Optional[str]:
    """The weakest enabled level the active route accepts when it cannot switch reasoning OFF, else
    None (OFF goes on the wire as configured). Static knowledge only; a route that 400s on the disable
    anyway is learned by turn_recovery and remembered by ``end_voice_turn_route``."""
    from agent.auxiliary_reasoning_floor import REASONING_FLOOR_EFFORT, known_reasoning_floor
    off = {"enabled": False}
    if known_reasoning_floor(off, agent.provider, agent.base_url, agent.model, TASK) is not off:
        return REASONING_FLOOR_EFFORT
    if agent.api_mode == "codex_responses":
        from agent.codex_responses_adapter import classify_responses_route
        from agent.transports.codex import _resolve_reasoning
        route = classify_responses_route(agent)
        # An enabled "none" clamps to "none" where the route has it, else to its weakest level
        # (None: the route takes no reasoning field at all).
        effort, _ = _resolve_reasoning(agent.model, {
            "provider": agent.provider, "base_url": agent.base_url, "is_codex_backend": route.is_codex_backend,
            "is_xai_responses": route.is_xai_responses, "reasoning_config": {"enabled": True, "effort": "none"}})
        return None if effort in (None, "none") else effort
    if agent.api_mode == "anthropic_messages":
        from agent.anthropic_adapter import _MANDATORY_THINKING_CLAUDE_SUBSTRINGS, _model_matches
        # Mandatory-thinking Claude omits the disable and thinks at its default effort.
        return REASONING_FLOOR_EFFORT if _model_matches(agent.model, _MANDATORY_THINKING_CLAUDE_SUBSTRINGS) else None
    return None


def _voice_reasoning(agent: Any, effort: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """``effort`` for the active route: reasoning OFF becomes the lowest valid level where OFF is
    not an option."""
    if not (isinstance(effort, dict) and effort.get("enabled") is False):
        return effort
    lowest = _lowest_effort(agent)
    return {"enabled": True, "effort": lowest} if lowest else effort


def _capture(agent: Any) -> Dict[str, Any]:
    from agent.agent_runtime_helpers import _build_primary_runtime_snapshot
    return {
        "snapshot": _build_primary_runtime_snapshot(agent, agent.api_mode),
        **{name: getattr(agent, name, None) for name in _EXTRA_FIELDS},
    }


def _reinstall(agent: Any, state: Dict[str, Any]) -> None:
    from agent.route_binding import reinstall_runtime_snapshot
    reinstall_runtime_snapshot(agent, state["snapshot"])
    for name in _EXTRA_FIELDS:
        setattr(agent, name, state[name])


def _warn_once(agent: Any, target: Dict[str, Any], reason: str) -> None:
    logger.warning("auxiliary.%s route %s/%s not used: %s; the voice turn runs on %s/%s",
                   TASK, target["provider"], target["model"], reason, agent.provider, agent.model)
    if getattr(agent, "_warned_voice_route", False):
        return
    agent._warned_voice_route = True
    emit = getattr(agent, "_emit_warning", None)
    if callable(emit):
        emit(f"⚠ Voice chat model {target['model']} via {target['provider']} unavailable ({reason}); "
             f"using {agent.model}.")


def begin_voice_turn_route(agent: Any, messages: List[Dict[str, Any]], system_prompt: Any) -> Any:
    """Bind the voice route for a marked turn; returns the system prompt the turn should send."""
    if not getattr(agent, "_voice_turn_pending", False):
        return system_prompt
    agent._voice_turn_pending = False
    from agent.auxiliary_task_config import _get_auxiliary_task_config
    from hermes_constants import parse_reasoning_effort
    cfg = _get_auxiliary_task_config(TASK)
    effort = parse_reasoning_effort(cfg.get("reasoning_effort"))
    target = _route_target(agent, cfg)
    if target is None and effort is None:
        return system_prompt
    state: Dict[str, Any] = {"reasoning_config": copy.deepcopy(getattr(agent, "reasoning_config", None))}
    agent._voice_route_state = state
    if target is not None:
        state.update(_capture(agent))
        reason = ""
        try:
            from agent.route_binding import bind_route_entry
            if bind_route_entry(agent, target, target["provider"], target["model"]) is None:
                reason = "provider not configured"
        except Exception as exc:  # health: allow BLE001 -- any bind failure falls back to the main model with a notice
            reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        if not reason and not _fits(agent, messages, system_prompt):
            reason = "the conversation is larger than its context window"
        if reason:
            _reinstall(agent, state)
            _warn_once(agent, target, reason)
            agent._voice_route_state = {"reasoning_config": state["reasoning_config"]}
            return agent._cached_system_prompt or system_prompt
        from agent.chat_completion_helpers import rewrite_prompt_model_identity
        rewrite_prompt_model_identity(agent, agent.model, agent.provider)
        for name in _REJECTION_FIELDS:
            setattr(agent, name, False)
        agent._turn_route_task = TASK
        logger.info("Voice turn routed to %s (%s)", agent.model, agent.provider)
    if effort is not None:
        agent.reasoning_config = _voice_reasoning(agent, effort)
    return agent._cached_system_prompt or system_prompt


def end_voice_turn_route(agent: Any) -> None:
    """Undo ``begin_voice_turn_route`` (idempotent; every turn exit calls it)."""
    state = getattr(agent, "_voice_route_state", None)
    if not state:
        return
    agent._voice_route_state = None
    agent._turn_route_task = ""
    try:
        if "snapshot" in state:
            if getattr(agent, "_reasoning_floor_required", False):
                # The voice model 400'd on the disable this turn: start the next voice turn at the floor.
                from agent.auxiliary_reasoning_floor import remember_reasoning_floor
                remember_reasoning_floor(agent.provider, agent.base_url, {"model": agent.model}, None)
            _reinstall(agent, state)
    except Exception:
        logger.warning("Voice turn route restore failed; the next turn re-resolves the main runtime",
                       exc_info=True)
        agent._fallback_activated = True  # restore_primary_runtime takes it from here
    agent.reasoning_config = state["reasoning_config"]
