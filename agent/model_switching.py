"""Adaptive model switching between tool iterations (opt-in).

The initial model (``model.default: auto``) is chosen by the semantic router. During a long
tool task the loop can escalate to a harder configured model when a generation keeps failing
(repeated tool failures — terminal/test exit codes, JSON ``error``/``success: false`` results),
and descend back to the cheaper model only at a safe boundary (a run of clean rounds with no
pending failure). The ladder is declared under ``agent.model_switching`` in config.yaml and is
OFF until ``escalate_to`` is set.

Switches are scheduled only between generations: the decision is made after a tool round has
fully completed (``observe_tool_round``), and applied at the start of the NEXT iteration with
nothing in flight (``apply_pending_model_switch``), mirroring ``agent/nous_wire.py``. A
generation is therefore never switched mid-stream, and the transcript is never mutated.

The adaptive path is intentionally limited to Hermes's configured custom Manifest endpoint. It
changes only ``agent.model``, which becomes the next request's model parameter. It must not call
``switch_model``: that full runtime operation invalidates the cached system prompt and persists
the destination in ``_primary_runtime``. Other provider modes are rejected safely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_ESCALATE_AFTER_FAILURES = 3
DEFAULT_DESCEND_AFTER_SUCCESSES = 2
DEFAULT_MAX_SWITCHES = 3

_CONFIG_KEY = "agent.model_switching"


@dataclass(frozen=True)
class ModelSwitchingSettings:
    """The opt-in ladder. ``enabled`` is False until ``escalate_to`` is set."""

    escalate_to: Optional[str] = None
    descend_to: Optional[str] = None
    escalate_after_failures: int = DEFAULT_ESCALATE_AFTER_FAILURES
    descend_after_successes: int = DEFAULT_DESCEND_AFTER_SUCCESSES
    max_switches: int = DEFAULT_MAX_SWITCHES

    @property
    def enabled(self) -> bool:
        return bool(self.escalate_to)


@dataclass
class SwitchState:
    """Per-turn escalation state; reset at the start of every user turn."""

    consecutive_failures: int = 0
    consecutive_successes: int = 0
    escalated: bool = False
    switches: int = 0


def _warn_invalid(key: str, raw: Any, default: Any) -> None:
    logger.warning(
        "Invalid %s in config.yaml: %r — falling back to default %r.", key, raw, default
    )


def _clean_model_id(raw: Any, key: str) -> Optional[str]:
    """A model id must be a non-empty string; anything else disables that rung."""
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        _warn_invalid(key, raw, None)
        return None
    return raw.strip()


def _resolve_positive_int(raw: Any, *, default: int, key: str) -> int:
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        value = -1
    if value <= 0:
        _warn_invalid(key, raw, default)
        return default
    return value


def resolve_model_switching_settings(
    config: Optional[Dict[str, Any]] = None,
) -> ModelSwitchingSettings:
    """Resolve the ``agent.model_switching`` ladder. Invalid values warn and fall back;
    never raises."""
    agent_cfg = config.get("agent") if isinstance(config, dict) else None
    raw_section = agent_cfg.get("model_switching") if isinstance(agent_cfg, dict) else None
    section: Dict[str, Any] = raw_section if isinstance(raw_section, dict) else {}
    if raw_section is not None and not isinstance(raw_section, dict):
        _warn_invalid(_CONFIG_KEY, raw_section, None)

    escalate_to = _clean_model_id(section.get("escalate_to"), f"{_CONFIG_KEY}.escalate_to")
    descend_to = _clean_model_id(section.get("descend_to"), f"{_CONFIG_KEY}.descend_to")
    return ModelSwitchingSettings(
        escalate_to=escalate_to,
        descend_to=descend_to,
        escalate_after_failures=_resolve_positive_int(
            section.get("escalate_after_failures", DEFAULT_ESCALATE_AFTER_FAILURES),
            default=DEFAULT_ESCALATE_AFTER_FAILURES, key=f"{_CONFIG_KEY}.escalate_after_failures",
        ),
        descend_after_successes=_resolve_positive_int(
            section.get("descend_after_successes", DEFAULT_DESCEND_AFTER_SUCCESSES),
            default=DEFAULT_DESCEND_AFTER_SUCCESSES, key=f"{_CONFIG_KEY}.descend_after_successes",
        ),
        max_switches=_resolve_positive_int(
            section.get("max_switches", DEFAULT_MAX_SWITCHES),
            default=DEFAULT_MAX_SWITCHES, key=f"{_CONFIG_KEY}.max_switches",
        ),
    )


def decide_next_model(
    settings: ModelSwitchingSettings,
    state: SwitchState,
    *,
    current_model: Optional[str],
    round_failed: bool,
) -> Tuple[Optional[str], SwitchState]:
    """Pure escalation/descent core.

    Returns ``(target_model, new_state)`` where ``target_model`` is the model id to switch
    to, or ``None`` to hold. The caller must apply the switch only at a generation boundary.
    """
    if not settings.enabled:
        return None, state

    new = replace(state)
    current = (current_model or "").strip()

    if round_failed:
        new.consecutive_failures += 1
        new.consecutive_successes = 0
    else:
        new.consecutive_failures = 0
        new.consecutive_successes += 1

    target: Optional[str] = None
    if (
        new.consecutive_failures >= settings.escalate_after_failures
        and not new.escalated
        and settings.escalate_to
        and settings.escalate_to != current
        and new.switches < settings.max_switches
    ):
        target = settings.escalate_to
        new.escalated = True
        new.switches += 1
        new.consecutive_failures = 0
        new.consecutive_successes = 0
    elif (
        new.escalated
        and settings.descend_to
        and new.consecutive_successes >= settings.descend_after_successes
        and settings.descend_to != current
        and new.switches < settings.max_switches
    ):
        target = settings.descend_to
        new.escalated = False
        new.switches += 1
        new.consecutive_successes = 0
        new.consecutive_failures = 0

    return target, new


def _settings_for(agent: Any) -> ModelSwitchingSettings:
    """Resolve the ladder once per agent and cache it (``None`` sentinel → resolved)."""
    settings = getattr(agent, "_model_switching_settings", None)
    if settings is not None:
        return settings
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly() or {}
    except Exception:
        config = {}
    settings = resolve_model_switching_settings(config)
    try:
        agent._model_switching_settings = settings
    except Exception:
        pass
    return settings


def reset_model_switching_turn(agent: Any) -> None:
    """Restore the semantic router for a new user turn and clear per-turn streaks.

    Escalation is deliberately temporary. Without this restoration, an escalated explicit model
    would bypass the semantic router on every later user request when ``descend_to`` is omitted.
    """
    original = getattr(agent, "_adaptive_model_switch_original_model", None)
    if original is None:
        original = getattr(agent, "model", None)
        agent._adaptive_model_switch_original_model = original
    else:
        agent.model = original
    agent._model_switch_state = None
    agent._adaptive_model_switch_pending = None


def _newest_tool_round(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The contiguous ``role: "tool"`` tail — exactly the just-executed round's results."""
    results: List[Dict[str, Any]] = []
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            results.append(msg)
        else:
            break
    return results


def _round_failed(round_results: List[Dict[str, Any]]) -> bool:
    """A round failed if ANY of its tool results carries a canonical failure signal."""
    from agent.display import _detect_tool_failure

    for msg in round_results:
        name = str(msg.get("name") or msg.get("tool_name") or "")
        content = msg.get("content")
        if isinstance(content, list):  # multimodal blocks — treat as non-failure
            continue
        is_failure, _ = _detect_tool_failure(name, content)
        if is_failure:
            return True
    return False


def observe_tool_round(agent: Any, messages: List[Dict[str, Any]]) -> None:
    """After a tool round completes, classify its results and schedule a switch if the
    escalation/descent ladder demands one. Only *schedules* — nothing is applied here."""
    settings = _settings_for(agent)
    if not settings.enabled:
        return

    state = getattr(agent, "_model_switch_state", None)
    if not isinstance(state, SwitchState):
        state = SwitchState()

    target, new_state = decide_next_model(
        settings, state, current_model=getattr(agent, "model", None),
        round_failed=_round_failed(_newest_tool_round(messages)),
    )
    agent._model_switch_state = new_state
    if target:
        agent._adaptive_model_switch_pending = target


def apply_pending_model_switch(agent: Any) -> bool:
    """Apply a scheduled model parameter change at an iteration boundary.

    This narrow path is valid only for the custom OpenAI-compatible endpoint, where every
    candidate is exposed behind one transport. It deliberately leaves the client, cached prompt,
    compressor and primary runtime untouched.
    """
    pending = getattr(agent, "_adaptive_model_switch_pending", None)
    if not pending:
        return False
    agent._adaptive_model_switch_pending = None

    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    api_mode = str(getattr(agent, "api_mode", "") or "").strip().lower()
    if provider not in {"custom", "openai-compatible"} or api_mode != "chat_completions":
        logger.warning(
            "adaptive model switching requires a custom endpoint in chat_completions mode; "
            "staying on %s",
            getattr(agent, "model", None),
        )
        return False

    agent.model = pending
    logger.info(
        "adaptive model request parameter -> %s (session=%s)",
        pending, getattr(agent, "session_id", None) or "?",
    )
    return True
