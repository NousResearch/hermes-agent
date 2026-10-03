"""Session budget guardrail for additive runaway-cost protection.

The guardrail observes real provider usage after a successful response and
latches a pause decision before the next provider call. It is intentionally
config-driven and additive: when disabled or absent, behavior is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


DEFAULT_SESSION_BUDGET_GUARDRAIL = {
    "enabled": False,
    "soft_prompt_tokens": 180_000,
    "hard_prompt_tokens": 240_000,
    "hard_consecutive_soft_hits": 3,
    "hard_projected_cost_usd": 25.0,
    "pause_and_ask": True,
}

CHOICE_CONTINUE_ONCE = "Continue once"
CHOICE_COMPRESS_THEN_CONTINUE = "Compress then continue"
CHOICE_STOP = "Stop"
CHOICES = [CHOICE_CONTINUE_ONCE, CHOICE_COMPRESS_THEN_CONTINUE, CHOICE_STOP]


@dataclass
class SessionBudgetGuardrailState:
    consecutive_soft_hits: int = 0
    last_prompt_tokens: int = 0
    last_projected_cost_usd: float = 0.0
    pending_hard_breach: bool = False
    reasons: list[str] = field(default_factory=list)
    fired: bool = False
    continue_once_armed: bool = False


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _as_positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return normalized guardrail config from a Hermes config dict."""
    agent_cfg = (config or {}).get("agent")
    if not isinstance(agent_cfg, dict):
        agent_cfg = {}
    raw = agent_cfg.get("session_budget_guardrail")
    if not isinstance(raw, dict):
        raw = {}
    merged = dict(DEFAULT_SESSION_BUDGET_GUARDRAIL)
    merged.update(raw)
    return {
        "enabled": _as_bool(merged.get("enabled"), False),
        "soft_prompt_tokens": _as_positive_int(
            merged.get("soft_prompt_tokens"),
            DEFAULT_SESSION_BUDGET_GUARDRAIL["soft_prompt_tokens"],
        ),
        "hard_prompt_tokens": _as_positive_int(
            merged.get("hard_prompt_tokens"),
            DEFAULT_SESSION_BUDGET_GUARDRAIL["hard_prompt_tokens"],
        ),
        "hard_consecutive_soft_hits": _as_positive_int(
            merged.get("hard_consecutive_soft_hits"),
            DEFAULT_SESSION_BUDGET_GUARDRAIL["hard_consecutive_soft_hits"],
        ),
        "hard_projected_cost_usd": _as_positive_float(
            merged.get("hard_projected_cost_usd"),
            DEFAULT_SESSION_BUDGET_GUARDRAIL["hard_projected_cost_usd"],
        ),
        "pause_and_ask": _as_bool(merged.get("pause_and_ask"), True),
    }


def reset_state(agent: Any) -> SessionBudgetGuardrailState:
    state = SessionBudgetGuardrailState()
    agent.session_budget_guardrail_state = state
    return state


def initialize_agent(agent: Any, config: dict[str, Any] | None) -> None:
    agent.session_budget_guardrail_config = resolve_config(config)
    reset_state(agent)


def _reason_join(reasons: Iterable[str]) -> str:
    return "; ".join(str(r) for r in reasons if r)


def record_usage(agent: Any, *, prompt_tokens: int, projected_cost_usd: float | None = None) -> SessionBudgetGuardrailState:
    """Observe one successful provider call and latch if thresholds are breached."""
    cfg = getattr(agent, "session_budget_guardrail_config", None) or {}
    state = getattr(agent, "session_budget_guardrail_state", None)
    if state is None:
        state = reset_state(agent)
    if not cfg.get("enabled"):
        return state

    prompt_tokens = int(prompt_tokens or 0)
    projected_cost = float(projected_cost_usd or 0.0)
    state.last_prompt_tokens = prompt_tokens
    state.last_projected_cost_usd = projected_cost

    reasons: list[str] = []
    soft_threshold = int(cfg.get("soft_prompt_tokens") or 0)
    hard_threshold = int(cfg.get("hard_prompt_tokens") or 0)
    soft_limit = int(cfg.get("hard_consecutive_soft_hits") or 0)
    hard_cost = float(cfg.get("hard_projected_cost_usd") or 0.0)

    if soft_threshold > 0 and prompt_tokens >= soft_threshold:
        state.consecutive_soft_hits += 1
    else:
        state.consecutive_soft_hits = 0

    if hard_threshold > 0 and prompt_tokens >= hard_threshold:
        reasons.append(f"prompt_tokens {prompt_tokens:,} >= hard_prompt_tokens {hard_threshold:,}")
    if soft_limit > 0 and state.consecutive_soft_hits >= soft_limit:
        reasons.append(
            f"consecutive soft prompt-token hits {state.consecutive_soft_hits} >= {soft_limit}"
        )
    if hard_cost > 0 and projected_cost >= hard_cost:
        reasons.append(
            f"projected session cost ${projected_cost:.4f} >= ${hard_cost:.4f}"
        )

    if reasons:
        state.pending_hard_breach = True
        state.fired = True
        state.reasons = reasons
    return state


def has_pending_breach(agent: Any) -> bool:
    cfg = getattr(agent, "session_budget_guardrail_config", None) or {}
    state = getattr(agent, "session_budget_guardrail_state", None)
    return bool(cfg.get("enabled") and state and state.pending_hard_breach)


def build_pause_question(agent: Any) -> str:
    state = getattr(agent, "session_budget_guardrail_state", None) or SessionBudgetGuardrailState()
    reason_text = _reason_join(getattr(state, "reasons", []) or ["session budget threshold breached"])
    return (
        "Session budget guardrail reached before the next provider call.\n\n"
        f"Reason: {reason_text}.\n"
        f"Last prompt tokens: {int(getattr(state, 'last_prompt_tokens', 0) or 0):,}.\n"
        f"Estimated session cost: ${float(getattr(state, 'last_projected_cost_usd', 0.0) or 0.0):.4f}.\n\n"
        "Choose how to proceed."
    )


def resolve_choice(raw_choice: Any) -> str:
    if isinstance(raw_choice, (list, tuple)) and raw_choice:
        raw_choice = raw_choice[0]
    text = str(raw_choice or "").strip().lower()
    if text in {"continue once", "continue", "once", "1"}:
        return CHOICE_CONTINUE_ONCE
    if text in {"compress then continue", "compress", "compact", "2"}:
        return CHOICE_COMPRESS_THEN_CONTINUE
    if text in {"stop", "halt", "cancel", "3"}:
        return CHOICE_STOP
    return CHOICE_STOP


def ask_for_decision(agent: Any) -> str:
    """Ask the platform/user callback. Missing callback fails safe to Stop."""
    cfg = getattr(agent, "session_budget_guardrail_config", None) or {}
    if not cfg.get("pause_and_ask", True):
        return CHOICE_STOP
    cb = getattr(agent, "clarify_callback", None)
    if not callable(cb):
        return CHOICE_STOP
    try:
        return resolve_choice(cb(build_pause_question(agent), CHOICES, multi_select=False))
    except TypeError:
        try:
            return resolve_choice(cb(build_pause_question(agent), CHOICES))
        except Exception:
            return CHOICE_STOP
    except Exception:
        return CHOICE_STOP


def clear_pending_for_continue_once(agent: Any) -> None:
    state = getattr(agent, "session_budget_guardrail_state", None)
    if state is None:
        state = reset_state(agent)
    state.pending_hard_breach = False
    state.continue_once_armed = True
    state.reasons = []


def clear_pending_after_compress(agent: Any) -> None:
    state = getattr(agent, "session_budget_guardrail_state", None)
    if state is None:
        state = reset_state(agent)
    state.pending_hard_breach = False
    state.continue_once_armed = False
    state.consecutive_soft_hits = 0
    state.reasons = []


def _try_compress_context(agent: Any, *, messages: Any = None, conversation_history: Any = None) -> bool:
    """Best-effort compression hook used only after an explicit guardrail choice.

    Different Hermes surfaces expose compression through slightly different
    methods. Treat unknown or deferred compression as not-compressed so the
    hard-breach latch remains armed.
    """
    compressor = getattr(agent, "context_compressor", None)
    candidates = []
    for owner in (agent, compressor):
        if owner is None:
            continue
        for name in ("manual_compress", "compress_context", "compress_conversation", "maybe_compress"):
            method = getattr(owner, name, None)
            if callable(method):
                candidates.append(method)
    for method in candidates:
        for kwargs in (
            {"messages": messages, "conversation_history": conversation_history},
            {"messages": messages},
            {"conversation_history": conversation_history},
            {},
        ):
            try:
                result = method(**{k: v for k, v in kwargs.items() if v is not None})
            except TypeError:
                continue
            except Exception:
                return False
            if result is False or result is None:
                return False
            return True
    return False


def enforce_before_provider_call(agent: Any, *, messages: Any = None, conversation_history: Any = None) -> str | None:
    """Apply a latched hard-breach decision before the next provider call.

    Returns ``None`` to continue the turn. Returns a user-facing stop message
    when the safe outcome is to stop instead of spending another provider call.
    """
    if not has_pending_breach(agent):
        return None
    choice = ask_for_decision(agent)
    if choice == CHOICE_CONTINUE_ONCE:
        clear_pending_for_continue_once(agent)
        return None
    if choice == CHOICE_COMPRESS_THEN_CONTINUE:
        if _try_compress_context(agent, messages=messages, conversation_history=conversation_history):
            clear_pending_after_compress(agent)
            return None
        return "Session budget guardrail stopped before the next provider call because compression was unavailable or deferred."
    return "Session budget guardrail stopped before the next provider call."
