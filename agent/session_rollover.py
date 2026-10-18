"""Runtime pressure accounting and safe-point decisions for session rollover.

The guard is deliberately opt-in: a gateway binds one to ``agent._session_rollover``.
It never changes the conversation prefix or tool surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

_PHASES = frozenset({"watching", "requested", "authoring", "invalidated"})
_API_MODES = frozenset({
    "chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse",
})

SESSION_ROLLOVER_REQUESTED_RESPONSE = (
    "This turn is paused for a safe session handoff; the accepted input is preserved."
)


class RolloverBudgetUnavailable(RuntimeError):
    """The bound runtime cannot supply a trustworthy rollover budget."""


@dataclass(frozen=True)
class RolloverBudget:
    """Resolved runtime budgets; no model-family or auxiliary-window defaults are used."""

    context_window: int
    threshold_tokens: int
    emergency_threshold_tokens: int
    reserve_tokens: int
    rollover_threshold_tokens: int
    hygiene_message_limit: int


@dataclass(frozen=True)
class RolloverSnapshot:
    """One read-only pressure measurement and its safe-point decision."""

    context_window: int
    threshold_tokens: int
    emergency_threshold_tokens: int
    reserve_tokens: int
    rollover_threshold_tokens: int
    pressure_tokens: int
    message_count: int
    hygiene_message_limit: int
    reason: Optional[str]
    rollover_requested: bool
    rollover_fallback: bool
    skip_hygiene: bool
    skip_proactive_compaction: bool


def _positive_runtime_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RolloverBudgetUnavailable(f"runtime {field} is unavailable or invalid")
    return value


def _configured_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def resolve_rollover_budget(agent: Any, guard: RolloverGuard) -> RolloverBudget:
    """Resolve the earliest real compaction/hygiene threshold from this agent's runtime."""
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        raise RolloverBudgetUnavailable("bound agent has no resolved context compressor")
    try:
        context_window = _positive_runtime_int(
            getattr(compressor, "context_length", None), "context window"
        )
    except Exception as exc:
        if isinstance(exc, RolloverBudgetUnavailable):
            raise
        raise RolloverBudgetUnavailable(f"runtime context window could not be resolved: {exc}") from exc

    compression_enabled = getattr(agent, "compression_enabled", None)
    if not isinstance(compression_enabled, bool):
        raise RolloverBudgetUnavailable("runtime compression-enabled state is unavailable")

    api_mode = getattr(agent, "api_mode", None)
    if not isinstance(api_mode, str) or api_mode not in _API_MODES:
        raise RolloverBudgetUnavailable("runtime API mode is unsupported or unavailable")

    thresholds = [guard.hygiene_threshold_tokens]
    if compression_enabled:
        try:
            thresholds.append(
                _positive_runtime_int(
                    getattr(compressor, "threshold_tokens", None), "local compaction threshold"
                )
            )
        except Exception as exc:
            if isinstance(exc, RolloverBudgetUnavailable):
                raise
            raise RolloverBudgetUnavailable(
                f"runtime local compaction threshold could not be resolved: {exc}"
            ) from exc

    if api_mode == "codex_responses":
        try:
            from agent.codex_responses_adapter import classify_responses_route
            from agent.native_compaction import native_compaction_context_management

            native_settings = native_compaction_context_management(
                agent, **classify_responses_route(agent)._asdict()
            )
        except Exception as exc:
            raise RolloverBudgetUnavailable(
                f"native provider compaction threshold could not be resolved: {exc}"
            ) from exc
        if native_settings:
            if len(native_settings) != 1 or not isinstance(native_settings[0], dict):
                raise RolloverBudgetUnavailable("native provider returned an unsupported compaction policy")
            thresholds.append(
                _positive_runtime_int(
                    native_settings[0].get("compact_threshold"),
                    "native compaction threshold",
                )
            )

    threshold_tokens = min(thresholds)
    # The compressor trigger already accounts for its configured output reservation.
    # Cap it at the runtime's hard input limit as well; for providers with an implicit
    # native default, reuse the same resolver used when the compressor was initialized.
    if not hasattr(agent, "max_tokens"):
        raise RolloverBudgetUnavailable("agent output-token limit is unavailable")
    try:
        from agent.agent_init import _compressor_max_tokens

        max_tokens = _compressor_max_tokens(agent)
    except Exception as exc:
        raise RolloverBudgetUnavailable(
            f"runtime output-token limit could not be resolved: {exc}"
        ) from exc
    if max_tokens is None:
        input_window = context_window
    elif not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
        raise RolloverBudgetUnavailable("runtime output-token limit is invalid")
    else:
        input_window = context_window - max_tokens
    if input_window <= 0:
        raise RolloverBudgetUnavailable("runtime output reservation leaves no input budget")
    emergency_threshold_tokens = min(threshold_tokens, input_window)
    reserve_tokens = guard.runway_tokens + guard.safety_margin_tokens
    if reserve_tokens >= emergency_threshold_tokens:
        raise RolloverBudgetUnavailable(
            "measured rollover reserve does not fit below the runtime compaction threshold"
        )
    return RolloverBudget(
        context_window=context_window,
        threshold_tokens=threshold_tokens,
        emergency_threshold_tokens=emergency_threshold_tokens,
        reserve_tokens=reserve_tokens,
        rollover_threshold_tokens=emergency_threshold_tokens - reserve_tokens,
        hygiene_message_limit=guard.hygiene_message_limit,
    )


def measure_rollover_pressure(
    agent: Any,
    messages: Sequence[Any],
    *,
    system_prompt: Optional[str] = None,
    request_pressure_tokens: Any = None,
    pending_message: Any = None,
) -> int:
    """Measure anchored usage plus appended growth, or accept the assembled request estimate.

    ``pending_message`` is a not-yet-appended accepted input and is measured from a
    copy; the caller's transcript is never modified.
    """
    if request_pressure_tokens is not None:
        if not isinstance(request_pressure_tokens, int) or isinstance(request_pressure_tokens, bool) or request_pressure_tokens < 0:
            raise RolloverBudgetUnavailable("assembled request pressure is unavailable or invalid")
        return request_pressure_tokens
    if not isinstance(messages, (list, tuple)):
        raise RolloverBudgetUnavailable("provider-visible conversation messages are unavailable")

    measured_messages = list(messages)
    if pending_message is not None:
        measured_messages.append(pending_message)
    if system_prompt is None:
        cached_prompt = getattr(agent, "_cached_system_prompt", None)
        if isinstance(cached_prompt, str):
            system_prompt = cached_prompt
        else:
            from agent.usage_anchor import anchored_context_tokens

            if anchored_context_tokens(measured_messages, getattr(agent, "_usage_anchor", None)) is None:
                raise RolloverBudgetUnavailable(
                    "an unanchored pressure estimate requires the resolved system prompt"
                )
            system_prompt = ""
    if not isinstance(system_prompt, str):
        raise RolloverBudgetUnavailable("resolved system prompt is unavailable")
    try:
        from agent.turn_context import _preflight_request_tokens

        pressure = _preflight_request_tokens(agent, measured_messages, system_prompt)
    except Exception as exc:
        raise RolloverBudgetUnavailable(f"runtime request pressure could not be measured: {exc}") from exc
    if not isinstance(pressure, int) or isinstance(pressure, bool) or pressure < 0:
        raise RolloverBudgetUnavailable("runtime request pressure is unavailable or invalid")
    return pressure


class RolloverGuard:
    """Mutable lifecycle guard bound only to the opted-in runtime agent."""

    def __init__(
        self,
        runway_tokens: int,
        safety_margin_tokens: int,
        hygiene_threshold_tokens: int,
        hygiene_message_limit: int,
    ) -> None:
        self.runway_tokens = _configured_int(runway_tokens, "runway_tokens")
        self.safety_margin_tokens = _configured_int(
            safety_margin_tokens, "safety_margin_tokens", allow_zero=True
        )
        self.hygiene_threshold_tokens = _configured_int(
            hygiene_threshold_tokens, "hygiene_threshold_tokens"
        )
        self.hygiene_message_limit = _configured_int(
            hygiene_message_limit, "hygiene_message_limit"
        )
        self.phase = "watching"
        self.pressure_tokens: Optional[int] = None
        self.threshold_tokens: Optional[int] = None
        self.context_window: Optional[int] = None
        self.reason: Optional[str] = None
        self.rollover_fallback = False
        self._diagnostic_emitted = False

    def inspect(
        self,
        agent: Any,
        messages: Sequence[Any],
        *,
        system_prompt: Optional[str] = None,
        request_pressure_tokens: Any = None,
        pending_message: Any = None,
    ) -> RolloverSnapshot:
        """Inspect pressure; ``pending_message`` is counted only when not already in ``messages``."""
        if self.phase not in _PHASES:
            raise RuntimeError(f"invalid rollover phase: {self.phase!r}")
        budget = resolve_rollover_budget(agent, self)
        pressure = measure_rollover_pressure(
            agent,
            messages,
            system_prompt=system_prompt,
            request_pressure_tokens=request_pressure_tokens,
            pending_message=pending_message,
        )
        message_count = len(messages) + (pending_message is not None)
        self.pressure_tokens = pressure
        self.threshold_tokens = budget.threshold_tokens
        self.context_window = budget.context_window
        reason = self.reason

        emergency_reason = None
        if pressure >= budget.emergency_threshold_tokens:
            emergency_reason = "emergency_context_pressure"
        elif self.phase == "authoring" and message_count >= budget.hygiene_message_limit:
            emergency_reason = "emergency_message_limit"

        if self.phase != "invalidated" and emergency_reason:
            self.invalidate(emergency_reason, agent=agent)
            reason = self.reason
        elif self.phase == "watching":
            if message_count >= budget.hygiene_message_limit:
                self.phase = "requested"
                self.reason = reason = "message_limit"
            elif pressure >= budget.rollover_threshold_tokens:
                self.phase = "requested"
                self.reason = reason = "token_pressure"
        elif self.phase == "requested":
            reason = self.reason or "token_pressure"
        elif self.phase == "authoring":
            reason = self.reason or "authoring"

        rollover_requested = self.phase == "requested"
        skip_proactive = rollover_requested or self.phase == "authoring"
        return RolloverSnapshot(
            context_window=budget.context_window,
            threshold_tokens=budget.threshold_tokens,
            emergency_threshold_tokens=budget.emergency_threshold_tokens,
            reserve_tokens=budget.reserve_tokens,
            rollover_threshold_tokens=budget.rollover_threshold_tokens,
            pressure_tokens=pressure,
            message_count=message_count,
            hygiene_message_limit=budget.hygiene_message_limit,
            reason=reason,
            rollover_requested=rollover_requested,
            rollover_fallback=self.rollover_fallback,
            skip_hygiene=skip_proactive,
            skip_proactive_compaction=skip_proactive,
        )

    def invalidate(self, reason: str, *, agent: Any = None) -> None:
        """Invalidate rollover before the caller enters native emergency recovery."""
        if self.phase == "invalidated":
            return
        self.phase = "invalidated"
        self.reason = str(reason or "emergency_recovery")
        self.rollover_fallback = True
        if not self._diagnostic_emitted:
            self._diagnostic_emitted = True
            notice = "Session rollover was invalidated; native context recovery is being used."
            emit = getattr(agent, "_emit_diagnostic_status", None) if agent is not None else None
            if callable(emit):
                try:
                    emit(notice)
                except Exception:
                    logger.warning("Could not emit session rollover fallback notice", exc_info=True)
            else:
                logger.warning("%s reason=%s", notice, self.reason)

    def take_result_flags(self) -> tuple[bool, bool]:
        """Return result flags once; the gateway releases an invalidated hygiene fence."""
        requested = self.phase == "requested"
        fallback = self.rollover_fallback
        self.rollover_fallback = False
        return requested, fallback
