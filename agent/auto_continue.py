"""Bounded fresh-turn continuation after max-iteration exhaustion.

The continuation policy lives outside ``conversation_loop.run_conversation``.
That inner function owns exactly one turn and exactly one ``IterationBudget``;
when an eligible turn exhausts, its finalizer closes and persists the turn, then
this coordinator starts another turn through the normal turn-context prologue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


AUTO_CONTINUE_ON_MAX_ITERATIONS_MARKER = (
    "[Continuing after max-iteration exhaustion]"
)
DEFAULT_AUTO_CONTINUE_ON_MAX_ITERATIONS_PROMPT = (
    "Continue autonomously from the current state. Do not repeat completed work. "
    "Stop and summarize if blocked, if approval is required, or before "
    "destructive or externally visible actions."
)


@dataclass(frozen=True)
class AutoContinueConfig:
    """Normalized, fail-closed auto-continuation policy."""

    enabled: bool = False
    max_auto_continues: int = 0
    prompt: str = DEFAULT_AUTO_CONTINUE_ON_MAX_ITERATIONS_PROMPT

    def can_continue(self, *, used: int) -> bool:
        return (
            self.enabled
            and self.max_auto_continues > 0
            and max(0, int(used)) < self.max_auto_continues
        )


def resolve_auto_continue_config(config: Any) -> AutoContinueConfig:
    """Normalize the nested config without ever creating an unbounded policy."""
    if not isinstance(config, dict):
        return AutoContinueConfig()
    agent_config = config.get("agent")
    if not isinstance(agent_config, dict):
        return AutoContinueConfig()
    raw = agent_config.get("auto_continue_on_max_iterations")
    if raw is True:
        raw = {"enabled": True}
    elif raw is False or raw is None:
        raw = {}
    if not isinstance(raw, dict):
        return AutoContinueConfig()

    enabled = raw.get("enabled") is True
    raw_maximum = raw.get("max_auto_continues", 0)
    try:
        maximum = 0 if isinstance(raw_maximum, bool) else int(raw_maximum)
    except (TypeError, ValueError, OverflowError):
        maximum = 0
    maximum = max(0, maximum)

    raw_prompt = raw.get("prompt")
    prompt = raw_prompt.strip() if isinstance(raw_prompt, str) else ""
    if not prompt:
        prompt = DEFAULT_AUTO_CONTINUE_ON_MAX_ITERATIONS_PROMPT
    return AutoContinueConfig(
        enabled=enabled,
        max_auto_continues=maximum,
        prompt=prompt,
    )


def load_auto_continue_config() -> AutoContinueConfig:
    """Load the policy once for one public conversation invocation."""
    from hermes_cli.config import load_config

    return resolve_auto_continue_config(load_config() or {})


def build_auto_continue_user_message(config: AutoContinueConfig) -> str:
    return f"{AUTO_CONTINUE_ON_MAX_ITERATIONS_MARKER}\n{config.prompt}"


def is_auto_continue_on_max_iterations_prompt(content: Any) -> bool:
    """Return whether content is the persisted synthetic continuation marker."""
    if not isinstance(content, str):
        return False
    return content.startswith(AUTO_CONTINUE_ON_MAX_ITERATIONS_MARKER)


def is_real_user_message(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and message.get("role") == "user"
        and not is_auto_continue_on_max_iterations_prompt(message.get("content"))
    )


def find_last_real_user_message_index(messages: list[dict[str, Any]]) -> int | None:
    """Find the last user-authored turn, excluding synthetic continuations."""
    for index in range(len(messages) - 1, -1, -1):
        if is_real_user_message(messages[index]):
            return index
    return None


def _cycle_api_calls(result: dict[str, Any]) -> int:
    raw = result.get("cycle_api_calls", result.get("api_calls", 0))
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def run_with_auto_continue(
    agent: Any,
    run_turn: Callable[..., dict[str, Any]],
    *,
    user_message: Any,
    system_message: str | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    task_id: str | None = None,
    stream_callback: Callable[..., Any] | None = None,
    persist_user_message: Any | None = None,
    persist_user_timestamp: float | None = None,
    moa_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one public request as one or more independently bounded turns."""
    config = load_auto_continue_config()
    if not config.can_continue(used=0):
        return run_turn(
            agent,
            user_message,
            system_message,
            conversation_history,
            task_id,
            stream_callback,
            persist_user_message,
            persist_user_timestamp=persist_user_timestamp,
            moa_config=moa_config,
        )

    current_user_message = user_message
    current_history = conversation_history
    current_persist_message = persist_user_message
    current_persist_timestamp = persist_user_timestamp
    total_api_calls = 0
    calls_by_cycle: list[int] = []
    auto_continues_used = 0
    cleanup_errors: list[str] = []
    completion_hook_errors: list[str] = []

    while True:
        continuation_available = config.can_continue(used=auto_continues_used)
        result = run_turn(
            agent,
            current_user_message,
            system_message,
            current_history,
            task_id,
            stream_callback,
            current_persist_message,
            persist_user_timestamp=current_persist_timestamp,
            moa_config=moa_config,
            _defer_iteration_limit_fallback=continuation_available,
            _total_api_call_offset=total_api_calls,
        )

        cycle_calls = _cycle_api_calls(result)
        calls_by_cycle.append(cycle_calls)
        total_api_calls += cycle_calls
        cleanup_errors.extend(result.get("cleanup_errors") or [])
        completion_hook_errors.extend(result.get("completion_hook_errors") or [])

        continuation_ready = bool(
            result.get("iteration_limit_continuation_ready", False)
        )
        if not continuation_ready or not continuation_available:
            if auto_continues_used:
                result["api_calls"] = total_api_calls
                result["cycle_api_calls"] = cycle_calls
                result["api_calls_by_cycle"] = calls_by_cycle
                result["auto_continues_used"] = auto_continues_used
                if cleanup_errors:
                    result["cleanup_errors"] = cleanup_errors
                if completion_hook_errors:
                    result["completion_hook_errors"] = completion_hook_errors
            result.pop("iteration_limit_continuation_ready", None)
            return result

        auto_continues_used += 1
        current_history = result.get("messages") or current_history
        current_user_message = build_auto_continue_user_message(config)
        # Synthetic continuations are deliberate durable user-role boundaries;
        # never rewrite them with the original turn's API-only persistence value.
        current_persist_message = None
        current_persist_timestamp = None

        try:
            agent._touch_activity(
                "starting fresh max-iteration continuation turn "
                f"({auto_continues_used}/{config.max_auto_continues})"
            )
            agent._emit_status(
                "🔁 Auto-continuing in a fresh turn after iteration exhaustion "
                f"({auto_continues_used}/{config.max_auto_continues})"
            )
        except Exception:
            pass
