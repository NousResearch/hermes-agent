"""Opt-in Runtime bridge for the State Answer Shadow contract.

This module deliberately has no authoritative-state lookup. A caller must inject a
collector on the agent as ``_state_answer_shadow_collector``. Until a Runtime owner
for current state/evidence/requested keys exists, the provider reports
``input_unavailable`` and never guesses from turn metadata.
"""
from __future__ import annotations

import uuid
from typing import Any

from .state_answer_shadow import (
    InputStatus,
    ShadowInput,
    TerminalStatus,
    observe_state_answer_shadow,
    observe_state_answer_shadow_terminal,
)


def _opaque_id() -> str:
    return uuid.uuid4().hex


class UnavailableStateInputProvider:
    """Explicit fail-closed provider used until Runtime SSOT ownership is wired."""

    def __init__(self, *, run_id: str, turn_id: str) -> None:
        self._run_id = run_id
        self._turn_id = turn_id

    def get_input(self) -> ShadowInput:
        return ShadowInput(
            status=InputStatus.INPUT_UNAVAILABLE,
            run_id=self._run_id,
            turn_id=self._turn_id,
        )


def start_runtime_shadow(agent: Any) -> str | None:
    """Start an injected Shadow observation; return None when not explicitly enabled."""
    collector = getattr(agent, "_state_answer_shadow_collector", None)
    if collector is None:
        return None
    run_id = _opaque_id()
    turn_id = _opaque_id()
    return observe_state_answer_shadow(
        UnavailableStateInputProvider(run_id=run_id, turn_id=turn_id),
        collector,
    )


def finish_runtime_shadow(agent: Any, event_id: str | None, result: Any) -> None:
    """Complete a Shadow observation from the actual finalized turn result."""
    collector = getattr(agent, "_state_answer_shadow_collector", None)
    if collector is None or event_id is None or not isinstance(result, dict):
        return
    interrupted = result.get("interrupted") is True
    failed = result.get("failed") is True
    reason = str(result.get("turn_exit_reason") or result.get("failure_reason") or "").lower()
    if interrupted:
        terminal = TerminalStatus.INTERRUPTED
    elif "preflight" in reason:
        terminal = TerminalStatus.PREFLIGHT_RETURN
    elif "retry" in reason:
        terminal = TerminalStatus.RETRY_EXHAUSTED
    elif "provider" in reason or "api" in reason:
        terminal = TerminalStatus.PROVIDER_ERROR
    elif failed:
        terminal = TerminalStatus.EXCEPTION
    else:
        terminal = TerminalStatus.COMPLETED
    model_status = (
        "not_called" if result.get("api_calls") == 0
        else "failed" if failed
        else "observed"
    )
    observe_state_answer_shadow_terminal(
        collector,
        event_id,
        terminal_status=terminal,
        model_call_status=model_status,
        persistence_confirmed=(
            result.get("persistence_confirmed")
            if isinstance(result.get("persistence_confirmed"), bool)
            else None
        ),
    )


def settle_runtime_shadow(agent: Any, event_id: str | None, result: Any) -> Any:
    """Emit the terminal Shadow event and preserve the runtime result unchanged."""
    finish_runtime_shadow(agent, event_id, result)
    return result
