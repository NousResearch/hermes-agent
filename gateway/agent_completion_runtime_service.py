"""Shared runtime helpers for gateway agent-completion handling."""

from __future__ import annotations

from typing import Any, Awaitable, Callable


def apply_gateway_reasoning_display(
    *,
    response: str,
    show_reasoning: bool,
    last_reasoning: Any,
) -> str:
    """Optionally prepend collapsed reasoning to the visible response."""

    if not show_reasoning or not response or not last_reasoning:
        return response

    lines = str(last_reasoning).strip().splitlines()
    if len(lines) > 15:
        display_reasoning = "\n".join(lines[:15])
        display_reasoning += f"\n_... ({len(lines) - 15} more lines)_"
    else:
        display_reasoning = str(last_reasoning).strip()
    return f"💭 **Reasoning:**\n```\n{display_reasoning}\n```\n\n{response}"


def drain_pending_process_watchers(
    *,
    process_registry: Any,
    run_process_watcher: Callable[[dict[str, Any]], Awaitable[None]],
    create_task: Callable[[Awaitable[None]], Any],
    logger: Any | None = None,
    resumed_log_template: str | None = None,
) -> int:
    """Schedule all pending background-process watchers and return the count."""

    scheduled = 0
    while process_registry.pending_watchers:
        watcher = process_registry.pending_watchers.pop(0)
        create_task(run_process_watcher(watcher))
        scheduled += 1
        if logger is not None and resumed_log_template:
            logger.info(resumed_log_template, watcher.get("session_id"))
    return scheduled
