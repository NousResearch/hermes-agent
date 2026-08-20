"""Metadata-only autonomous-task lifecycle hooks.

These hooks give profile/runtime integrations a deterministic signal when an
agent turn is near or at the tool/turn budget boundary. The payload is
intentionally tiny and metadata-only so hook consumers can update external DOD
or continuation state without receiving prompts, transcript text, tool outputs,
credentials, environment dumps, or customer data.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

HOOK_NAME = "autonomous_task_turn_budget"
PREEMPTIVE_TURN_BUDGET_THRESHOLD = 70
ALLOWED_FIELDS = frozenset({"event", "turns_used", "max_turns", "session_id"})
FORBIDDEN_PAYLOAD_FIELDS = frozenset(
    {
        "prompts",
        "messages",
        "conversation_history",
        "tool_outputs",
        "secrets",
        "cookies",
        "customer_data",
        "environment",
        "full_environment",
    }
)


def _coerce_non_negative_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def build_turn_budget_metadata(
    *,
    event: str,
    turns_used: Any,
    max_turns: Any,
    session_id: Any,
    **extra: Any,
) -> dict[str, Any]:
    """Return the exact metadata-only payload allowed for the hook.

    ``extra`` is accepted defensively so call sites cannot accidentally widen
    the hook by passing transcript/tool/secret-bearing objects. Known forbidden
    keys and all other non-contract keys are ignored rather than forwarded.
    """

    del extra  # never forward non-contract payloads
    metadata = {
        "event": str(event or ""),
        "turns_used": _coerce_non_negative_int(turns_used),
        "max_turns": _coerce_non_negative_int(max_turns),
        "session_id": str(session_id or ""),
    }
    assert set(metadata) <= ALLOWED_FIELDS
    return metadata


def notify_turn_budget_or_closeout(
    *,
    event: str,
    turns_used: Any,
    max_turns: Any,
    session_id: Any,
    **extra: Any,
) -> bool:
    """Invoke the lifecycle hook with metadata only, fail-open.

    Returns ``True`` when hook dispatch completed, ``False`` when a consumer or
    lifecycle import failed. Failures are logged but must never block final
    response delivery or session persistence.
    """

    # Deliberately build from the allowlist, not from caller kwargs.
    metadata = build_turn_budget_metadata(
        event=event,
        turns_used=turns_used,
        max_turns=max_turns,
        session_id=session_id,
        **extra,
    )
    try:
        from hermes_cli.lifecycle import invoke_hook

        invoke_hook(HOOK_NAME, **metadata)
        return True
    except Exception:
        logger.warning("autonomous task turn-budget hook failed", exc_info=True)
        return False


def event_for_turn_budget(
    *,
    turns_used: Any,
    max_turns: Any,
    iteration_limit_fallback: bool = False,
    budget_exhausted: bool = False,
    threshold: int = PREEMPTIVE_TURN_BUDGET_THRESHOLD,
) -> str | None:
    """Classify whether this turn should emit an autonomous-task hook event."""

    used = _coerce_non_negative_int(turns_used)
    maximum = _coerce_non_negative_int(max_turns)
    if iteration_limit_fallback or budget_exhausted or (maximum and used >= maximum):
        return "max_turn_or_tool_iteration_closeout"
    if used >= threshold:
        return "preemptive_turn_budget_threshold"
    return None


__all__ = [
    "ALLOWED_FIELDS",
    "FORBIDDEN_PAYLOAD_FIELDS",
    "HOOK_NAME",
    "PREEMPTIVE_TURN_BUDGET_THRESHOLD",
    "build_turn_budget_metadata",
    "event_for_turn_budget",
    "notify_turn_budget_or_closeout",
]
