"""Lightweight capture helpers for eval-lab tool-call records.

This intentionally does not patch the core conversation loop. Existing hooks can
call this helper when they need a redacted TrajectoryStep.
"""

from __future__ import annotations

from typing import Any

from agent.eval_lab.redaction import redact_secrets
from agent.eval_lab.schemas import TrajectoryStep


def capture_tool_call(
    *,
    tool_name: str,
    tool_args: dict[str, Any] | None,
    duration_ms: int | None,
    status: str,
    error: str | None = None,
) -> TrajectoryStep:
    """Build a redacted tool-call step without changing tool execution behavior."""

    return TrajectoryStep(
        role="tool",
        content=f"tool_status={status}",
        tool_name=tool_name,
        tool_args_redacted=redact_secrets(tool_args or {}),
        duration_ms=duration_ms,
        error=redact_secrets(error) if error else None,
    )
