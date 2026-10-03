"""Default-off receipt emission at completed tool-dispatch boundaries."""

from __future__ import annotations

import logging
from typing import Any

from tools.terminal_tool_lifecycle import get_active_env

logger = logging.getLogger(__name__)


def _maybe_record_action_receipt(
    agent,
    *,
    function_name: str,
    function_args: Any,
    result: Any,
    effective_task_id: str,
    tool_call_id: str,
    duration_s: float,
    exit_status: str,
) -> None:
    """Record one shadow action receipt for a tool call that actually ran.

    Default-off observability, gated on ``observability.action_receipts.enabled``.
    When that setting is absent or false this returns after a single boolean
    check — no envelope is built and no database is opened or created.

    Every failure here is swallowed. This is telemetry: it must never change
    what a tool did, what the model sees, or the order it sees it in. Callers
    are responsible for invoking it only for calls that actually started —
    malformed, preflight-blocked, and skipped-before-start calls never reach it.
    """
    try:
        from agent.action_receipts import ActionReceiptLedger, is_enabled

        if not is_enabled():
            return

        from agent.task_envelope import build_shadow_envelope

        try:
            cwd = getattr(get_active_env(effective_task_id), "cwd", None)
        except Exception:
            cwd = None

        session_id = getattr(agent, "session_id", "") or None
        envelope = build_shadow_envelope(
            tool_name=function_name,
            args=function_args,
            cwd=str(cwd) if cwd else None,
            session_id=session_id,
            task_id=effective_task_id or None,
            tool_call_id=tool_call_id or None,
        )
        receipt_cwd = getattr(envelope, "cwd", None)
        ActionReceiptLedger().record_receipt(
            tool_name=function_name,
            args=function_args,
            output=result,
            exit_status=exit_status,
            duration_ms=int(duration_s * 1000),
            session_id=session_id,
            task_id=effective_task_id or None,
            tool_call_id=tool_call_id or None,
            cwd=receipt_cwd,
            envelope=envelope,
        )
    except Exception as exc:
        logger.warning("action receipt write failed for %s: %s", function_name, exc)
