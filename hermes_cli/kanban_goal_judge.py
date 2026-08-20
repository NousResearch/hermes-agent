"""Shared completion-gate decisions for Kanban goal-mode tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.redact import redact_sensitive_text


@dataclass(frozen=True)
class GateDecision:
    action: str  # pass | reject | block
    reason: str = ""


def _safe_reason(reason: object, fallback: str) -> str:
    """Return a persistence/display-safe judge reason."""
    text = str(reason or "").strip()
    return redact_sensitive_text(text, force=True) if text else fallback


def evaluate_goal_completion(
    task,
    response: str,
    *,
    judge_available: Callable[[], bool],
    judge: Callable[..., tuple],
) -> GateDecision:
    """Evaluate once while preserving legacy best-effort fail-open behavior."""
    # ``getattr`` keeps old fixtures/plugins that construct Task-like objects
    # compatible with the additive column's default.
    required = getattr(task, "goal_judge_policy", "best_effort") == "required"
    if not task.goal_mode:
        return GateDecision("pass")
    try:
        available = judge_available()
    except Exception:
        return GateDecision("block", "goal judge availability check failed") if required else GateDecision("pass")
    if not available:
        return GateDecision("block", "goal judge is unavailable") if required else GateDecision("pass")
    try:
        verdict, reason, parse_failed, wait_directive, transport_failed = judge(
            goal=f"{task.title}\n\n{task.body or ''}".strip(),
            last_response=(response or "").strip(),
        )
    except Exception:
        return GateDecision("block", "goal judge evaluation failed") if required else GateDecision("pass")

    if required:
        if transport_failed:
            return GateDecision("block", _safe_reason(reason, "goal judge transport failed"))
        if parse_failed:
            return GateDecision("block", _safe_reason(reason, "goal judge response could not be parsed"))
        if verdict == "done":
            return GateDecision("pass", _safe_reason(reason, "accepted"))
        if verdict == "continue":
            return GateDecision("reject", _safe_reason(reason, "goal is not complete"))
        if verdict == "wait":
            wait_reason = (
                wait_directive.get("reason")
                if isinstance(wait_directive, dict)
                else None
            )
            return GateDecision(
                "reject",
                _safe_reason(wait_reason or reason, "goal judge requested a wait"),
            )
        return GateDecision("block", _safe_reason(reason, "unsupported goal judge verdict"))

    # Exact legacy mapping: once a judge is reachable and returns normally,
    # only the verdict controls the existing best-effort gate.
    if verdict == "done":
        return GateDecision("pass", _safe_reason(reason, "accepted"))
    return GateDecision("reject", _safe_reason(reason, "goal is not complete"))
