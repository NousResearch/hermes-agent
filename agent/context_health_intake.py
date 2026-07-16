"""Phase 2 pre-turn Context Health intake adapter.

This module is intentionally small and deterministic: it does not call a model,
provider, plugin, retriever, gateway, or task-boundary authority.  It only
classifies the current inbound user prompt with the Phase 1 policy model and,
when explicitly enabled for runtime behavior, replaces a long prompt with a
safe summary/path pointer before the raw prompt enters active history.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4

from agent.context_health_policy import (
    ContextHealthPolicy,
    classify_prompt_for_intake,
    default_context_health_policy,
    normalize_context_health_policy,
)


@dataclass(frozen=True)
class PreTurnIntakeResult:
    """Adapter result consumed by ``agent.turn_context``."""

    action: str
    user_message: str
    persist_user_message: str | None
    packet_dir: str | None = None
    summary_path: str | None = None
    intake_path: str | None = None
    task_state_path: str | None = None
    reason: str = ""
    signals: tuple[str, ...] = ()

    @property
    def replaced(self) -> bool:
        return self.action == "replace"


class PreTurnIntakeHold(RuntimeError):
    """Fail-closed HOLD raised before raw prompt append/persist/provider flow."""

    def __init__(self, user_response: str, *, reason: str = "", signals: tuple[str, ...] = ()):
        super().__init__(user_response)
        self.user_response = user_response
        self.reason = reason
        self.signals = signals


def run_pre_turn_intake(
    *,
    agent: Any,
    user_message: str,
    persist_user_message: str | None,
    session_id: str | None,
    task_id: str,
    turn_id: str,
) -> PreTurnIntakeResult:
    """Apply Phase 2 pre-turn intake or return exact pass-through.

    Runtime behavior is gated by all three policy switches:
    ``context_health.enabled``, ``runtime_behavior_enabled``, and
    ``pre_model_intake.enabled``.  Disabled mode returns without writing files
    or mutating messages, even if later adapter logic would fail.
    """

    policy = _resolve_policy(agent)
    if not _runtime_intake_enabled(policy):
        return PreTurnIntakeResult(
            action="pass",
            user_message=user_message,
            persist_user_message=persist_user_message,
            reason="context_health_disabled",
        )

    try:
        decision = classify_prompt_for_intake(user_message, policy)
        if decision.action == "pass":
            return PreTurnIntakeResult(
                action="pass",
                user_message=user_message,
                persist_user_message=persist_user_message,
                reason=decision.reason,
                signals=decision.signals,
            )

        if decision.action == "hold":
            raise PreTurnIntakeHold(
                _hold_response(decision.reason),
                reason=decision.reason,
                signals=decision.signals,
            )

        if decision.action != "force_md_intake":
            raise PreTurnIntakeHold(
                _hold_response("unsupported_intake_decision"),
                reason="unsupported_intake_decision",
                signals=decision.signals,
            )

        packet = _write_intake_packet(
            agent=agent,
            user_message=user_message,
            session_id=session_id,
            task_id=task_id,
            turn_id=turn_id,
            reason=decision.reason,
            signals=decision.signals,
        )
        replacement = _replacement_message(packet, reason=decision.reason, signals=decision.signals)
        return PreTurnIntakeResult(
            action="replace",
            user_message=replacement,
            persist_user_message=replacement,
            packet_dir=str(packet["packet_dir"]),
            summary_path=str(packet["summary_path"]),
            intake_path=str(packet["intake_path"]),
            task_state_path=str(packet["task_state_path"]),
            reason=decision.reason,
            signals=decision.signals,
        )
    except PreTurnIntakeHold:
        raise
    except Exception:
        # Enabled runtime mode is fail-closed by decision record: do not let an
        # adapter exception fall through to raw append/persist/provider context.
        raise PreTurnIntakeHold(
            _hold_response("intake_adapter_failed"),
            reason="intake_adapter_failed",
            signals=(),
        )


def _resolve_policy(agent: Any) -> ContextHealthPolicy:
    explicit = getattr(agent, "_context_health_policy", None)
    if isinstance(explicit, ContextHealthPolicy):
        return explicit
    explicit = getattr(agent, "context_health_policy", None)
    if isinstance(explicit, ContextHealthPolicy):
        return explicit
    raw = getattr(agent, "context_health", None)
    if isinstance(raw, Mapping):
        return _normalize_runtime_policy(raw)
    cfg = getattr(agent, "config", None)
    if isinstance(cfg, Mapping):
        return _normalize_runtime_policy(cfg.get("context_health"))
    return default_context_health_policy()


def _normalize_runtime_policy(raw: Any) -> ContextHealthPolicy:
    policy = normalize_context_health_policy(raw if isinstance(raw, Mapping) else None)
    if isinstance(raw, Mapping) and isinstance(raw.get("runtime_behavior_enabled"), bool):
        return replace(policy, runtime_behavior_enabled=raw["runtime_behavior_enabled"])
    return policy


def _runtime_intake_enabled(policy: ContextHealthPolicy) -> bool:
    return bool(
        policy.enabled
        and policy.runtime_behavior_enabled
        and policy.pre_model_intake.enabled
    )


def _write_intake_packet(
    *,
    agent: Any,
    user_message: str,
    session_id: str | None,
    task_id: str,
    turn_id: str,
    reason: str,
    signals: tuple[str, ...],
) -> dict[str, Path]:
    root = Path(
        getattr(agent, "_context_health_intake_dir", None)
        or Path.home() / ".hermes" / "context-health" / "intake"
    )
    packet_dir = root / _safe_slug(session_id or "session") / _safe_slug(task_id) / _safe_slug(turn_id, suffix=uuid4().hex[:8])
    packet_dir.mkdir(parents=True, exist_ok=False)

    intake_path = packet_dir / "intake.md"
    summary_path = packet_dir / "summary.md"
    task_state_path = packet_dir / "task-state.md"

    line_count = user_message.count("\n") + 1
    char_count = len(user_message)
    signal_text = ", ".join(signals) if signals else "none"

    intake_path.write_text(
        "# Context Health Intake\n\n"
        "## Metadata\n\n"
        f"- session_id: `{session_id or 'none'}`\n"
        f"- task_id: `{task_id}`\n"
        f"- turn_id: `{turn_id}`\n"
        f"- reason: `{reason}`\n"
        f"- signals: `{signal_text}`\n"
        f"- char_count: `{char_count}`\n"
        f"- line_count: `{line_count}`\n\n"
        "## Original Prompt\n\n"
        "```text\n"
        f"{user_message}\n"
        "```\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        "# Context Health Intake Summary\n\n"
        "A long inbound user prompt was moved to an intake packet before it "
        "entered active conversation history. Provider-visible context should "
        "use this summary and the packet path, not the raw prompt body.\n\n"
        f"- reason: `{reason}`\n"
        f"- signals: `{signal_text}`\n"
        f"- char_count: `{char_count}`\n"
        f"- line_count: `{line_count}`\n"
        f"- intake_path: `{intake_path}`\n",
        encoding="utf-8",
    )
    task_state_path.write_text(
        "# Context Health Task State\n\n"
        "- phase: Phase 2 pre-turn intake\n"
        "- state: raw prompt externalized before append/persist\n"
        "- provider_payload_contract: summary/path pointer only\n"
        "- full_task_boundary_firewall: not implemented in Phase 2\n"
        f"- summary_path: `{summary_path}`\n",
        encoding="utf-8",
    )
    return {
        "packet_dir": packet_dir,
        "intake_path": intake_path,
        "summary_path": summary_path,
        "task_state_path": task_state_path,
    }


def _replacement_message(packet: dict[str, Path], *, reason: str, signals: tuple[str, ...]) -> str:
    signal_text = ", ".join(signals) if signals else "none"
    return (
        "[Context Health Intake]\n"
        "The current inbound prompt was long enough to require Phase 2 pre-turn intake.\n"
        "The raw prompt body was moved to an MD intake packet before active history append/persist.\n"
        f"Reason: {reason}; signals: {signal_text}.\n"
        f"Summary: {packet['summary_path']}\n"
        f"Intake packet: {packet['packet_dir']}\n"
        "Use the summary/path pointer for this turn; do not reconstruct the raw prompt in provider context."
    )


def _hold_response(reason: str) -> str:
    return (
        "Context Health HOLD: pre-turn intake could not safely continue. "
        "The raw prompt was not added to active history, provider context, or the current-turn persisted message. "
        f"Reason: {reason}."
    )


def _safe_slug(value: str, *, suffix: str | None = None) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.:-]+", "-", value).strip("-._")[:96] or "item"
    if suffix:
        return f"{slug}-{suffix}"
    return slug


__all__ = ["PreTurnIntakeHold", "PreTurnIntakeResult", "run_pre_turn_intake"]
