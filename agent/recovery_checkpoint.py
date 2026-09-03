"""Deterministic recovery checkpoints derived from persisted tool history.

The gateway persists each assistant tool-call block before dispatch and each tool
result immediately after completion.  This module turns that append-only ledger
into a small restart handoff without asking an LLM to summarize its own work.
Only tool names, call IDs, and result dispositions are included; arguments and
result bodies are deliberately excluded because they may contain secrets or
untrusted content.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from agent.tool_result_classification import tool_may_have_side_effect

MODE_CONTINUE_SAFE = "CONTINUE_SAFE"
MODE_RECONCILE_REQUIRED = "RECONCILE_REQUIRED"

_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9_.:-]+")
_UNKNOWN_ORPHAN_MARKERS = (
    "effect is unknown",
    "its effect is unknown",
    "may have executed",
)


@dataclass(frozen=True)
class RecoveryAction:
    tool_name: str
    tool_call_id: str


@dataclass(frozen=True)
class RecoveryCheckpoint:
    mode: str
    recorded_result_ids: tuple[str, ...] = ()
    unknown_effects: tuple[RecoveryAction, ...] = ()
    retryable_read_only_ids: tuple[str, ...] = ()


def _safe_label(value: Any, *, fallback: str, limit: int = 96) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    cleaned = _SAFE_LABEL_RE.sub("_", text).strip("_")
    return (cleaned or fallback)[:limit]


def _tool_call_id(call: Mapping[str, Any]) -> str:
    return _safe_label(
        call.get("id") or call.get("call_id"),
        fallback="unknown_call",
    )


def _tool_name(call: Mapping[str, Any]) -> str:
    function = call.get("function")
    raw_name = function.get("name") if isinstance(function, Mapping) else None
    return _safe_label(raw_name, fallback="unknown_tool")


def _active_turn(history: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return the last user turn and everything persisted after it."""
    start = 0
    for index, message in enumerate(history):
        if message.get("role") == "user":
            start = index
    return list(history[start:])


def _result_reports_failure(result: Mapping[str, Any]) -> bool:
    content = result.get("content")
    payload: Any = content
    if isinstance(content, str):
        stripped = content.strip()
        try:
            payload = json.loads(stripped)
        except (TypeError, ValueError):
            lowered = stripped.lower()
            return (
                lowered.startswith("error executing tool")
                or lowered.startswith("[command interrupted")
                or "timed out after" in lowered
            )
    if not isinstance(payload, Mapping):
        return False
    if payload.get("error"):
        return True
    exit_code = payload.get("exit_code")
    if (
        isinstance(exit_code, int)
        and not isinstance(exit_code, bool)
        and exit_code != 0
    ):
        return True
    if payload.get("success") is False:
        return True
    status = str(payload.get("status") or "").strip().lower()
    return status in {"error", "failed", "failure", "timeout", "timed_out"}


def _result_is_unknown(tool_name: str, result: Mapping[str, Any]) -> bool:
    disposition = str(result.get("effect_disposition") or "").strip().lower()
    if disposition == "none":
        return False
    if disposition == "unknown":
        return True
    content = result.get("content")
    if isinstance(content, str):
        lowered = content.lower()
        if "orphan recovery" in lowered and any(
            marker in lowered for marker in _UNKNOWN_ORPHAN_MARKERS
        ):
            return True
    return tool_may_have_side_effect(tool_name) and _result_reports_failure(result)


def _iter_calls(
    messages: Iterable[Mapping[str, Any]],
) -> Iterable[tuple[str, str]]:
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if isinstance(call, Mapping):
                yield _tool_call_id(call), _tool_name(call)


def build_recovery_checkpoint(
    history: Sequence[Mapping[str, Any]] | None,
) -> RecoveryCheckpoint:
    """Build a deterministic checkpoint for the interrupted active user turn.

    A recorded non-unknown tool result is authoritative and must not be replayed.
    An unknown or unanswered side-effecting call requires read-only external-state
    reconciliation.  An unanswered read-only call is safe to retry.
    """
    active = _active_turn(list(history or []))
    results: dict[str, Mapping[str, Any]] = {}
    for message in active:
        if message.get("role") != "tool":
            continue
        call_id = _safe_label(message.get("tool_call_id"), fallback="unknown_call")
        results[call_id] = message

    recorded: list[str] = []
    unknown: list[RecoveryAction] = []
    retryable: list[str] = []
    seen: set[str] = set()

    for call_id, tool_name in _iter_calls(active):
        if call_id in seen:
            continue
        seen.add(call_id)
        result = results.get(call_id)
        if result is not None:
            if _result_is_unknown(tool_name, result):
                unknown.append(RecoveryAction(tool_name, call_id))
            else:
                recorded.append(call_id)
            continue
        if tool_may_have_side_effect(tool_name):
            unknown.append(RecoveryAction(tool_name, call_id))
        else:
            retryable.append(call_id)

    mode = MODE_RECONCILE_REQUIRED if unknown else MODE_CONTINUE_SAFE
    return RecoveryCheckpoint(
        mode=mode,
        recorded_result_ids=tuple(recorded),
        unknown_effects=tuple(unknown),
        retryable_read_only_ids=tuple(retryable),
    )


def render_recovery_checkpoint(checkpoint: RecoveryCheckpoint) -> str:
    """Render a bounded, secret-free recovery contract for the model."""
    lines = [
        "[Recovery checkpoint — deterministic from the persisted execution ledger]",
        f"mode={checkpoint.mode}",
        f"recorded_results={len(checkpoint.recorded_result_ids)}",
        f"unknown_effects={len(checkpoint.unknown_effects)}",
        f"retryable_read_only={len(checkpoint.retryable_read_only_ids)}",
    ]

    if checkpoint.unknown_effects:
        lines.append("unknown_effect_calls:")
        for action in checkpoint.unknown_effects[:8]:
            lines.append(f"- {action.tool_name} call_id={action.tool_call_id}")
        if len(checkpoint.unknown_effects) > 8:
            lines.append(f"- and {len(checkpoint.unknown_effects) - 8} more")
        lines.extend([
            "Recovery protocol:",
            "1. Do not repeat an unknown-effect action.",
            "2. Inspect the exact external target with read-only/status tools.",
            "3. If inspection proves the action landed, continue after it.",
            "4. If inspection proves it did not land, retry only through normal approval gates.",
            "5. If inspection cannot determine the state, stop and ask the user.",
        ])
    else:
        lines.extend([
            "Recovery protocol:",
            "1. Do not repeat recorded tool calls.",
            "2. Retry unanswered read-only calls only when still needed.",
            "3. Continue after the last recorded result and finish the task.",
        ])

    return "\n".join(lines)
