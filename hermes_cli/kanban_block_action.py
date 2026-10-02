"""Deterministic blocked-task action contracts shared by CLI and notifications."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional


_BLOCK_EVENT_KINDS = frozenset(
    {"blocked", "gave_up", "block_loop_detected", "dependency_wait", "timed_out"}
)


@dataclass(frozen=True)
class BlockActionContract:
    disposition: str
    action_required: bool
    owner: str
    action: str
    reply_format: str
    consequence_if_no_action: str
    next_action: str
    retry_condition: str
    auto_resume: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reason_from_events(events: Iterable[Any]) -> Optional[str]:
    """Return the reason attached to the event that created the current block.

    Stop at the newest block event even when it has no reason. Continuing into
    older history would revive a repaired block's stale action after a retry.
    """
    for event in reversed(list(events)):
        kind = _get(event, "kind")
        if kind not in _BLOCK_EVENT_KINDS:
            continue
        payload = _get(event, "payload")
        if isinstance(payload, str):
            import json
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                payload = None
        if isinstance(payload, Mapping):
            reason = payload.get("reason") or payload.get("error")
            if reason:
                return str(reason).strip()
        return None
    return None


def latest_block_event_id(events: Iterable[Any]) -> Optional[int]:
    """Return the newest block-related event id, if one exists."""
    for event in reversed(list(events)):
        if _get(event, "kind") in _BLOCK_EVENT_KINDS:
            event_id = _get(event, "id")
            return int(event_id) if event_id is not None else None
    return None


def latest_block_run_id(events: Iterable[Any]) -> Optional[int]:
    """Return the run that owns the newest block-related event, if any."""
    for event in reversed(list(events)):
        if _get(event, "kind") in _BLOCK_EVENT_KINDS:
            run_id = _get(event, "run_id")
            return int(run_id) if run_id is not None else None
    return None


def build_block_action(task: Any, *, reason: Optional[str] = None) -> BlockActionContract:
    """Build the current required-action contract from persisted task state."""
    kind = (_get(task, "block_kind") or "").strip().lower()
    assignee = _get(task, "assignee") or _get(task, "created_by") or "kanban dispatcher"
    # ``last_failure_error`` is historical attempt metadata, not necessarily the
    # reason for the current block. Only the current block event may supply it.
    reason = (reason or "").strip()

    if kind == "needs_input":
        disposition, owner = "Matt action required", "Matt"
    elif kind == "dependency":
        disposition, owner = "Dependency", str(assignee)
    elif kind == "capability":
        disposition, owner = "Internal owner action", str(assignee)
    else:
        disposition, owner = "Stale/recovery", str(assignee)
    return BlockActionContract(**_defaults(disposition, owner, reason, assignee))


def render_block_action(contract: BlockActionContract, *, indent: str = "  ") -> list[str]:
    reply = contract.reply_format if contract.action_required else "No reply required."
    return [
        f"{indent}Disposition: {contract.disposition}",
        f"{indent}Owner: {contract.owner}",
        f"{indent}Required action: {contract.action}",
        f"{indent}Reply format: {reply}",
        f"{indent}If no action: {contract.consequence_if_no_action}",
        f"{indent}Next action: {contract.next_action}",
        f"{indent}Retry condition: {contract.retry_condition}",
        f"{indent}Auto-resume: {'yes' if contract.auto_resume else 'no'}",
    ]


def _defaults(disposition: str, owner: str, reason: str, assignee: Any) -> dict[str, Any]:
    detail = reason or "Resolve the blocker recorded on the task."
    if disposition == "Matt action required":
        return {
            "disposition": disposition, "action_required": True, "owner": owner,
            "action": detail,
            "reply_format": "Reply with the requested decision or input in plain text.",
            "consequence_if_no_action": "The task remains blocked and no further work is dispatched.",
            "next_action": f"{assignee} will apply Matt's reply, unblock the task, and continue.",
            "retry_condition": "Matt provides the requested decision or input.", "auto_resume": False,
        }
    if disposition == "Dependency":
        return {
            "disposition": disposition, "action_required": False, "owner": owner,
            "action": detail, "reply_format": "No reply required.",
            "consequence_if_no_action": "The task remains gated until its dependency is complete.",
            "next_action": f"{owner} will continue after the parent dependency completes.",
            "retry_condition": "All parent dependencies are done.", "auto_resume": True,
        }
    if disposition == "Internal owner action":
        return {
            "disposition": disposition, "action_required": False, "owner": owner,
            "action": detail, "reply_format": "No reply required.",
            "consequence_if_no_action": "The task remains blocked until the internal capability gap is resolved.",
            "next_action": f"{owner} will resolve or re-route the capability gap, then unblock the task.",
            "retry_condition": "The required internal capability or access is available.", "auto_resume": False,
        }
    return {
        "disposition": disposition, "action_required": False, "owner": owner,
        "action": detail, "reply_format": "No reply required.",
        "consequence_if_no_action": "The task remains blocked pending recovery or operator triage.",
        "next_action": f"{owner} will inspect the failure evidence, recover or re-route the task, then unblock it.",
        "retry_condition": "The transient failure is cleared or the stale task state is repaired.", "auto_resume": False,
    }


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)
