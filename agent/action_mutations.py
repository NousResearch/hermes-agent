"""Reviewed mutation capabilities and safe post-tool projections.

This module deliberately never persists tool arguments or results. The
observer receives them only long enough to select a reviewed capability and
construct the bounded event projection sent to the profile-local journal.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Mapping
from uuid import UUID, uuid4, uuid5

from gateway.action_journal import ActionJournal, MutationEvent, MutationStatus, MutationType

_EVENT_NAMESPACE = UUID("f5b3a5bc-3d4a-5b8c-9a1f-8aa2f2b8bd0f")
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{0,127}$")
_JOURNAL_LOCK = threading.RLock()
_JOURNALS: dict[str, ActionJournal] = {}


@dataclass(frozen=True, slots=True)
class MutationSpec:
    """Trusted, bounded metadata for a state-changing connected tool."""

    action_type: str
    provider: str
    operation: str
    one_shot: bool = False
    requires_receipt: bool = False
    destination_arg: str | None = None

    def __post_init__(self) -> None:
        if self.action_type not in {item.value for item in MutationType}:
            raise ValueError("unsupported mutation action_type")
        for name, value, maximum in (
            ("provider", self.provider, 80),
            ("operation", self.operation, 80),
        ):
            if not isinstance(value, str) or not 1 <= len(value) <= maximum:
                raise ValueError(f"invalid mutation {name}")
        if not isinstance(self.one_shot, bool) or not isinstance(self.requires_receipt, bool):
            raise ValueError("mutation flags must be bool")
        if self.destination_arg is not None and (
            not isinstance(self.destination_arg, str)
            or not _LABEL_RE.fullmatch(self.destination_arg)
        ):
            raise ValueError("invalid destination argument name")


def _spec(action_type: str, provider: str, operation: str, *, one_shot: bool = False) -> MutationSpec:
    return MutationSpec(
        action_type=action_type,
        provider=provider,
        operation=operation,
        one_shot=one_shot,
        requires_receipt=one_shot,
    )


# Exact names are intentional. A name that merely contains "create" or
# "update" is not evidence of an external mutation.
REVIEWED_MUTATIONS: Mapping[str, MutationSpec] = {
    "mcp_google_calendar_create_event": _spec("calendar", "Google Calendar", "create_event", one_shot=True),
    "mcp_google_calendar_add_event": _spec("calendar", "Google Calendar", "create_event", one_shot=True),
    "mcp_todoist_add_tasks": _spec("todoist", "Todoist", "create_task", one_shot=True),
    "mcp_todoist_create_task": _spec("todoist", "Todoist", "create_task", one_shot=True),
    "mcp_obsidian_create_note": _spec("note", "Obsidian", "create_note", one_shot=True),
    "mcp_apple_notes_create_note": _spec("note", "Apple Notes", "create_note", one_shot=True),
    "mcp_google_docs_create_document": _spec("note", "Google Docs", "create_note", one_shot=True),
    "ha_call_service": _spec("home_automation", "Home Assistant", "call_service"),
    "mcp_homeassistant_call_service": _spec("home_automation", "Home Assistant", "call_service"),
    "mcp_home_assistant_call_service": _spec("home_automation", "Home Assistant", "call_service"),
    "mcp_opnsense_apply_firewall_rule": _spec("network", "Firewall", "apply_rule"),
    "mcp_opnsense_update_firewall_rule": _spec("network", "Firewall", "update_rule"),
    "mcp_unifi_update_firewall_rule": _spec("network", "Firewall", "update_rule"),
}


def reviewed_mutation_spec(tool_name: str) -> MutationSpec | None:
    return REVIEWED_MUTATIONS.get(str(tool_name).strip().casefold())


def _resolve_spec(
    function_name: str,
    *,
    registry: Any | None,
) -> MutationSpec | None:
    active_registry = registry
    if active_registry is None:
        try:
            from tools.registry import registry as active_registry
        except Exception:
            active_registry = None
    if active_registry is not None:
        try:
            entry = active_registry.get_entry(function_name)
        except Exception:
            entry = None
        candidate = getattr(entry, "mutation", None)
        if isinstance(candidate, MutationSpec):
            return candidate
    return reviewed_mutation_spec(function_name)


def _event_key(
    *,
    function_name: str,
    session_id: str,
    turn_id: str,
    tool_call_id: str,
    api_request_id: str,
) -> UUID:
    identity = "\0".join(
        (
            str(session_id or ""),
            str(turn_id or ""),
            str(tool_call_id or ""),
            str(api_request_id or ""),
            str(function_name),
        )
    )
    if not any((session_id, turn_id, tool_call_id, api_request_id)):
        return uuid4()
    return uuid5(_EVENT_NAMESPACE, identity)


def _default_journal() -> ActionJournal:
    from hermes_constants import get_hermes_home, hermes_home_key

    home = get_hermes_home()
    key = hermes_home_key(home)
    with _JOURNAL_LOCK:
        journal = _JOURNALS.get(key)
        if journal is None:
            journal = ActionJournal(home / "gateway" / "becky-actions.sqlite3", profile_key=key)
            _JOURNALS[key] = journal
        return journal


def get_action_journal() -> ActionJournal:
    """Return the process-local profile journal used by mutation observers."""
    return _default_journal()


def close_action_journals() -> None:
    global _JOURNALS
    with _JOURNAL_LOCK:
        journals, _JOURNALS = dict(_JOURNALS), {}
    for journal in journals.values():
        try:
            journal.close()
        except Exception:
            pass


def _safe_destination(spec: MutationSpec, arguments: Mapping[str, Any]) -> str | None:
    if not spec.destination_arg:
        return None
    value = arguments.get(spec.destination_arg)
    if not isinstance(value, str) or not _LABEL_RE.fullmatch(value.strip()):
        return None
    normalized = value.strip().casefold()
    allowed = {
        "obsidian": "Obsidian",
        "apple_notes": "Apple Notes",
        "apple notes": "Apple Notes",
        "google_docs": "Google Docs",
        "google docs": "Google Docs",
        "personal calendar": "Personal calendar",
    }
    return allowed.get(normalized)


def record_tool_mutation(
    *,
    function_name: str,
    function_args: Mapping[str, Any] | None,
    result: Any,
    status: str | None,
    session_id: str = "",
    turn_id: str = "",
    tool_call_id: str = "",
    api_request_id: str = "",
    journal: ActionJournal | None = None,
    registry: Any | None = None,
    now: Callable[[], datetime] | None = None,
) -> MutationEvent | None:
    """Record one machine-classified mutating call as a safe event.

    ``result`` is accepted to make the observer boundary explicit but is never
    parsed, interpolated, logged, or stored here. ``status`` is supplied by
    Hermes's machine result classifier, not by assistant prose.
    """
    del result
    if status not in {"ok", "error", "succeeded", "failed"}:
        return None
    spec = _resolve_spec(function_name, registry=registry)
    if spec is None:
        return None
    arguments = function_args if isinstance(function_args, Mapping) else {}
    key = _event_key(
        function_name=function_name,
        session_id=session_id,
        turn_id=turn_id,
        tool_call_id=tool_call_id,
        api_request_id=api_request_id,
    )
    target = journal or _default_journal()
    existing = target.get(key)
    if existing is not None:
        return existing
    final_status = MutationStatus.SUCCEEDED if status in {"ok", "succeeded"} else MutationStatus.FAILED
    outcome = "succeeded" if final_status is MutationStatus.SUCCEEDED else "failed"
    operation_label = spec.operation.replace("_", " ")
    title = f"{spec.provider} {operation_label}"[:128]
    description = f"{spec.provider} {operation_label} {outcome}."[:1_000]
    timestamp = (now or (lambda: datetime.now(UTC)))()
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    event = MutationEvent(
        source_event_key=key,
        status=final_status,
        action_type=spec.action_type,
        title=title,
        description=description,
        provider=spec.provider,
        operation=spec.operation,
        destination=_safe_destination(spec, arguments),
        occurred_at=timestamp,
        context=description,
        requires_receipt=spec.requires_receipt,
    )
    stored, _ = target.append(event)
    return stored


__all__ = [
    "MutationSpec",
    "REVIEWED_MUTATIONS",
    "close_action_journals",
    "get_action_journal",
    "record_tool_mutation",
    "reviewed_mutation_spec",
]
