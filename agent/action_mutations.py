"""Reviewed mutation capabilities and safe post-tool projections.

This module deliberately never persists tool arguments or results. The
observer receives them only long enough to select a reviewed capability and
construct the bounded event projection sent to the profile-local journal.
"""

from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Mapping
from uuid import UUID, uuid4, uuid5

from gateway.action_journal import ActionJournal, MutationEvent, MutationStatus, MutationType

_EVENT_NAMESPACE = UUID("f5b3a5bc-3d4a-5b8c-9a1f-8aa2f2b8bd0f")
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{0,127}$")
_METADATA_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _./:-]{0,79}$")
_JOURNAL_LOCK = threading.RLock()
_JOURNALS: dict[str, ActionJournal] = {}
_EVENT_KEY_OVERRIDE: ContextVar[UUID | None] = ContextVar(
    "becky_action_event_key_override", default=None
)


@dataclass(frozen=True, slots=True)
class MutationSpec:
    """Trusted, bounded metadata for a state-changing connected tool."""

    action_type: str
    provider: str
    operation: str
    one_shot: bool = False
    requires_receipt: bool = False
    destination_arg: str | None = None
    write_action_arg: str | None = None
    write_actions: frozenset[str] = frozenset()
    write_predicate: Callable[[Mapping[str, Any]], bool] | None = None

    def __post_init__(self) -> None:
        if self.action_type not in {item.value for item in MutationType}:
            raise ValueError("unsupported mutation action_type")
        for name, value, maximum in (
            ("provider", self.provider, 80),
            ("operation", self.operation, 80),
        ):
            if (
                not isinstance(value, str)
                or not 1 <= len(value) <= maximum
                or _METADATA_LABEL_RE.fullmatch(value) is None
            ):
                raise ValueError(f"invalid mutation {name}")
        if not isinstance(self.one_shot, bool) or not isinstance(self.requires_receipt, bool):
            raise ValueError("mutation flags must be bool")
        if self.destination_arg is not None and (
            not isinstance(self.destination_arg, str)
            or not _LABEL_RE.fullmatch(self.destination_arg)
        ):
            raise ValueError("invalid destination argument name")
        if self.write_action_arg is not None and (
            not isinstance(self.write_action_arg, str)
            or not _LABEL_RE.fullmatch(self.write_action_arg)
        ):
            raise ValueError("invalid write action argument name")
        if self.write_action_arg is None and self.write_actions:
            raise ValueError("write actions require an argument name")
        if any(
            not isinstance(action, str)
            or not 1 <= len(action) <= 40
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]*", action) is None
            for action in self.write_actions
        ):
            raise ValueError("invalid write action")

    def allows_arguments(self, arguments: Mapping[str, Any]) -> bool:
        """Return whether this mixed MCP operation is a state-changing call."""
        if self.write_predicate is not None and not self.write_predicate(arguments):
            return False
        if self.write_action_arg is None:
            return True
        value = arguments.get(self.write_action_arg)
        return isinstance(value, str) and value.strip().casefold() in {
            action.casefold() for action in self.write_actions
        }


def _spec(
    action_type: str,
    provider: str,
    operation: str,
    *,
    one_shot: bool = False,
    write_action_arg: str | None = None,
    write_actions: frozenset[str] = frozenset(),
    write_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> MutationSpec:
    return MutationSpec(
        action_type=action_type,
        provider=provider,
        operation=operation,
        one_shot=one_shot,
        requires_receipt=one_shot,
        write_action_arg=write_action_arg,
        write_actions=write_actions,
        write_predicate=write_predicate,
    )


def _home_assistant_app_write(arguments: Mapping[str, Any]) -> bool:
    action = arguments.get("action")
    if isinstance(action, str) and action.casefold() in {
        "add", "configure", "create", "delete", "deploy", "disable", "enable",
        "flash", "install", "rebuild", "remove", "restart", "set", "start",
        "stop", "uninstall", "update", "upload",
    }:
        return True
    # Proxy reads are allowed through this tool, but an array patch is an
    # explicit write mode. Configuration fields are also write-only modes.
    if arguments.get("array_patch") is not None:
        return True
    if arguments.get("path") is not None:
        return False
    return any(
        key in arguments
        for key in (
            "options", "network", "boot", "watchdog", "auto_update", "supervisor",
            "ingress", "websocket", "api", "rest", "config", "settings",
        )
    )


# Exact names are intentional. A name that merely contains "create" or
# "update" is not evidence of an external mutation.
REVIEWED_MUTATIONS: Mapping[str, MutationSpec] = {
    "mcp_google_calendar_create_event": _spec("calendar", "Google Calendar", "create_event", one_shot=True),
    "mcp_google_calendar_add_event": _spec("calendar", "Google Calendar", "create_event", one_shot=True),
    "mcp__google_calendar__create_event": _spec("calendar", "Google Calendar", "create_event", one_shot=True),
    "mcp__google_calendar__add_event": _spec("calendar", "Google Calendar", "create_event", one_shot=True),
    "mcp_todoist_add_tasks": _spec("todoist", "Todoist", "create_task", one_shot=True),
    "mcp_todoist_create_task": _spec("todoist", "Todoist", "create_task", one_shot=True),
    "mcp__todoist__add_tasks": _spec("todoist", "Todoist", "create_task", one_shot=True),
    "mcp__todoist__create_task": _spec("todoist", "Todoist", "create_task", one_shot=True),
    "mcp__todoist__add_comments": _spec("todoist", "Todoist", "add_comment"),
    "mcp__todoist__add_filters": _spec("todoist", "Todoist", "add_filter"),
    "mcp__todoist__add_labels": _spec("todoist", "Todoist", "add_label"),
    "mcp__todoist__add_projects": _spec("todoist", "Todoist", "add_project"),
    "mcp__todoist__add_reminders": _spec("todoist", "Todoist", "add_reminder"),
    "mcp__todoist__add_sections": _spec("todoist", "Todoist", "add_section"),
    "mcp__todoist__complete_tasks": _spec("todoist", "Todoist", "complete_task"),
    "mcp__todoist__delete_object": _spec("todoist", "Todoist", "delete_object"),
    "mcp__todoist__manage_assignments": _spec("todoist", "Todoist", "manage_assignment"),
    "mcp__todoist__project_management": _spec("todoist", "Todoist", "manage_project"),
    "mcp__todoist__project_move": _spec("todoist", "Todoist", "move_project"),
    "mcp__todoist__reorder_objects": _spec("todoist", "Todoist", "reorder_object"),
    "mcp__todoist__reschedule_tasks": _spec("todoist", "Todoist", "reschedule_task"),
    "mcp__todoist__uncomplete_tasks": _spec("todoist", "Todoist", "uncomplete_task"),
    "mcp__todoist__update_comments": _spec("todoist", "Todoist", "update_comment"),
    "mcp__todoist__update_filters": _spec("todoist", "Todoist", "update_filter"),
    "mcp__todoist__update_labels": _spec("todoist", "Todoist", "update_label"),
    "mcp__todoist__update_projects": _spec("todoist", "Todoist", "update_project"),
    "mcp__todoist__update_reminders": _spec("todoist", "Todoist", "update_reminder"),
    "mcp__todoist__update_sections": _spec("todoist", "Todoist", "update_section"),
    "mcp__todoist__update_tasks": _spec("todoist", "Todoist", "update_task"),
    "mcp_obsidian_create_note": _spec("note", "Obsidian", "create_note", one_shot=True),
    "mcp_apple_notes_create_note": _spec("note", "Apple Notes", "create_note", one_shot=True),
    "mcp_google_docs_create_document": _spec("note", "Google Docs", "create_note", one_shot=True),
    "mcp__obsidian__create_note": _spec("note", "Obsidian", "create_note", one_shot=True),
    "mcp__apple_notes__create_note": _spec("note", "Apple Notes", "create_note", one_shot=True),
    "mcp__google_docs__create_document": _spec("note", "Google Docs", "create_note", one_shot=True),
    "mcp__obsidian__append_note": _spec("note", "Obsidian", "append_note"),
    "mcp__obsidian__update_note": _spec("note", "Obsidian", "update_note"),
    "mcp__obsidian__delete_note": _spec("note", "Obsidian", "delete_note"),
    "mcp__apple_notes__append_note": _spec("note", "Apple Notes", "append_note"),
    "mcp__apple_notes__update_note": _spec("note", "Apple Notes", "update_note"),
    "mcp__apple_notes__delete_note": _spec("note", "Apple Notes", "delete_note"),
    "mcp__google_docs__update_document": _spec("note", "Google Docs", "update_note"),
    "mcp__google_docs__delete_document": _spec("note", "Google Docs", "delete_note"),
    "mcp__google_calendar__update_event": _spec("calendar", "Google Calendar", "update_event"),
    "mcp__google_calendar__delete_event": _spec("calendar", "Google Calendar", "delete_event"),
    "ha_call_service": _spec("home_automation", "Home Assistant", "call_service"),
    "mcp_homeassistant_call_service": _spec("home_automation", "Home Assistant", "call_service"),
    "mcp_home_assistant_call_service": _spec("home_automation", "Home Assistant", "call_service"),
    "mcp__homeassistant__ha_call_service": _spec("home_automation", "Home Assistant", "call_service"),
    "mcp__homeassistant__call_service": _spec("home_automation", "Home Assistant", "call_service"),
    "mcp__homeassistant__ha_call_event": _spec("home_automation", "Home Assistant", "call_event"),
    "mcp__homeassistant__ha_bulk_control": _spec("home_automation", "Home Assistant", "bulk_control"),
    "mcp__homeassistant__ha_manage_app": _spec("home_automation", "Home Assistant", "manage_app", write_predicate=_home_assistant_app_write),
    "mcp__homeassistant__ha_manage_pipeline": _spec("home_automation", "Home Assistant", "manage_pipeline", write_action_arg="action", write_actions=frozenset({"create", "update", "set_preferred", "process"})),
    "mcp__homeassistant__ha_manage_blueprints": _spec("home_automation", "Home Assistant", "manage_blueprints", write_action_arg="action", write_actions=frozenset({"import", "save", "delete"})),
    "mcp__homeassistant__ha_manage_energy_prefs": _spec("home_automation", "Home Assistant", "manage_energy_preferences", write_action_arg="mode", write_actions=frozenset({"set", "add_device", "remove_device", "add_source"})),
    "mcp__homeassistant__ha_manage_hacs": _spec("home_automation", "Home Assistant", "manage_hacs", write_action_arg="action", write_actions=frozenset({"download", "remove", "add_repository", "update_information"})),
    "mcp__homeassistant__ha_manage_radio": _spec("home_automation", "Home Assistant", "manage_radio", write_action_arg="action", write_actions=frozenset({"include", "commission", "remove", "heal", "rebuild_routes", "reconfigure", "update_firmware", "provision", "restore_network", "change_channel", "hard_reset", "remove_fabric"})),
    "mcp__homeassistant__ha_manage_theme": _spec("home_automation", "Home Assistant", "manage_theme", write_action_arg="action", write_actions=frozenset({"set", "set_engine_theme"})),
    "mcp__homeassistant__ha_manage_updates": _spec("home_automation", "Home Assistant", "manage_updates", write_action_arg="action", write_actions=frozenset({"install", "skip", "unskip"})),
    "mcp__homeassistant__ha_manage_backup": _spec("home_automation", "Home Assistant", "manage_backup", write_action_arg="action", write_actions=frozenset({"create", "restore", "delete"})),
    "mcp__homeassistant__ha_set_config": _spec("home_automation", "Home Assistant", "set_config"),
    "mcp__homeassistant__ha_remove_config": _spec("home_automation", "Home Assistant", "remove_config"),
    "mcp__homeassistant__ha_reload": _spec("home_automation", "Home Assistant", "reload"),
    "mcp__homeassistant__ha_reload_core": _spec("home_automation", "Home Assistant", "reload_core"),
    "mcp__homeassistant__ha_restart": _spec("home_automation", "Home Assistant", "restart"),
    "mcp__homeassistant__ha_config_delete_dashboard": _spec("home_automation", "Home Assistant", "delete_dashboard"),
    "mcp__homeassistant__ha_config_delete_dashboard_resource": _spec("home_automation", "Home Assistant", "delete_dashboard_resource"),
    "mcp__homeassistant__ha_config_remove_automation": _spec("home_automation", "Home Assistant", "remove_automation"),
    "mcp__homeassistant__ha_config_remove_calendar_event": _spec("home_automation", "Home Assistant", "remove_calendar_event"),
    "mcp__homeassistant__ha_config_remove_category": _spec("home_automation", "Home Assistant", "remove_category"),
    "mcp__homeassistant__ha_config_remove_group": _spec("home_automation", "Home Assistant", "remove_group"),
    "mcp__homeassistant__ha_config_remove_label": _spec("home_automation", "Home Assistant", "remove_label"),
    "mcp__homeassistant__ha_config_remove_scene": _spec("home_automation", "Home Assistant", "remove_scene"),
    "mcp__homeassistant__ha_config_remove_script": _spec("home_automation", "Home Assistant", "remove_script"),
    "mcp__homeassistant__ha_config_set_automation": _spec("home_automation", "Home Assistant", "set_automation"),
    "mcp__homeassistant__ha_config_set_calendar_event": _spec("home_automation", "Home Assistant", "set_calendar_event"),
    "mcp__homeassistant__ha_config_set_category": _spec("home_automation", "Home Assistant", "set_category"),
    "mcp__homeassistant__ha_config_set_dashboard": _spec("home_automation", "Home Assistant", "set_dashboard"),
    "mcp__homeassistant__ha_config_set_dashboard_resource": _spec("home_automation", "Home Assistant", "set_dashboard_resource"),
    "mcp__homeassistant__ha_config_set_group": _spec("home_automation", "Home Assistant", "set_group"),
    "mcp__homeassistant__ha_config_set_helper": _spec("home_automation", "Home Assistant", "set_helper"),
    "mcp__homeassistant__ha_config_set_label": _spec("home_automation", "Home Assistant", "set_label"),
    "mcp__homeassistant__ha_config_set_scene": _spec("home_automation", "Home Assistant", "set_scene"),
    "mcp__homeassistant__ha_config_set_script": _spec("home_automation", "Home Assistant", "set_script"),
    "mcp__homeassistant__ha_config_set_yaml": _spec("home_automation", "Home Assistant", "set_yaml"),
    "mcp__homeassistant__ha_remove_area_or_floor": _spec("home_automation", "Home Assistant", "remove_area_or_floor"),
    "mcp__homeassistant__ha_remove_device": _spec("home_automation", "Home Assistant", "remove_device"),
    "mcp__homeassistant__ha_remove_entity": _spec("home_automation", "Home Assistant", "remove_entity"),
    "mcp__homeassistant__ha_remove_helpers_integrations": _spec("home_automation", "Home Assistant", "remove_helpers_integrations"),
    "mcp__homeassistant__ha_remove_todo_item": _spec("home_automation", "Home Assistant", "remove_todo_item"),
    "mcp__homeassistant__ha_remove_zone": _spec("home_automation", "Home Assistant", "remove_zone"),
    "mcp__homeassistant__ha_set_area_or_floor": _spec("home_automation", "Home Assistant", "set_area_or_floor"),
    "mcp__homeassistant__ha_set_device": _spec("home_automation", "Home Assistant", "set_device"),
    "mcp__homeassistant__ha_set_entity": _spec("home_automation", "Home Assistant", "set_entity"),
    "mcp__homeassistant__ha_set_integration": _spec("home_automation", "Home Assistant", "set_integration"),
    "mcp__homeassistant__ha_set_todo_item": _spec("home_automation", "Home Assistant", "set_todo_item"),
    "mcp__homeassistant__ha_set_zone": _spec("home_automation", "Home Assistant", "set_zone"),
    "mcp_opnsense_apply_firewall_rule": _spec("network", "Firewall", "apply_rule"),
    "mcp_opnsense_update_firewall_rule": _spec("network", "Firewall", "update_rule"),
    "mcp_unifi_update_firewall_rule": _spec("network", "Firewall", "update_rule"),
    "mcp__opnsense__apply_firewall_rule": _spec("network", "Firewall", "apply_rule"),
    "mcp__opnsense__update_firewall_rule": _spec("network", "Firewall", "update_rule"),
    "mcp__unifi__update_firewall_rule": _spec("network", "Firewall", "update_rule"),
}


def reviewed_mutation_spec(tool_name: str) -> MutationSpec | None:
    return REVIEWED_MUTATIONS.get(str(tool_name).strip().casefold())


def _resolve_spec(
    function_name: str,
    *,
    registry: Any | None,
) -> MutationSpec | None:
    reviewed = reviewed_mutation_spec(function_name)
    if reviewed is not None:
        return reviewed
    active_registry: Any | None = registry
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
    return None


def _event_key(
    *,
    function_name: str,
    session_id: str,
    turn_id: str,
    tool_call_id: str,
    api_request_id: str,
) -> UUID:
    override = _EVENT_KEY_OVERRIDE.get()
    if override is not None:
        return override
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


@contextmanager
def action_event_key_override(key: UUID):
    """Bind the authenticated one-shot key to the common tool observer.

    One-shot execution uses the same tool path as a normal turn.  The
    request key therefore has to flow through that path, otherwise the
    observer would create a second journal row under its session/tool-call
    identity.  Context-local state keeps concurrent requests isolated.
    """
    token = _EVENT_KEY_OVERRIDE.set(key)
    try:
        yield
    finally:
        _EVENT_KEY_OVERRIDE.reset(token)


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
    if not spec.allows_arguments(arguments):
        return None
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
        action_type=MutationType(spec.action_type),
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
    "action_event_key_override",
]
