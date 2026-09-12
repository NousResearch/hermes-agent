"""Permission-filtered references over existing Hermes state owners.

References are coordinates, not capabilities. Every detailed lookup reopens the
owning store read-only and repeats the authority check that made the reference
visible. No discovery path initializes schemas, recovers leases, consumes a
queue, or changes readiness.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Optional


_UNKNOWN = "Unknown or unavailable discovery reference."


@dataclass(frozen=True)
class SharedDiscoveryScope:
    """Session-frozen local coordinates; room authority remains gateway-owned."""

    profile_name: str
    profile_home: str
    kanban_board: str
    kanban_db_path: str
    kanban_task_id: str = ""
    room_provider: Any = None


def build_local_discovery_scope(*, room_provider: Any = None) -> SharedDiscoveryScope:
    """Freeze profile and board coordinates without opening their stores."""
    from hermes_constants import get_hermes_home
    from hermes_cli.kanban_db import get_current_board, kanban_db_path
    from hermes_cli.profiles import get_active_profile_name

    home = Path(get_hermes_home()).resolve()
    try:
        profile = get_active_profile_name() or "default"
    except Exception:
        profile = "default"
    board = get_current_board()
    return SharedDiscoveryScope(
        profile_name=str(profile), profile_home=str(home), kanban_board=str(board),
        kanban_db_path=str(kanban_db_path(board=board).resolve()),
        kanban_task_id=str(os.environ.get("HERMES_KANBAN_TASK") or ""),
        room_provider=room_provider,
    )


def _scope(agent: Any) -> SharedDiscoveryScope:
    scope = getattr(agent, "_shared_discovery_scope", None)
    if isinstance(scope, SharedDiscoveryScope):
        return scope
    scope = build_local_discovery_scope()
    agent._shared_discovery_scope = scope
    return scope


def _worker_discovery(agent: Any, reference: Optional[str]) -> Mapping[str, Any]:
    from agent.subagent_lifecycle import SubagentLifecycleService
    return SubagentLifecycleService(lambda: agent).discover_readonly(reference)


def _bot_references(agent: Any, reference: Optional[str]) -> Mapping[str, Any]:
    from tools.bot_mode_dm import message_agent_authorized, _resolve_local_name
    if not message_agent_authorized(agent):
        if reference:
            raise PermissionError(_UNKNOWN)
        return {"references": []}
    from tools.bot_mode_probe import _handle, _hermes_root, _profile_name, _roster

    home = Path(getattr(agent, "_shared_discovery_scope").profile_home)
    roster = dict(_roster(_hermes_root(home)))
    me = _profile_name(home)

    def item(name: str) -> Mapping[str, Any]:
        handle = _handle(name)
        return {
            "reference": f"bot:{handle}", "kind": "bot", "label": f"@{handle}",
            "availability": "available", "freshness": "live",
            "actions": ["message"],
            "scope": {"kind": "profile_roster", "profile": _scope(agent).profile_name},
        }

    if reference:
        raw = reference.partition(":")[2]
        resolved = _resolve_local_name(raw, list(roster)) if raw else None
        if not resolved or resolved == me:
            raise PermissionError(_UNKNOWN)
        return {"reference": item(resolved)}
    return {"references": [item(name) for name in roster if name != me]}


def _task_access(agent: Any, scope: SharedDiscoveryScope) -> tuple[bool, set[str]]:
    tools = set(getattr(agent, "_executable_tool_names", ()) or ())
    if "kanban_list" in tools:
        return True, tools
    return bool("kanban_show" in tools and scope.kanban_task_id), tools


def _task_references(agent: Any, reference: Optional[str]) -> Mapping[str, Any]:
    scope = _scope(agent)
    allowed, tools = _task_access(agent, scope)
    if not allowed:
        if reference:
            raise PermissionError(_UNKNOWN)
        return {"references": []}
    from hermes_cli.kanban_db import get_current_board, get_task, kanban_db_path, list_tasks
    from hermes_cli.kanban_db_connect import connect_existing_readonly

    # Board is a session capability: config/env changes require a new session.
    current_board = str(get_current_board())
    current_path = str(kanban_db_path(board=current_board).resolve())
    if current_board != scope.kanban_board or current_path != scope.kanban_db_path:
        raise PermissionError(_UNKNOWN)
    conn = connect_existing_readonly(Path(scope.kanban_db_path))
    if conn is None:
        if reference:
            raise PermissionError(_UNKNOWN)
        return {"references": []}

    def item(task: Any) -> Mapping[str, Any]:
        actions = ["inspect"] if "kanban_show" in tools else []
        return {
            "reference": f"task:{task.id}", "kind": "task", "label": str(task.title),
            "availability": "available", "freshness": "stored", "actions": actions,
            "status": str(task.status), "assignee": task.assignee,
            "scope": {"kind": "kanban_board", "board": scope.kanban_board},
        }

    try:
        if reference:
            task_id = reference.partition(":")[2]
            if not task_id or (scope.kanban_task_id and "kanban_list" not in tools
                               and task_id != scope.kanban_task_id):
                raise PermissionError(_UNKNOWN)
            task = get_task(conn, task_id)
            if task is None:
                raise PermissionError(_UNKNOWN)
            return {"reference": item(task)}
        tasks = list_tasks(conn)
        if scope.kanban_task_id and "kanban_list" not in tools:
            tasks = [task for task in tasks if task.id == scope.kanban_task_id]
        return {"references": [item(task) for task in tasks]}
    finally:
        conn.close()


def _room_references(agent: Any, reference: Optional[str]) -> Mapping[str, Any]:
    provider = _scope(agent).room_provider
    if provider is None:
        if reference:
            raise PermissionError(_UNKNOWN)
        return {"references": []}
    return provider.resolve(agent, reference)


_RESOLVERS = {
    "worker": _worker_discovery,
    "run": _worker_discovery,
    "bot": _bot_references,
    "room": _room_references,
    "task": _task_references,
}


def discover_shared_references(agent: Any, reference: Optional[str] = None) -> Mapping[str, Any]:
    """List visible typed references or inspect one exact typed reference."""
    _scope(agent)
    if reference:
        kind, sep, object_id = str(reference).partition(":")
        resolver = _RESOLVERS.get(kind)
        if not sep or not object_id or resolver is None:
            raise PermissionError(_UNKNOWN)
        try:
            resolved = resolver(agent, str(reference))
            return {
                "reference_detail": resolved["reference"],
                **{key: value for key, value in resolved.items() if key != "reference"},
            }
        except (KeyError, PermissionError, ValueError):
            raise PermissionError(_UNKNOWN) from None

    references = []
    for resolver in (_worker_discovery, _bot_references, _room_references, _task_references):
        try:
            references.extend(resolver(agent, None).get("references", ()))
        except (PermissionError, ValueError):
            continue
    return {
        "references": references,
        "reference_types": ["worker", "run", "bot", "room", "task"],
    }
