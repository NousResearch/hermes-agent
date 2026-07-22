"""Mission Control dry-run preview CLI.

The first Mission Control slice is intentionally preview-only.  It turns a
small graph spec into a deterministic "would create" envelope that can be
reviewed before any Kanban tasks, cron jobs, subscriptions, or live sends are
created.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:  # PyYAML is an optional runtime dependency in some minimal installs.
    import yaml
except Exception:  # pragma: no cover - exercised only when yaml is absent
    yaml = None  # type: ignore[assignment]


class MissionSpecError(ValueError):
    """Raised when a mission spec cannot be safely previewed."""


def _read_spec_text(path: str | None) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        # Never surface the raw OSError: it embeds the local filesystem path
        # (and, for permission errors, can hint at what's on disk).
        raise MissionSpecError("mission spec file could not be read") from exc


def load_spec(path: str | None) -> dict[str, Any]:
    """Load a JSON/YAML mission spec from *path* or stdin.

    JSON is attempted first so the command works even without PyYAML.  YAML is
    accepted when the dependency is present.
    """

    text = _read_spec_text(path)
    if not text.strip():
        raise MissionSpecError("mission spec is empty")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise MissionSpecError("mission spec is not valid JSON and PyYAML is unavailable")
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            # yaml.YAMLError messages can echo back offending file content
            # (including secret-like values); keep the CLI-facing error generic.
            raise MissionSpecError("mission spec is not valid JSON/YAML") from exc

    if not isinstance(data, dict):
        raise MissionSpecError("mission spec must be a JSON/YAML object")
    return data


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise MissionSpecError(f"{name} must be an object")
    return value


def _as_list(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise MissionSpecError(f"{name} must be a list")
    return value


def _stringish(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    raise MissionSpecError("mission string fields must be strings")


def _task_preview(raw: Any, index: int) -> dict[str, Any]:
    if isinstance(raw, str):
        raw = {"title": raw}
    if not isinstance(raw, dict):
        raise MissionSpecError(f"task #{index + 1} must be an object or string")

    title = _stringish(raw.get("title") or raw.get("name") or raw.get("task"))
    if not title:
        raise MissionSpecError(f"task #{index + 1} is missing title/name/task")

    depends_on = raw.get("depends_on", raw.get("needs", []))
    if isinstance(depends_on, str):
        depends_on = [depends_on]
    if not isinstance(depends_on, list):
        raise MissionSpecError(f"task #{index + 1} depends_on must be a string or list")
    normalized_deps = []
    for dep in depends_on:
        try:
            normalized_dep = _stringish(dep)
        except MissionSpecError as exc:
            raise MissionSpecError(
                f"task #{index + 1} depends_on entries must be non-empty strings"
            ) from exc
        if not normalized_dep:
            raise MissionSpecError(f"task #{index + 1} depends_on entries must be non-empty strings")
        normalized_deps.append(normalized_dep)

    out: dict[str, Any] = {
        "index": index + 1,
        "title": title,
        "assignee": _stringish(raw.get("assignee") or raw.get("profile")),
        "depends_on": normalized_deps,
    }
    task_id = _stringish(raw.get("id"))
    if task_id:
        out["id"] = task_id
    body = _stringish(raw.get("body") or raw.get("description"))
    if body:
        out["body"] = body
    return out


def build_preview(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic dry-run mission preview.

    The accepted input is deliberately small and upstream-generic:

    - top-level ``mission`` object for metadata, or metadata at the root
    - ``graph``/``tasks`` list for task previews
    - optional ``ack`` and ``watchdog`` objects
    """

    mission = _as_mapping(spec.get("mission"), "mission")
    root = {**spec, **mission}
    graph = _as_list(root.get("graph", root.get("tasks")), "graph/tasks")
    if not graph:
        raise MissionSpecError("mission spec must include at least one graph/task item")

    origin = _stringish(root.get("origin"))
    return_to = _stringish(root.get("return_to") or root.get("returnTo") or origin)
    name = _stringish(root.get("name") or root.get("title") or root.get("objective")) or "unnamed mission"
    objective = _stringish(root.get("objective") or root.get("description")) or name

    ack = _as_mapping(root.get("ack"), "ack")
    watchdog = _as_mapping(root.get("watchdog"), "watchdog")
    task_previews = [_task_preview(item, index) for index, item in enumerate(graph)]
    final_task = _stringish(ack.get("final_task") or ack.get("task_id"))
    if final_task is None and task_previews:
        final_task = task_previews[-1].get("id") or task_previews[-1]["title"]

    return {
        "status": "mission_dry_run_preview",
        "sent": False,
        "created": False,
        "live_dispatch": False,
        "mission": {
            "name": name,
            "objective": objective,
            "origin": origin,
            "return_to": return_to,
        },
        "would_create_tasks": task_previews,
        "would_subscribe_final_ack": {
            "enabled": bool(return_to),
            "origin": origin,
            "return_to": return_to,
            "final_task": final_task,
            "verdict_schema": ack.get("verdict_schema", ["GO", "BLOCK", "NEED_MORE"]),
        },
        "would_create_watchdog": {
            "enabled": bool(watchdog),
            "schedule": watchdog.get("schedule"),
            "material_change_only": watchdog.get("material_change_only", True),
            "deliver": watchdog.get("deliver", return_to),
        },
        "safety": {
            "dry_run_only": True,
            "would_send": False,
            "would_write_kanban": False,
            "would_create_cron": False,
            "would_trigger_agent": False,
        },
    }


def mission_create(args: argparse.Namespace) -> int:
    if not getattr(args, "dry_run", False):
        print(
            "Error: mission create currently supports preview only; pass --dry-run.",
            file=sys.stderr,
        )
        return 2

    try:
        preview = build_preview(load_spec(getattr(args, "file", None)))
    except MissionSpecError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(preview, indent=2, sort_keys=True))
    else:
        mission = preview["mission"]
        print(f"Mission dry-run: {mission['name']}")
        print(f"Objective: {mission['objective']}")
        print(f"Origin: {mission.get('origin') or '-'}")
        print(f"Return-to: {mission.get('return_to') or '-'}")
        print("Would create tasks:")
        for task in preview["would_create_tasks"]:
            assignee = f" [{task['assignee']}]" if task.get("assignee") else ""
            deps = f" depends_on={task['depends_on']}" if task.get("depends_on") else ""
            print(f"  {task['index']}. {task['title']}{assignee}{deps}")
        ack = preview["would_subscribe_final_ack"]
        print(f"Would subscribe final ACK: {ack['enabled']} -> {ack.get('return_to') or '-'}")
        wd = preview["would_create_watchdog"]
        print(f"Would create watchdog: {wd['enabled']} schedule={wd.get('schedule') or '-'}")
    return 0


def mission_command(args: argparse.Namespace) -> int:
    command = getattr(args, "mission_command", None)
    if command == "create":
        return mission_create(args)
    print("usage: hermes mission create --dry-run --file SPEC", file=sys.stderr)
    return 2


def build_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "mission",
        help="Preview Mission Control task graphs and ACK/watchdog contracts",
        description="Mission Control preview wrapper for durable task graphs",
    )
    mission_subparsers = parser.add_subparsers(dest="mission_command")

    create = mission_subparsers.add_parser(
        "create",
        help="Preview a mission graph without creating tasks, subscriptions, or watchdogs",
    )
    create.add_argument("--dry-run", action="store_true", help="Preview only; required in M0")
    create.add_argument("--file", "-f", help="JSON/YAML mission spec path; omit or '-' for stdin")
    create.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.set_defaults(func=mission_command)
    return parser
