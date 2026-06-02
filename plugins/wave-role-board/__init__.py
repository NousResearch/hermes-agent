"""Wave role-board plugin.

This bundled plugin provides a lightweight, profile-aware event hub for Wave
Terminal role dashboards. It intentionally does not require Wave to be running:
messages are appended to ``$HERMES_HOME/wave-hub/messages.jsonl`` and any
viewer can render them.

Tools registered:
- wave_progress: append a role progress/status/log/memo event
- wave_board_status: inspect latest role status and viewer process state
- wave_board_restore: run ``$HERMES_HOME/wave-hub/restore_wave.sh`` if present
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROLES = ("Coda", "Clara", "Mira", "Nova")
ROLE_ALIASES = {r.lower(): r for r in ROLES}
ROLE_ALIASES.update({
    "codex": "Coda",
    "claude": "Clara",
    "research": "Mira",
    "ops": "Nova",
    "hugo": "Nova",
    "hermes": "Nova",
})
VALID_KINDS = {"progress", "agent", "log", "memo", "status", "system"}


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except Exception:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).expanduser()


def _hub() -> Path:
    return Path(os.environ.get("WAVE_HUB_HOME", _hermes_home() / "wave-hub")).expanduser()


def _now_time() -> str:
    return datetime.now().astimezone().strftime("%H:%M:%S")


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _role(value: str) -> str:
    role = ROLE_ALIASES.get((value or "").strip().lower())
    if role not in ROLES:
        raise ValueError(f"role must be one of: {', '.join(ROLES)}")
    return role


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _paths() -> Dict[str, Path]:
    hub = _hub()
    return {
        "hub": hub,
        "messages": hub / "messages.jsonl",
        "events": hub / "events.jsonl",
        "agents": hub / "agents.json",
        "context": hub / "current_context.json",
        "current_project": hub / "current_project.json",
        "restore": hub / "restore_wave.sh",
        "role_view": hub / "role_view.py",
    }


def _ensure_hub() -> None:
    paths = _paths()
    paths["hub"].mkdir(parents=True, exist_ok=True)
    if not paths["agents"].exists():
        _write_json(paths["agents"], {
            role: {"status": "idle", "task": None, "message": "", "updated_at": None}
            for role in ROLES
        })
    if not paths["context"].exists():
        _write_json(paths["context"], {
            "scope": "global",
            "mode": "chat",
            "project_name": None,
            "project_path": None,
            "set_at": _now_iso(),
            "source": "wave-role-board",
        })


def _append_event(kind: str, payload: Dict[str, Any]) -> None:
    _ensure_hub()
    paths = _paths()
    rec = {"ts": _now_time(), "at": _now_iso(), "kind": kind, **payload}
    with paths["events"].open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_agents() -> Dict[str, Any]:
    _ensure_hub()
    agents = _read_json(_paths()["agents"], {})
    if not isinstance(agents, dict):
        agents = {}
    for role in ROLES:
        agents.setdefault(role, {"status": "idle", "task": None, "message": "", "updated_at": None})
    return agents


def _update_agent(role: str, *, status: Optional[str], task: Optional[str], message: str, source: str) -> Dict[str, Any]:
    agents = _load_agents()
    rec = agents.setdefault(role, {})
    if status:
        rec["status"] = status
    elif rec.get("status") in {None, "idle"}:
        rec["status"] = "running"
    if task is not None:
        rec["task"] = task or None
    rec["message"] = message
    rec["source"] = source
    rec["updated_at"] = _now_iso()
    agents[role] = rec
    _write_json(_paths()["agents"], agents)
    return rec


def append_progress(
    role: str,
    message: str,
    *,
    kind: str = "progress",
    status: Optional[str] = None,
    task: Optional[str] = None,
    source: str = "wave-role-board",
    mode: Optional[str] = None,
    scope: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _ensure_hub()
    role = _role(role)
    if kind not in VALID_KINDS:
        kind = "progress"
    context = _read_json(_paths()["context"], {})
    rec = {
        "ts": _now_time(),
        "at": _now_iso(),
        "role": role,
        "text": message,
        "kind": kind,
        "status": status,
        "task": task,
        "source": source,
        "mode": mode or context.get("mode") or "chat",
        "scope": scope or context.get("scope") or "global",
        "project_name": context.get("project_name"),
        "project_path": context.get("project_path"),
        "metadata": metadata or {},
    }
    with _paths()["messages"].open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _update_agent(role, status=status, task=task, message=message, source=source)
    _append_event("message", {"role": role, "kind": kind, "status": status, "task": task, "source": source})
    return rec


def _viewer_processes() -> list[Dict[str, Any]]:
    try:
        out = subprocess.check_output(["/bin/ps", "-axo", "pid=,command="], text=True, errors="replace")
    except Exception:
        return []
    role_view = str(_paths()["role_view"])
    rows: list[Dict[str, Any]] = []
    for line in out.splitlines():
        if role_view not in line:
            continue
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        cmd = parts[1] if len(parts) > 1 else ""
        role = next((r for r in ROLES if f" {r}" in cmd or cmd.endswith(r)), None)
        rows.append({"pid": int(parts[0]), "role": role, "command": cmd})
    return rows


def board_status() -> Dict[str, Any]:
    _ensure_hub()
    procs = _viewer_processes()
    roles_running = sorted({p["role"] for p in procs if p.get("role")})
    return {
        "hub": str(_paths()["hub"]),
        "agents": _load_agents(),
        "viewer_processes": procs,
        "roles_running": roles_running,
        "all_roles_running": all(r in roles_running for r in ROLES),
        "messages_path": str(_paths()["messages"]),
        "restore_script": str(_paths()["restore"]),
        "restore_available": _paths()["restore"].exists(),
    }


def restore_board(force: bool = False) -> Dict[str, Any]:
    status = board_status()
    if status["all_roles_running"] and not force:
        return {"restored": False, "reason": "all viewers running", "status": status}
    restore = _paths()["restore"]
    if not restore.exists():
        return {"restored": False, "reason": f"missing {restore}", "status": status}
    proc = subprocess.run(["/bin/bash", str(restore)], text=True, capture_output=True, timeout=120)
    return {
        "restored": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
        "status_before": status,
        "status_after": board_status(),
    }


def wave_progress_handler(args: Dict[str, Any], **_: Any) -> str:
    message = str(args.get("message") or args.get("text") or "")
    if not message:
        return json.dumps({"success": False, "error": "message is required"}, ensure_ascii=False)
    try:
        rec = append_progress(
            str(args.get("role") or "Nova"),
            message,
            kind=str(args.get("kind") or "progress"),
            status=(str(args.get("status")) if args.get("status") is not None else None),
            task=(str(args.get("task")) if args.get("task") is not None else None),
            source=str(args.get("source") or "wave_progress"),
            mode=(str(args.get("mode")) if args.get("mode") else None),
            scope=(str(args.get("scope")) if args.get("scope") else None),
        )
        return json.dumps({"success": True, "data": rec}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


def wave_board_status_handler(args: Dict[str, Any], **_: Any) -> str:
    return json.dumps({"success": True, "data": board_status()}, ensure_ascii=False)


def wave_board_restore_handler(args: Dict[str, Any], **_: Any) -> str:
    return json.dumps({"success": True, "data": restore_board(force=bool(args.get("force")))}, ensure_ascii=False)


def _summarize_tool(tool_name: str, args: Dict[str, Any]) -> tuple[str, str] | None:
    if tool_name == "delegate_task":
        return "Coda", "delegate_task started: launching subagent work."
    if tool_name == "terminal":
        cmd = str(args.get("command") or "").strip().replace("\n", " ")
        if not cmd:
            return None
        lower = cmd.lower()
        if "codex" in lower:
            return "Coda", f"Codex/Coda command started: {cmd[:160]}"
        if "claude" in lower:
            return "Clara", f"Claude/Clara command started: {cmd[:160]}"
        if "kanban" in lower:
            return "Nova", f"Kanban command started: {cmd[:160]}"
        return "Nova", f"Terminal command started: {cmd[:160]}"
    if tool_name in {"write_file", "patch"}:
        path = args.get("path") or ("multi-file patch" if args.get("mode") == "patch" else "unknown")
        return "Coda", f"File change: {path}"
    if tool_name in {"read_file", "search_files"}:
        return "Mira", f"Inspection/search: {tool_name}"
    return None


def _pre_tool_call(tool_name: str = "", args: Optional[Dict[str, Any]] = None, **_: Any) -> None:
    try:
        summarized = _summarize_tool(tool_name, args or {})
        if summarized:
            role, msg = summarized
            append_progress(role, msg, kind="agent", status="running", task=tool_name, source="hook:pre_tool_call")
    except Exception:
        pass


def _post_tool_call(tool_name: str = "", args: Optional[Dict[str, Any]] = None, result: Any = None, **_: Any) -> None:
    try:
        if tool_name not in {"delegate_task", "terminal", "write_file", "patch", "read_file", "search_files"}:
            return
        role = "Clara" if tool_name in {"terminal", "delegate_task"} else "Coda"
        rendered = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        status = "needs_review" if any(s in rendered.lower() for s in ("error", "traceback", "failed")) else "done"
        append_progress(role, f"{tool_name} completed: {status}", kind="status", status=status, task=tool_name, source="hook:post_tool_call")
    except Exception:
        pass


def _on_session_start(**_: Any) -> None:
    try:
        append_progress("Nova", "Hermes session started: Wave role-board plugin active.", kind="system", status="running", task="session", source="hook:on_session_start")
    except Exception:
        pass


def _on_session_end(completed: bool = True, interrupted: bool = False, **_: Any) -> None:
    try:
        status = "done" if completed and not interrupted else "interrupted"
        append_progress("Nova", f"Hermes session ended: {status}", kind="system", status=status, task="session", source="hook:on_session_end")
    except Exception:
        pass


def register(ctx) -> None:
    ctx.register_tool(
        name="wave_progress",
        toolset="wave_role_board",
        schema={
            "name": "wave_progress",
            "description": "Append a progress/status/log/memo message to the Wave Terminal T2 role board for Coda, Clara, Mira, or Nova.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": list(ROLES)},
                    "message": {"type": "string"},
                    "kind": {"type": "string", "enum": sorted(VALID_KINDS), "default": "progress"},
                    "status": {"type": "string"},
                    "task": {"type": "string"},
                    "source": {"type": "string"},
                    "mode": {"type": "string"},
                    "scope": {"type": "string"},
                },
                "required": ["role", "message"],
            },
        },
        handler=wave_progress_handler,
        description="Emit progress to Wave T2 role board",
        emoji="📟",
    )
    ctx.register_tool(
        name="wave_board_status",
        toolset="wave_role_board",
        schema={"name": "wave_board_status", "description": "Inspect Wave T2 role-board viewer and agent status.", "parameters": {"type": "object", "properties": {}}},
        handler=wave_board_status_handler,
        description="Inspect Wave T2 role board",
        emoji="📊",
    )
    ctx.register_tool(
        name="wave_board_restore",
        toolset="wave_role_board",
        schema={"name": "wave_board_restore", "description": "Restore the Wave T2 four-pane role board if a restore script is available.", "parameters": {"type": "object", "properties": {"force": {"type": "boolean", "default": False}}}},
        handler=wave_board_restore_handler,
        description="Restore Wave T2 role board",
        emoji="🛠️",
    )
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)
