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
VALID_SCOPES = {"global", "scratch", "project"}
VALID_MODES = {"chat", "council", "project", "scratch"}
MODE_POLICIES = {
    "chat": {"scope": "global", "real_agents": "none_or_minimal", "description": "Fast global conversation; short role-aware T2 notes only."},
    "council": {"scope": "global_or_project", "real_agents": "recommended", "description": "Serious multi-perspective decision; collect Coda/Clara/Mira/Nova opinions and synthesize as Hugo."},
    "project": {"scope": "project", "real_agents": "as_needed", "description": "Concrete project work with files/tests/browser checks; target project/path required."},
    "scratch": {"scope": "scratch", "real_agents": "none_or_minimal", "description": "Idea development before a project folder exists; planning/risk/ops perspectives."},
}
COUNCIL_TRIGGERS = ("council", "카운슬", "회의", "여러 에이전트", "다 의견", "의견 받고", "coda/clara/mira/nova", "코다", "클라라")
SCRATCH_TRIGGERS = ("scratch", "스크래치", "아이디어", "폴더 없이", "구조만 논의", "새 프로젝트")
PROJECT_TRIGGERS = ("구현", "수정", "테스트", "고쳐", "repo", "레포", "프로젝트", "파일", "브라우저", "배포", "/users/", "~/", "shopping-crawler")
CHAT_TRIGGERS = ("어떻게 생각", "추천", "이해했", "말해줘", "설명", "판단해")


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
        "projects": hub / "projects.json",
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
    if not paths["projects"].exists():
        _write_json(paths["projects"], {})


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


def _load_context() -> Dict[str, Any]:
    _ensure_hub()
    ctx = _read_json(_paths()["context"], {})
    if not isinstance(ctx, dict):
        ctx = {}
    ctx.setdefault("scope", "global")
    ctx.setdefault("mode", "chat")
    ctx.setdefault("project_name", None)
    ctx.setdefault("project_path", None)
    return ctx


def _save_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    ctx = {**ctx, "set_at": ctx.get("set_at") or _now_iso()}
    _write_json(_paths()["context"], ctx)
    _append_event("context", ctx)
    return ctx


def _load_current_project() -> Optional[Dict[str, Any]]:
    data = _read_json(_paths()["current_project"], None)
    return data if isinstance(data, dict) else None


def _save_current_project(name: str, path: str, source: str) -> Dict[str, Any]:
    rec = {"project_name": name, "project_path": path, "set_at": _now_iso(), "source": source}
    _write_json(_paths()["current_project"], rec)
    _append_event("current_project", rec)
    return rec


def _resolve_project(value: str) -> tuple[str, str]:
    if value == "active":
        active = _load_current_project()
        if active and active.get("project_path"):
            return str(active.get("project_name") or Path(str(active["project_path"])).name), str(active["project_path"])
        raise ValueError("active project is not set")
    projects = _read_json(_paths()["projects"], {})
    if isinstance(projects, dict) and value in projects:
        return value, str(projects[value])
    path_obj = Path(value).expanduser()
    if path_obj.exists():
        return path_obj.name, str(path_obj.resolve())
    raise ValueError(f"unknown project alias or missing path: {value}")


def _recommended_action(mode: str, needs_project: bool = False) -> str:
    if needs_project:
        return "ask_for_target_project"
    if mode == "chat":
        return "answer_directly_and_optionally_emit_short_role_notes"
    if mode == "council":
        return "gather_coda_clara_mira_nova_opinions_then_synthesize"
    if mode == "project":
        return "use_project_path_and_emit_progress_at_stage_boundaries"
    if mode == "scratch":
        return "develop_idea_without_creating_project_until_approved"
    return "answer_directly"


def classify_request(text: str, *, explicit_project: Optional[str] = None, active_project: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw = text or ""
    lower = raw.lower()
    active_project = active_project if active_project is not None else _load_current_project()
    mode, reason, confidence = "chat", "default_chat", 0.55
    if any(t in lower for t in COUNCIL_TRIGGERS):
        mode, reason, confidence = "council", "explicit_council_trigger", 0.9
    elif any(t in lower for t in SCRATCH_TRIGGERS):
        mode, reason, confidence = "scratch", "scratch_or_ideation_trigger", 0.82
    elif explicit_project or any(t in lower for t in PROJECT_TRIGGERS):
        mode, reason, confidence = "project", "project_work_trigger", 0.78
    elif any(t in lower for t in CHAT_TRIGGERS):
        mode, reason, confidence = "chat", "short_chat_trigger", 0.72

    scope = "scratch" if mode == "scratch" else "global"
    project_name = project_path = None
    needs_project = False
    if mode == "project":
        scope = "project"
        if explicit_project:
            try:
                project_name, project_path = _resolve_project(explicit_project)
            except Exception:
                needs_project = True
        elif active_project and active_project.get("project_path"):
            project_name = str(active_project.get("project_name") or Path(str(active_project.get("project_path"))).name)
            project_path = str(active_project.get("project_path"))
        else:
            needs_project = True
    elif mode == "council":
        if explicit_project:
            try:
                project_name, project_path = _resolve_project(explicit_project)
                scope = "project"
            except Exception:
                needs_project = True
        elif active_project and any(t in lower for t in PROJECT_TRIGGERS):
            scope = "project"
            project_name = str(active_project.get("project_name") or Path(str(active_project.get("project_path"))).name)
            project_path = str(active_project.get("project_path"))
    return {"mode": mode, "scope": scope, "reason": reason, "confidence": confidence, "needs_project": needs_project, "project_name": project_name, "project_path": project_path, "policy": MODE_POLICIES[mode], "recommended_action": _recommended_action(mode, needs_project)}


def set_mode(mode: str, *, project: Optional[str] = None, source: str = "wave_set_mode") -> Dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(VALID_MODES))}")
    if mode == "project":
        name, path = _resolve_project(project or "active")
        _save_current_project(name, path, source)
        return _save_context({"scope": "project", "mode": "project", "project_name": name, "project_path": path, "source": source})
    if mode == "council" and project:
        name, path = _resolve_project(project)
        return _save_context({"scope": "project", "mode": "council", "project_name": name, "project_path": path, "source": source})
    scope = "scratch" if mode == "scratch" else "global"
    return _save_context({"scope": scope, "mode": mode, "project_name": None, "project_path": None, "source": source})


def apply_route(text: str, *, project: Optional[str] = None, source: str = "wave_route_request", emit_notes: bool = False) -> Dict[str, Any]:
    route = classify_request(text, explicit_project=project)
    ctx = _load_context()
    if not route["needs_project"]:
        if route["mode"] in {"project", "council"} and route.get("project_path"):
            ctx = _save_context({"scope": route["scope"], "mode": route["mode"], "project_name": route["project_name"], "project_path": route["project_path"], "source": source})
            if route["mode"] == "project":
                _save_current_project(str(route["project_name"]), str(route["project_path"]), source)
        else:
            ctx = set_mode(str(route["mode"]), source=source)
    if emit_notes:
        emit_mode_notes(route)
    return {"route": route, "context": ctx}


def emit_mode_notes(route: Dict[str, Any]) -> None:
    mode = str(route.get("mode") or "chat")
    scope = str(route.get("scope") or "global")
    if mode == "chat":
        notes = {"Coda": "Implementation view: direct answer; no code work yet.", "Clara": "Review view: no four-subagent call unless decision risk is high.", "Mira": "Context view: global chat, no project required.", "Nova": "Ops view: short role notes only."}
    elif mode == "council":
        notes = {r: "Council mode ready: waiting for meaningful role opinion." for r in ROLES}
    elif mode == "project":
        proj = route.get("project_name") or "target project required"
        notes = {"Coda": f"Project implementation ready: {proj}", "Clara": f"Project tests/regression ready: {proj}", "Mira": f"Project requirements/context ready: {proj}", "Nova": f"Project operations/progress tracking ready: {proj}"}
    else:
        notes = {"Coda": "Scratch build view: compare structures before creating files.", "Clara": "Scratch risk view: check complexity and failure modes.", "Mira": "Scratch planning view: define user value and problem.", "Nova": "Scratch ops view: identify when to turn this into a project."}
    for role, message in notes.items():
        append_progress(role, message, kind="memo", status="noted", task=f"{mode}-mode", source="mode-router", mode=mode, scope=scope)


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
        "context": _load_context(),
        "active_project": _load_current_project(),
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


def wave_set_mode_handler(args: Dict[str, Any], **_: Any) -> str:
    try:
        data = set_mode(str(args.get("mode") or "chat"), project=(str(args.get("project")) if args.get("project") else None))
        return json.dumps({"success": True, "data": data}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


def wave_route_request_handler(args: Dict[str, Any], **_: Any) -> str:
    try:
        data = apply_route(str(args.get("text") or args.get("message") or ""), project=(str(args.get("project")) if args.get("project") else None), emit_notes=bool(args.get("emit_notes")))
        return json.dumps({"success": True, "data": data}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


def wave_council_note_handler(args: Dict[str, Any], **_: Any) -> str:
    try:
        ctx = set_mode("council", project=(str(args.get("project")) if args.get("project") else None), source=str(args.get("source") or "wave_council_note"))
        emitted = []
        for role, key in (("Coda", "coda"), ("Clara", "clara"), ("Mira", "mira"), ("Nova", "nova")):
            if args.get(key):
                append_progress(role, str(args[key]), kind="agent", status="opinion", task=str(args.get("topic") or "council"), source=str(args.get("source") or "wave_council_note"), mode="council", scope=ctx.get("scope", "global"))
                emitted.append(role)
        return json.dumps({"success": True, "data": {"context": ctx, "emitted_roles": emitted}}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


def _pre_llm_call(user_message: str = "", platform: str = "", **_: Any) -> Optional[Dict[str, str]]:
    try:
        payload = apply_route(user_message or "", emit_notes=False)
        route = payload.get("route", {})
        mode = route.get("mode", "chat")
        scope = route.get("scope", "global")
        if mode in {"chat", "scratch"}:
            emit_mode_notes(route)
        context = (
            "Wave role-board routing for this turn:\n"
            f"- mode: {mode}\n"
            f"- scope: {scope}\n"
            f"- reason: {route.get('reason')}\n"
            f"- needs_project: {route.get('needs_project')}\n"
            f"- recommended_action: {route.get('recommended_action')}\n"
            "Rules: chat/scratch mode should avoid automatic four-subagent calls and only emit short T2 role notes; "
            "council mode should gather meaningful Coda/Clara/Mira/Nova opinions and show them in T2; "
            "project mode must use an explicit or active project path and ask if ambiguous."
        )
        return {"context": context}
    except Exception:
        return None


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
    ctx.register_tool(
        name="wave_set_mode",
        toolset="wave_role_board",
        schema={"name": "wave_set_mode", "description": "Set Wave T2 interaction mode: chat, council, project, or scratch.", "parameters": {"type": "object", "properties": {"mode": {"type": "string", "enum": sorted(VALID_MODES)}, "project": {"type": "string"}}, "required": ["mode"]}},
        handler=wave_set_mode_handler,
        description="Set Wave mode",
        emoji="🎛️",
    )
    ctx.register_tool(
        name="wave_route_request",
        toolset="wave_role_board",
        schema={"name": "wave_route_request", "description": "Classify a request into chat/council/project/scratch mode and optionally emit short T2 notes.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "project": {"type": "string"}, "emit_notes": {"type": "boolean", "default": False}}, "required": ["text"]}},
        handler=wave_route_request_handler,
        description="Route Wave request mode",
        emoji="🧭",
    )
    ctx.register_tool(
        name="wave_council_note",
        toolset="wave_role_board",
        schema={"name": "wave_council_note", "description": "Emit council-mode Coda/Clara/Mira/Nova opinions to T2 after real or synthesized consultation.", "parameters": {"type": "object", "properties": {"topic": {"type": "string"}, "project": {"type": "string"}, "coda": {"type": "string"}, "clara": {"type": "string"}, "mira": {"type": "string"}, "nova": {"type": "string"}, "source": {"type": "string"}}}},
        handler=wave_council_note_handler,
        description="Emit council opinions",
        emoji="🧑‍⚖️",
    )
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("pre_llm_call", _pre_llm_call)
