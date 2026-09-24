"""Shared Agent Roster helpers for role enforcement and dashboard APIs.

The plugin is intentionally read-mostly: profile roles are sourced from
``profile_role`` in each profile's config.yaml, Kanban drift is surfaced as
violations, and hook-time enforcement is driven by explicit role metadata.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from hermes_constants import get_default_hermes_root, get_hermes_home

PLUGIN_NAME = "agent-roster"
DEFAULT_ROLE_CONFIG_KEY = "profile_role"
DEFAULT_STRICT_MODE = "audit"
AUDIT_LOG_NAME = "agent-roster-audit.jsonl"

PIPELINE_STAGES: list[dict[str, Any]] = [
    {"key": "idea", "label": "Idea", "types": ["IDEA"], "profiles": ["default"]},
    {"key": "research", "label": "Research", "types": ["RESEARCH", "SOURCECHECK", "BENCHMARK"], "profiles": ["researcher"]},
    {"key": "analysis", "label": "Analysis", "types": ["ANALYSIS", "ANALYST"], "profiles": ["analyst"]},
    {"key": "spec", "label": "Spec / PM", "types": ["SPEC", "PM", "PRD"], "profiles": ["pm"]},
    {"key": "write", "label": "Writing / UX", "types": ["WRITE", "UX", "COPY"], "profiles": ["writer"]},
    {"key": "build", "label": "Build", "types": ["BUILD", "CODE", "DEV", "FRONTEND", "BACKEND"], "profiles": ["default", "frontend-eng", "backend-eng"]},
    {"key": "qa", "label": "QA / Review", "types": ["QA", "REVIEW", "TEST"], "profiles": ["reviewer"]},
    {"key": "launch", "label": "Launch", "types": ["LAUNCH", "DEPLOY", "OPS"], "profiles": ["default", "ops"]},
]

DEFAULT_PRD_REQUIRED_FIELDS: dict[str, list[str]] = {
    "goal": ["goal", "ziel"],
    "non_goals": ["non-goal", "non goal", "nicht-ziel", "nicht ziel", "nichtziele"],
    "target_user": ["target user", "user", "nutzer", "zielgruppe", "persona"],
    "acceptance_criteria": ["acceptance criteria", "akzeptanzkriterien", "akzeptanz"],
    "success_metric": ["success metric", "success metrics", "erfolgskriterium", "erfolgsmetrik"],
    "risks": ["risk", "risiko", "risiken"],
    "dependencies": ["dependency", "dependencies", "abhängigkeit", "abhaengigkeit", "abhängigkeiten", "abhaengigkeiten"],
    "definition_of_done": ["definition of done", "dod", "done-kriter", "fertig wenn"],
}

QUALITY_GATE_TYPES = {"BUILD", "CODE", "DEV", "FRONTEND", "BACKEND", "LAUNCH", "DEPLOY", "OPS"}
REVIEW_GATE_TYPES = QUALITY_GATE_TYPES


def _now() -> int:
    return int(time.time())


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _root() -> Path:
    return get_default_hermes_root()


def _active_home() -> Path:
    return get_hermes_home()


def current_profile_name() -> str:
    explicit = os.environ.get("HERMES_PROFILE", "").strip()
    if explicit:
        return explicit
    home = _active_home()
    if home.parent.name == "profiles":
        return home.name
    return "default"


def root_config(root: Optional[Path] = None) -> dict[str, Any]:
    return _read_yaml((root or _root()) / "config.yaml")


def roster_config(root: Optional[Path] = None) -> dict[str, Any]:
    cfg = root_config(root)
    dashboard = cfg.get("dashboard") if isinstance(cfg.get("dashboard"), dict) else {}
    raw = dashboard.get("agent_roster") if isinstance(dashboard.get("agent_roster"), dict) else {}
    out = {
        "enabled": True,
        "source": "profile_config",
        "role_config_key": DEFAULT_ROLE_CONFIG_KEY,
        "strict_mode": DEFAULT_STRICT_MODE,
        "check_kanban_assignments": True,
        "check_task_prefix_compatibility": True,
        "check_quality_gates": True,
        "show_profiles_without_roles": True,
        "inject_outside_kanban": False,
        "audit_enabled": True,
        "enforce_completion_gates": True,
    }
    out.update(raw)
    mode = str(out.get("strict_mode") or DEFAULT_STRICT_MODE).strip().lower()
    # Backward-compatible alias: early Agent Roster drafts used ``warn`` for
    # audit-only behavior.  The runtime has no user-visible warning channel for
    # pre-tool hooks, so expose the behavior honestly as ``audit`` instead of
    # silently pretending to warn.
    if mode == "warn":
        mode = "audit"
    if mode not in {"off", "audit", "block"}:
        mode = DEFAULT_STRICT_MODE
    out["strict_mode"] = mode
    return out


def is_enabled(root: Optional[Path] = None) -> bool:
    return roster_config(root).get("enabled") is not False


def _profile_home(root: Path, profile: str) -> Path:
    return root if profile == "default" else root / "profiles" / profile


def _iter_profile_homes(root: Path) -> list[tuple[str, Path]]:
    homes: list[tuple[str, Path]] = []
    if (root / "config.yaml").exists():
        homes.append(("default", root))
    profiles_dir = root / "profiles"
    if profiles_dir.is_dir():
        for child in sorted(profiles_dir.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir() and (child / "config.yaml").exists():
                homes.append((child.name, child))
    if not homes:
        homes.append(("default", root))
    return homes


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Iterable):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _normalize_role(raw: Any, profile: str) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    role = dict(raw)
    role_id = str(role.get("id") or profile).strip() or profile
    role["id"] = role_id
    role.setdefault("display_name", role_id)
    role.setdefault("status", "active")
    role["activity_fields"] = _clean_list(role.get("activity_fields"))
    role["allowed_task_types"] = [t.upper() for t in _clean_list(role.get("allowed_task_types"))]
    role["allowed_boards"] = _clean_list(role.get("allowed_boards"))
    role["forbidden"] = [f.lower() for f in _clean_list(role.get("forbidden"))]
    role["escalation"] = _clean_list(role.get("escalation"))
    output_contract = role.get("output_contract") if isinstance(role.get("output_contract"), dict) else {}
    output_contract["required_sections"] = _clean_list(output_contract.get("required_sections"))
    role["output_contract"] = output_contract
    return role


def collect_profiles(root: Optional[Path] = None) -> list[dict[str, Any]]:
    root = root or _root()
    cfg = roster_config(root)
    role_key = str(cfg.get("role_config_key") or DEFAULT_ROLE_CONFIG_KEY)
    profiles: list[dict[str, Any]] = []
    for name, home in _iter_profile_homes(root):
        raw_cfg = _read_yaml(home / "config.yaml")
        role = _normalize_role(raw_cfg.get(role_key), name)
        profiles.append(
            {
                "name": name,
                "home": str(home),
                "is_default": name == "default",
                "exists": home.exists(),
                "role": role,
                "has_role_metadata": role is not None,
                "status": (role or {}).get("status", "missing"),
                "toolsets": _clean_list(raw_cfg.get("toolsets")),
            }
        )
    return profiles


def profiles_by_name(profiles: Optional[list[dict[str, Any]]] = None) -> dict[str, dict[str, Any]]:
    return {p["name"]: p for p in (profiles if profiles is not None else collect_profiles())}


def _extract_task_type(title: str, body: Optional[str] = None) -> str:
    text = (title or "").strip()
    match = re.match(r"^\s*(?:\[\s*)?([A-Za-z][A-Za-z0-9_-]{1,24})(?:\s*\])?\s*[:\-]", text)
    if match:
        return match.group(1).upper().replace("-", "_")
    bracket = re.match(r"^\s*\[\s*([A-Za-z][A-Za-z0-9_-]{1,24})\s*\]", text)
    if bracket:
        return bracket.group(1).upper().replace("-", "_")
    hay = f"{title}\n{body or ''}".lower()
    keyword_map = {
        "build": "BUILD",
        "code": "CODE",
        "deploy": "DEPLOY",
        "launch": "LAUNCH",
        "review": "REVIEW",
        "qa": "QA",
        "research": "RESEARCH",
        "analyse": "ANALYSIS",
        "analysis": "ANALYSIS",
        "prd": "PRD",
        "spec": "SPEC",
        "write": "WRITE",
    }
    for key, value in keyword_map.items():
        if key in hay:
            return value
    return "GENERAL"


def _task_stage(task_type: str, assignee: Optional[str]) -> str:
    typ = task_type.upper()
    for stage in PIPELINE_STAGES:
        if typ in stage["types"]:
            return stage["key"]
    if assignee:
        for stage in PIPELINE_STAGES:
            if assignee in stage["profiles"]:
                return stage["key"]
    return "other"


def _task_to_dict(kb: Any, conn: Any, task: Any, board: str) -> dict[str, Any]:
    parents = kb.parent_ids(conn, task.id)
    children = kb.child_ids(conn, task.id)
    task_type = _extract_task_type(task.title, task.body)
    return {
        "id": task.id,
        "board": board,
        "title": task.title,
        "body": task.body,
        "assignee": task.assignee,
        "status": task.status,
        "priority": task.priority,
        "tenant": task.tenant,
        "created_by": task.created_by,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "workspace_kind": task.workspace_kind,
        "workspace_path": task.workspace_path,
        "task_type": task_type,
        "stage": _task_stage(task_type, task.assignee),
        "parents": parents,
        "children": children,
    }


def collect_boards(root: Optional[Path] = None) -> list[dict[str, Any]]:
    # kanban_db resolves its own root via HERMES_KANBAN_HOME / get_default_hermes_root.
    from hermes_cli import kanban_db as kb

    boards: list[dict[str, Any]] = []
    try:
        board_meta = kb.list_boards(include_archived=False)
    except Exception:
        board_meta = [{"slug": "default", "name": "Default", "archived": False}]
    for meta in board_meta:
        slug = str(meta.get("slug") or "default")
        conn = None
        tasks: list[dict[str, Any]] = []
        stats: dict[str, Any] = {}
        try:
            conn = kb.connect(board=slug)
            tasks = [_task_to_dict(kb, conn, t, slug) for t in kb.list_tasks(conn, include_archived=False, limit=500)]
            try:
                stats = kb.board_stats(conn)
            except Exception:
                stats = {}
        except Exception as exc:
            stats = {"error": str(exc)}
        finally:
            if conn is not None:
                conn.close()
        boards.append({"slug": slug, "metadata": meta, "tasks": tasks, "stats": stats})
    return boards


def _violation(code: str, severity: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "created_at": _now(),
        **extra,
    }


def _body_field_presence(body: Optional[str]) -> tuple[list[str], list[str]]:
    hay = (body or "").lower()
    present: list[str] = []
    missing: list[str] = []
    for field, aliases in DEFAULT_PRD_REQUIRED_FIELDS.items():
        if any(alias.lower() in hay for alias in aliases):
            present.append(field)
        else:
            missing.append(field)
    return present, missing


def _is_reviewer_profile(profile: Optional[dict[str, Any]]) -> bool:
    if not profile:
        return False
    role = profile.get("role") or {}
    values = [profile.get("name"), role.get("id"), role.get("category"), role.get("display_name")]
    return any(str(v or "").strip().lower() in {"reviewer", "review", "qa"} for v in values)


def _has_reviewer_gate(task: dict[str, Any], board_tasks: dict[str, dict[str, Any]], profile_map: dict[str, dict[str, Any]]) -> bool:
    related = set(task.get("parents") or []) | set(task.get("children") or [])
    for rid in related:
        other = board_tasks.get(rid)
        if not other:
            continue
        assignee = other.get("assignee")
        if _is_reviewer_profile(profile_map.get(str(assignee)) if assignee else None):
            return True
        title = str(other.get("title") or "").lower()
        if "review" in title or "qa" in title:
            return True
    return False


def collect_violations(
    profiles: list[dict[str, Any]],
    boards: list[dict[str, Any]],
    root: Optional[Path] = None,
) -> list[dict[str, Any]]:
    cfg = roster_config(root)
    role_key = str(cfg.get("role_config_key") or DEFAULT_ROLE_CONFIG_KEY)
    profile_map = profiles_by_name(profiles)
    violations: list[dict[str, Any]] = []

    for profile in profiles:
        role = profile.get("role")
        if not role:
            if cfg.get("show_profiles_without_roles", True):
                violations.append(
                    _violation(
                        "profile_missing_role_metadata",
                        "warning",
                        f"Profile {profile['name']} has no {role_key} metadata.",
                        profile=profile["name"],
                    )
                )
            continue
        role_status = str(role.get("status") or "").lower()
        if role_status in {"disabled", "blocked"}:
            violations.append(
                _violation(
                    "profile_not_active",
                    "high",
                    f"Profile {profile['name']} role status is {role.get('status')}.",
                    profile=profile["name"],
                )
            )
        if role_status == "restricted":
            continue
        if not role.get("allowed_task_types"):
            violations.append(
                _violation(
                    "role_missing_allowed_task_types",
                    "warning",
                    f"Profile {profile['name']} has no allowed_task_types.",
                    profile=profile["name"],
                )
            )
        if not role.get("allowed_boards"):
            violations.append(
                _violation(
                    "role_missing_allowed_boards",
                    "warning",
                    f"Profile {profile['name']} has no allowed_boards.",
                    profile=profile["name"],
                )
            )

    for board in boards:
        slug = board.get("slug") or "default"
        board_tasks = {t["id"]: t for t in board.get("tasks", [])}
        for task in board.get("tasks", []):
            status = task.get("status")
            assignee = task.get("assignee")
            task_type = str(task.get("task_type") or "GENERAL").upper()
            if status in {"ready", "running"} and not assignee:
                violations.append(
                    _violation(
                        "ready_task_missing_assignee",
                        "high",
                        f"Task {task['id']} is {status} but has no assignee.",
                        board=slug,
                        task_id=task["id"],
                    )
                )
            if assignee and assignee not in profile_map:
                violations.append(
                    _violation(
                        "task_assignee_profile_missing",
                        "high",
                        f"Task {task['id']} is assigned to missing profile {assignee}.",
                        board=slug,
                        task_id=task["id"],
                        assignee=assignee,
                    )
                )
                continue
            profile = profile_map.get(str(assignee)) if assignee else None
            role = (profile or {}).get("role") or {}
            if not role:
                continue
            allowed_boards = set(role.get("allowed_boards") or [])
            if cfg.get("check_kanban_assignments", True) and allowed_boards and slug not in allowed_boards:
                violations.append(
                    _violation(
                        "task_board_not_allowed_for_role",
                        "medium",
                        f"Task {task['id']} is on board {slug}, not in {assignee}'s allowed_boards.",
                        board=slug,
                        task_id=task["id"],
                        assignee=assignee,
                    )
                )
            allowed_types = set(str(t).upper() for t in (role.get("allowed_task_types") or []))
            if (
                cfg.get("check_task_prefix_compatibility", True)
                and allowed_types
                and task_type != "GENERAL"
                and task_type not in allowed_types
            ):
                violations.append(
                    _violation(
                        "task_type_not_allowed_for_role",
                        "medium",
                        f"Task {task['id']} type {task_type} is not allowed for {assignee}.",
                        board=slug,
                        task_id=task["id"],
                        assignee=assignee,
                        task_type=task_type,
                    )
                )
            if cfg.get("check_quality_gates", True) and task_type in QUALITY_GATE_TYPES:
                present, missing = _body_field_presence(task.get("body"))
                if missing:
                    violations.append(
                        _violation(
                            "prd_lite_missing",
                            "high",
                            f"Task {task['id']} is {task_type} but lacks PRD-lite fields: {', '.join(missing)}.",
                            board=slug,
                            task_id=task["id"],
                            assignee=assignee,
                            task_type=task_type,
                            present_fields=present,
                            missing_fields=missing,
                        )
                    )
            if (
                cfg.get("check_quality_gates", True)
                and task_type in REVIEW_GATE_TYPES
                and not _has_reviewer_gate(task, board_tasks, profile_map)
            ):
                violations.append(
                    _violation(
                        "reviewer_gate_missing",
                        "high",
                        f"Task {task['id']} is {task_type} but has no linked reviewer/QA gate.",
                        board=slug,
                        task_id=task["id"],
                        assignee=assignee,
                        task_type=task_type,
                    )
                )
    return violations


def build_pipeline(boards: list[dict[str, Any]]) -> dict[str, Any]:
    stage_defs = {stage["key"]: {**stage, "task_count": 0, "ready_count": 0, "blocked_count": 0, "running_count": 0, "done_count": 0, "tasks": []} for stage in PIPELINE_STAGES}
    stage_defs["other"] = {"key": "other", "label": "Other", "types": [], "profiles": [], "task_count": 0, "ready_count": 0, "blocked_count": 0, "running_count": 0, "done_count": 0, "tasks": []}
    for board in boards:
        for task in board.get("tasks", []):
            key = task.get("stage") or "other"
            stage = stage_defs.setdefault(key, {"key": key, "label": key.title(), "types": [], "profiles": [], "task_count": 0, "ready_count": 0, "blocked_count": 0, "running_count": 0, "done_count": 0, "tasks": []})
            stage["task_count"] += 1
            status = task.get("status") or ""
            status_key = f"{status}_count"
            if status_key in stage:
                stage[status_key] += 1
            stage["tasks"].append({k: task.get(k) for k in ("id", "board", "title", "assignee", "status", "task_type")})
    ordered = [stage_defs[s["key"]] for s in PIPELINE_STAGES] + [stage_defs["other"]]
    return {"stages": ordered, "count": len(ordered)}


def build_roster(root: Optional[Path] = None) -> dict[str, Any]:
    root = root or _root()
    profiles = collect_profiles(root)
    boards = collect_boards(root)
    violations = collect_violations(profiles, boards, root)
    pipeline = build_pipeline(boards)
    severity_counts: dict[str, int] = {}
    for violation in violations:
        sev = str(violation.get("severity") or "unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    return {
        "generated_at": _now(),
        "config": roster_config(root),
        "summary": {
            "profile_count": len(profiles),
            "role_count": sum(1 for p in profiles if p.get("role")),
            "profiles_missing_role_metadata": sum(1 for p in profiles if not p.get("role")),
            "board_count": len(boards),
            "task_count": sum(len(b.get("tasks", [])) for b in boards),
            "violation_count": len(violations),
            "severity_counts": severity_counts,
        },
        "profiles": profiles,
        "boards": boards,
        "violations": violations,
        "pipeline": pipeline,
    }


def role_context_for_current_turn() -> Optional[str]:
    if not is_enabled():
        return None
    cfg = roster_config()
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if not task_id and not cfg.get("inject_outside_kanban", False):
        return None

    profile_name = current_profile_name()
    profile = profiles_by_name().get(profile_name)
    role = (profile or {}).get("role")
    if not role:
        return None

    lines = [
        "## Agent Roster Role Context",
        f"Profile: {profile_name}",
        f"Role: {role.get('display_name') or role.get('id')}",
        f"Status: {role.get('status')}",
    ]
    mission = str(role.get("mission") or "").strip()
    if mission:
        lines.append(f"Mission: {mission}")
    if role.get("activity_fields"):
        lines.append("Activity fields: " + ", ".join(role["activity_fields"]))
    if role.get("allowed_task_types"):
        lines.append("Allowed task types: " + ", ".join(role["allowed_task_types"]))
    if role.get("allowed_boards"):
        lines.append("Allowed boards: " + ", ".join(role["allowed_boards"]))
    if role.get("forbidden"):
        lines.append("Forbidden actions: " + ", ".join(role["forbidden"]))
    required = ((role.get("output_contract") or {}).get("required_sections") or [])
    if required:
        lines.append("Required output sections: " + ", ".join(required))
    if role.get("escalation"):
        lines.append("Escalate when: " + ", ".join(role["escalation"]))

    if task_id:
        task_context = _current_task_context(task_id)
        if task_context:
            lines.append(f"Task: {task_id} — {task_context['title']}")
            lines.append(f"Task type: {task_context['task_type']}; board: {task_context['board']}; status: {task_context['status']}")
    lines.append("Policy: stay inside this role. If the task conflicts with the role or lacks required gate data, block and explain the missing input instead of improvising.")
    return "\n".join(lines)


def _current_task_context(task_id: str) -> Optional[dict[str, Any]]:
    from hermes_cli import kanban_db as kb

    board = os.environ.get("HERMES_KANBAN_BOARD", "default") or "default"
    conn = None
    try:
        conn = kb.connect(board=board)
        task = kb.get_task(conn, task_id)
        if not task:
            return None
        return _task_to_dict(kb, conn, task, board)
    except Exception:
        return None
    finally:
        if conn is not None:
            conn.close()


def classify_tool_actions(tool_name: str, args: Optional[dict[str, Any]] = None) -> list[str]:
    args = args or {}
    name = str(tool_name or "").strip()
    actions: set[str] = set()
    if name in {"write_file", "patch"}:
        actions.add("code_changes")
        paths = _tool_target_paths(name, args)
        if any(_is_config_like_path(path) for path in paths):
            actions.add("config_changes")
    if name == "execute_code":
        actions.update({"code_changes", "shell"})
    if name == "terminal":
        actions.add("shell")
        command = str(args.get("command") or "").lower()
        if re.search(r"\bgit\s+push\b", command):
            actions.add("git_push")
        if re.search(r"\bgit\s+commit\b", command):
            actions.add("git_commit")
        if re.search(r"\b(hermes\s+config|hermes\s+profile|hermes\s+tools|hermes\s+plugins|vim\s+.*config|nano\s+.*config)\b", command):
            actions.add("config_changes")
        if re.search(r"\b(wrangler\s+deploy|vercel\s+deploy|netlify\s+deploy|fly\s+deploy|railway\s+up|docker\s+push|kubectl\s+apply|terraform\s+apply)\b", command):
            actions.add("deployment")
        if re.search(r"\b(npm|pnpm|yarn)\s+run\s+(build|deploy|release)\b", command):
            actions.add("deployment")
        if re.search(r"\b(sed|perl|python|python3|node)\b.*\b(write_text|open\(|Path\(|sed -i|>\s*[^&])", command):
            actions.add("code_changes")
    if name.startswith("browser_"):
        actions.add("browser_automation")
        actions.add("network")
    if name.startswith("web_"):
        actions.add("network")
    if name in {"memory", "viking_remember"}:
        actions.add("memory_changes")
    if name in {"skill_manage"}:
        actions.add("skill_changes")
    if name in {"cronjob"}:
        actions.add("scheduling")
    if name in {"send_message"}:
        actions.add("external_message")
    if name.startswith("kanban_"):
        actions.add("kanban")
        if name in {"kanban_create", "kanban_link", "kanban_unblock"}:
            actions.add("kanban_routing")
        if name == "kanban_complete":
            actions.add("task_completion")
    return sorted(actions)


def _tool_target_paths(tool_name: str, args: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    direct = args.get("path")
    if direct:
        paths.append(str(direct))
    if tool_name == "patch":
        patch_text = str(args.get("patch") or "")
        for match in re.finditer(r"^\*\*\*\s+(?:Update|Add|Delete) File:\s+(.+?)\s*$", patch_text, re.MULTILINE):
            paths.append(match.group(1).strip())
    return paths


def _is_config_like_path(path: str) -> bool:
    lowered = path.lower()
    markers = (
        "config.yaml",
        "config.yml",
        ".env",
        "auth.json",
        "soul.md",
        "agents.md",
        "claude.md",
        ".hermes.md",
    )
    return any(marker in lowered for marker in markers)


def write_audit(event: str, **fields: Any) -> None:
    try:
        if not roster_config().get("audit_enabled", True):
            return
        root = _root()
        path = root / "logs" / AUDIT_LOG_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": _now(), "event": event, **fields}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # Hooks must never break the agent loop.
        return


def evaluate_tool_policy(
    tool_name: str,
    args: Optional[dict[str, Any]] = None,
    *,
    session_id: str = "",
    task_id: str = "",
    tool_call_id: str = "",
) -> Optional[dict[str, str]]:
    if not is_enabled():
        return None
    cfg = roster_config()
    mode = str(cfg.get("strict_mode") or DEFAULT_STRICT_MODE).lower()
    if mode == "off":
        return None
    profile_name = current_profile_name()
    if mode == "block" and cfg.get("enforce_completion_gates", True):
        gate_message = _completion_gate_block_message(tool_name, args, profile_name)
        if gate_message:
            write_audit(
                "completion_gate_block",
                profile=profile_name,
                tool_name=tool_name,
                session_id=session_id,
                task_id=task_id or os.environ.get("HERMES_KANBAN_TASK", ""),
                tool_call_id=tool_call_id,
                mode=mode,
            )
            return {"action": "block", "message": gate_message}
    profile = profiles_by_name().get(profile_name)
    role = (profile or {}).get("role") or {}
    forbidden = _forbidden_action_set(role)
    if not forbidden:
        return None
    actions = set(classify_tool_actions(tool_name, args))
    matched = sorted(actions & forbidden)
    if not matched:
        return None
    message = (
        f"Agent Roster policy: profile {profile_name!r} is forbidden from "
        f"{', '.join(matched)}; blocked tool {tool_name!r}. Reassign this "
        "task or ask the supervisor to create an allowed follow-up task."
    )
    audit_fields = {
        "profile": profile_name,
        "role_id": role.get("id"),
        "tool_name": tool_name,
        "actions": sorted(actions),
        "matched_forbidden": matched,
        "session_id": session_id,
        "task_id": task_id or os.environ.get("HERMES_KANBAN_TASK", ""),
        "tool_call_id": tool_call_id,
        "mode": mode,
    }
    if mode == "block":
        write_audit("tool_policy_block", **audit_fields)
        return {"action": "block", "message": message}
    write_audit("tool_policy_audit", **audit_fields)
    return None


def _completion_gate_block_message(tool_name: str, args: Optional[dict[str, Any]], profile_name: str) -> Optional[str]:
    if str(tool_name or "") != "kanban_complete":
        return None
    args = args or {}
    task_id = str(args.get("task_id") or os.environ.get("HERMES_KANBAN_TASK", "")).strip()
    if not task_id:
        return None
    board = str(args.get("board") or os.environ.get("HERMES_KANBAN_BOARD", "default") or "default")
    from hermes_cli import kanban_db as kb

    conn = None
    try:
        conn = kb.connect(board=board)
        task = kb.get_task(conn, task_id)
        if not task:
            return None
        task_dict = _task_to_dict(kb, conn, task, board)
        task_type = str(task_dict.get("task_type") or "GENERAL").upper()
        if task_type not in QUALITY_GATE_TYPES:
            return None
        profiles = collect_profiles(_root())
        profile_map = profiles_by_name(profiles)
        board_tasks = {
            t.id: _task_to_dict(kb, conn, t, board)
            for t in kb.list_tasks(conn, include_archived=False, limit=500)
        }
        _, missing = _body_field_presence(task_dict.get("body"))
        if missing:
            return (
                f"Agent Roster completion gate: task {task_id} is {task_type} but lacks "
                f"PRD-lite fields: {', '.join(missing)}. Add goal, target user, "
                "acceptance criteria, risks, dependencies, success metric, and definition of done before completion."
            )
        if not _has_reviewer_gate(task_dict, board_tasks, profile_map):
            return (
                f"Agent Roster completion gate: task {task_id} is {task_type} but has no linked reviewer/QA gate. "
                "Create or link a QA/reviewer task before marking this done."
            )
    except Exception as exc:
        write_audit(
            "completion_gate_check_error",
            profile=profile_name,
            tool_name=tool_name,
            task_id=task_id,
            board=board,
            error=str(exc),
        )
        return None
    finally:
        if conn is not None:
            conn.close()
    return None


def _forbidden_action_set(role: dict[str, Any]) -> set[str]:
    """Return canonical policy actions from free-form role metadata."""
    out: set[str] = set()
    for item in role.get("forbidden") or []:
        raw = str(item).strip().lower()
        if not raw:
            continue
        out.add(raw)
        if "git push" in raw or "push" in raw:
            out.add("git_push")
        if "git commit" in raw or "commit" in raw:
            out.add("git_commit")
        if "deploy" in raw or "deployment" in raw or "veröffentlich" in raw or "veroeffentlich" in raw:
            out.add("deployment")
        if "config" in raw or "konfiguration" in raw or "eigenmächtig code/config" in raw or "eigenmaechtig code/config" in raw:
            out.add("config_changes")
        if "code" in raw or "änderung" in raw or "aenderung" in raw or "ändern" in raw or "aendern" in raw:
            out.add("code_changes")
        if "memory" in raw or "gedächtnis" in raw or "gedaechtnis" in raw:
            out.add("memory_changes")
        if "skill" in raw:
            out.add("skill_changes")
        if "message" in raw or "nachricht" in raw:
            out.add("external_message")
    return out


def audit_tool_result(
    tool_name: str,
    args: Optional[dict[str, Any]] = None,
    result: Any = None,
    *,
    session_id: str = "",
    task_id: str = "",
    tool_call_id: str = "",
    duration_ms: int = 0,
) -> None:
    if not is_enabled():
        return
    write_audit(
        "tool_call",
        profile=current_profile_name(),
        tool_name=tool_name,
        actions=classify_tool_actions(tool_name, args),
        session_id=session_id,
        task_id=task_id or os.environ.get("HERMES_KANBAN_TASK", ""),
        tool_call_id=tool_call_id,
        duration_ms=duration_ms,
        result_type=type(result).__name__,
        result_redacted=True,
    )


def read_audit(limit: int = 200) -> list[dict[str, Any]]:
    path = _root() / "logs" / AUDIT_LOG_NAME
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, min(limit, 1000)):]
        out = []
        for line in lines:
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    out.append(row)
            except Exception:
                continue
        return out
    except Exception:
        return []
