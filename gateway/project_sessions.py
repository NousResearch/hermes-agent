"""Telegram/tmux project-session routing helpers.

This module intentionally keeps the routing layer small and deterministic:
Gateway commands discover/register tmux panes, store a lightweight profile-local
registry, and send user prompts to the selected Hermes CLI pane via /steer.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

STATE_VERSION = 1
STATE_FILENAME = "project_sessions.json"
_PROJECT_PREFIX_RE = re.compile(r"^\s*\[([^\]\n]{1,80})\]\s*(.+?)\s*$", re.DOTALL)
_BROAD_ROOT_NAMES = {"", "/", "~", "home", "work", "workspace", "desktop", "documents", "downloads"}


@dataclass
class ProjectSession:
    label: str
    tmux_target: str
    workdir: str
    aliases: list[str] = field(default_factory=list)
    handoff_path: str | None = None
    status: str = "active"
    notify: bool = False
    last_seen_at: float = field(default_factory=time.time)


@dataclass
class ProjectRoutingState:
    version: int = STATE_VERSION
    mode: str = "office"
    active_by_chat: dict[str, str] = field(default_factory=dict)
    projects: dict[str, ProjectSession] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


def state_path() -> Path:
    return get_hermes_home() / STATE_FILENAME


def _source_key(source: Any) -> str:
    platform = getattr(getattr(source, "platform", None), "value", None) or getattr(source, "platform", None) or "unknown"
    chat_id = getattr(source, "chat_id", None) or getattr(source, "user_id", None) or "unknown"
    thread_id = getattr(source, "thread_id", None)
    return f"{platform}:{chat_id}:{thread_id}" if thread_id else f"{platform}:{chat_id}"


def load_state(path: Path | None = None) -> ProjectRoutingState:
    path = path or state_path()
    if not path.exists():
        return ProjectRoutingState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ProjectRoutingState()
    projects: dict[str, ProjectSession] = {}
    for label, item in (raw.get("projects") or {}).items():
        if isinstance(item, dict):
            try:
                projects[label] = ProjectSession(**item)
            except TypeError:
                continue
    return ProjectRoutingState(
        version=int(raw.get("version") or STATE_VERSION),
        mode=str(raw.get("mode") or "office"),
        active_by_chat=dict(raw.get("active_by_chat") or {}),
        projects=projects,
        updated_at=float(raw.get("updated_at") or time.time()),
    )


def save_state(state: ProjectRoutingState, path: Path | None = None) -> None:
    path = path or state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = time.time()
    payload = asdict(state)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _run_tmux(args: list[str]) -> str:
    proc = subprocess.run(["tmux", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
    if proc.returncode != 0:
        return ""
    return proc.stdout


def _looks_like_project_dir(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except Exception:
        resolved = path
    name = resolved.name.lower()
    if name in _BROAD_ROOT_NAMES:
        return False
    markers = [resolved / ".git", resolved / ".hermes", resolved / "AGENTS.md", resolved / "CLAUDE.md"]
    return any(marker.exists() for marker in markers)


def _default_label(session: str, window: str, cwd: Path, used: set[str]) -> str:
    # Prefer tmux window names when they look human-meaningful; otherwise use the
    # project directory name. Deduplicate deterministically.
    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", (window or cwd.name or session).strip()).strip("-._")
    if not base or base in {"zsh", "bash", "python", "python3", "-"}:
        base = re.sub(r"[^A-Za-z0-9_.-]+", "-", (cwd.name or session).strip()).strip("-._") or "project"
    label = base
    i = 2
    while label in used:
        label = f"{base}-{i}"
        i += 1
    used.add(label)
    return label


def discover_tmux_projects() -> list[ProjectSession]:
    fmt = "#{session_name}\t#{window_index}\t#{window_name}\t#{pane_index}\t#{pane_current_path}\t#{pane_current_command}"
    output = _run_tmux(["list-panes", "-a", "-F", fmt])
    if not output.strip():
        return []
    sessions: list[ProjectSession] = []
    used: set[str] = set()
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        session, window_index, window_name, pane_index, cwd_s, command = parts[:6]
        cwd = Path(cwd_s).expanduser()
        if not _looks_like_project_dir(cwd):
            continue
        label = _default_label(session, window_name, cwd, used)
        tmux_target = f"{session}:{window_index}.{pane_index}"
        handoff = cwd / ".hermes" / "handoff" / "current.md"
        aliases = [] if cwd.name == label else [cwd.name]
        sessions.append(
            ProjectSession(
                label=label,
                tmux_target=tmux_target,
                workdir=str(cwd),
                aliases=aliases,
                handoff_path=str(handoff),
                status="active",
                notify=False,
            )
        )
    return sessions


def refresh_from_tmux(state: ProjectRoutingState | None = None) -> ProjectRoutingState:
    state = state or load_state()
    for discovered in discover_tmux_projects():
        existing = state.projects.get(discovered.label)
        if existing:
            existing.tmux_target = discovered.tmux_target
            existing.workdir = discovered.workdir
            existing.handoff_path = discovered.handoff_path
            existing.status = "active"
            existing.last_seen_at = time.time()
            for alias in discovered.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
        else:
            state.projects[discovered.label] = discovered
    return state


def find_project(state: ProjectRoutingState, label_or_alias: str) -> ProjectSession | None:
    needle = (label_or_alias or "").strip().lower()
    if not needle:
        return None
    for label, project in state.projects.items():
        names = [label, project.label, *project.aliases]
        if any(needle == str(name).strip().lower() for name in names):
            return project
    # Allow numeric selection in current sorted display order.
    if needle.isdigit():
        idx = int(needle) - 1
        values = sorted(state.projects.values(), key=lambda p: p.label.lower())
        if 0 <= idx < len(values):
            return values[idx]
    return None


def format_projects(state: ProjectRoutingState, source: Any | None = None) -> str:
    if not state.projects:
        return "[프로젝트 세션]\n등록/감지된 tmux 프로젝트 세션이 없습니다. 프로젝트 디렉토리에서 Hermes를 실행한 뒤 다시 시도하세요."
    active = state.active_by_chat.get(_source_key(source)) if source is not None else None
    lines = ["[프로젝트 세션]", f"mode: {state.mode}", ""]
    for i, project in enumerate(sorted(state.projects.values(), key=lambda p: p.label.lower()), 1):
        marker = " *active*" if active and project.label == active else ""
        aliases = f" / aliases: {', '.join(project.aliases)}" if project.aliases else ""
        lines.extend([
            f"{i}. [{project.label}]{marker}{aliases}",
            f"   - tmux: {project.tmux_target}",
            f"   - cwd: {project.workdir}",
            f"   - notify: {'on' if project.notify else 'off'}",
        ])
    lines.extend(["", "사용:", " /switch <번호|label>", " /psend <지시>", " [label] <지시>"])
    return "\n".join(lines)


def switch_project(source: Any, target: str, *, path: Path | None = None) -> str:
    state = refresh_from_tmux(load_state(path))
    project = find_project(state, target)
    if not project:
        save_state(state, path)
        return f"프로젝트 `{target}`를 찾지 못했습니다.\n\n{format_projects(state, source)}"
    state.active_by_chat[_source_key(source)] = project.label
    save_state(state, path)
    return f"[전환 완료]\n현재 Telegram 대상: [{project.label}]\n- tmux: {project.tmux_target}\n- cwd: {project.workdir}"


def current_project(source: Any, *, path: Path | None = None) -> str:
    state = refresh_from_tmux(load_state(path))
    label = state.active_by_chat.get(_source_key(source))
    project = find_project(state, label or "") if label else None
    save_state(state, path)
    if not project:
        return "현재 선택된 프로젝트가 없습니다. /projects 후 /switch <번호|label>을 사용하세요."
    return f"[현재 프로젝트]\n[{project.label}]\n- tmux: {project.tmux_target}\n- cwd: {project.workdir}\n- handoff: {project.handoff_path or '-'}\n- notify: {'on' if project.notify else 'off'}"


def parse_project_prefix(text: str) -> tuple[str, str] | None:
    m = _PROJECT_PREFIX_RE.match(text or "")
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def _send_to_tmux(project: ProjectSession, prompt: str) -> None:
    subprocess.run(
        ["tmux", "send-keys", "-t", project.tmux_target, prompt, "Enter"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5,
        check=True,
    )


def _away_instruction(project: ProjectSession) -> str:
    return (
        "퇴근모드로 전환합니다. 현재 작업을 중단하지 말고, 완료/막힘/결정 필요 시 "
        f"Telegram에 프로젝트 prefix [{project.label}]를 붙여 짧게 보고하세요. "
        "긴 로그는 요약하고, 다음 사용자가 답할 수 있는 선택지를 포함하세요."
    )


def _office_instruction(project: ProjectSession) -> str:
    return (
        "출근모드로 전환합니다. 이후 proactive Telegram 결과 보고는 중지하고, "
        "응답은 현재 tmux 터미널 중심으로 남기세요. 필요한 handoff만 최신화하세요."
    )


def send_prompt_to_project(source: Any, message: str, *, explicit_label: str | None = None, path: Path | None = None) -> str:
    state = refresh_from_tmux(load_state(path))
    label = explicit_label or state.active_by_chat.get(_source_key(source))
    project = find_project(state, label or "") if label else None
    if not project:
        save_state(state, path)
        return "대상 프로젝트가 없습니다. `[label] 지시` 형식으로 보내거나 /projects 후 /switch를 먼저 사용하세요."
    safe_message = (message or "").strip()
    if not safe_message:
        return "전달할 지시가 비어 있습니다. 예: [dev-1] 다음 작업 진행해줘"
    prompt = f"/steer {safe_message}"
    try:
        _send_to_tmux(project, prompt)
    except Exception as exc:
        save_state(state, path)
        return f"[{project.label}] 전달 실패 — tmux target 확인 필요: {exc}"
    state.active_by_chat[_source_key(source)] = project.label
    save_state(state, path)
    return f"[{project.label}] 전달 완료\n- tmux: {project.tmux_target}\n- 방식: /steer"


def handle_prefixed_message(source: Any, text: str, *, path: Path | None = None) -> str | None:
    parsed = parse_project_prefix(text)
    if not parsed:
        return None
    label, message = parsed
    state = refresh_from_tmux(load_state(path))
    project = find_project(state, label)
    if not project:
        save_state(state, path)
        return f"프로젝트 prefix `{label}`를 찾지 못했습니다.\n\n{format_projects(state, source)}"
    if state.mode != "away" or not project.notify:
        return None
    # Avoid hijacking approval command payloads; approval routing is separate.
    if message.strip().lower().startswith(("/approve", "/deny")):
        return None
    return send_prompt_to_project(source, message, explicit_label=project.label, path=path)


def set_mode(source: Any, mode: str, target: str = "all", *, path: Path | None = None) -> str:
    state = refresh_from_tmux(load_state(path))
    mode = "away" if mode == "away" else "office"
    scope = (target or "all").strip().lower() or "all"
    if scope not in {"all", "current"}:
        return "사용: /afterwork [all|current] 또는 /office [all|current]"
    projects: Iterable[ProjectSession]
    if scope == "current":
        label = state.active_by_chat.get(_source_key(source))
        project = find_project(state, label or "") if label else None
        projects = [project] if project else []
    else:
        projects = sorted(state.projects.values(), key=lambda p: p.label.lower())
    changed: list[ProjectSession] = []
    failures: list[str] = []
    for project in projects:
        if not project:
            continue
        old_notify = project.notify
        instruction = _away_instruction(project) if mode == "away" else _office_instruction(project)
        try:
            _send_to_tmux(project, f"/steer {instruction}")
        except Exception as exc:
            project.notify = old_notify
            failures.append(f"[{project.label}] {exc}")
            continue
        project.notify = mode == "away"
        changed.append(project)
    if changed:
        state.mode = mode
    save_state(state, path)
    title = "퇴근모드 시작" if mode == "away" else "출근모드 전환 완료"
    lines = [f"[{title}]", ""]
    if changed:
        lines.append("대상:")
        for p in changed:
            lines.append(f"- [{p.label}] tmux={p.tmux_target} notify={'on' if p.notify else 'off'}")
    else:
        lines.append("대상 프로젝트가 없습니다. /projects 후 /switch 하거나 프로젝트 tmux 세션을 확인하세요.")
    if mode == "away":
        lines.extend(["", "회신 형식:", "[label] 다음 작업 진행해줘", "[label] 멈추고 handoff 정리해줘"])
    if failures:
        lines.extend(["", "전달 실패:", *failures])
    return "\n".join(lines)
