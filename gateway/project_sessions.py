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
_HERMES_COMMAND_RE = re.compile(r"^(?:hermes(?:\.exe)?|python(?:\d+(?:\.\d+)*)?|pythonw)$", re.IGNORECASE)
_SHELL_COMMANDS = {"sh", "bash", "zsh", "fish", "pwsh", "powershell", "cmd", "nu"}


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
    command: str = ""
    routable: bool = False
    last_away_snapshot_path: str | None = None


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
    # Older builds stored project routing as:
    # {"active_by_user": {"858...": "label"}, "pinned": {...}}.
    # Keep reading that file instead of treating it as empty so a user's first
    # /projects or /current after upgrading does not silently discard intent.
    legacy_active = raw.get("active_by_user") if isinstance(raw, dict) else None
    legacy_active_by_chat: dict[str, str] = {}
    if isinstance(legacy_active, dict) and not raw.get("active_by_chat"):
        for user_id, label in legacy_active.items():
            if user_id and label and str(label).strip().lower() not in _BROAD_ROOT_NAMES:
                legacy_active_by_chat[f"telegram:{user_id}"] = str(label)
    projects: dict[str, ProjectSession] = {}
    for label, item in (raw.get("projects") or {}).items():
        if isinstance(item, dict):
            try:
                projects[label] = ProjectSession(**item)
            except TypeError:
                # Older records may have unknown keys in future/downgrade paths;
                # keep the known fields instead of discarding the project.
                known = {f.name for f in ProjectSession.__dataclass_fields__.values()}
                filtered = {k: v for k, v in item.items() if k in known}
                try:
                    projects[label] = ProjectSession(**filtered)
                except TypeError:
                    continue
    return ProjectRoutingState(
        version=int(raw.get("version") or STATE_VERSION),
        mode=str(raw.get("mode") or "office"),
        active_by_chat=dict(raw.get("active_by_chat") or legacy_active_by_chat),
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


def _looks_like_hermes_pane(command: str) -> bool:
    """Return whether a tmux pane is safe to receive Hermes /steer input.

    tmux exposes only the current foreground command, not the full argv.  Hermes
    normally appears as ``hermes`` or as the Python interpreter that launched the
    installed CLI.  Shell/editor/server panes are intentionally not routable even
    if their cwd is a valid project directory.
    """
    name = Path((command or "").strip()).name.lower()
    if not name or name in _SHELL_COMMANDS:
        return False
    return bool(_HERMES_COMMAND_RE.match(name))


def _status_for(command: str, *, discovered: bool = True) -> str:
    if not discovered:
        return "stale"
    return "ready" if _looks_like_hermes_pane(command) else "shell-only"


def _status_label(project: ProjectSession) -> str:
    if project.status == "ready" and project.routable:
        return "READY"
    if project.status == "stale":
        return "STALE"
    return "SHELL-ONLY"


def _non_routable_reason(project: ProjectSession) -> str | None:
    if project.status == "stale":
        return "tmux target이 현재 감지되지 않는 STALE 프로젝트입니다. /projects로 목록을 갱신한 뒤 다시 선택하세요."
    if not project.routable:
        command = project.command or "unknown"
        return f"현재 tmux pane은 Hermes 세션이 아닙니다(command={command}). tmux에서 Hermes를 실행한 뒤 다시 시도하세요."
    return None


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
        status = _status_for(command)
        sessions.append(
            ProjectSession(
                label=label,
                tmux_target=tmux_target,
                workdir=str(cwd),
                aliases=aliases,
                handoff_path=str(handoff),
                status=status,
                notify=False,
                command=command,
                routable=status == "ready",
            )
        )
    return sessions


def refresh_from_tmux(state: ProjectRoutingState | None = None) -> ProjectRoutingState:
    state = state or load_state()
    discovered = {project.label: project for project in discover_tmux_projects()}
    seen: set[str] = set()
    for label, project in discovered.items():
        seen.add(label)
        existing = state.projects.get(label)
        if existing:
            existing.tmux_target = project.tmux_target
            existing.workdir = project.workdir
            existing.handoff_path = project.handoff_path
            existing.status = project.status
            existing.command = project.command
            existing.routable = project.routable
            existing.last_seen_at = time.time()
            # A shell-only/stale pane must not keep a previous away-mode relay.
            if not existing.routable:
                existing.notify = False
            for alias in project.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
        else:
            state.projects[label] = project
    for label, project in state.projects.items():
        if label in seen:
            continue
        project.status = "stale"
        project.routable = False
        project.notify = False
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


def commute_help_text() -> str:
    return """[퇴근모드/출근모드 도움말]

개념:
- 퇴근모드: tmux의 READY Hermes 프로젝트 세션은 계속 터미널에서 돌고, Telegram으로도 결과/막힘/결정 필요 사항을 받는 모드
- 출근모드: Telegram relay를 끄고 다시 tmux 터미널 중심으로 보는 모드
- 승인 요청은 별도입니다. 작업 지시는 [label] 형식, 승인은 /approve 또는 /deny를 사용하세요.

기본 순서:
1. /projects
   - 감지된 프로젝트 label과 READY/SHELL-ONLY/STALE 상태 확인
2. 퇴근모드 또는 /afterwork all
   - READY 프로젝트만 퇴근모드로 전환하고, SHELL-ONLY/STALE 프로젝트는 제외
3. [label] <지시>
   - 퇴근모드 중 READY 프로젝트에 지시 전달
   예: [llm-eval-pipeline] 다음 작업 진행해줘
4. 출근모드 또는 /office all
   - READY 프로젝트를 출근모드로 복귀

현재 프로젝트에만 지시:
- /switch <번호|label> 후 /psend <지시>
- /psend는 선택된 READY 프로젝트 tmux pane에 /steer로 전달합니다.

고급/테스트 옵션:
- /afterwork current: 선택한 READY 프로젝트만 퇴근모드
- /office current: 선택한 READY 프로젝트만 출근모드

자연어 단축:
- Telegram DM: 퇴근모드 → /afterwork all, 출근모드 → /office all
- tmux CLI: 퇴근모드 → /afterwork all, 출근모드 → /office all

주의:
- SHELL-ONLY는 프로젝트 디렉토리의 일반 shell pane입니다. Hermes 세션이 아니므로 /psend와 퇴근모드 전달 대상에서 제외됩니다.
- 긴 로그 확인은 tmux/Termius가 적합하고, Telegram은 요약/결정/짧은 지시용입니다.
""".strip()


def format_projects(state: ProjectRoutingState, source: Any | None = None) -> str:
    if not state.projects:
        return "[프로젝트 세션]\n등록/감지된 tmux 프로젝트 세션이 없습니다. 프로젝트 디렉토리에서 Hermes를 실행한 뒤 다시 시도하세요."
    active = state.active_by_chat.get(_source_key(source)) if source is not None else None
    lines = ["[프로젝트 세션]", f"mode: {state.mode}", ""]
    for i, project in enumerate(sorted(state.projects.values(), key=lambda p: p.label.lower()), 1):
        marker = " *active*" if active and project.label == active else ""
        aliases = f" / aliases: {', '.join(project.aliases)}" if project.aliases else ""
        state_label = _status_label(project)
        lines.extend([
            f"{i}. [{project.label}] {state_label}{marker}{aliases}",
            f"   - tmux: {project.tmux_target}",
            f"   - command: {project.command or '-'}",
            f"   - cwd: {project.workdir}",
            f"   - notify: {'on' if project.notify else 'off'}",
        ])
        reason = _non_routable_reason(project)
        if reason:
            lines.append(f"   - Telegram 지시: 불가 — {reason}")
        else:
            lines.append("   - Telegram 지시: 가능")
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
    reason = _non_routable_reason(project)
    if reason:
        return (
            f"[전환 완료: 주의]\n현재 Telegram 대상: [{project.label}]\n"
            f"- 상태: {_status_label(project)}\n- tmux: {project.tmux_target}\n"
            f"- command: {project.command or '-'}\n- cwd: {project.workdir}\n\n주의: {reason}"
        )
    return (
        f"[전환 완료]\n현재 Telegram 대상: [{project.label}]\n"
        f"- 상태: READY\n- tmux: {project.tmux_target}\n- cwd: {project.workdir}"
    )


def current_project(source: Any, *, path: Path | None = None) -> str:
    state = refresh_from_tmux(load_state(path))
    label = state.active_by_chat.get(_source_key(source))
    project = find_project(state, label or "") if label else None
    save_state(state, path)
    if not project:
        return "현재 선택된 프로젝트가 없습니다. /projects 후 /switch <번호|label>을 사용하세요."
    reason = _non_routable_reason(project)
    lines = [
        "[현재 프로젝트]",
        f"[{project.label}] {_status_label(project)}",
        f"- tmux: {project.tmux_target}",
        f"- command: {project.command or '-'}",
        f"- cwd: {project.workdir}",
        f"- handoff: {project.handoff_path or '-'}",
        f"- notify: {'on' if project.notify else 'off'}",
    ]
    if reason:
        lines.extend(["", f"주의: {reason}"])
    return "\n".join(lines)


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


def _capture_pane(project: ProjectSession, lines: int = 80) -> str:
    if not project.tmux_target:
        return ""
    output = _run_tmux(["capture-pane", "-p", "-t", project.tmux_target, "-S", f"-{lines}"])
    return output.strip()[-4000:]


def _write_away_snapshot(project: ProjectSession, *, mode: str, sent: bool, error: str | None = None) -> str | None:
    try:
        root = Path(project.workdir).expanduser()
        snapshot = root / ".hermes" / "handoff" / "telegram-away-snapshot.md"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        pane = _capture_pane(project)
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        snapshot.write_text(
            "# Telegram away-mode snapshot\n\n"
            f"- updated_at: {now}\n"
            f"- project: {project.label}\n"
            f"- mode: {mode}\n"
            f"- status: {_status_label(project)}\n"
            f"- routable: {project.routable}\n"
            f"- tmux_target: {project.tmux_target}\n"
            f"- command: {project.command or '-'}\n"
            f"- workdir: {project.workdir}\n"
            f"- steer_sent: {sent}\n"
            f"- error: {error or '-'}\n\n"
            "## Recent tmux output\n\n"
            f"```text\n{pane}\n```\n",
            encoding="utf-8",
        )
        project.last_away_snapshot_path = str(snapshot)
        return str(snapshot)
    except Exception:
        return None


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
    reason = _non_routable_reason(project)
    if reason:
        save_state(state, path)
        return f"[차단]\n[{project.label}] Telegram 지시를 보낼 수 없습니다.\n- 상태: {_status_label(project)}\n- tmux: {project.tmux_target}\n- command: {project.command or '-'}\n\n{reason}"
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
    excluded: list[str] = []
    for project in projects:
        if not project:
            continue
        reason = _non_routable_reason(project)
        if reason:
            project.notify = False
            if mode == "away":
                _write_away_snapshot(project, mode=mode, sent=False, error=reason)
            excluded.append(f"[{project.label}] {_status_label(project)} — {reason}")
            continue
        old_notify = project.notify
        instruction = _away_instruction(project) if mode == "away" else _office_instruction(project)
        try:
            _send_to_tmux(project, f"/steer {instruction}")
        except Exception as exc:
            project.notify = old_notify
            if mode == "away":
                _write_away_snapshot(project, mode=mode, sent=False, error=str(exc))
            failures.append(f"[{project.label}] {exc}")
            continue
        project.notify = mode == "away"
        if mode == "away":
            _write_away_snapshot(project, mode=mode, sent=True)
        changed.append(project)
    if changed:
        state.mode = mode
    save_state(state, path)
    title = "퇴근모드 시작" if mode == "away" else "출근모드 전환 완료"
    lines = [f"[{title}]", ""]
    if changed:
        lines.append("대상:")
        for p in changed:
            extra = f" snapshot={p.last_away_snapshot_path}" if mode == "away" and p.last_away_snapshot_path else ""
            lines.append(f"- [{p.label}] tmux={p.tmux_target} notify={'on' if p.notify else 'off'}{extra}")
    else:
        lines.append("전환된 READY 프로젝트가 없습니다. /projects로 상태를 확인하거나 프로젝트 tmux pane에서 Hermes를 실행하세요.")
    if excluded:
        lines.extend(["", "제외:", *excluded])
    if mode == "away":
        lines.extend(["", "회신 형식:", "[label] 다음 작업 진행해줘", "[label] 멈추고 handoff 정리해줘"])
    if failures:
        lines.extend(["", "전달 실패:", *failures])
    return "\n".join(lines)
