from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .utils import (
    atomic_write,
    command_exists,
    file_exists,
    filter_input_box,
    get_project_root,
    iso_now,
    project_hash,
    project_slug,
    read_text,
    run_cmd,
)
from .runtime_layout import runtime_provider

STATE_SCHEMA_VERSION = 1
DEFAULT_WIDTH = 200
DEFAULT_HEIGHT = 50
REMAIN_ON_EXIT = "on"
PLACEHOLDER_COMMAND = ("/bin/sleep", "86400")
ARTIFACT_TTL_SECONDS = 24 * 60 * 60
RECONCILE_GRACE_SECONDS = 1.0
RUNNER_MODE_ENV = "SA_TMUX_RUNTIME"
VALID_RUNTIME_MODES = {"legacy", "runner", "auto"}
SIGNAL_EXIT_CODES = {130, 131, 143}
SESSION_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,160}$")


@dataclass(frozen=True)
class SessionPaths:
    state: Path
    command: Path
    runner: Path
    output: Path


@dataclass(frozen=True)
class PaneSnapshot:
    exists: bool
    pane_id: str
    pane_pid: int
    dead: bool
    dead_status: int | None


def runtime_mode() -> str:
    value = os.environ.get(RUNNER_MODE_ENV, "auto").strip().lower()
    return value if value in VALID_RUNTIME_MODES else "auto"


def resolve_command_shell() -> str:
    for candidate in (_tmux_default_shell(), os.environ.get("SHELL", "").strip(), shutil.which("bash") or ""):
        resolved = _resolve_shell_path(candidate)
        if resolved:
            return resolved
    return "/bin/sh"


def generate_session_name(step: str, epic: str, story_id: str, cycle: str = "") -> str:
    stamp = time.strftime("%y%m%d-%H%M%S", time.localtime())
    suffix = story_id.replace(".", "-")
    name = f"sa-{project_slug()}-{stamp}-e{epic}-s{suffix}-{step}"
    if cycle:
        name += f"-r{cycle}"
    return name


def agent_type() -> str:
    value = os.environ.get("AI_AGENT", "").strip().lower()
    if value in {"claude", "codex"}:
        return value
    return runtime_provider()


def agent_cli(agent: str, model: str = "") -> str:
    model = (model or "").strip()
    if agent == "codex":
        base = "codex exec"
    else:
        base = "claude --dangerously-skip-permissions"
    if model:
        base = f"{base} --model {shlex.quote(model)}"
    return base


def skill_prefix(agent: str) -> str:
    return "none" if agent == "codex" else "bmad-"


def _artifact_base_dir() -> Path:
    return Path(tempfile.gettempdir())


def session_paths(session: str, project_root: str | None = None) -> SessionPaths:
    session = _validated_session_name(session)
    hash_value = project_hash(project_root)
    base = _artifact_base_dir()
    return SessionPaths(
        state=base / f".sa-{hash_value}-session-{session}-state.json",
        command=base / f".sa-{hash_value}-session-{session}-command.sh",
        runner=base / f".sa-{hash_value}-session-{session}-runner.sh",
        output=base / f"sa-{hash_value}-output-{session}.txt",
    )


def tmux_has_session(session: str) -> bool:
    return command_exists("tmux") and run_cmd("tmux", "has-session", "-t", session)[1] == 0


def tmux_display(session: str, fmt: str) -> str:
    output, _ = run_cmd("tmux", "display-message", "-t", session, "-p", fmt)
    return output.strip()


def tmux_show_environment(session: str, key: str) -> str:
    output, code = run_cmd("tmux", "show-environment", "-t", session, key)
    if code != 0:
        return ""
    parts = output.strip().split("=", 1)
    return parts[1] if len(parts) == 2 else ""


def tmux_list_sessions(project_only: bool) -> tuple[list[str], int]:
    if not command_exists("tmux"):
        return ([], 1)
    output, code = run_cmd("tmux", "list-sessions", "-F", "#{session_name}")
    if code != 0:
        return ([], code)
    sessions = [line.strip() for line in output.splitlines() if line.strip().startswith("sa-")]
    if project_only:
        prefix = f"sa-{project_slug()}-"
        sessions = [line for line in sessions if line.startswith(prefix)]
    return (sessions, 0)


def load_session_state(path: str | Path) -> dict[str, object]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        raw = json.loads(read_text(target))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def save_session_state(path: str | Path, payload: dict[str, object]) -> None:
    _write_private_text(Path(path), json.dumps(payload, separators=(",", ":")), 0o600)


def update_session_state(path: str | Path, **updates: object) -> dict[str, object]:
    target = Path(path)
    state = load_session_state(target)
    state.update(updates)
    state["updatedAt"] = iso_now()
    save_session_state(target, state)
    return state


def _wait_for_terminal_state(
    path: str | Path,
    max_wait: float = RECONCILE_GRACE_SECONDS,
    tick: float = 0.05,
) -> dict[str, object]:
    deadline = time.time() + max_wait
    state = load_session_state(path)
    while not _is_terminal_state(state) and time.time() < deadline:
        time.sleep(tick)
        state = load_session_state(path)
    return state


def cleanup_runtime_artifacts(session: str, project_root: str | None = None) -> None:
    paths = session_paths(session, project_root)
    for path in (paths.state, paths.command, paths.runner, paths.output):
        path.unlink(missing_ok=True)
    fallback = paths.output.with_name(f"{paths.output.stem}-fallback{paths.output.suffix}")
    fallback.unlink(missing_ok=True)


def cleanup_stale_terminal_artifacts(project_root: str | None = None, ttl_seconds: int = ARTIFACT_TTL_SECONDS) -> None:
    root_hash = project_hash(project_root)
    cutoff = time.time() - ttl_seconds
    tmp_dir = _artifact_base_dir()
    protected_sessions: set[str] = set()
    state_paths = tmp_dir.glob(f".sa-{root_hash}-session-*-state.json")
    for state_path in state_paths:
        session = _session_name_from_state_path(state_path)
        if not session:
            state_path.unlink(missing_ok=True)
            continue
        state = load_session_state(state_path)
        if not _is_terminal_state(state):
            protected_sessions.add(session)
        try:
            if state_path.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        if _is_terminal_state(state):
            cleanup_runtime_artifacts(session, project_root)
    for pattern in (
        f".sa-{root_hash}-session-*-command.sh",
        f".sa-{root_hash}-session-*-runner.sh",
        f"sa-{root_hash}-output-*.txt",
    ):
        for path in tmp_dir.glob(pattern):
            try:
                if path.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            session = _session_name_from_artifact_path(path, root_hash)
            if session and session in protected_sessions:
                continue
            path.unlink(missing_ok=True)


def tmux_kill_session(session: str, project_root: str | None = None) -> None:
    if command_exists("tmux"):
        run_cmd("tmux", "kill-session", "-t", session)
    cleanup_runtime_artifacts(session, project_root)


def spawn_session(
    session: str,
    command: str,
    selected_agent: str,
    project_root: str | None = None,
    mode: str | None = None,
) -> tuple[str, int]:
    resolved_mode = _resolve_spawn_mode(mode)
    if resolved_mode == "legacy":
        return _spawn_legacy(session, command, selected_agent, project_root)
    return _spawn_runner(session, command, selected_agent, project_root)


def heartbeat_check(
    session: str,
    selected_agent: str,
    *,
    project_root: str | None = None,
    mode: str | None = None,
) -> tuple[str, float, str, str]:
    if not session:
        return ("error", 0.0, "", "no_session")

    resolved_mode = _status_mode(session, project_root, mode)
    if resolved_mode == "legacy":
        return _legacy_heartbeat_check(session, selected_agent)

    prompt = _check_prompt_visible(session) if tmux_has_session(session) else "false"
    state_path = session_paths(session, project_root).state
    state = load_session_state(state_path)
    if not state:
        return ("error", 0.0, "", "state_missing")

    if _is_terminal_state(state):
        pid = str(state.get("childPid") or "")
        status = "completed" if str(state.get("result") or "") == "success" else "dead"
        return (status, 0.0, pid, prompt)

    child_pid = _safe_int(state.get("childPid"))
    if child_pid <= 0:
        return ("idle", 0.0, "", prompt)

    cpu = _process_cpu(child_pid)
    if _pid_alive(child_pid):
        return (("alive" if cpu > 0.1 else "idle"), cpu, str(child_pid), prompt)

    _wait_for_terminal_state(state_path)
    status = session_status(session, full=False, codex=selected_agent == "codex", project_root=project_root, mode=resolved_mode)
    public = str(status["session_state"])
    if public == "completed":
        return ("completed", 0.0, str(child_pid), prompt)
    if public in {"crashed", "stuck", "not_found"}:
        return ("dead", 0.0, str(child_pid), prompt)
    return ("idle", 0.0, str(child_pid), prompt)


def session_status(
    session: str,
    *,
    full: bool,
    codex: bool,
    project_root: str | None = None,
    mode: str | None = None,
) -> dict[str, str | int]:
    resolved_mode = _status_mode(session, project_root, mode)
    if resolved_mode == "legacy":
        return _legacy_session_status(session, full=full, codex=codex, project_root=project_root)
    return _runner_session_status(session, full=full, codex=codex, project_root=project_root)


def pane_status(session: str) -> str:
    pane = _pane_snapshot(session)
    if not pane.exists:
        return "missing"
    if pane.dead:
        if pane.dead_status is None:
            return "crashed:unknown"
        if pane.dead_status != 0:
            return f"crashed:{pane.dead_status}"
        return "exited:0"
    return "alive"


def verify_or_create_output(output_file: str, session_name: str, hash_value: str, *, project_root: str | None = None) -> str:
    if output_file and file_exists(output_file) and Path(output_file).stat().st_size > 0:
        return output_file
    expected = _artifact_base_dir() / f"sa-{hash_value}-output-{session_name}.txt"
    if tmux_has_session(session_name):
        capture = _capture_text(session_name, start=-300)
        if capture:
            _write_private_text(expected, "\n".join(capture.splitlines()[:200]), 0o600)
            if expected.stat().st_size > 0:
                return str(expected)
    if expected.exists() and expected.stat().st_size > 0:
        return str(expected)
    if project_root is not None:
        fallback = session_paths(session_name, project_root).output
        if fallback.exists() and fallback.stat().st_size > 0:
            return str(fallback)
    return ""


def extract_active_task(capture: str) -> str:
    pattern = re.compile(r"(?i)(Musing|Thinking|Working|Running|Loading|Creating|Galloping|Beaming|Razzmatazzing|ctrl\+c to interrupt|✻|·|⏺)")
    active = ""
    for line in capture.splitlines():
        if pattern.search(line):
            active = line.strip()
    active = re.sub(r"[·✳⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏✶✻⏺]", "", active)
    active = re.sub(r"\(ctrl\+c.*", "", active).strip()
    return active[:80]


def detect_codex_session(session: str, capture: str) -> str:
    if tmux_show_environment(session, "AI_AGENT") == "codex":
        return "codex"
    if re.search(r"(?i)OpenAI Codex|codex exec|gpt-[0-9]+-codex|tokens used|codex-cli", capture):
        return "codex"
    return "claude"


def estimate_wait(task: str, done: int, total: int) -> int:
    lower = task.lower()
    if re.search(r"loading|reading|searching|parsing|launching|starting", lower):
        return 30
    if re.search(r"presenting|waiting|menu|select|choose", lower):
        return 15
    if re.search(r"running tests|testing|building|compiling|installing", lower):
        return 120
    if re.search(r"writing|editing|updating|creating|fixing", lower):
        return 60
    if total > 0:
        progress = 100 * done // total
        if progress < 25:
            return 90
        if progress < 50:
            return 75
        if progress < 75:
            return 60
        return 30
    return 60


def _spawn_runner(session: str, command: str, selected_agent: str, project_root: str | None) -> tuple[str, int]:
    if not command_exists("tmux"):
        return ("tmux not found\n", 1)
    bash_path = shutil.which("bash")
    if not bash_path:
        return ("bash not found\n", 1)
    command_shell = resolve_command_shell()

    root = Path(project_root or get_project_root()).resolve()
    cleanup_stale_terminal_artifacts(str(root))
    paths = session_paths(session, str(root))
    cleanup_runtime_artifacts(session, str(root))

    _write_private_text(paths.command, _command_file_content(command), 0o700)
    _write_private_text(paths.runner, _runner_file_content(paths, bash_path, command_shell, str(root)), 0o700)

    create_out, create_code = run_cmd(
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        "-x",
        str(DEFAULT_WIDTH),
        "-y",
        str(DEFAULT_HEIGHT),
        "-c",
        str(root),
        "-e",
        "STORY_AUTOMATOR_CHILD=true",
        "-e",
        f"AI_AGENT={selected_agent}",
        "-e",
        "CLAUDECODE=",
        "-e",
        "BASH_ENV=",
        *PLACEHOLDER_COMMAND,
    )
    if create_code != 0:
        cleanup_runtime_artifacts(session, str(root))
        return (create_out, create_code)

    pane_id = tmux_display(session, "#{pane_id}")
    if not pane_id:
        tmux_kill_session(session, str(root))
        return ("failed to resolve tmux pane id\n", 1)

    set_out, set_code = run_cmd("tmux", "set-option", "-t", pane_id, "remain-on-exit", REMAIN_ON_EXIT)
    if set_code != 0:
        tmux_kill_session(session, str(root))
        return (set_out, set_code)

    created_at = iso_now()
    pane_pid = _safe_int(tmux_display(session, "#{pane_pid}"))
    if pane_pid <= 0:
        tmux_kill_session(session, str(root))
        return ("failed to resolve tmux pane pid\n", 1)

    save_session_state(
        paths.state,
        {
            "schemaVersion": STATE_SCHEMA_VERSION,
            "session": session,
            "agent": selected_agent,
            "projectRoot": str(root),
            "paneId": pane_id,
            "panePid": pane_pid,
            "runnerPid": "",
            "childPid": "",
            "commandFile": str(paths.command),
            "outputHint": str(paths.output),
            "createdAt": created_at,
            "startedAt": "",
            "finishedAt": "",
            "updatedAt": created_at,
            "lifecycle": "created",
            "result": "",
            "exitCode": "",
            "failureReason": "",
        },
    )

    respawn_out, respawn_code = run_cmd("tmux", "respawn-pane", "-k", "-t", pane_id, bash_path, str(paths.runner))
    if respawn_code != 0:
        tmux_kill_session(session, str(root))
        return (respawn_out, respawn_code)

    deadline = time.time() + 1.0
    respawned_pane_pid = 0
    while time.time() < deadline:
        respawned_pane_pid = _safe_int(tmux_display(session, "#{pane_pid}"))
        if respawned_pane_pid > 0:
            break
        time.sleep(0.05)
    if respawned_pane_pid <= 0:
        tmux_kill_session(session, str(root))
        return ("failed to resolve respawned tmux pane pid\n", 1)

    update_session_state(
        paths.state,
        paneId=pane_id,
        panePid=respawned_pane_pid,
    )
    return ("", 0)


def _spawn_legacy(session: str, command: str, selected_agent: str, project_root: str | None) -> tuple[str, int]:
    if not command_exists("tmux"):
        return ("tmux not found\n", 1)
    root = project_root or get_project_root()
    paths = session_paths(session, root)
    paths.state.unlink(missing_ok=True)
    output, code = run_cmd(
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        "-x",
        str(DEFAULT_WIDTH),
        "-y",
        str(DEFAULT_HEIGHT),
        "-c",
        root,
        "-e",
        "STORY_AUTOMATOR_CHILD=true",
        "-e",
        f"AI_AGENT={selected_agent}",
        "-e",
        "CLAUDECODE=",
    )
    if code != 0:
        return (output, code)
    if len(command) > 500:
        _write_private_text(paths.command, "#!/bin/bash\n" + command + "\n", 0o700)
        run_cmd("tmux", "send-keys", "-t", session, f"bash {paths.command}", "Enter")
    else:
        run_cmd("tmux", "send-keys", "-t", session, command, "Enter")
    return ("", 0)


def _runner_session_status(
    session: str,
    *,
    full: bool,
    codex: bool,
    project_root: str | None,
) -> dict[str, str | int]:
    root = str(Path(project_root or get_project_root()).resolve())
    paths = session_paths(session, root)
    state = load_session_state(paths.state)
    if not state:
        if tmux_has_session(session):
            return _legacy_session_status(session, full=full, codex=codex, project_root=root)
        return _not_found_status()

    if _is_terminal_state(state):
        return _terminal_runner_status(session, state, full=full, project_root=root)

    pane = _pane_snapshot(session)
    child_pid = _safe_int(state.get("childPid"))
    runner_pid = _safe_int(state.get("runnerPid"))
    child_alive = child_pid > 0 and _pid_alive(child_pid)
    runner_alive = runner_pid > 0 and _pid_alive(runner_pid)

    if str(state.get("lifecycle") or "") == "running" and not child_alive:
        refreshed = _wait_for_terminal_state(paths.state)
        if _is_terminal_state(refreshed):
            return _terminal_runner_status(session, refreshed, full=full, project_root=root)
        state = _reconcile_runner_state(paths, refreshed or state, pane)
        if _is_terminal_state(state):
            return _terminal_runner_status(session, state, full=full, project_root=root)
        child_pid = _safe_int(state.get("childPid"))
        runner_pid = _safe_int(state.get("runnerPid"))
        child_alive = child_pid > 0 and _pid_alive(child_pid)
        runner_alive = runner_pid > 0 and _pid_alive(runner_pid)
    else:
        state = _reconcile_runner_state(paths, state, pane)
        if _is_terminal_state(state):
            return _terminal_runner_status(session, state, full=full, project_root=root)

    capture = _capture_text(session, start=-120)
    todos_done, todos_total = _todo_counts(capture)
    active_task = extract_active_task(capture)
    lifecycle = str(state.get("lifecycle") or "")
    prompt_visible = _check_prompt_visible(session)

    if _runner_claude_prompt_completed(paths, state, capture, prompt_visible):
        state = load_session_state(paths.state)
        return _terminal_runner_status(session, state, full=full, project_root=root)

    if lifecycle == "running" and child_alive:
        label = active_task or ("Codex working" if codex else "Claude working")
        return {
            "status": "active",
            "todos_done": todos_done,
            "todos_total": todos_total,
            "active_task": label,
            "wait_estimate": estimate_wait(label, todos_done, todos_total),
            "session_state": "in_progress",
        }

    if lifecycle in {"created", "launching"} or runner_alive or pane.exists:
        label = active_task or ("launching codex" if codex else "launching claude")
        return {
            "status": "idle",
            "todos_done": todos_done,
            "todos_total": todos_total,
            "active_task": label,
            "wait_estimate": 15,
            "session_state": "just_started",
        }

    output = _export_output_file(session, root) if full else ""
    return {
        "status": "idle",
        "todos_done": todos_done,
        "todos_total": todos_total,
        "active_task": output,
        "wait_estimate": 0,
        "session_state": "stuck",
    }


def _terminal_runner_status(
    session: str,
    state: dict[str, object],
    *,
    full: bool,
    project_root: str,
) -> dict[str, str | int]:
    paths = session_paths(session, project_root)
    text = _capture_text(session, start=-300) or _output_text(paths.output)
    todos_done, todos_total = _todo_counts(text)
    result = str(state.get("result") or "")
    failure_reason = str(state.get("failureReason") or "")
    exit_code = _safe_int(state.get("exitCode"))
    output = _export_output_file(session, project_root) if full else ""

    if result == "success":
        return {
            "status": "idle",
            "todos_done": max(todos_done, 1 if text else 0),
            "todos_total": max(todos_total, 1 if text else 0),
            "active_task": output,
            "wait_estimate": 0,
            "session_state": "completed",
        }

    if result == "unknown" and failure_reason == "launch_never_succeeded":
        return {
            "status": "idle",
            "todos_done": todos_done,
            "todos_total": todos_total,
            "active_task": output,
            "wait_estimate": 0,
            "session_state": "stuck",
        }

    return {
        "status": "crashed",
        "todos_done": todos_done,
        "todos_total": todos_total,
        "active_task": output,
        "wait_estimate": exit_code or 1,
        "session_state": "crashed",
    }


def _reconcile_runner_state(paths: SessionPaths, state: dict[str, object], pane: PaneSnapshot | None = None) -> dict[str, object]:
    if _is_terminal_state(state):
        return state

    pane_snapshot = pane or _pane_snapshot(str(state.get("session") or ""))
    lifecycle = str(state.get("lifecycle") or "")
    child_pid = _safe_int(state.get("childPid"))
    runner_pid = _safe_int(state.get("runnerPid"))
    child_alive = child_pid > 0 and _pid_alive(child_pid)
    runner_alive = runner_pid > 0 and _pid_alive(runner_pid)

    if pane_snapshot.exists and pane_snapshot.dead:
        result, reason = _result_from_exit_code(pane_snapshot.dead_status)
        finished = iso_now()
        save_session_state(
            paths.state,
            {
                **state,
                "finishedAt": str(state.get("finishedAt") or finished),
                "updatedAt": finished,
                "lifecycle": "finished",
                "result": result,
                "exitCode": pane_snapshot.dead_status if pane_snapshot.dead_status is not None else "",
                "failureReason": reason,
            },
        )
        return load_session_state(paths.state)

    if lifecycle == "running" and not child_alive and not runner_alive:
        finished = iso_now()
        save_session_state(
            paths.state,
            {
                **state,
                "finishedAt": str(state.get("finishedAt") or finished),
                "updatedAt": finished,
                "lifecycle": "finished",
                "result": "interrupted",
                "exitCode": _safe_int(state.get("exitCode")) or 1,
                "failureReason": "runner_finalization_missing",
            },
        )
        return load_session_state(paths.state)

    if lifecycle in {"created", "launching"} and not pane_snapshot.exists and _state_age_seconds(state) > RECONCILE_GRACE_SECONDS:
        save_session_state(
            paths.state,
            {
                **state,
                "updatedAt": iso_now(),
                "lifecycle": "finished",
                "result": "unknown",
                "exitCode": "",
                "failureReason": "launch_never_succeeded",
                "finishedAt": iso_now(),
            },
        )
        return load_session_state(paths.state)

    return state


def _runner_claude_prompt_completed(
    paths: SessionPaths,
    state: dict[str, object],
    capture: str,
    prompt_visible: str,
) -> bool:
    if str(state.get("agent") or "") != "claude":
        return False
    if str(state.get("lifecycle") or "") != "running":
        return False
    if prompt_visible != "true":
        return False
    if not _claude_completion_marker_present(capture):
        return False

    finished = iso_now()
    save_session_state(
        paths.state,
        {
            **state,
            "finishedAt": str(state.get("finishedAt") or finished),
            "updatedAt": finished,
            "lifecycle": "finished",
            "result": "success",
            "exitCode": 0,
            "failureReason": "",
        },
    )
    return True


def _claude_completion_marker_present(capture: str) -> bool:
    """Return True if a tmux pane capture contains a Claude CLI completion marker.

    Claude CLI uses ~185 random verbs in its spinner (Baked, Cogitated, Cooked,
    etc.). When work completes the tense switches from present-participle (-ing)
    to past-tense (-ed). This function detects that switch by matching past-tense
    verbs followed by ``for Xm`` after a line start or the ``✻`` spinner glyph,
    rejecting both in-progress forms and generic text like "tests passed for 3m".
    """
    if not capture:
        return False
    return bool(
        re.search(
            r"(?im)(?:^|✻\s+)(?:\w+ed|Done)\s+for\s+\d+m(?:\s+\d+s)?\b",
            capture,
        )
    )


def _legacy_session_status(
    session: str,
    *,
    full: bool,
    codex: bool,
    project_root: str | None,
) -> dict[str, str | int]:
    if codex:
        return _legacy_codex_session_status(session, full=full, project_root=project_root)
    return _legacy_claude_session_status(session, full=full, project_root=project_root)


def _legacy_codex_session_status(
    session: str,
    *,
    full: bool,
    project_root: str | None,
) -> dict[str, str | int]:
    if not session:
        return {"status": "error", "todos_done": 0, "todos_total": 0, "active_task": "no_session", "wait_estimate": 30, "session_state": "error"}
    if not tmux_has_session(session):
        return _not_found_status()
    capture = _capture_text(session, start=-120)
    todos_done, todos_total = _todo_counts(capture)
    if re.search(r"tokens used|❯\s*(\d+[smh]\s*)?\d{1,2}:\d{2}:\d{2}\s*$", capture):
        output = _write_capture(session, capture, project_root=project_root)
        return {"status": "idle", "todos_done": max(1, todos_done), "todos_total": max(1, todos_total or 1), "active_task": output if full else "", "wait_estimate": 0, "session_state": "completed"}
    heartbeat, cpu, _, prompt = _legacy_heartbeat_check(session, "codex")
    if heartbeat == "alive":
        label = extract_active_task(capture) or f"Codex working (CPU: {cpu:.1f}%)"
        return {"status": "active", "todos_done": todos_done, "todos_total": todos_total, "active_task": label, "wait_estimate": 90, "session_state": "in_progress"}
    if prompt == "true":
        output = _write_capture(session, capture, project_root=project_root)
        return {"status": "idle", "todos_done": max(1, todos_done), "todos_total": max(1, todos_total or 1), "active_task": output if full else "", "wait_estimate": 0, "session_state": "completed"}
    if todos_done or todos_total:
        output = _write_capture(session, capture, project_root=project_root)
        return {"status": "idle", "todos_done": todos_done, "todos_total": todos_total, "active_task": output if full else "", "wait_estimate": 0, "session_state": "completed"}
    output = _write_capture(session, capture, project_root=project_root)
    return {"status": "idle", "todos_done": 0, "todos_total": 0, "active_task": output if full else "", "wait_estimate": 0, "session_state": "stuck"}


def _legacy_claude_session_status(
    session: str,
    *,
    full: bool,
    project_root: str | None,
) -> dict[str, str | int]:
    if not session:
        return {"status": "error", "todos_done": 0, "todos_total": 0, "active_task": "no_session", "wait_estimate": 30, "session_state": "error"}

    root = project_root or get_project_root()
    state_path = session_paths(session, root).state

    if not tmux_has_session(session):
        state_path.unlink(missing_ok=True)
        return _not_found_status()

    current_pane_state = pane_status(session)
    if current_pane_state.startswith("crashed:"):
        exit_code = current_pane_state.removeprefix("crashed:")
        capture = _capture_text(session, start=-200)
        output = _write_capture(session, capture, project_root=root, max_lines=150)
        state_path.unlink(missing_ok=True)
        wait_estimate = int(exit_code) if exit_code.isdigit() else 1
        return {
            "status": "crashed",
            "todos_done": 0,
            "todos_total": 0,
            "active_task": output if full else "",
            "wait_estimate": wait_estimate,
            "session_state": "crashed",
        }

    state = _load_legacy_state(state_path)
    state["poll_count"] = int(state["poll_count"]) + 1

    capture = _capture_text(session, start=-50)
    if not capture:
        return {"status": "error", "todos_done": 0, "todos_total": 0, "active_task": "capture_failed", "wait_estimate": 30, "session_state": "error"}

    current_status_time = _parse_statusline_time(capture)
    todos_done, todos_total = _todo_counts(capture)

    if re.search(r"for [0-9]+m [0-9]+s", capture):
        _save_legacy_state(
            state_path,
            poll_count=int(state["poll_count"]),
            has_active=True,
            done=int(state["last_todos_done"]),
            total=int(state["last_todos_total"]),
            status_time=current_status_time,
        )
        output = _write_full_capture(session, project_root=root) if full else ""
        return {
            "status": "idle",
            "todos_done": int(state["last_todos_done"]),
            "todos_total": int(state["last_todos_total"]),
            "active_task": output,
            "wait_estimate": 0,
            "session_state": "completed",
        }

    pane_pid = _safe_int(tmux_display(session, "#{pane_pid}"))
    claude_running = pane_pid > 0 and run_cmd("pgrep", "-P", str(pane_pid), "-f", "claude")[1] == 0
    activity_detected = bool(
        re.search(
            r"(?i)ctrl\+c to interrupt|Musing|Thinking|Working|Running|Loading|Beaming|Galloping|Razzmatazzing|Creating|⏺|✻|·",
            capture,
        )
    )

    if activity_detected or claude_running:
        active_task = extract_active_task(capture) or "Claude working"
        wait_estimate = estimate_wait(active_task, todos_done, todos_total)
        _save_legacy_state(
            state_path,
            poll_count=int(state["poll_count"]),
            has_active=True,
            done=todos_done,
            total=todos_total,
            status_time=current_status_time,
        )
        return {
            "status": "active",
            "todos_done": todos_done,
            "todos_total": todos_total,
            "active_task": active_task,
            "wait_estimate": wait_estimate,
            "session_state": "in_progress",
        }

    session_state = "stuck"
    if bool(state["has_ever_been_active"]):
        session_state = "completed"
    elif int(state["poll_count"]) <= 10:
        session_state = "just_started"
    elif current_status_time and str(state["last_statusline_time"]):
        session_state = "just_started" if current_status_time != str(state["last_statusline_time"]) else "stuck"
    elif current_status_time:
        session_state = "just_started"

    output = _write_full_capture(session, project_root=root) if full else ""
    if full and current_pane_state.startswith("exited:"):
        session_state = "completed"

    _save_legacy_state(
        state_path,
        poll_count=int(state["poll_count"]),
        has_active=bool(state["has_ever_been_active"]),
        done=int(state["last_todos_done"]),
        total=int(state["last_todos_total"]),
        status_time=current_status_time,
    )
    return {
        "status": "idle",
        "todos_done": int(state["last_todos_done"]),
        "todos_total": int(state["last_todos_total"]),
        "active_task": output,
        "wait_estimate": 0,
        "session_state": session_state,
    }


def _legacy_heartbeat_check(session: str, selected_agent: str) -> tuple[str, float, str, str]:
    if not session:
        return ("error", 0.0, "", "no_session")
    if not tmux_has_session(session):
        return ("error", 0.0, "", "session_not_found")
    capture = _capture_text(session, start=-40)
    lines = capture.splitlines()
    last_line = lines[-1] if lines else ""
    prompt = "true" if re.search(r"(❯|\$|#|%)\s*$", last_line) else "false"
    pane_pid = _safe_int(tmux_display(session, "#{pane_pid}"))
    if pane_pid <= 0:
        return ("completed" if prompt == "true" else "dead", 0.0, "", prompt)
    pattern = "codex" if selected_agent == "codex" else "claude"
    agent_pid = _find_agent_pid(str(pane_pid), pattern, 0)
    if not agent_pid:
        return ("completed" if prompt == "true" else "dead", 0.0, "", prompt)
    cpu = _process_cpu(int(agent_pid))
    status = "alive" if cpu > 0.1 else "idle"
    if prompt == "true":
        status = "completed"
    return (status, cpu, agent_pid, prompt)


def _status_mode(session: str, project_root: str | None, mode: str | None) -> str:
    configured = (mode or runtime_mode()).strip().lower()
    if configured not in VALID_RUNTIME_MODES:
        configured = "auto"
    if configured in {"legacy", "runner"}:
        return configured
    state = load_session_state(session_paths(session, project_root).state)
    if int(state.get("schemaVersion") or 0) == STATE_SCHEMA_VERSION:
        return "runner"
    return "legacy"


def _resolve_spawn_mode(mode: str | None) -> str:
    configured = (mode or runtime_mode()).strip().lower()
    if configured == "legacy":
        return "legacy"
    return "runner"


def _runner_file_content(paths: SessionPaths, bash_path: str, command_shell: str, project_root: str) -> str:
    state_file = shlex.quote(str(paths.state))
    command_file = shlex.quote(str(paths.command))
    resolved_command_shell = shlex.quote(command_shell)
    root = shlex.quote(project_root)
    python_bin = shlex.quote(sys.executable)
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd -- {root}

STATE_FILE={state_file}
COMMAND_FILE={command_file}
COMMAND_SHELL={resolved_command_shell}
PYTHON_BIN={python_bin}

write_state() {{
  STATE_LIFECYCLE="$1" \\
  STATE_RESULT="$2" \\
  STATE_EXIT_CODE="$3" \\
  STATE_FAILURE_REASON="$4" \\
  STATE_RUNNER_PID="${{5:-}}" \\
  STATE_CHILD_PID="${{6:-}}" \\
  STATE_STARTED_AT="${{7:-}}" \\
  STATE_FINISHED_AT="${{8:-}}" \\
  STATE_FILE="$STATE_FILE" \\
  "$PYTHON_BIN" <<'PY'
from __future__ import annotations
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["STATE_FILE"])
state = {{}}
if path.exists():
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            state = loaded
    except Exception:
        state = {{}}

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

state["schemaVersion"] = {STATE_SCHEMA_VERSION}
state["lifecycle"] = os.environ["STATE_LIFECYCLE"]
state["result"] = os.environ["STATE_RESULT"]
state["failureReason"] = os.environ["STATE_FAILURE_REASON"]
state["updatedAt"] = now_iso()

runner_pid = os.environ.get("STATE_RUNNER_PID", "")
child_pid = os.environ.get("STATE_CHILD_PID", "")
exit_code = os.environ.get("STATE_EXIT_CODE", "")
started_at = os.environ.get("STATE_STARTED_AT", "")
finished_at = os.environ.get("STATE_FINISHED_AT", "")

if runner_pid:
    state["runnerPid"] = int(runner_pid)
if child_pid:
    state["childPid"] = int(child_pid)
if started_at:
    state["startedAt"] = started_at
if finished_at:
    state["finishedAt"] = finished_at
if exit_code != "":
    state["exitCode"] = int(exit_code)

path.parent.mkdir(parents=True, exist_ok=True)
fd, tmp_name = tempfile.mkstemp(prefix=f".{{path.name}}.", suffix=".tmp", dir=str(path.parent))
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(state, separators=(",", ":")))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_name, path)
    os.chmod(path, 0o600)
finally:
    try:
        os.unlink(tmp_name)
    except FileNotFoundError:
        pass
PY
}}

now_iso() {{
  date -u +%Y-%m-%dT%H:%M:%SZ
}}

runner_pid=$$
write_state "launching" "" "" "" "$runner_pid" "" "" ""

if [[ ! -f "$COMMAND_FILE" ]]; then
  finished_at="$(now_iso)"
  write_state "finished" "spawn_error" "127" "command_file_missing" "$runner_pid" "" "" "$finished_at"
  exit 127
fi

run_payload() {{
  "$COMMAND_SHELL" "$COMMAND_FILE"
}}

set +e
run_payload &
child_pid=$!
started_at="$(now_iso)"
write_state "running" "" "" "" "$runner_pid" "$child_pid" "$started_at" ""
wait "$child_pid"
exit_code=$?
set -e

result="success"
failure_reason=""
if [[ "$exit_code" -eq 0 ]]; then
  result="success"
elif [[ "$exit_code" -eq 126 || "$exit_code" -eq 127 ]]; then
  result="spawn_error"
  failure_reason="runner_exec_failed"
elif [[ "$exit_code" -eq 130 || "$exit_code" -eq 131 || "$exit_code" -eq 143 ]]; then
  result="interrupted"
  failure_reason="signal_terminated"
else
  result="failure"
  failure_reason="exit_nonzero"
fi

finished_at="$(now_iso)"
write_state "finished" "$result" "$exit_code" "$failure_reason" "$runner_pid" "$child_pid" "$started_at" "$finished_at"
exit "$exit_code"
"""


def _command_file_content(command: str) -> str:
    return command.rstrip() + "\n"


def _write_private_text(path: Path, data: str, mode: int) -> None:
    atomic_write(path, data)
    path.chmod(mode)


def _tmux_default_shell() -> str:
    if not command_exists("tmux"):
        return ""
    output, code = run_cmd("tmux", "show-options", "-gv", "default-shell")
    if code != 0:
        return ""
    return output.strip()


def _resolve_shell_path(candidate: str) -> str:
    if not candidate:
        return ""
    if os.path.isabs(candidate):
        return candidate if os.path.isfile(candidate) and os.access(candidate, os.X_OK) else ""
    return shutil.which(candidate) or ""


def _capture_text(session: str, *, start: int) -> str:
    if not tmux_has_session(session):
        return ""
    capture, code = run_cmd("tmux", "capture-pane", "-t", session, "-p", "-S", str(start))
    if code != 0:
        return ""
    return filter_input_box(capture)


def _write_capture(session: str, capture: str, *, project_root: str | None = None, max_lines: int = 200) -> str:
    path = session_paths(session, project_root).output
    lines = capture.splitlines()[:max_lines]
    _write_private_text(path, "\n".join(lines), 0o600)
    return str(path)


def _write_full_capture(session: str, *, project_root: str | None = None) -> str:
    return _write_capture(session, _capture_text(session, start=-300), project_root=project_root)


def _export_output_file(session: str, project_root: str) -> str:
    paths = session_paths(session, project_root)
    hash_value = project_hash(project_root)
    return verify_or_create_output(str(paths.output), session, hash_value, project_root=project_root)


def _output_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return read_text(path)
    except OSError:
        return ""


def _todo_counts(text: str) -> tuple[int, int]:
    done = text.count("☒")
    total = done + text.count("☐")
    return done, total


def _pane_snapshot(session: str) -> PaneSnapshot:
    if not tmux_has_session(session):
        return PaneSnapshot(False, "", 0, False, None)
    pane_id = tmux_display(session, "#{pane_id}")
    pane_pid = _safe_int(tmux_display(session, "#{pane_pid}"))
    pane_dead = tmux_display(session, "#{pane_dead}") == "1"
    dead_status = tmux_display(session, "#{pane_dead_status}")
    return PaneSnapshot(True, pane_id, pane_pid, pane_dead, _safe_int(dead_status) if dead_status else None)


def _load_legacy_state(path: Path) -> dict[str, str | int | bool]:
    state: dict[str, str | int | bool] = {
        "poll_count": 0,
        "has_ever_been_active": False,
        "last_todos_done": 0,
        "last_todos_total": 0,
        "last_statusline_time": "",
    }
    raw = load_session_state(path)
    if not raw or raw.get("schemaVersion"):
        return state
    state["poll_count"] = int(raw.get("pollCount", 0) or 0)
    state["has_ever_been_active"] = bool(raw.get("hasEverBeenActive", False))
    state["last_todos_done"] = int(raw.get("lastTodosDone", 0) or 0)
    state["last_todos_total"] = int(raw.get("lastTodosTotal", 0) or 0)
    state["last_statusline_time"] = str(raw.get("lastStatuslineTime", "") or "")
    return state


def _save_legacy_state(
    path: Path,
    *,
    poll_count: int,
    has_active: bool,
    done: int,
    total: int,
    status_time: str,
) -> None:
    save_session_state(
        path,
        {
            "pollCount": poll_count,
            "hasEverBeenActive": has_active,
            "lastTodosDone": done,
            "lastTodosTotal": total,
            "lastStatuslineTime": status_time,
            "lastPollAt": iso_now(),
        },
    )


def _parse_statusline_time(capture: str) -> str:
    matches = re.findall(r"\| [0-9]{2}:[0-9]{2}:[0-9]{2}", capture)
    if not matches:
        return ""
    return matches[-1].replace("|", "").strip()


def _check_prompt_visible(session: str) -> str:
    capture = _capture_text(session, start=-20)
    for line in reversed(capture.splitlines()[-8:]):
        current = line.rstrip()
        if not current:
            continue
        if re.search(r"❯\s*([0-9]+[smh]\s*)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}\s*$", current):
            return "true"
        if re.search(r"(❯|\$|#|%)\s*$", current):
            return "true"
    return "false"


def _find_agent_pid(parent: str, pattern: str, depth: int) -> str:
    if depth > 4:
        return ""
    output, code = run_cmd("pgrep", "-P", parent)
    if code != 0:
        return ""
    for child in [line.strip() for line in output.splitlines() if line.strip()]:
        command, _ = run_cmd("ps", "-o", "comm=", "-p", child)
        if pattern.lower() in command.lower():
            return child
        nested = _find_agent_pid(child, pattern, depth + 1)
        if nested:
            return nested
    return ""


def _process_cpu(pid: int) -> float:
    output, code = run_cmd("ps", "-o", "%cpu=", "-p", str(pid))
    if code != 0:
        return 0.0
    try:
        return float((output or "").strip() or "0")
    except ValueError:
        return 0.0


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _result_from_exit_code(exit_code: int | None) -> tuple[str, str]:
    if exit_code == 0:
        return ("success", "")
    if exit_code is None:
        return ("unknown", "pane_dead_unknown_status")
    if exit_code in SIGNAL_EXIT_CODES:
        return ("interrupted", "pane_dead_signal")
    if exit_code in {126, 127}:
        return ("spawn_error", "runner_exec_failed")
    return ("failure", "pane_dead_exit")


def _is_terminal_state(state: dict[str, object]) -> bool:
    return str(state.get("lifecycle") or "") == "finished"


def _state_age_seconds(state: dict[str, object]) -> float:
    for key in ("updatedAt", "createdAt"):
        value = str(state.get(key) or "")
        if not value:
            continue
        parsed = _parse_iso(value)
        if parsed is not None:
            return max(0.0, time.time() - parsed.timestamp())
    return 0.0


def _parse_iso(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _safe_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _not_found_status() -> dict[str, str | int]:
    return {"status": "not_found", "todos_done": 0, "todos_total": 0, "active_task": "", "wait_estimate": 0, "session_state": "not_found"}


def _validated_session_name(session: str) -> str:
    if not SESSION_NAME_RE.fullmatch(session):
        raise ValueError(f"invalid session name: {session!r}")
    return session


def _session_name_from_state_path(path: Path) -> str:
    match = re.match(r"\.sa-[^-]+-session-(.+)-state\.json$", path.name)
    if not match:
        return ""
    try:
        return _validated_session_name(match.group(1))
    except ValueError:
        return ""


def _session_name_from_artifact_path(path: Path, root_hash: str) -> str:
    patterns = (
        rf"\.sa-{re.escape(root_hash)}-session-(.+)-(?:command|runner)\.sh$",
        rf"sa-{re.escape(root_hash)}-output-(.+)\.txt$",
    )
    for pattern in patterns:
        match = re.match(pattern, path.name)
        if not match:
            continue
        try:
            return _validated_session_name(match.group(1))
        except ValueError:
            return ""
    return ""
