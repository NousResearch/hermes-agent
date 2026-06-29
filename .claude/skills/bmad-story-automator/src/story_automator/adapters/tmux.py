from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.common import ensure_dir, run_cmd, trim_lines
from ..core.tmux_runtime import (
    agent_cli,
    agent_type,
    detect_codex_session,
    estimate_wait,
    extract_active_task,
    generate_session_name,
    heartbeat_check,
    load_session_state,
    pane_status,
    project_hash,
    project_slug,
    save_session_state,
    session_status,
    skill_prefix,
    tmux_display,
    tmux_has_session,
    tmux_kill_session,
    tmux_list_sessions as _tmux_list_sessions,
    tmux_show_environment,
    verify_or_create_output,
)


@dataclass
class TmuxStatus:
    status: str
    todos_done: int
    todos_total: int
    active_task: str
    wait_estimate: int
    session_state: str

def tmux_new_session(session: str, root: str | Path, selected_agent: str) -> tuple[str, int]:
    return run_cmd(
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        "-x",
        "200",
        "-y",
        "50",
        "-c",
        str(root),
        "-e",
        "STORY_AUTOMATOR_CHILD=true",
        "-e",
        f"AI_AGENT={selected_agent}",
        "-e",
        "CLAUDECODE=",
    )


def tmux_send_keys(session: str, command: str, enter: bool = True) -> tuple[str, int]:
    args = ["tmux", "send-keys", "-t", session, command]
    if enter:
        args.append("Enter")
    return run_cmd(*args)


def tmux_list_sessions(project_only: bool = False) -> list[str]:
    sessions, _ = _tmux_list_sessions(project_only)
    return sessions


def load_json_state(path: str | Path) -> dict[str, object]:
    return load_session_state(path)


def save_json_state(path: str | Path, payload: dict[str, object]) -> None:
    ensure_dir(Path(path).parent)
    save_session_state(path, payload)


def count_rune(text: str, target: str) -> int:
    return sum(1 for char in text if char == target)


def find_first_todo_line(capture: str) -> int:
    for index, line in enumerate(trim_lines(capture), start=1):
        if "☒" in line or "☐" in line:
            return index
    return 999
