from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from pathlib import Path

from ..core.runtime_layout import active_marker_path, runtime_provider
from ..core.stop_hooks import HookConfigError, ensure_stop_hook
from ..core.utils import (
    get_project_slug,
    run_cmd,
    write_json,
)


def _workflow_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _workflow_doc_relative(doc_name: str) -> str:
    doc_path = _workflow_root() / "data" / doc_name
    project_root = Path(os.environ.get("PROJECT_ROOT") or os.getcwd()).resolve()
    try:
        return str(doc_path.resolve().relative_to(project_root))
    except ValueError:
        return str(doc_path.resolve())


def _stop_hook_command(command: str, project_root: Path) -> str:
    command_parts = shlex.split(command)
    if not command_parts:
        return command
    candidates = [
        _workflow_root() / "scripts" / "story-automator",
        Path(shutil.which("story-automator")) if shutil.which("story-automator") else None,
        Path(sys.argv[0]).resolve() if Path(sys.argv[0]).exists() and os.access(Path(sys.argv[0]), os.X_OK) else None,
    ]
    for candidate in candidates:
        if candidate and candidate.exists() and os.access(candidate, os.X_OK):
            command_parts[0] = str(candidate.resolve())
            return shlex.join(["env", f"PROJECT_ROOT={project_root}", *command_parts])
    return shlex.join(["env", f"PROJECT_ROOT={project_root}", shutil.which("python3") or "python3", "-m", "story_automator", *command_parts[1:]])


def cmd_derive_project_slug(args: list[str]) -> int:
    if args and args[0] in {"--help", "-h"}:
        print("Usage: derive-project-slug [--project-root PATH]")
        return 0
    project_root = os.getcwd()
    for idx, arg in enumerate(args):
        if arg == "--project-root" and idx + 1 < len(args):
            project_root = args[idx + 1]
    write_json({"ok": True, "slug": get_project_slug(project_root), "projectRoot": project_root})
    return 0


def cmd_ensure_marker_gitignore(args: list[str]) -> int:
    gitignore = ""
    entry = ""
    for idx, arg in enumerate(args):
        if arg == "--gitignore" and idx + 1 < len(args):
            gitignore = args[idx + 1]
        if arg == "--entry" and idx + 1 < len(args):
            entry = args[idx + 1]
    if not gitignore or not entry:
        write_json({"ok": False, "error": "missing_args"})
        return 1
    path = Path(gitignore)
    if not path.exists():
        path.write_text("")
    content = path.read_text()
    for line in content.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped == entry:
            write_json({"ok": True, "changed": False, "path": str(path)})
            return 0
    prefix = "" if not content or content.endswith("\n") else "\n"
    with path.open("a") as handle:
        handle.write(f"{prefix}{entry}\n")
    write_json({"ok": True, "changed": True, "path": str(path)})
    return 0


def cmd_ensure_stop_hook(args: list[str]) -> int:
    settings = ""
    command = ""
    timeout = 10
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "--settings" and idx + 1 < len(args):
            settings = args[idx + 1]
            idx += 2
        elif arg == "--command" and idx + 1 < len(args):
            idx += 1
            command_parts: list[str] = []
            while idx < len(args) and not args[idx].startswith("--"):
                command_parts.append(args[idx])
                idx += 1
            if command_parts:
                command = command_parts[0] if len(command_parts) == 1 else shlex.join(command_parts)
        elif arg == "--timeout" and idx + 1 < len(args):
            timeout = int(args[idx + 1])
            idx += 2
        else:
            idx += 1
    if not command:
        write_json({"ok": False, "error": "missing_required_args"})
        return 1
    project_root = Path(os.environ.get("PROJECT_ROOT") or os.getcwd()).resolve()
    provider = runtime_provider(project_root)
    if provider == "claude" and not settings:
        write_json({"ok": False, "error": "missing_required_args"})
        return 1
    command = _stop_hook_command(command, project_root)
    settings_path = Path(settings).expanduser().resolve() if settings else None
    try:
        result = ensure_stop_hook(
            provider=provider,
            project_root=project_root,
            settings_path=settings_path,
            command=command,
            timeout=timeout,
        )
    except HookConfigError as exc:
        write_json(
            {
                "ok": False,
                "error": exc.code,
                "path": str(exc.path),
                "provider": provider,
                "message": exc.message,
            }
        )
        return 1
    write_json({"ok": True, **result})
    return 0


def cmd_stop_hook(_: list[str]) -> int:
    sys.stdin.read()
    if os.environ.get("STORY_AUTOMATOR_CHILD", "").lower() == "true":
        return 0
    marker = active_marker_path()
    if not marker.exists():
        return 0
    try:
        payload = json.loads(marker.read_text())
    except json.JSONDecodeError:
        return 0
    remaining = payload.get("storiesRemaining", 0)
    if isinstance(remaining, str) and remaining.isdigit():
        remaining = int(remaining)
    if not remaining:
        return 0
    reason = (
        "Story Automator active "
        f"({remaining} stories remaining). Read "
        + _workflow_doc_relative("stop-hook-recovery.md")
    )
    print(json.dumps({"decision": "block", "reason": reason}, indent=2))
    return 0


def cmd_commit_story(args: list[str]) -> int:
    repo = ""
    story = ""
    title = ""
    for idx, arg in enumerate(args):
        if arg == "--repo" and idx + 1 < len(args):
            repo = args[idx + 1]
        elif arg == "--story" and idx + 1 < len(args):
            story = args[idx + 1]
        elif arg == "--title" and idx + 1 < len(args):
            title = args[idx + 1]
    if not repo or not story or not title:
        write_json({"ok": False, "error": "missing_args"})
        return 1
    if not Path(repo).is_dir():
        write_json({"ok": False, "error": "repo_not_found"})
        return 1
    status = run_cmd("git", "-C", repo, "status", "--porcelain")
    if status.exit_code != 0:
        write_json({"ok": False, "error": "git_status_failed"})
        return 1
    lines = [line for line in status.output.strip().splitlines() if line.strip()]
    if not lines:
        write_json({"ok": False, "error": "no_changes"})
        return 0
    if run_cmd("git", "-C", repo, "add", "-A").exit_code != 0:
        write_json({"ok": False, "error": "git_add_failed"})
        return 1
    message = f"feat(story-{story}): {title}"
    commit = run_cmd("git", "-C", repo, "commit", "-m", message)
    if commit.exit_code != 0:
        write_json({"ok": False, "error": "commit_failed"})
        return 1
    sha = run_cmd("git", "-C", repo, "rev-parse", "HEAD").output.strip()
    write_json({"ok": True, "commit": sha})
    return 0


def cmd_list_sessions(args: list[str]) -> int:
    if args and args[0] in {"--help", "-h"}:
        print("Usage: list-sessions --slug SLUG")
        return 0
    slug = ""
    for idx, arg in enumerate(args):
        if arg == "--slug" and idx + 1 < len(args):
            slug = args[idx + 1]
    if not slug:
        write_json({"ok": False, "error": "missing_slug"})
        return 1
    if shutil.which("tmux") is None:
        write_json({"ok": False, "error": "tmux_not_found", "sessions": [], "count": 0})
        return 0
    result = run_cmd("tmux", "list-sessions", "-F", "#{session_name}")
    if result.exit_code != 0:
        write_json({"ok": True, "sessions": [], "count": 0})
        return 0
    prefix = f"sa-{slug}-"
    sessions = [line for line in result.output.splitlines() if line.startswith(prefix)]
    write_json({"ok": True, "sessions": sessions, "count": len(sessions)})
    return 0
