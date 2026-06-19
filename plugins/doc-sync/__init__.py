"""doc-sync plugin — auto-sync documentation after code changes.

Wires two behaviours:

1. ``post_tool_call`` hook — tracks files modified by ``write_file``,
   ``patch``, and ``terminal`` (git operations) during each turn.

2. ``on_session_end`` hook — when code files were modified during the
   session, spawns a background thread to run doc-sync (CHANGELOG
   update, INDEX rebuild, route map progress check).

Zero agent compliance required. Docs update automatically.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Doc-sync script path
DOC_SYNC_SCRIPT = Path.home() / ".hermes" / "scripts" / "doc-sync.py"

# Directories to watch for code changes (relative to home)
# Files in these dirs trigger doc-sync
WATCHED_PROJECTS = {
    "ticketpilot": Path.home() / "ticketpilot",
    "hermes-agent": Path.home() / ".hermes" / "hermes-agent",
    "crawlweaver": Path.home() / "ai-crawler",
    "compound-system": Path.home() / "workspace" / "compound-system",
    "resume-match": Path.home() / "resume-match",
}

# File extensions that count as "code changes"
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".vue", ".svelte",
    ".go", ".rs", ".java", ".kt", ".swift",
    ".yaml", ".yml", ".toml", ".json",
    ".sh", ".bash",
    ".md",  # Markdown docs also count
}

# Files that are documentation themselves (skip syncing docs-about-docs)
DOC_PATTERNS = {
    "CHANGELOG.md", "INDEX.md", "README.md",
    "ARCHITECTURE.md", "CONTRIBUTING.md",
}

# Per-task set of modified code files. Keyed by task_id/session_id.
_modified_files: Dict[str, Set[str]] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tracker_key(task_id: str, session_id: str) -> str:
    return task_id or session_id or "default"


def _is_code_file(path_str: str) -> bool:
    """Check if a file is a code file worth syncing docs for."""
    p = Path(path_str)
    if p.name in DOC_PATTERNS:
        return False
    if p.suffix in CODE_EXTENSIONS:
        return True
    # Check if it's inside a watched project
    for project_path in WATCHED_PROJECTS.values():
        try:
            p.relative_to(project_path)
            return True
        except ValueError:
            continue
    return False


def _detect_project(path_str: str) -> Optional[str]:
    """Detect which project a file belongs to."""
    p = Path(path_str)
    for name, project_path in WATCHED_PROJECTS.items():
        try:
            p.relative_to(project_path)
            return name
        except ValueError:
            continue
    return None


def _record_modification(task_id: str, session_id: str, path: str) -> None:
    """Record that a file was modified during this turn."""
    if not _is_code_file(path):
        return
    key = _tracker_key(task_id, session_id)
    with _lock:
        _modified_files.setdefault(key, set()).add(path)


def _drain(task_id: str, session_id: str) -> Set[str]:
    """Pop the set of modified files for this turn."""
    key = _tracker_key(task_id, session_id)
    with _lock:
        return _modified_files.pop(key, set())


def _extract_paths_from_write_file(args: Dict[str, Any]) -> Set[str]:
    path = args.get("path")
    return {path} if isinstance(path, str) and path else set()


def _extract_paths_from_patch(args: Dict[str, Any]) -> Set[str]:
    path = args.get("path")
    return {path} if isinstance(path, str) and path else set()


def _extract_paths_from_terminal(args: Dict[str, Any]) -> Set[str]:
    """Extract file paths from terminal commands (git mv, touch, etc.)."""
    paths: Set[str] = set()
    cmd = args.get("command") or ""
    if not isinstance(cmd, str) or not cmd:
        return paths

    # Git operations that modify files
    git_patterns = [
        r"git\s+(?:add|mv|rm|checkout)\s+(.+)",
        r"git\s+commit\s+.*(?:--(?:file|only|all))\s+(.+)",
    ]
    for pattern in git_patterns:
        match = re.search(pattern, cmd)
        if match:
            try:
                for tok in shlex.split(match.group(1)):
                    if tok.startswith(("-",)):
                        continue
                    paths.add(tok)
            except ValueError:
                pass

    # touch command
    if re.match(r"^\s*touch\s+", cmd):
        try:
            for tok in shlex.split(cmd)[1:]:
                if not tok.startswith("-"):
                    paths.add(tok)
        except ValueError:
            pass

    return paths


def _run_doc_sync(projects: Set[str], files: Set[str]) -> None:
    """Run doc-sync for modified projects. Runs in background thread."""
    import subprocess

    for project in projects:
        try:
            result = subprocess.run(
                ["python3", str(DOC_SYNC_SCRIPT), "sync", project],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info("doc-sync: %s updated (%d files)", project, len(files))
            else:
                logger.warning("doc-sync: %s failed: %s", project, result.stderr[:200])
        except subprocess.TimeoutExpired:
            logger.warning("doc-sync: %s timed out", project)
        except Exception as e:
            logger.error("doc-sync: %s error: %s", project, e)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _on_post_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    **_: Any,
) -> None:
    """Track files modified by tool calls."""
    if not isinstance(args, dict):
        return

    candidates: Set[str] = set()
    if tool_name == "write_file":
        candidates = _extract_paths_from_write_file(args)
    elif tool_name == "patch":
        candidates = _extract_paths_from_patch(args)
    elif tool_name == "terminal":
        candidates = _extract_paths_from_terminal(args)
    else:
        return

    for path_str in candidates:
        if path_str:
            _record_modification(task_id, session_id, path_str)


def _on_session_end(
    session_id: str = "",
    completed: bool = True,
    interrupted: bool = False,
    **_: Any,
) -> None:
    """Run doc-sync when session ends with modified files."""
    if interrupted or not completed:
        return

    # Use empty task_id since we don't have it in on_session_end
    modified = _drain("", session_id)
    if not modified:
        return

    # Detect which projects were affected
    projects: Set[str] = set()
    for f in modified:
        project = _detect_project(f)
        if project:
            projects.add(project)

    if not projects:
        return

    logger.info(
        "doc-sync: %d files modified in %s, triggering sync",
        len(modified), ", ".join(sorted(projects)),
    )

    # Run in background thread to not block session teardown
    thread = threading.Thread(
        target=_run_doc_sync,
        args=(projects, modified),
        name="doc-sync",
        daemon=True,
    )
    thread.start()


def register(ctx) -> None:
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("on_session_end", _on_session_end)
