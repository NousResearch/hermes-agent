"""Helpers for the gateway /cc command.

The command starts a detached Claude Code Remote Control session in a git repo,
so users on messaging platforms can kick off a real interactive Claude Code TUI
and then drive it from Anthropic's official app.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_DEFAULT_WORKSPACE_ROOTS = (
    "~/work",
    "~/worktrees",
    "~/projects",
    "~/repos",
    "~/src",
    "~/.hermes/hermes-agent",
)
_SKIP_DIR_NAMES = {
    ".cache",
    ".claude",
    ".git",
    ".hermes",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
    "venv",
}
_MAX_SCAN_DEPTH = 4


@dataclass(frozen=True)
class ClaudeCodeLaunch:
    """Result of launching a Claude Code Remote Control tmux session."""

    repo_path: Path
    tmux_session: str
    remote_name: str
    command: str


def _slug(value: str, *, fallback: str = "repo", limit: int = 48) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    if not slug:
        slug = fallback
    return slug[:limit].strip("-._") or fallback


def _coerce_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split(os.pathsep) if part.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        return [str(part) for part in value if str(part).strip()]
    return []


def _config_get(config: Any, dotted: str) -> Any:
    cur = config
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def workspace_roots(config: Any | None = None) -> list[Path]:
    """Return configured/default roots searched by /cc.

    Operators can set ``gateway.claude_code.workspace_roots`` in config.yaml.
    Defaults cover the common development directories without introducing a new
    environment variable for non-secret behavior.
    """

    configured = _coerce_path_list(
        _config_get(config, "gateway.claude_code.workspace_roots") if config is not None else None
    )
    terminal_cwd = _config_get(config, "terminal.cwd") if config is not None else None
    candidates: list[str] = []
    if terminal_cwd and str(terminal_cwd).strip() not in {"", "."}:
        candidates.append(str(terminal_cwd))
    candidates.extend(configured or list(_DEFAULT_WORKSPACE_ROOTS))

    roots: list[Path] = []
    seen: set[Path] = set()
    for raw in candidates:
        expanded = Path(os.path.expandvars(os.path.expanduser(str(raw)))).resolve()
        if expanded in seen or not expanded.exists() or not expanded.is_dir():
            continue
        seen.add(expanded)
        roots.append(expanded)
    return roots


def _git_root(path: Path) -> Path | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    root = proc.stdout.strip()
    return Path(root).resolve() if root else None


def _iter_candidate_dirs(root: Path) -> Iterable[Path]:
    yield root
    root_depth = len(root.parts)
    for current, dirs, _files in os.walk(root):
        cur_path = Path(current)
        depth = len(cur_path.parts) - root_depth
        if depth >= _MAX_SCAN_DEPTH:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in _SKIP_DIR_NAMES]
        for dirname in dirs:
            yield cur_path / dirname


def find_repo(repo_name: str, config: Any | None = None) -> tuple[Path | None, str | None]:
    """Resolve a repo name/path to a git repository root.

    Returns ``(path, None)`` on success, ``(None, message)`` on failure.
    """

    query = (repo_name or "").strip()
    if not query:
        return None, "Usage: /cc <repo-name>"

    raw_path = Path(os.path.expandvars(os.path.expanduser(query)))
    if raw_path.is_absolute() or any(sep in query for sep in ("/", os.sep)):
        candidate = raw_path.resolve()
        if not candidate.exists():
            return None, f"No such path: `{candidate}`"
        git_root = _git_root(candidate)
        if git_root:
            return git_root, None
        return None, f"Path is not inside a git repository: `{candidate}`"

    normalized = query.casefold()
    matches: list[Path] = []
    seen: set[Path] = set()
    for root in workspace_roots(config):
        for candidate in _iter_candidate_dirs(root):
            if candidate.name.casefold() != normalized:
                continue
            git_root = _git_root(candidate)
            if git_root and git_root not in seen:
                seen.add(git_root)
                matches.append(git_root)

    if not matches:
        roots_text = ", ".join(f"`{root}`" for root in workspace_roots(config)) or "no existing roots"
        return None, f"Repo `{query}` not found under configured workspace roots ({roots_text})."
    if len(matches) > 1:
        choices = "\n".join(f"- `{path}`" for path in matches[:10])
        return None, f"Repo name `{query}` is ambiguous. Use an absolute path:\n{choices}"
    return matches[0], None


def _require_command(name: str) -> str | None:
    path = shutil.which(name)
    if not path:
        return f"`{name}` is not installed or not on PATH."
    return None


def _tmux_session_exists(session_name: str) -> bool:
    proc = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def _unique_tmux_session(base: str) -> str:
    session = base
    if not _tmux_session_exists(session):
        return session
    stamp = int(time.time())
    for idx in range(1, 20):
        session = f"{base}-{stamp}-{idx}"
        if not _tmux_session_exists(session):
            return session
    raise RuntimeError("Could not allocate a unique tmux session name")


def launch_claude_code(repo_path: Path, repo_name: str) -> ClaudeCodeLaunch:
    """Start Claude Code Remote Control in a detached tmux session."""

    for cmd in ("tmux", "claude"):
        err = _require_command(cmd)
        if err:
            raise RuntimeError(err)

    repo_path = repo_path.resolve()
    if _git_root(repo_path) is None:
        raise RuntimeError(f"Not a git repository: {repo_path}")

    remote_name = _slug(repo_path.name or repo_name, fallback="repo")
    tmux_session = _unique_tmux_session(f"cc-{remote_name}")
    claude_command = (
        "claude --dangerously-skip-permissions "
        f"--remote-control {shlex_quote(remote_name)}"
    )
    proc = subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            tmux_session,
            "-c",
            str(repo_path),
            claude_command,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "tmux new-session failed").strip()
        raise RuntimeError(detail)

    return ClaudeCodeLaunch(
        repo_path=repo_path,
        tmux_session=tmux_session,
        remote_name=remote_name,
        command=claude_command,
    )


def shlex_quote(value: str) -> str:
    """Small local quote helper to avoid importing shlex on hot paths elsewhere."""

    if re.fullmatch(r"[A-Za-z0-9_@%+=:,./-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"
