"""LLM-free handling for simple project/session attach messages.

The goal is intentionally narrow: when the user's first message is only
"너는 <project> 세션이야" / "<project> 붙어", answer with a tiny live repo
snapshot and do not load project docs, skills, or tool schemas through the
agent loop.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


_ATTACH_PATTERNS = (
    re.compile(r"^\s*너는\s+(?P<project>.+?)\s*(?:세션이야|세션이다|야|이야)\s*[.!?。]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?P<project>.+?)\s*(?:세션이야|세션이다|붙어|연결)\s*[.!?。]*\s*$", re.IGNORECASE),
)
_REJECT_PROJECTS = {"무슨", "어떤", "뭔", "어느", "세션"}
_PROJECT_TOKEN_RE = re.compile(r"^[\w .@+\-/가-힣]+$")


@dataclass(frozen=True)
class AttachIntent:
    project: str


@dataclass(frozen=True)
class AttachLightResult:
    response: str
    project: str
    workdir: str
    resolved: bool


def _clean_project(raw: str) -> str:
    value = (raw or "").strip().strip("'\"`“”‘’[](){}")
    value = re.sub(r"\s+", " ", value)
    return value


def detect_attach_intent(text: Any) -> Optional[AttachIntent]:
    """Return a project attach intent for simple status-only attach phrases."""
    if not isinstance(text, str):
        return None
    if "\n" in text or len(text.strip()) > 80:
        return None
    stripped = text.strip()
    if not stripped or stripped.endswith("?") or stripped.endswith("？"):
        return None

    for pattern in _ATTACH_PATTERNS:
        match = pattern.match(stripped)
        if not match:
            continue
        project = _clean_project(match.group("project"))
        if not project or project.lower() in _REJECT_PROJECTS:
            return None
        if len(project) > 50 or not _PROJECT_TOKEN_RE.match(project):
            return None
        return AttachIntent(project=project)
    return None


def _norm_name(value: str) -> str:
    return re.sub(r"[\s_.-]+", "", value).lower()


def _iter_project_roots(config: Optional[Mapping[str, Any]], cwd: Path) -> Iterable[Path]:
    seen: set[str] = set()

    def add(pathish: Any):
        if not pathish:
            return
        try:
            path = Path(os.path.expandvars(os.path.expanduser(str(pathish)))).resolve()
        except Exception:
            return
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        yield path

    cfg_roots: list[Any] = []
    if isinstance(config, Mapping):
        attach_cfg = config.get("attach_light") or {}
        if isinstance(attach_cfg, Mapping):
            raw = attach_cfg.get("project_roots") or []
            if isinstance(raw, (str, Path)):
                cfg_roots.append(raw)
            elif isinstance(raw, Iterable):
                cfg_roots.extend(list(raw))
    env_roots = os.getenv("HERMES_ATTACH_PROJECT_ROOTS", "")
    if env_roots:
        cfg_roots.extend([p for p in env_roots.split(os.pathsep) if p])

    defaults = [
        cwd,
        cwd.parent,
        Path.home() / "projects",
        Path.home() / "work",
        Path.home() / "tohohowsl" / "projects",
        Path.home() / "tohohowsl" / "OMX" / "projects",
    ]
    for item in [*cfg_roots, *defaults]:
        yield from add(item)


def resolve_project_workdir(
    project: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    cwd: Optional[Path | str] = None,
) -> tuple[Path, bool]:
    """Resolve a project name to a directory, falling back to cwd."""
    base = Path(cwd or os.getcwd()).expanduser().resolve()
    cleaned = _clean_project(project)
    as_path = Path(os.path.expanduser(cleaned))
    if as_path.is_absolute() and as_path.exists() and as_path.is_dir():
        return as_path.resolve(), True

    target_norm = _norm_name(cleaned)
    for root in _iter_project_roots(config, base):
        if not root.exists() or not root.is_dir():
            continue
        if _norm_name(root.name) == target_norm:
            return root.resolve(), True
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir() and _norm_name(child.name) == target_norm:
                return child.resolve(), True
    return base, _norm_name(base.name) == target_norm


def _run_git(args: list[str], cwd: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _git_snapshot(workdir: Path) -> dict[str, Any]:
    inside = _run_git(["rev-parse", "--is-inside-work-tree"], workdir) == "true"
    if not inside:
        return {"is_git": False}

    branch_line = _run_git(["status", "--short", "--branch"], workdir).splitlines()
    branch = branch_line[0].replace("## ", "") if branch_line else "unknown"
    changes = [line for line in branch_line[1:] if line.strip()]
    head = _run_git(["rev-parse", "--short", "HEAD"], workdir) or "unknown"
    remote = _run_git(["remote", "get-url", "origin"], workdir)
    return {
        "is_git": True,
        "branch": branch,
        "head": head,
        "remote": remote,
        "change_count": len(changes),
        "state": "clean" if not changes else f"dirty ({len(changes)} change{'s' if len(changes) != 1 else ''})",
    }


def render_attach_light_status(
    text: Any,
    *,
    config: Optional[Mapping[str, Any]] = None,
    cwd: Optional[Path | str] = None,
) -> Optional[AttachLightResult]:
    intent = detect_attach_intent(text)
    if not intent:
        return None

    workdir, resolved = resolve_project_workdir(intent.project, config=config, cwd=cwd)
    git = _git_snapshot(workdir)
    lines = [
        f"[attach-light] {intent.project} 세션 연결",
        f"path: {workdir}",
    ]
    if not resolved:
        lines.append("resolution: requested project name not found; using current working directory")
    if git.get("is_git"):
        lines.append(f"branch: {git.get('branch')}")
        lines.append(f"HEAD: {git.get('head')}")
        if git.get("remote"):
            lines.append(f"remote: {git.get('remote')}")
        lines.append(f"state: {git.get('state')}")
    else:
        lines.append("git: not a git repository")
    lines.append("mode: low-token attach; docs/skills not loaded")
    return AttachLightResult(
        response="\n".join(lines),
        project=intent.project,
        workdir=str(workdir),
        resolved=resolved,
    )


__all__ = [
    "AttachIntent",
    "AttachLightResult",
    "detect_attach_intent",
    "resolve_project_workdir",
    "render_attach_light_status",
]
