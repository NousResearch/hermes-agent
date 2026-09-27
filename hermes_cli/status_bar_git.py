"""Cached workspace and rich Git status for terminal status bars."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

_TTL_SECONDS = 5.0
_cache: dict[str, tuple[float, dict]] = {}
_EMPTY = {
    "branch": "", "ahead": 0, "behind": 0, "staged": 0,
    "modified": 0, "untracked": 0, "conflicted": 0,
}


def _resolve_git_dir(start: Path) -> Optional[Path]:
    """Nearest enclosing git dir for ``start``, following worktree pointer files."""
    for parent in (start, *start.parents):
        dotgit = parent / ".git"
        if dotgit.is_dir():
            return dotgit
        if dotgit.is_file():
            try:
                line = dotgit.read_text(encoding="utf-8-sig", errors="replace").strip()
            except OSError:
                return None
            if line.startswith("gitdir:"):
                target = (parent / line.split(":", 1)[1].strip()).resolve()
                return target if target.is_dir() else None
            return None
    return None


def _branch_from_head(base: Path) -> str:
    """Branch (or abbreviated detached commit) straight from ``.git/HEAD``."""
    git_dir = _resolve_git_dir(base)
    if git_dir is None:
        return ""
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8-sig", errors="replace").strip()
    except OSError:
        return ""
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        return ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref.rsplit("/", 1)[-1]
    return f"{head[:8]}…" if head else ""


def parse_porcelain_v2(text: str) -> dict:
    """Parse ``git status --porcelain=v2 --branch`` into compact counters."""
    result = dict(_EMPTY)
    for line in text.splitlines():
        if line.startswith("# branch.head "):
            branch = line[len("# branch.head "):].strip()
            result["branch"] = "detached" if branch == "(detached)" else branch
        elif line.startswith("# branch.ab "):
            match = re.search(r"\+(\d+)\s+-(\d+)", line)
            if match:
                result["ahead"], result["behind"] = map(int, match.groups())
        elif line.startswith(("1 ", "2 ")):
            fields = line.split()
            xy = fields[1] if len(fields) > 1 else ".."
            if len(xy) >= 2:
                result["staged"] += int(xy[0] != ".")
                result["modified"] += int(xy[1] != ".")
        elif line.startswith("u "):
            result["conflicted"] += 1
        elif line.startswith("? "):
            result["untracked"] += 1
    return result


def current_git_status(cwd: Optional[str] = None) -> dict:
    """Rich Git status for ``cwd``; failures/non-repositories return an empty snapshot."""
    try:
        base = Path(cwd or os.getcwd()).resolve()
    except OSError:
        return dict(_EMPTY)
    key, now = str(base), time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < _TTL_SECONDS:
        return dict(hit[1])
    result = dict(_EMPTY)
    try:
        proc = subprocess.run(
            ["git", "-C", str(base), "--no-optional-locks", "status", "--porcelain=v2", "--branch"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=0.75,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode == 0:
            result = parse_porcelain_v2(proc.stdout)
    except Exception:
        pass
    if not result["branch"]:
        # git unavailable/too slow (or a bare .git fixture): fall back to the
        # cheap HEAD read so the branch segment still resolves.
        result["branch"] = _branch_from_head(base)
    _cache[key] = (now, result)
    return dict(result)


def current_git_branch(cwd: Optional[str] = None) -> str:
    """Backward-compatible branch-only view of :func:`current_git_status`."""
    return str(current_git_status(cwd).get("branch") or "")


def format_git_status(status: dict) -> str:
    """Claude/Starship-style branch plus deterministic state counters."""
    branch = str(status.get("branch") or "")
    if not branch:
        return ""
    pieces = [f" {branch}"]
    for key, glyph in (
        ("conflicted", "="), ("staged", "+"), ("modified", "!"),
        ("untracked", "?"), ("ahead", "⇡"), ("behind", "⇣"),
    ):
        value = int(status.get(key) or 0)
        if value:
            pieces.append(f"{glyph}{value}")
    return " ".join(pieces)


def format_status_path(cwd: Optional[str] = None, *, keep: int = 1, fish_len: int = 1) -> str:
    """Fish-style path: contract every parent and preserve the final component."""
    try:
        path = str(Path(cwd or os.getcwd()).resolve())
    except OSError:
        path = cwd or os.getcwd()
    home = str(Path.home())
    if path == home:
        return "~"
    prefix = "~" if path.startswith(home + os.sep) else os.sep
    rel = path[len(home) + 1:] if prefix == "~" else path.lstrip(os.sep)
    parts = [p for p in rel.split(os.sep) if p]
    if len(parts) > keep:
        front = []
        for part in parts[:-keep]:
            width = fish_len + 1 if part.startswith(".") else fish_len
            front.append(part[:width] if len(part) > width else part)
        parts = front + parts[-keep:]
    body = "/".join(parts)
    return f"~/{body}" if prefix == "~" else f"/{body}" if body else "/"
