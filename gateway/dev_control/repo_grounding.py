"""Bounded read-only repository grounding for Dev clarification sessions."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_TIME_BUDGET_SECONDS = 8.0
MAX_TIME_BUDGET_SECONDS = 10.0
MAX_FILES = 60
MAX_TOTAL_BYTES = 200 * 1024
MAX_BYTES_PER_FILE = 8 * 1024
MAX_TREE_ENTRIES_PER_REPO = 120

_SKIP_DIRS = {
    ".build",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}
_TEXT_SUFFIXES = {
    "",
    ".c",
    ".cc",
    ".css",
    ".go",
    ".h",
    ".html",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".py",
    ".rs",
    ".sh",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
_CONVENTION_NAMES = ("AGENTS.md", "CLAUDE.md", "README.md", "README.txt", "README")


def collect_repo_grounding(
    *,
    repositories: Iterable[Dict[str, Any]],
    vision_brief: str,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    allowed_roots: Optional[Iterable[Path]] = None,
) -> Dict[str, Any]:
    """Collect compact repo evidence without mutating the target repositories."""

    start = time.monotonic()
    budget = max(0.1, min(float(time_budget_seconds or DEFAULT_TIME_BUDGET_SECONDS), MAX_TIME_BUDGET_SECONDS))
    deadline = start + budget
    warnings: list[str] = []
    provenance: list[str] = []
    repo_summaries: list[Dict[str, Any]] = []
    total_bytes = 0
    total_files = 0
    truncated = False
    roots = list(allowed_roots) if allowed_roots is not None else _allowed_roots()
    keywords = _keywords(vision_brief)

    for raw_repo in repositories or []:
        if time.monotonic() >= deadline:
            truncated = True
            warnings.append("Repo grounding stopped at time budget before all repositories were scanned.")
            break
        path_text = str((raw_repo or {}).get("path") or "").strip()
        label = str((raw_repo or {}).get("label") or "").strip() or None
        if not path_text:
            continue
        try:
            path = Path(path_text).expanduser().resolve(strict=False)
        except Exception as exc:
            warnings.append(f"Skipping repository {path_text!r}: could not resolve path ({exc}).")
            continue
        if not path.exists() or not path.is_dir():
            warnings.append(f"Skipping repository {path}: path does not exist or is not a directory.")
            continue
        if not _is_under_any(path, roots):
            warnings.append(f"Skipping repository {path}: outside allowed grounding roots.")
            continue

        summary, used_bytes, used_files, repo_truncated = _scan_repo(
            path=path,
            label=label,
            keywords=keywords,
            deadline=deadline,
            remaining_files=max(0, MAX_FILES - total_files),
            remaining_bytes=max(0, MAX_TOTAL_BYTES - total_bytes),
        )
        repo_summaries.append(summary)
        total_bytes += used_bytes
        total_files += used_files
        provenance.extend(summary.get("provenance") or [])
        truncated = truncated or repo_truncated
        if total_files >= MAX_FILES or total_bytes >= MAX_TOTAL_BYTES:
            truncated = True
            warnings.append("Repo grounding stopped at file or byte budget.")
            break

    grounding = {
        "repositories": repo_summaries,
        "evidence_token_budget": MAX_TOTAL_BYTES // 4,
        "truncated": bool(truncated),
        "warnings": warnings,
        "allowed_roots": [str(root) for root in roots],
        "time_budget_seconds": budget,
        "file_budget": MAX_FILES,
        "byte_budget": MAX_TOTAL_BYTES,
    }
    return {
        "grounding": grounding,
        "provenance": sorted(dict.fromkeys(provenance)),
        "warnings": warnings,
    }


def _scan_repo(
    *,
    path: Path,
    label: Optional[str],
    keywords: set[str],
    deadline: float,
    remaining_files: int,
    remaining_bytes: int,
) -> tuple[Dict[str, Any], int, int, bool]:
    tree: list[str] = []
    candidates: list[tuple[int, Path]] = []
    conventions: list[Dict[str, str]] = []
    provenance: list[str] = []
    used_bytes = 0
    used_files = 0
    truncated = False

    for root, dirs, files in os.walk(path, topdown=True, followlinks=False):
        if time.monotonic() >= deadline:
            truncated = True
            break
        dirs[:] = sorted(dirname for dirname in dirs if dirname not in _SKIP_DIRS and not dirname.startswith(".cache"))
        root_path = Path(root)
        rel_root = _relative(root_path, path)
        depth = 0 if rel_root == "." else len(Path(rel_root).parts)
        if depth >= 3:
            dirs[:] = []
        for dirname in dirs:
            if len(tree) < MAX_TREE_ENTRIES_PER_REPO:
                tree.append(f"{_relative(root_path / dirname, path)}/")
        for filename in sorted(files):
            file_path = root_path / filename
            rel = _relative(file_path, path)
            if len(tree) < MAX_TREE_ENTRIES_PER_REPO:
                tree.append(rel)
            if filename in _CONVENTION_NAMES or _is_text_file(file_path):
                candidates.append((_score_path(rel, keywords), file_path))
        if len(tree) >= MAX_TREE_ENTRIES_PER_REPO and candidates:
            truncated = True

    for name in _CONVENTION_NAMES:
        convention_path = path / name
        if convention_path.exists() and convention_path.is_file():
            excerpt, byte_count = _read_excerpt(convention_path, remaining_bytes - used_bytes)
            if excerpt:
                conventions.append({"path": str(convention_path), "excerpt": excerpt})
                provenance.append(str(convention_path))
                used_bytes += byte_count
                used_files += 1

    key_files: list[Dict[str, str]] = []
    seen = {item["path"] for item in conventions}
    for _, file_path in sorted(candidates, key=lambda item: (-item[0], _relative(item[1], path))):
        if time.monotonic() >= deadline or used_files >= remaining_files or used_bytes >= remaining_bytes:
            truncated = True
            break
        if str(file_path) in seen or not file_path.is_file():
            continue
        excerpt, byte_count = _read_excerpt(file_path, remaining_bytes - used_bytes)
        if not excerpt:
            continue
        key_files.append({"path": str(file_path), "excerpt": excerpt})
        provenance.append(str(file_path))
        seen.add(str(file_path))
        used_bytes += byte_count
        used_files += 1
        if len(key_files) >= 12:
            break

    summary = {
        "label": label,
        "path": str(path),
        "summary": f"Read-only grounding collected {len(key_files)} key file(s) from {path.name}.",
        "tree": tree[:MAX_TREE_ENTRIES_PER_REPO],
        "key_files": key_files,
        "conventions": conventions,
        "provenance": sorted(dict.fromkeys(provenance)),
        "truncated": bool(truncated),
    }
    return summary, used_bytes, used_files, truncated


def _allowed_roots() -> list[Path]:
    roots: list[Path] = [
        Path("~/Projects").expanduser(),
        Path("~/projects").expanduser(),
        Path("~/Documents/Oryn.ai").expanduser(),
        Path("~/.codex/worktrees").expanduser(),
        Path.cwd(),
    ]
    raw_env = os.getenv("ORYN_REPO_GROUNDING_ROOTS", "")
    for raw in raw_env.replace(",", os.pathsep).split(os.pathsep):
        if raw.strip():
            roots.append(Path(raw.strip()).expanduser())
    resolved: list[Path] = []
    for root in roots:
        try:
            resolved.append(root.resolve(strict=False))
        except Exception:
            continue
    return sorted(dict.fromkeys(resolved), key=str)


def _is_under_any(path: Path, roots: Iterable[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _keywords(text: str) -> set[str]:
    words = {
        word.lower()
        for word in "".join(char if char.isalnum() else " " for char in str(text or "")).split()
        if len(word) >= 4
    }
    return set(list(words)[:40])


def _score_path(relative_path: str, keywords: set[str]) -> int:
    lowered = relative_path.lower()
    score = 0
    for keyword in keywords:
        if keyword in lowered:
            score += 3
    if Path(relative_path).name in _CONVENTION_NAMES:
        score += 5
    if any(part in lowered for part in ("test", "model", "api", "client", "gateway", "clarification", "plan")):
        score += 1
    return score


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_SUFFIXES


def _read_excerpt(path: Path, remaining_bytes: int) -> tuple[str, int]:
    if remaining_bytes <= 0 or not _is_text_file(path):
        return "", 0
    try:
        data = path.read_bytes()[: min(MAX_BYTES_PER_FILE, remaining_bytes)]
    except Exception:
        return "", 0
    if b"\x00" in data:
        return "", 0
    text = data.decode("utf-8", errors="replace")
    lines = [line.rstrip() for line in text.splitlines()[:80]]
    excerpt = "\n".join(line for line in lines if line.strip())[:MAX_BYTES_PER_FILE]
    return excerpt, len(data)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
