#!/usr/bin/env python3
"""Repository guidance discovery with AGENTS.md support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

AGENTS_MD_FILENAME = "AGENTS.md"
GUIDANCE_DIRS = ("docs/architecture", "docs/engineering", "docs/operations")
MAX_CONTENT_BYTES = 64 * 1024


def _file_info(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "name": path.name,
            "size_bytes": stat.st_size,
            "readable": True,
        }
    except Exception:
        return None


def detect_repo_guidance(repo_root: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "repo_root": str(repo_root),
        "agents_md": _file_info(repo_root / AGENTS_MD_FILENAME),
        "guidance_docs": [],
    }
    for rel in GUIDANCE_DIRS:
        directory = repo_root / rel
        if not directory.is_dir():
            continue
        for doc in sorted(directory.glob("*.md")):
            info = _file_info(doc)
            if info:
                manifest["guidance_docs"].append(info)
    return manifest


def read_agents_md(repo_root: Path) -> str | None:
    agents_path = repo_root / AGENTS_MD_FILENAME
    if not agents_path.is_file():
        return None
    try:
        content = agents_path.read_text(encoding="utf-8", errors="replace")
        return content[:MAX_CONTENT_BYTES]
    except Exception:
        return None


def bootstrap_agents_md(repo_root: Path, project_summary: str | None = None) -> dict[str, Any]:
    agents_path = repo_root / AGENTS_MD_FILENAME
    if agents_path.exists():
        return {
            "created": False,
            "path": str(agents_path),
            "reason": "AGENTS.md already exists",
        }
    if not repo_root.is_dir():
        return {
            "created": False,
            "path": str(agents_path),
            "reason": f"Repo root does not exist: {repo_root}",
        }
    summary = project_summary or "Add project-specific engineering guidance here."
    content = (
        "# AGENTS.md\n\n"
        "## Project summary\n"
        f"{summary}\n\n"
        "## Engineering Guidance\n"
        "- Keep changes scoped and testable.\n"
        "- Prefer existing patterns over new abstractions.\n"
        "- Document non-obvious decisions in docs/engineering when needed.\n"
    )
    try:
        agents_path.write_text(content, encoding="utf-8")
        return {"created": True, "path": str(agents_path), "reason": "Created AGENTS.md"}
    except Exception as exc:
        return {"created": False, "path": str(agents_path), "reason": str(exc)}


class RepoKnowledgeService:
    def detect(self, workspace_path: Path) -> dict[str, Any]:
        return detect_repo_guidance(workspace_path)

    def read_agents_md(self, workspace_path: Path) -> str | None:
        return read_agents_md(workspace_path)

    def bootstrap(self, workspace_path: Path, project_summary: str | None = None) -> dict[str, Any]:
        return bootstrap_agents_md(workspace_path, project_summary=project_summary)
