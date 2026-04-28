#!/usr/bin/env python3
"""
RepoKnowledge — AGENTS.md detection and repository guidance bootstrap.

Detects AGENTS.md and other guidance files at the repo root and in
canonical docs subdirectories. Never blindly loads large files into context.
Never overwrites an existing AGENTS.md.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

AGENTS_MD_FILENAME = "AGENTS.md"
CLAUDE_MD_FILENAME = "CLAUDE.md"
GEMINI_MD_FILENAME = "GEMINI.md"
CODEX_FILENAME = ".codex"

# Canonical guidance doc directories to scan
GUIDANCE_DIRS = [
    "docs/architecture",
    "docs/engineering",
    "docs/operations",
    "docs/adr",
]

# Max file size to return content for (avoids loading huge docs)
MAX_CONTENT_BYTES = 64 * 1024  # 64 KB


def _check_file(path: Path) -> Optional[Dict[str, Any]]:
    """Return file metadata dict if path exists and is readable."""
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


def detect_guidance_files(repo_root: Path) -> Dict[str, Any]:
    """Detect available guidance files at *repo_root*.

    Returns a dict describing what was found. Content is NOT loaded —
    callers decide what to read based on the manifest.
    """
    result: Dict[str, Any] = {
        "repo_root": str(repo_root),
        "agents_md": None,
        "claude_md": None,
        "gemini_md": None,
        "codex": None,
        "nested_agents_md": [],
        "guidance_docs": [],
    }

    result["agents_md"] = _check_file(repo_root / AGENTS_MD_FILENAME)
    result["claude_md"] = _check_file(repo_root / CLAUDE_MD_FILENAME)
    result["gemini_md"] = _check_file(repo_root / GEMINI_MD_FILENAME)
    result["codex"] = _check_file(repo_root / CODEX_FILENAME)

    # Nested AGENTS.md (one level deep in known directories)
    for candidate_dir in [repo_root / "docs", repo_root / "src", repo_root / "app"]:
        if candidate_dir.is_dir():
            nested = candidate_dir / AGENTS_MD_FILENAME
            info = _check_file(nested)
            if info:
                result["nested_agents_md"].append(info)

    # Scan canonical guidance directories
    for rel_dir in GUIDANCE_DIRS:
        dir_path = repo_root / rel_dir
        if not dir_path.is_dir():
            continue
        for md_file in sorted(dir_path.glob("*.md")):
            info = _check_file(md_file)
            if info:
                result["guidance_docs"].append(info)

    return result


def read_agents_md(repo_root: Path) -> Optional[str]:
    """Read AGENTS.md content if it exists and is within size limit."""
    agents_path = repo_root / AGENTS_MD_FILENAME
    if not agents_path.is_file():
        return None
    try:
        size = agents_path.stat().st_size
        if size > MAX_CONTENT_BYTES:
            logger.warning(
                "AGENTS.md is large (%d bytes) — returning truncated version", size
            )
        return agents_path.read_text(encoding="utf-8", errors="replace")[:MAX_CONTENT_BYTES]
    except Exception as exc:
        logger.warning("Failed to read AGENTS.md at %s: %s", repo_root, exc)
        return None


def bootstrap_agents_md(repo_root: Path, project_summary: Optional[str] = None) -> Dict[str, Any]:
    """Create a minimal AGENTS.md at *repo_root* ONLY if one does not exist.

    Returns {"created": bool, "path": str, "reason": str}.
    Never overwrites an existing AGENTS.md.
    """
    agents_path = repo_root / AGENTS_MD_FILENAME

    if agents_path.exists():
        return {
            "created": False,
            "path": str(agents_path),
            "reason": "AGENTS.md already exists — not overwriting",
        }

    if not repo_root.is_dir():
        return {
            "created": False,
            "path": str(agents_path),
            "reason": f"Repo root does not exist: {repo_root}",
        }

    summary_section = (
        f"\n{project_summary}\n" if project_summary else "\nBriefly describe the project here.\n"
    )

    # Detect what guidance docs are available to reference
    guidance = detect_guidance_files(repo_root)
    read_section_lines = []
    for doc in guidance["guidance_docs"][:10]:
        rel = Path(doc["path"]).relative_to(repo_root)
        read_section_lines.append(f"- {rel}")

    read_section = ""
    if read_section_lines:
        read_section = "\n## Before editing\nRead:\n" + "\n".join(read_section_lines) + "\n"

    hermes_section = ""
    if _check_file(repo_root / "docs" / "HERMES_CODE_MODE.md"):
        hermes_section = "\n## Hermes Code Mode\nRead:\n- docs/HERMES_CODE_MODE.md\n"

    content = f"""# AGENTS.md

## Project summary
{summary_section}{read_section}{hermes_section}
## Deployment

Read deployment and operations runbooks in `docs/operations/` if available.

---
_Generated by Hermes Code Mode. Edit this file to add project-specific guidance._
"""

    try:
        agents_path.write_text(content, encoding="utf-8")
        return {
            "created": True,
            "path": str(agents_path),
            "reason": "Created minimal AGENTS.md",
        }
    except Exception as exc:
        return {
            "created": False,
            "path": str(agents_path),
            "reason": f"Write failed: {exc}",
        }


class RepoKnowledgeService:
    """Service for AGENTS.md detection and repo guidance."""

    def detect(self, workspace_path: Path) -> Dict[str, Any]:
        """Return guidance file manifest for *workspace_path*."""
        return detect_guidance_files(workspace_path)

    def read_agents_md(self, workspace_path: Path) -> Optional[str]:
        """Read AGENTS.md if present."""
        return read_agents_md(workspace_path)

    def bootstrap(
        self, workspace_path: Path, project_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a minimal AGENTS.md only if one does not exist."""
        return bootstrap_agents_md(workspace_path, project_summary=project_summary)
