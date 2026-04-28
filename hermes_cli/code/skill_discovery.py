#!/usr/bin/env python3
"""
SkillDiscovery — bridge between built-in skills and SKILL.md-style folder skills.

Discovers skills from:
  1. Built-in SKILL_CATALOG (always present, from coding_skills.py)
  2. Global skill folder: ~/.hermes/skills/<skill-name>/SKILL.md
  3. Workspace skill folder: <workspace>/.hermes/skills/<skill-name>/SKILL.md

Each folder-based skill must contain a SKILL.md file. Scripts and resources
in the folder are available but not required.

SKILL.md format (recommended):
  # Skill Name

  ## Description
  What this skill does.

  ## Parameters
  - param1: description
  - param2: description

  ## Steps
  1. Step one
  2. Step two
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SKILL_MD_FILENAME = "SKILL.md"


def _parse_skill_md(skill_md_path: Path) -> Dict[str, Any]:
    """Extract metadata from SKILL.md. Best-effort; never raises."""
    try:
        content = skill_md_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}

    lines = content.splitlines()
    title: Optional[str] = None
    description_lines: list[str] = []
    in_description = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and title is None:
            title = stripped[2:].strip()
        elif stripped.startswith("## Description"):
            in_description = True
        elif stripped.startswith("## ") and in_description:
            break
        elif in_description and stripped:
            description_lines.append(stripped)

    return {
        "title": title,
        "description": " ".join(description_lines) if description_lines else None,
        "content": content,
    }


def _discover_skills_in_dir(skills_dir: Path, source: str) -> List[Dict[str, Any]]:
    """Walk *skills_dir* and return skill dicts for each valid skill folder."""
    results = []
    if not skills_dir.is_dir():
        return results

    for candidate in sorted(skills_dir.iterdir()):
        if not candidate.is_dir():
            continue
        skill_md = candidate / SKILL_MD_FILENAME
        if not skill_md.is_file():
            continue

        metadata = _parse_skill_md(skill_md)
        skill_name = candidate.name

        scripts_dir = candidate / "scripts"
        resources_dir = candidate / "resources"

        skill: Dict[str, Any] = {
            "name": skill_name,
            "title": metadata.get("title") or skill_name.replace("_", " ").title(),
            "description": metadata.get("description") or f"Skill from {source}",
            "source": source,
            "skill_md_path": str(skill_md),
            "skill_dir": str(candidate),
            "has_scripts": scripts_dir.is_dir(),
            "has_resources": resources_dir.is_dir(),
            "content": metadata.get("content", ""),
            "safe_only": True,
            "requires_workspace": True,
            "builtin": False,
        }
        results.append(skill)

    return results


def discover_global_skills() -> List[Dict[str, Any]]:
    """Discover skills from ~/.hermes/skills/."""
    try:
        from hermes_cli.config import get_hermes_home
        global_skills_dir = get_hermes_home() / "skills"
        return _discover_skills_in_dir(global_skills_dir, "global")
    except Exception as exc:
        logger.debug("global skill discovery failed: %s", exc)
        return []


def discover_workspace_skills(workspace_path: Path) -> List[Dict[str, Any]]:
    """Discover skills from <workspace>/.hermes/skills/."""
    try:
        skills_dir = workspace_path / ".hermes" / "skills"
        return _discover_skills_in_dir(skills_dir, f"workspace:{workspace_path.name}")
    except Exception as exc:
        logger.debug("workspace skill discovery failed for %s: %s", workspace_path, exc)
        return []


def get_builtin_skills() -> List[Dict[str, Any]]:
    """Return built-in skills from SKILL_CATALOG with builtin=True marker."""
    try:
        from hermes_cli.code.coding_skills import SKILL_CATALOG
        return [
            {**skill, "builtin": True, "source": "builtin", "skill_md_path": None, "skill_dir": None}
            for skill in SKILL_CATALOG
        ]
    except Exception as exc:
        logger.warning("Failed to load built-in skills: %s", exc)
        return []


def discover_all_skills(workspace_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return merged list: built-in skills first, then global, then workspace.

    Folder skills with the same name as a built-in override the built-in entry.
    """
    builtin = get_builtin_skills()
    global_skills = discover_global_skills()
    workspace_skills = discover_workspace_skills(workspace_path) if workspace_path else []

    # Merge: later entries win by name
    by_name: Dict[str, Dict[str, Any]] = {}
    for skill in builtin:
        by_name[skill["name"]] = skill
    for skill in global_skills:
        by_name[skill["name"]] = skill
    for skill in workspace_skills:
        by_name[skill["name"]] = skill

    return list(by_name.values())


class SkillDiscoveryService:
    """Service wrapper for skill discovery."""

    def list_skills(self, workspace_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Return all available skills, merged from all sources."""
        return discover_all_skills(workspace_path=workspace_path)

    def get_skill(self, name: str, workspace_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Return a single skill by name, or None if not found."""
        for skill in self.list_skills(workspace_path=workspace_path):
            if skill["name"] == name:
                return skill
        return None

    def read_skill_instructions(self, skill_name: str, workspace_path: Optional[Path] = None) -> Optional[str]:
        """Return the SKILL.md content for a folder-based skill, or None."""
        skill = self.get_skill(skill_name, workspace_path=workspace_path)
        if skill and skill.get("skill_md_path"):
            try:
                return Path(skill["skill_md_path"]).read_text(encoding="utf-8")
            except Exception:
                pass
        return None
