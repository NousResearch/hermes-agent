#!/usr/bin/env python3
"""Code Mode skill discovery bridge for built-in and SKILL.md folder skills."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.skill_commands import scan_skill_commands

SKILL_MD_FILENAME = "SKILL.md"
_MAX_SKILL_MD_BYTES = 32 * 1024


def _parse_skill_md(skill_md_path: Path) -> dict[str, Any]:
    try:
        content = skill_md_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    if len(content.encode("utf-8", errors="ignore")) > _MAX_SKILL_MD_BYTES:
        content = content[:_MAX_SKILL_MD_BYTES]
    title = None
    description = None
    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()
        if not title and stripped.startswith("# "):
            title = stripped.removeprefix("# ").strip()
        if not description and stripped and not stripped.startswith("#"):
            description = stripped[:160]
        if title and description:
            break
    return {"title": title, "description": description}


def _discover_skills_in_dir(skills_dir: Path, source: str) -> list[dict[str, Any]]:
    if not skills_dir.is_dir():
        return []
    discovered: list[dict[str, Any]] = []
    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir():
            continue
        skill_md = child / SKILL_MD_FILENAME
        if not skill_md.is_file():
            continue
        meta = _parse_skill_md(skill_md)
        discovered.append(
            {
                "name": child.name,
                "title": meta.get("title") or child.name,
                "description": meta.get("description") or "",
                "source": source,
                "skill_md_path": str(skill_md),
                "skill_dir": str(child),
                "has_scripts": (child / "scripts").is_dir(),
                "has_resources": (child / "resources").is_dir(),
                "builtin": False,
            }
        )
    return discovered


def discover_global_skills() -> list[dict[str, Any]]:
    from tools.skills_tool import SKILLS_DIR
    skills = _discover_skills_in_dir(SKILLS_DIR, "global")
    try:
        from agent.skill_utils import get_external_skills_dirs

        for directory in get_external_skills_dirs():
            skills.extend(_discover_skills_in_dir(directory, f"external:{directory.name}"))
    except Exception:
        pass
    return skills


def discover_workspace_skills(workspace_path: Path) -> list[dict[str, Any]]:
    return _discover_skills_in_dir(workspace_path / ".hermes" / "skills", f"workspace:{workspace_path.name}")


def discover_builtin_skills() -> list[dict[str, Any]]:
    skills = []
    for command, info in scan_skill_commands().items():
        name = command.lstrip("/")
        skills.append(
            {
                "name": name,
                "title": info.get("name") or name,
                "description": info.get("description") or "",
                "source": "builtin_bridge",
                "skill_md_path": info.get("skill_md_path"),
                "skill_dir": info.get("skill_dir"),
                "has_scripts": bool(info.get("skill_dir") and Path(info["skill_dir"]).joinpath("scripts").is_dir()),
                "has_resources": bool(info.get("skill_dir") and Path(info["skill_dir"]).joinpath("resources").is_dir()),
                "builtin": True,
            }
        )
    return skills


class SkillDiscoveryService:
    def list_skills(self, workspace_path: Path | None = None) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for skill in discover_builtin_skills():
            merged[skill["name"]] = skill
        for skill in discover_global_skills():
            merged[skill["name"]] = skill
        if workspace_path:
            for skill in discover_workspace_skills(workspace_path):
                merged[skill["name"]] = skill
        return sorted(merged.values(), key=lambda s: s["name"])
