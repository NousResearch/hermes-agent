"""Skill mutation policy helpers.

This module is intentionally lightweight so both ``skill_manage`` and Curator
telemetry/archive paths can consult the same mutation-boundary policy without
importing the tool registry or agent runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


READONLY_EXTERNAL_SKILL_ERROR = (
    "Skill '{name}' is a pinned/read-only external skill at {path}; "
    "refusing {operation}. External Atlas/shared skills must be unpinned or "
    "explicitly opted into mutability before mutation."
)


def is_under(path: Path, root: Path) -> bool:
    """Return True if *path* resolves under *root*."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def is_external_skill_path(skill_dir: Path) -> bool:
    """Return True if *skill_dir* lives under a configured external skill root."""
    try:
        from agent.skill_utils import get_external_skills_dirs
    except Exception:
        return False

    for root in get_external_skills_dirs():
        if is_under(skill_dir, root):
            return True
    return False


def readonly_external_skill_error(
    name: str,
    skill_dir: Path,
    operation: str,
) -> Optional[str]:
    """Return a fail-closed message for pinned external skills, else None.

    The current policy is deliberately narrow: local pinned skills keep their
    existing behavior, while skills resolved under configured external dirs are
    treated as read-only when pinned in the usage sidecar.

    This check must fail closed when the usage sidecar exists but is unreadable:
    otherwise a corrupt ``.usage.json`` silently erases the pin signal and makes
    shared external skills mutable again.
    """
    if not is_external_skill_path(skill_dir):
        return None
    try:
        from tools import skill_usage
    except Exception:
        return None

    usage_path = skill_usage._usage_file()
    if not usage_path.exists():
        return None

    try:
        data = json.loads(usage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (
            f"Skill '{name}' is an external skill at {skill_dir}; refusing {operation} "
            "because skills/.usage.json is unreadable and pin state cannot be "
            "verified safely. Repair or remove the broken usage sidecar first."
        )
    if not isinstance(data, dict):
        return None

    rec = data.get(name)
    if not isinstance(rec, dict) or not rec.get("pinned"):
        return None
    return READONLY_EXTERNAL_SKILL_ERROR.format(
        name=name,
        path=skill_dir,
        operation=operation,
    )


def find_skill_dir_across_roots(skill_name: str) -> Optional[Path]:
    """Locate a skill directory by directory name or frontmatter name.

    Searches local skills first, then configured external skill dirs, mirroring
    normal skill resolution order.
    """
    try:
        from agent.skill_utils import EXCLUDED_SKILL_DIRS, get_all_skills_dirs
        from agent.skill_utils import parse_frontmatter
    except Exception:
        return None

    for root in get_all_skills_dirs():
        if not root.exists():
            continue
        for skill_md in root.rglob("SKILL.md"):
            if any(part in EXCLUDED_SKILL_DIRS for part in skill_md.parts):
                continue
            if skill_md.parent.name == skill_name:
                return skill_md.parent
            try:
                frontmatter, _body = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            except OSError:
                continue
            if str(frontmatter.get("name", "")).strip() == skill_name:
                return skill_md.parent
    return None
