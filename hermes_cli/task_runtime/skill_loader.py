"""Skill Loader for Task Runtime.

Loads skill metadata (name, description, path, presence) for the
skills suggested by IntentResolver. Does NOT execute skill code; it
only confirms that each suggested skill is installed (or not) and
returns metadata for TaskContract.

NEVER mutates ~/.hermes/skills/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os


@dataclass(frozen=True)
class SkillRecord:
    """Metadata for a single skill."""
    skill_name: str
    description: str = ""
    installed: bool = False
    skill_path: str | None = None
    source: str = "unknown"  # "bundled" | "user" | "optional" | "missing"


def _skills_root(hermes_home: Path) -> Path:
    return hermes_home / "skills"


def _find_skill(hermes_home: Path, name: str) -> SkillRecord:
    """Look for a skill named `name` under the standard skill directories."""
    roots = [
        _skills_root(hermes_home),
        Path(os.environ.get("HERMES_REPO", "")) / "skills" if os.environ.get("HERMES_REPO") else None,
        Path(os.environ.get("HERMES_REPO", "")) / "optional-skills" if os.environ.get("HERMES_REPO") else None,
    ]
    roots = [r for r in roots if r is not None and r.exists()]
    for root in roots:
        for sub in (root.rglob(f"SKILL.md") if root.name == "skills" else [root / "SKILL.md"]):
            try:
                text = sub.read_text(encoding="utf-8", errors="ignore")
                m_name = re.search(r"^name:\s*(\S+)", text, re.M)
                if m_name and m_name.group(1) == name:
                    m_desc = re.search(r"^description:\s*(.+)$", text, re.M)
                    desc = m_desc.group(1).strip() if m_desc else ""
                    if "optional-skills" in str(sub):
                        source = "optional"
                    elif str(sub).startswith(str(Path(os.environ.get("HERMES_REPO", "")) / "skills")):
                        source = "bundled"
                    else:
                        source = "user"
                    return SkillRecord(
                        skill_name=name,
                        description=desc,
                        installed=True,
                        skill_path=str(sub),
                        source=source,
                    )
            except Exception:
                continue
    return SkillRecord(skill_name=name, description="", installed=False, skill_path=None, source="missing")


import re


def load(resolved_intent, context: dict[str, Any]) -> list[SkillRecord]:
    """Load SkillRecord list for the suggested skills.

    Pure metadata: returns SkillRecord entries (installed or not) without
    executing any skill code.
    """
    hermes_home = Path(context.get("hermes_home", str(Path.home() / ".hermes")))
    out: list[SkillRecord] = []
    seen: set[str] = set()
    for name in resolved_intent.suggested_skills:
        if name in seen:
            continue
        seen.add(name)
        out.append(_find_skill(hermes_home, name))
    return out