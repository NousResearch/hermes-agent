"""Task skill-list normalization for the Kanban DB create path."""
from __future__ import annotations

from typing import Iterable, Optional

from toolsets import get_toolset_names


def normalize_task_skills(skills: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Strip/dedupe skills and reject comma-joined names or toolset names."""
    if skills is None:
        return None
    known_toolsets = {name.casefold() for name in get_toolset_names()}
    cleaned: list[str] = []
    seen: set[str] = set()
    toolset_typos: list[str] = []
    for item in skills:
        if not item:
            continue
        name = str(item).strip()
        if not name:
            continue
        if "," in name:
            raise ValueError(f"skill name cannot contain comma: {name!r} (pass a list of separate names instead of a comma-joined string)")
        if name.casefold() in known_toolsets:
            toolset_typos.append(name)
            continue
        if name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    if toolset_typos:
        quoted = ", ".join(repr(name) for name in toolset_typos)
        noun = "is a toolset name" if len(toolset_typos) == 1 else "are toolset names"
        raise ValueError(
            f"{quoted} {noun}, not skill name(s). Put toolsets in the assignee profile's `toolsets:` config instead of per-task skills. "
            "Skills are named skill bundles (e.g. `blogwatcher`, `github-code-review`); toolsets are runtime capabilities (e.g. `web`, `browser`, `terminal`)."
        )
    return cleaned
