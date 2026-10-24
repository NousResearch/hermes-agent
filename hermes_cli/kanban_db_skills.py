"""Task-skills normalization for the Kanban DB layer.

Split out of ``hermes_cli.kanban_db`` (FILE_LINES offset): the skills
normalizer and its toolset-name set live here; the facade imports the
normalizer for ``create_task``.
"""

from __future__ import annotations

from typing import Iterable, Optional

from toolsets import get_toolset_names

KNOWN_TOOLSET_NAMES = frozenset(name.casefold() for name in get_toolset_names())


def _normalize_task_skills(skills: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Strip/dedupe a skills list. Commas and slashes are refused (a joined
    string must not land in one argv slot — a slash-joined phrase is two names,
    not a path-style skill reference); toolset names are rejected all at once
    because agents that confuse the two usually pass several."""
    if skills is None:
        return None
    cleaned: list[str] = []
    seen: set[str] = set()
    toolset_typos: list[str] = []
    for s in skills:
        if not s:
            continue
        name = str(s).strip()
        if not name:
            continue
        if "," in name:
            raise ValueError(
                f"skill name cannot contain comma: {name!r} "
                f"(pass a list of separate names instead of a comma-joined string)"
            )
        if "/" in name:
            raise ValueError(
                f"skill name cannot contain '/': {name!r} "
                "(a slash-joined phrase is not one skill — pass separate names, "
                "e.g. ['github-pr-workflow', 'github-code-review'])"
            )
        if name.casefold() in KNOWN_TOOLSET_NAMES:
            toolset_typos.append(name)
            continue
        if name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    if toolset_typos:
        quoted = ", ".join(repr(n) for n in toolset_typos)
        noun = "is a toolset name" if len(toolset_typos) == 1 else "are toolset names"
        raise ValueError(
            f"{quoted} {noun}, not skill name(s). "
            "Put toolsets in the assignee profile's `toolsets:` config "
            "instead of per-task skills. Skills are named skill bundles "
            "(e.g. `blogwatcher`, `github-code-review`); toolsets are runtime "
            "capabilities (e.g. `web`, `browser`, `terminal`)."
        )
    return cleaned
