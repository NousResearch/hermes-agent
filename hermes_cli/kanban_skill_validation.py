"""Fail-closed validation for skills explicitly pinned to Kanban workers."""
from __future__ import annotations

from contextlib import suppress
from typing import Iterable


def unavailable_profile_skills(assignee: str | None, skills: Iterable[str] | None) -> list[str]:
    """Return requested skills that are not effective for ``assignee``."""
    requested = list(dict.fromkeys(str(s).strip() for s in (skills or ()) if str(s).strip()))
    if not requested:
        return []
    if not assignee:
        return requested

    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(get_profile_dir(assignee))
    skills_tool = None
    try:
        from tools import skills_tool

        # Discovery caches are process-global while profile selection is context-local.
        skills_tool._SKILLS_CACHE.clear()
        available = {item["name"] for item in skills_tool._find_all_skills()}
        with suppress(Exception):
            from hermes_cli.plugins import discover_plugins, get_plugin_manager

            discover_plugins()
            available.update(
                item["name"] for item in get_plugin_manager().list_plugin_skill_metadata()
                if not skills_tool._is_skill_disabled(item["name"])
            )
        return [name for name in requested if name not in available]
    finally:
        if skills_tool is not None:
            with suppress(Exception):
                skills_tool._SKILLS_CACHE.clear()
        reset_hermes_home_override(token)


def validate_profile_skills(assignee: str | None, skills: Iterable[str] | None) -> None:
    missing = unavailable_profile_skills(assignee, skills)
    if missing:
        profile = assignee or "<unassigned>"
        raise ValueError(
            f"profile {profile!r} cannot load explicit skill(s): {', '.join(missing)}"
        )
