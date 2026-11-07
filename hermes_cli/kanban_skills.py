"""Resolve a Kanban card's per-task skills against the ASSIGNEE profile's own tree.

Skills are profile islands: the board is shared, but a worker resolves ``skills`` from its own
profile's HERMES_HOME. A card naming a skill its assignee cannot load boots a worker that dies
before its first turn (``kanban_skills_unavailable``), spending the card's dispatch-failure budget
on the same opaque boot — the 2026-11-02 Wyze Scale incident. The create surfaces validate here so
such a card is refused with the skill and profile named instead of being queued unrunnable.
"""

from __future__ import annotations

from typing import Optional, Sequence

__all__ = ['KanbanSkillsUnavailable', 'skills_refusal', 'unavailable_error', 'unresolvable_skills']


class KanbanSkillsUnavailable(ValueError):
    """The card named skills its assignee profile cannot load.

    Only raised with a message this module composed (skill names + profile), so the owner may
    persist it on the board: unlike an arbitrary runtime exception, it carries no credentials.
    """


def _assignee_home(assignee: str):
    """The assignee profile's HERMES_HOME, or ``None`` when this host has no live such profile."""
    from hermes_cli.profiles import get_profile_dir, profile_exists

    name = (assignee or '').strip()
    if not name or not profile_exists(name):
        return None
    try:
        return get_profile_dir(name)
    except ValueError:
        return None


def unresolvable_skills(assignee: str, skills: Sequence[str]) -> list[str]:
    """Identifiers in *skills* the *assignee* profile's own tree cannot resolve, in input order.

    Mirrors a worker boot's lookup (same roots and name/path aliases as ``skill_view``) but leaves
    out the creator's trusted project dirs: the worker runs in the card's workspace, so a skill
    only the creator's repo carries is not loadable there. An assignee with no live profile on
    this host is left to the dispatcher's own unknown-assignee refusal, and ``plugin:`` skills are
    skipped — they are not profile-tree files.
    """
    wanted = [s.strip() for s in (skills or ()) if isinstance(s, str) and s.strip()]
    if not wanted:
        return []
    home = _assignee_home(assignee)
    if home is None:
        return []
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        return _unresolved_in(home, wanted)
    finally:
        reset_hermes_home_override(token)


def _unresolved_in(home, wanted: Sequence[str]) -> list[str]:
    """*wanted* identifiers missing from the skill roots of the profile whose home is *home*."""
    from agent.skill_utils import get_skill_search_roots, normalize_skill_lookup_name
    from tools.skills_tool import locate_skill

    roots = get_skill_search_roots(home / 'skills', include_project=False)
    missing = []
    for identifier in wanted:
        if ':' in identifier:  # plugin-provided skill: not a profile-tree file, cannot judge here
            continue
        error, _skill_dir, skill_md = locate_skill(normalize_skill_lookup_name(identifier), None, roots)
        if error is not None or skill_md is None:
            missing.append(identifier)
    return missing


def skills_refusal(assignee: str, skills: Sequence[str]) -> Optional[str]:
    """Actionable create-time refusal for a card the assignee could not boot, else ``None``."""
    missing = unresolvable_skills(assignee, skills)
    if not missing:
        return None
    names = ', '.join(repr(name) for name in missing)
    return (f"profile {assignee!r} cannot load skill(s) {names}: not in that profile's skill "
            f"tree, so a worker would die at boot (kanban_skills_unavailable). Install them "
            f"there (hermes -p {assignee} skills) or drop them from this card.")


def unavailable_error(profile: Optional[str], missing: Sequence[str]) -> KanbanSkillsUnavailable:
    """The boot failure a card raises when NONE of its requested skills resolved.

    Names every missing skill and the profile they were resolved against: this text reaches
    ``task_runs.error`` and the console line, so a human never sees a bare token again.
    """
    names = ', '.join(str(name) for name in missing)
    profile_name = profile or '?'
    return KanbanSkillsUnavailable(
        f'kanban_skills_unavailable: {names} missing on profile {profile_name!r}')
