"""Task-skills normalization for the Kanban DB layer.

Split out of ``hermes_cli.kanban_db`` (FILE_LINES offset): the skills
normalizer and its toolset-name set live here; the facade imports the
normalizer for ``create_task``.
"""

from __future__ import annotations

import json
from pathlib import Path
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


# --- assignee-profile resolution (create/assign-time guard) -------------------------
#
# A well-formed skill name that is installed only in the CREATING profile (or nowhere)
# used to persist fine and abort the dispatched worker at boot — ``ValueError:
# kanban_skills_unavailable`` in gateway.session_kanban — with zero messages, burning the
# card's entire retry budget. Profiles are islands: resolution must happen against the
# ASSIGNEE's trees, never the creating session's. Dispatch-time degradation is the safety
# net for names that slip past this guard (pre-existing cards, DB edits, disabled skills).


def _assignee_home(assignee: str) -> Optional[Path]:
    """HERMES_HOME the *assignee* runs under, or None when that profile is not live on
    this root. Reuses the dispatcher's own resolution (``resolve_profile_env`` requires a
    live named profile and raises ``FileNotFoundError`` otherwise) so create-time and
    dispatch-time resolution cannot drift."""
    from hermes_cli.profiles import resolve_profile_env

    try:
        return Path(resolve_profile_env(assignee))
    except (FileNotFoundError, ValueError):
        return None


def _assignee_skill_roots(home: Path) -> list[tuple[int, Path]]:
    """``(tier, dir)`` roots a worker running AS the assignee searches.

    Binding the assignee's home makes the shared helper read THAT profile's
    ``skills.create_dir`` / ``skills.external_dirs`` (profiles are islands). Project
    tiers are excluded: the card's workspace does not exist yet at create/assign time.
    """
    from agent.skill_utils import get_skill_search_roots
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(home)
    try:
        return get_skill_search_roots(home / "skills", include_project=False)
    finally:
        reset_hermes_home_override(token)


def validate_task_skills_for_assignee(
    skills: Optional[Iterable[str]], assignee: Optional[str],
) -> None:
    """Refuse a card whose skills cannot load where the card will run.

    Raises ``ValueError`` naming every unresolvable skill, the assignee profile and the
    searched dirs, so the mistake surfaces while the card is still being authored instead
    of as a zero-message worker crash-loop. Existence check only: a skill the assignee has
    DISABLED, a plugin-qualified ``plugin:skill`` name (resolved from the worker's plugin
    registry, not these trees) and project-tier skills of a workspace that does not exist
    yet are left to the dispatch-time degradation. No-op when the card carries no skills,
    is unassigned (validated on assign), or the assignee profile is not live on this root
    (fail open: cannot prove absence).
    """
    names = [n for n in (skills or []) if n and ":" not in n]
    if not names or not assignee:
        return
    home = _assignee_home(assignee)
    if home is None:
        return
    from tools.skills_tool import _locate_skill

    try:
        roots = _assignee_skill_roots(home)
        # ``_locate_skill`` is skill_view's own resolution step: cross-tier precedence,
        # same-tier collision refusal, frontmatter-name matching.
        missing = [n for n in names if _locate_skill(n, None, roots)[2] is None]
    except OSError:
        return  # unreadable trees: cannot prove absence -> degrade, never block authoring
    if missing:
        quoted = ", ".join(repr(n) for n in missing)
        searched = ", ".join(str(d) for _tier, d in roots)
        raise ValueError(
            f"task skill(s) {quoted} do not resolve in assignee profile '{assignee}' "
            f"(searched: {searched}). Install the skill into that profile, or remove it "
            "from the card's skills — the dispatched worker can only preload skills "
            "installed in its own profile."
        )


def assert_task_skills_for_assignee(
    conn, task_id: str, assignee: Optional[str],
) -> None:
    """Validate an existing card's stored skills against a NEW assignee.

    ``reassign_task`` calls this BEFORE it reclaims a live claim, so a refused handoff
    mutates nothing (``assign_task`` re-validates inside its own transaction).
    """
    row = conn.execute("SELECT skills FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None or not row["skills"]:
        return
    validate_task_skills_for_assignee(json.loads(row["skills"]), assignee)
