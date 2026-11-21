"""Dispatch-side worker readiness: skill pre-flight and CLI toolset resolution.

Skill pre-flight — never spawn a worker that would hard-exit at init. The review lane
force-loads ``sdlc-review`` onto reviewer workers. When that skill is not on the assignee's
profile, the worker used to die seconds after spawn with ``Error: Unknown skill(s)`` and
crash-loop the single spawn slot (``reason=unknown_skill`` in gateway-service/worker-deaths.log).
Resolution here runs the SAME code path the worker will run — including its transient-miss
retry — so a flaky read is retried, not believed; only a genuinely absent skill blocks the
dispatch, and then it is healed from the fleet source or parked ``needs_input`` instead of
being spawned into a wall.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


def _profile_home_for(assignee: str) -> Optional[str]:
    """Absolute profile home for *assignee*, or ``None`` when it cannot be resolved."""
    if not assignee:
        return None
    try:
        from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

        return resolve_profile_env(normalize_profile_name(assignee))
    except (ValueError, OSError):
        return None


def _preflight_resolve_skills(profile_home: str, skills: Iterable[str]) -> tuple[list[str], list[str]]:
    """``(loaded, missing)`` for *skills* under the ASSIGNEE's profile scope.

    ``build_preloaded_skills_prompt`` is exactly what ``hermes --skills`` runs in the worker, so
    this is a true pre-flight rather than a filename guess: it also catches disabled, ambiguous
    and platform-mismatched skills, and it inherits ``_load_skill_payload``'s bounded retry, so a
    momentary miss on a cold skills tree is retried before being called "missing".
    """
    from hermes_cli.kanban_db_dispatch import _worker_profile_scope

    names = [str(name) for name in skills if name]
    if not names:
        return [], []
    with _worker_profile_scope(profile_home, bind_home=True):
        from agent.skill_commands import build_preloaded_skills_prompt

        _prompt, loaded, missing = build_preloaded_skills_prompt(names)
    return list(loaded), list(missing)


def _fleet_skill_roots(profile_home: str) -> list[Path]:
    """Read-only fleet sources for a missing skill, in trust order: the platform-default home
    (the fleet baseline every profile is bootstrapped from), then every other profile's skills
    dir. The assignee's own dir is excluded — it is the one that is missing the skill.
    """
    from hermes_constants import _get_platform_default_hermes_home

    default_home = Path(_get_platform_default_hermes_home())
    try:
        current = Path(profile_home).resolve()
    except OSError:
        current = Path(profile_home)
    homes: list[Path] = [default_home]
    profiles_root = default_home / "profiles"
    try:
        candidates = sorted(profiles_root.iterdir()) if profiles_root.is_dir() else []
    except OSError:
        candidates = []
    for entry in candidates:
        if not entry.is_dir():
            continue
        try:
            if entry.resolve() == current:
                continue
        except OSError:
            continue
        homes.append(entry)
    return [home / "skills" for home in homes if (home / "skills").is_dir()]


def _find_fleet_skill(skill_id: str, roots: Iterable[Path]) -> Optional[tuple[Path, Path]]:
    """``(search_root, skill_dir_or_file)`` for *skill_id* in the first root that carries it.

    Mirrors skill_view's three lookup strategies — category-qualified path
    (``devops/sdlc-review``), bare directory name, and the frontmatter ``name:`` — so a skill
    that resolves for one profile resolves identically once installed for another.
    """
    from tools.skills_tool import _safe_frontmatter
    from agent.skill_utils import iter_skill_index_files

    identifier = (skill_id or "").strip().strip("/")
    if not identifier:
        return None
    for root in roots:
        try:
            direct = root / identifier
            if (direct / "SKILL.md").is_file():
                return root, direct
            flat = root / f"{identifier}.md"
            if flat.is_file():
                return root, flat
            for skill_md in iter_skill_index_files(root, "SKILL.md"):
                if skill_md.parent.name == identifier:
                    return root, skill_md.parent
                if str(_safe_frontmatter(skill_md).get("name") or "") == identifier:
                    return root, skill_md.parent
        except OSError:
            continue
    return None


def _install_skill_from_fleet(profile_home: str, skill_id: str) -> Optional[str]:
    """Copy *skill_id* from the fleet source into the assignee's skills dir.

    Returns the source path it was installed from, or ``None`` when no fleet root carries it
    (or the copy failed) — the caller then parks the card instead of spawning.
    """
    try:
        profile_skills = Path(profile_home) / "skills"
        profile_skills.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("kanban dispatcher: cannot prepare %s/skills for skill %s: %s",
                       profile_home, skill_id, exc)
        return None
    found = _find_fleet_skill(skill_id, _fleet_skill_roots(profile_home))
    if not found:
        return None
    root, source = found
    try:
        if source.is_dir():
            try:
                rel = source.resolve().relative_to(root.resolve())
            except (OSError, ValueError):
                rel = Path(skill_id)
            dest = profile_skills / rel
            if dest.resolve() == source.resolve():
                return None
            shutil.copytree(source, dest, dirs_exist_ok=True)
        else:
            dest = profile_skills / source.name
            if dest.resolve() == source.resolve():
                return None
            shutil.copy2(source, dest)
    except (OSError, shutil.Error) as exc:
        logger.warning("kanban dispatcher: fleet copy of skill %s -> %s failed: %s",
                       skill_id, profile_home, exc)
        return None
    logger.info("kanban dispatcher: installed fleet skill %s for profile %s (from %s)",
                skill_id, profile_home, source)
    return str(source)


def preflight_worker_skills(task: Any, profile_home: Optional[str]) -> Optional[str]:
    """Check every forced skill against the assignee's profile BEFORE the worker is spawned.

    ``None`` → all resolve (installing any that were missing) and the spawn proceeds. Otherwise
    the returned message is what parks the card: a config gap a retry cannot fix must surface to
    a human, never be rediscovered as a crash loop.
    """
    skills = [str(skill) for skill in (task.skills or []) if skill]
    if not skills or not profile_home:
        return None
    # Reasons are process-global: drop this check's stale entries first so the detail below
    # describes THIS profile's resolution, not a previous profile's.
    from agent.skill_commands import _skill_resolution_reasons, skill_resolution_reasons

    for name in skills:
        _skill_resolution_reasons.pop(name, None)
    _loaded, missing = _preflight_resolve_skills(profile_home, skills)
    if not missing:
        return None
    installed: list[str] = []
    for name in list(missing):
        source = _install_skill_from_fleet(profile_home, name)
        if source:
            installed.append(f"{name} (from {source})")
    if installed:
        # Believe nothing until it resolves: the copy must actually be loadable.
        _loaded, missing = _preflight_resolve_skills(profile_home, skills)
    if not missing:
        return None
    reasons = skill_resolution_reasons(missing)
    reason_text = (": " + "; ".join(f"{name} -> {reasons[name].rstrip('.')}"
                                    for name in missing if name in reasons)) if reasons else ""
    roots = _fleet_skill_roots(profile_home)
    return (
        f"Pre-flight skill check blocked this spawn — profile '{task.assignee}' cannot load "
        f"skill(s): {', '.join(missing)}{reason_text} "
        f"Searched {len(roots)} fleet source(s) ({', '.join(str(r) for r in roots[:4])}"
        f"{'…' if len(roots) > 4 else ''}) with nothing to install. "
        f"Needs input: install the skill on that profile or drop it from this task's skills — "
        f"the worker was not started, so this card cannot crash-loop on `Unknown skill(s)`."
    )


def block_for_skill_preflight(conn: sqlite3.Connection, task: Any, message: str, result: Any) -> None:
    """Park a card whose forced skill does not resolve: comment + ``needs_input`` block.

    Deliberately does NOT spend the card's retry budget — nothing about the task was tried, the
    gap is a coverage problem, and ``needs_input`` is sticky until a human fixes it.
    """
    from hermes_cli import kanban_db as _kb
    from hermes_cli.kanban_db_dispatch import _record_task_failure

    _kb._log.warning("kanban dispatcher: pre-flight skill check failed for %s: %s", task.id, message)
    try:
        _kb.add_comment(conn, task.id, "dispatcher", message)
    except sqlite3.Error as exc:
        _kb._log.debug("kanban dispatcher: pre-flight comment for %s failed: %s", task.id, exc)
    blocked = False
    try:
        blocked = _kb.block_task(
            conn, task.id, kind="needs_input", reason=message[:500],
            expected_run_id=task.current_run_id,
        )
    except sqlite3.Error as exc:
        _kb._log.warning("kanban dispatcher: pre-flight block for %s failed: %s", task.id, exc)
    if not blocked:
        # Defensive: release the claim immediately rather than letting it ride out its TTL.
        _record_task_failure(
            conn, task.id, message, outcome="spawn_failed", failure_limit=None,
            release_claim=True, end_run=True, force_trip=True,
        )
    result.auto_blocked.append(task.id)


def _resolve_worker_cli_toolsets(hermes_home: Optional[str]) -> Optional[list[str]]:
    """Return the assigned profile's effective CLI toolsets for a worker.

    Resolved at dispatch time and passed as an explicit ``--toolsets`` pin so
    worker startup cannot fall back to a stale root/active-profile config or a
    profile whose top-level ``toolsets`` is only the kanban orchestrator
    surface. ``model_tools`` still appends the task-scoped kanban lifecycle
    tools when ``HERMES_KANBAN_TASK`` is set.
    """
    if not hermes_home:
        return None
    try:
        from hermes_cli import kanban_db_dispatch as _dispatch
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        with _dispatch._worker_profile_scope(hermes_home):
            cfg = load_config()
            toolsets = sorted(_get_platform_tools(cfg, "cli"))
        return toolsets or None
    except Exception as exc:
        from hermes_cli import kanban_db as _kb

        _kb._log.debug(
            "kanban worker: could not resolve CLI toolsets for HERMES_HOME=%r (%s)",
            hermes_home,
            exc,
        )
        return None
