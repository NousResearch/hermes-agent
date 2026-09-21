"""Forced-skill validation for Kanban cards — one resolver, every creation surface.

A card's forced skills reach the worker as ``hermes --skills <name>`` and are loaded
by :func:`agent.skill_commands.build_preloaded_skills_prompt`, which reports a name it
cannot resolve as *missing* and lets the session continue. Nothing fails, so a typo'd
or category name costs a whole worker run in the wrong context and reads as a bad
model rather than a bad card. Validate once, before the card is queued, and once more
at the spawn boundary for cards written before that existed.

Resolution is scoped to the ASSIGNEE's profile: one gateway process serves many
profiles, and the creator's own skills say nothing about what the executor has
installed. The home is bound with ``set_hermes_home_override`` plus a secret scope for
the duration of the scan (the same scope-then-read ordering
``kanban_db_dispatch._resolve_worker_cli_toolsets`` uses), then released.

The validator fails OPEN whenever it cannot enumerate: an assignee with no profile
directory (control-plane lanes such as ``orion-cc``) or a scan that raises yields no
problems. Refusing a card because the check itself broke would be worse than the bug
it prevents.
"""

from __future__ import annotations

import difflib
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

_MAX_SUGGESTIONS = 3
_MAX_NAME_CHARS = 80
_MAX_REPORTED = 10


@dataclass(frozen=True)
class ForcedSkillProblem:
    """One forced-skill name that will not resolve on the assignee's profile."""

    name: str
    reason: str
    """``"unknown"``, ``"category"`` or ``"disabled"``."""
    suggestions: tuple[str, ...] = ()

    def describe(self) -> str:
        # Clipped: a legacy card's stored name is unbounded, and this text becomes a
        # durable block reason and a chat notification.
        shown = self.name[:_MAX_NAME_CHARS]
        hint = f" (did you mean: {', '.join(self.suggestions)}?)" if self.suggestions else ""
        if self.reason == "category":
            return (f"{shown!r} is a skill CATEGORY, not a skill — force-load one of "
                    f"the skills inside it{hint}")
        if self.reason == "disabled":
            return f"{shown!r} is installed but disabled on this profile"
        return f"{shown!r} is not installed{hint}"


@dataclass(frozen=True)
class _SkillIndex:
    names: frozenset[str]
    categories: frozenset[str]
    by_category: dict[str, tuple[str, ...]]
    disabled: frozenset[str]


def profile_home(assignee: Optional[str]) -> Optional[str]:
    """The assignee profile's ``HERMES_HOME``, or None when there is nothing to scan.

    A missing profile directory is not an error here: unassigned cards and
    control-plane lanes are both legitimate and both unenumerable.
    """
    name = (assignee or "").strip()
    if not name:
        return None
    try:
        from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

        return resolve_profile_env(normalize_profile_name(name))
    except Exception as exc:
        logger.debug("kanban: no profile home for assignee %r (%s)", assignee, exc)
        return None


@contextmanager
def _profile_scope(hermes_home: str) -> Iterator[None]:
    """Bind *hermes_home* (+ its secrets under multiplex) for the enclosed scan."""
    from agent.secret_scope import (
        build_profile_secret_scope, is_multiplex_active, reset_secret_scope, set_secret_scope)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(hermes_home)
    secret_token = None
    try:
        # Inside the try: building the scope reads the profile's ``.env`` and can
        # raise, and an escape here would strand the caller on the assignee's home.
        if is_multiplex_active():
            secret_token = set_secret_scope(build_profile_secret_scope(Path(hermes_home)))
        yield
    finally:
        if secret_token is not None:
            reset_secret_scope(secret_token)
        reset_hermes_home_override(token)


def _installed_skill_index() -> _SkillIndex:
    """Skills visible to the currently bound profile home."""
    from agent.skill_utils import get_disabled_skill_names
    from tools.skills_tool import _find_all_skills

    names: set[str] = set()
    by_category: dict[str, list[str]] = {}
    for entry in _find_all_skills(skip_disabled=True):
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        names.add(name)
        category = str(entry.get("category") or "").strip()
        if category:
            by_category.setdefault(category, []).append(name)
    try:
        disabled = {str(n) for n in get_disabled_skill_names()}
    except Exception:
        disabled = set()
    return _SkillIndex(
        names=frozenset(names),
        categories=frozenset(by_category),
        by_category={k: tuple(sorted(v)) for k, v in by_category.items()},
        disabled=frozenset(disabled),
    )


def _project_skill_names(project_path: Optional[str]) -> frozenset[str]:
    """Skill names a card's project repo contributes, honouring the trust gate.

    Project-local skills (``<repo>/.hermes/skills``) only load from a root listed in
    ``skills.trusted_project_dirs``; an untrusted repo's skills are a prompt-injection
    vector the worker will not load either, so they must not validate a card.
    ``find_project_root`` is given an explicit start: the validator's own cwd is the
    gateway's, not the card's.
    """
    if not project_path:
        return frozenset()
    from agent.skill_utils import (
        _candidate_project_skills_dirs, find_project_root, is_project_root_trusted,
        iter_skill_index_files, parse_frontmatter)

    root = find_project_root(Path(project_path))
    if root is None or not is_project_root_trusted(root):
        return frozenset()
    names: set[str] = set()
    for skills_dir in _candidate_project_skills_dirs(root):
        for skill_md in iter_skill_index_files(skills_dir, "SKILL.md"):
            try:
                frontmatter, _body = parse_frontmatter(skill_md.read_text(encoding="utf-8")[:4000])
            except Exception:
                continue
            names.add(str(frontmatter.get("name") or skill_md.parent.name))
    return frozenset(names)


def _resolves_directly(name: str) -> bool:
    """Last-chance lookup through the exact loader the worker will use.

    The index is the fast path; this catches names the enumerator does not list
    (``plugin:skill`` namespaces, path-shaped identifiers) so validation never refuses
    a card the worker would have loaded fine.
    """
    try:
        from agent.skill_commands import _load_skill_payload

        return _load_skill_payload(name) is not None
    except Exception:
        return False


def _classify(name: str, index: _SkillIndex, project_names: frozenset[str]) -> Optional[ForcedSkillProblem]:
    if name in index.names or name in project_names:
        return None
    if name in index.disabled:
        return ForcedSkillProblem(name, "disabled")
    if name in index.categories:
        return ForcedSkillProblem(name, "category", index.by_category.get(name, ())[:_MAX_SUGGESTIONS])
    if _resolves_directly(name):
        return None
    pool = sorted(index.names | project_names)
    return ForcedSkillProblem(
        name, "unknown", tuple(difflib.get_close_matches(name, pool, n=_MAX_SUGGESTIONS, cutoff=0.6)))


def check_forced_skills(
    skills: Optional[Iterable[str]], *, assignee: Optional[str], project_path: Optional[str] = None,
) -> list[ForcedSkillProblem]:
    """Every forced-skill name that will not resolve on *assignee*'s profile.

    Empty when the list is fine, when there is nothing to check, or when the profile
    could not be enumerated (see the module docstring on failing open).
    """
    wanted = [str(s).strip() for s in (skills or ()) if str(s or "").strip()]
    if not wanted:
        return []
    hermes_home = profile_home(assignee)
    if not hermes_home:
        return []
    try:
        with _profile_scope(hermes_home):
            index = _installed_skill_index()
            project_names = _project_skill_names(project_path)
            return [p for p in (_classify(n, index, project_names) for n in dict.fromkeys(wanted))
                    if p is not None]
    except Exception as exc:
        logger.debug(
            "kanban: forced-skill check skipped for assignee %r (%s)", assignee, exc, exc_info=True)
        return []


def forced_skill_error(problems: list[ForcedSkillProblem], *, assignee: Optional[str]) -> str:
    """One message naming EVERY bad entry — a list of typos must not take N round trips.

    Detail is capped: this text becomes a durable block reason and a chat ping, and a
    card can carry an arbitrary number of names.
    """
    who = (assignee or "").strip() or "the assignee"
    lead = "forced skill is" if len(problems) == 1 else "forced skills are"
    shown, extra = problems[:_MAX_REPORTED], max(0, len(problems) - _MAX_REPORTED)
    details = "; ".join(p.describe() for p in shown) + (f"; (+{extra} more)" if extra else "")
    return (f"{len(problems)} {lead} unavailable on profile {who!r}: {details}. "
            f"List what it has with `hermes -p {who} skills list`, or drop the name from the card.")


def validate_forced_skills(
    skills: Optional[Iterable[str]], *, assignee: Optional[str], project_path: Optional[str] = None,
) -> None:
    """Raise ``ValueError`` describing every unresolvable name, or return quietly."""
    problems = check_forced_skills(skills, assignee=assignee, project_path=project_path)
    if problems:
        raise ValueError(forced_skill_error(problems, assignee=assignee))
