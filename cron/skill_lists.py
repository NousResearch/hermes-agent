"""Skill-name normalization shared by the cron store, the prompt assembler, the CLI, and the
``cronjob`` tool (issue #132867).

Deliberately stdlib-only and free of ``cron.jobs``: a long-running scheduler can import this
module while the tree on disk is mid-upgrade, and importing the store from a tick path is the
skew failure ``tests/cron/test_upgrade_module_skew.py`` exists to prevent.
"""

from __future__ import annotations

import ast
from typing import Any


def _names_literal(text: str) -> list[str] | None:
    """``text`` as the skill names it spells, or ``None`` when it does not spell any.

    Only a Python list/tuple literal of bare strings unwraps: ``"['x', 'y']"`` -> ``["x", "y"]``.
    ``"[1, 2]"``, ``"[weird-name]"`` and ``"'x'"`` are left alone, so a skill that merely looks
    like a literal is never mangled into names it never had. An empty literal (``"[]"``) *does*
    spell a list — of no names — so it unwraps to ``[]`` rather than becoming a skill called
    ``"[]"``.
    """
    body = text.strip()
    if len(body) < 2 or not ((body[0] == "[" and body[-1] == "]")
                             or (body[0] == "(" and body[-1] == ")")):
        return None
    try:
        parsed = ast.literal_eval(body)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(parsed, (list, tuple)):
        return None
    names: list[str] = []
    for item in parsed:
        if not isinstance(item, str):
            return None
        nested = _names_literal(item)
        names.extend(nested if nested is not None else [item])
    return names


def unwrap_skill_repr(items: Any) -> Any:
    """Return ``items`` with repr-of-a-list skill names replaced by the names they spell.

    A model caller can pass ``skills="['x', 'y']"`` instead of a list. Normalizers used to
    ``str()`` that string into one bogus skill literally named ``['x', 'y']``, which nothing
    could load: the job ran without its skills, silently, and ``hermes cron doctor`` reported
    the record clean. Both shapes a store can hold are healed — the bare string, and the
    ``["['x', 'y']"]`` a pre-fix normalizer wrote when it wrapped the string in a list.

    Anything that is not such a literal comes back as the *same object*, so a caller can detect
    an unwrap by identity (``hermes_cli.cron._cron_doctor_issues_for_job`` does).
    """
    if isinstance(items, str):
        names = _names_literal(items)
        return items if names is None else names
    if isinstance(items, (list, tuple)):
        unwrapped = False
        names: list[str] = []
        for item in items:
            nested = _names_literal(item) if isinstance(item, str) else None
            unwrapped = unwrapped or nested is not None
            names.extend(nested if nested is not None else [item])
        return names if unwrapped else items
    return items


def normalize_skill_names(skill: str | None = None, skills: Any = None) -> list[str]:
    """The canonical skill list for a job record or a create/update payload.

    Accepts the legacy singular ``skill``, the plural ``skills`` (str, list or tuple), or both,
    and returns unique stripped names in order. A repr-shaped value is unwrapped first, so a job
    created from ``skills="['a', 'b']"`` gets the two skills it asked for, and a record already
    holding that bogus name heals as it is read (#132867).
    """
    skills = unwrap_skill_repr(skills)
    skill = unwrap_skill_repr(skill)
    # A repr-shaped `skill` can unwrap to a whole list ("['a', 'b']" -> ['a', 'b']);
    # treat that as the skills list rather than one entry wrapping it.
    if isinstance(skill, (list, tuple)):
        skills = list(skill) if skills is None else skills
        skill = None
    if skills is None:
        raw_items = [skill] if skill else []
    elif isinstance(skills, str):
        raw_items = [skills]
    else:
        raw_items = list(skills)
    normalized: list[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized
