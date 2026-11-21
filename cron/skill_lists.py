"""Skills-list healing, import-free on purpose.

A model calling the ``cronjob`` tool can pass ``skills`` as the python repr of the list
(``"['x']"``) instead of the list itself; the job then stores one bogus skill name and the
skills never load at run time. Both the job store (``cron.jobs``) and prompt assembly
(``cron.scheduler_prompt``) must heal that shape on read, and prompt assembly previously
reached into ``cron.jobs`` for it — a lazy ``cron.jobs`` import in a module that had no
dependency on the store, so an entrypoint that imports ``cron.scheduler_prompt`` without a
usable ``cron.jobs`` broke skill loading entirely.

This module depends only on the stdlib, like ``cron.constants`` and ``cron.env_settings``, so
any cron sibling can import the heal without going through the store.
"""

import ast
import contextlib
from typing import Any, Optional


def _skill_list_items(value: Any) -> list[Any]:
    """Flatten one ``skills`` entry that a caller stringified as a whole (an LLM passing
    ``"['x']"`` or ``['['x']']`` instead of ``['x']``). A str/tuple/list/dict literal that
    parses back to bare strings is unwrapped in place; anything else is kept verbatim so a
    skill name that merely looks like a literal is never mangled. Nested lists are flattened
    one level deep — the shape tool callers actually produce."""
    if not isinstance(value, str):
        if isinstance(value, (list, tuple)):
            flat: list[Any] = []
            for item in value:
                if isinstance(item, str):
                    flat.append(item)
                elif isinstance(item, (list, tuple)):
                    flat.extend(item)
                else:
                    flat.append(item)
            return flat
        return [value]
    text = value.strip()
    if not (text.startswith(("[", "{")) and text.endswith(("]", "}"))):
        return [value]
    with contextlib.suppress(Exception):
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            if all(isinstance(p, str) for p in parsed):
                return list(parsed)
        elif isinstance(parsed, dict):
            # A dict literal whose values are all strings: values are the skill names
            # (keys were positional artifacts of the caller).
            if parsed and all(isinstance(v, str) for v in parsed.values()):
                return list(parsed.values())
    return [value]


def _normalize_skill_list(skill: Optional[str] = None, skills: Optional[Any] = None) -> list[str]:
    """Normalize legacy/single-skill and multi-skill inputs into a unique ordered list."""
    if skills is None:
        raw_items = [skill] if skill else []
    elif isinstance(skills, str):
        raw_items = [skills]
    else:
        raw_items = list(skills)
    normalized: list[str] = []
    for item in raw_items:
        for sub in _skill_list_items(item):
            text = str(sub or "").strip()
            if text and text not in normalized:
                normalized.append(text)
    return normalized
