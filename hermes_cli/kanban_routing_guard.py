"""Routing guard: a card that edits a profile's own instruction surface never
routes to that profile.

Why this exists
---------------
The decomposer picks assignees by matching task text against profile
descriptions, so a card about "wordsmith" names wordsmith and matches
wordsmith - the seat that owns the file it edits. That failure is structural:
no prompt wording fixes it. On 2026-10-05 card ``t_e98a4f1a`` ("add the 'Sir'
passage to Wordsmith SOUL.md") spawned a wordsmith worker 38 seconds after
creation and had to be reclaimed before it wrote anything. Operator rule: a
seat cannot edit its own instruction surface (precedent: ``t_5062f83c``,
routed to devops).

This module is the mechanical guard. It detects instruction-surface paths in
card text; when the chosen assignee equals the profile that owns a matched
surface, it rewrites the route to the fallback lane
(``kanban.routing_guard_fallback_assignee``, default ``devops``), else to
``kanban.default_assignee``. When neither is a legal alternative the card is
left UNASSIGNED - the "never owned by the surface's profile" invariant wins
over "never unassigned". Every rewrite is auditable through a ``routing_guard``
task event plus a comment on the card (see :func:`record_guard_decision`).

Detected surfaces (case-insensitive file names, relative or absolute path):

* ``profiles/<name>/SOUL.md``
* ``profiles/<name>/USER.md``
* ``profiles/<name>/MEMORY.md`` (bare form)
* ``profiles/<name>/memories/MEMORY.md`` and ``.../memories/USER.md``
* the default profile's home-root equivalents: ``~/.hermes/SOUL.md``,
  ``~/.hermes/memories/MEMORY.md``, ``~/.hermes/memories/USER.md`` (profile
  ``default``)

Usage for callers that write an assignee:

    decision = guard_assignee(chosen, task.title, task.body)
    if decision.rewritten:
        apply_assignee(conn, task.id, decision.assignee)  # None = unassign
        record_guard_decision(conn, task.id, decision)

Import discipline: stdlib only at import time. ``hermes_cli.profiles``,
``hermes_cli.config`` and ``hermes_cli.kanban_db`` are imported lazily inside
functions so the decomposer (and any other consumer) can import this module
without cycles.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

__all__ = [
    "DEFAULT_FALLBACK_ASSIGNEE",
    "ROUTING_GUARD_EVENT_KIND",
    "GuardDecision",
    "InstructionSurfaceRef",
    "configured_fallback_assignee",
    "find_instruction_surface_refs",
    "guard_assignee",
    "record_guard_decision",
]

#: Fallback lane when ``kanban.routing_guard_fallback_assignee`` is unset or
#: unreadable. Mirrors the DEFAULT_CONFIG entry in ``config_defaults.py``.
DEFAULT_FALLBACK_ASSIGNEE = "devops"

#: Event kind written by :func:`record_guard_decision` on the affected card.
ROUTING_GUARD_EVENT_KIND = "routing_guard"

#: Canonical name of the default profile (home root ``~/.hermes``).
_DEFAULT_PROFILE = "default"

# lane_source values
_LANE_FALLBACK = "fallback"
_LANE_DEFAULT_ASSIGNEE = "default_assignee"
_LANE_NONE = "none"

#: Surface file name, case-insensitive, not followed by an alphanumeric (so
#: ``SOUL.md.bak`` still matches, ``SOUL.mdx`` does not).
_SURFACE_FILENAME = r"(?P<filename>(?i:(?:SOUL|USER|MEMORY)\.md))(?![A-Za-z0-9])"

#: ``profiles/<name>/[memories/]<surface>`` - relative or absolute. The
#: lookbehind keeps ``myprofiles/...`` from matching.
_PROFILE_SURFACE_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])profiles/"
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9_-]{0,63})/"
    r"(?:memories/)?" + _SURFACE_FILENAME
)

#: Home-root surface of the default profile: ``<...>/.hermes/[memories/]<surface>``.
#: ``~/.hermes/profiles/<name>/...`` does NOT match here (the profile regex owns it).
_HOME_SURFACE_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])\.hermes/"
    r"(?:memories/)?" + _SURFACE_FILENAME
)


@dataclass(frozen=True)
class InstructionSurfaceRef:
    """One instruction-surface reference found in card text."""

    profile: str          # canonical profile id that owns the surface
    matched_path: str     # the path text as written in the card
    surface: str          # "SOUL" | "USER" | "MEMORY"
    default_home: bool = False  # True for the home-root (default profile) form


@dataclass(frozen=True)
class GuardDecision:
    """Outcome of :func:`guard_assignee`.

    Callers must branch on ``rewritten`` (never on a string comparison): when
    it is True they must APPLY ``assignee`` - including ``None``, which means
    "leave the card unassigned" - and record the audit via
    :func:`record_guard_decision`.
    """

    assignee: Optional[str]       # resulting assignee; None = leave unassigned
    rewritten: bool
    original_assignee: Optional[str]
    reason: str = ""
    lane_source: str = ""         # "fallback" | "default_assignee" | "none" | ""
    matched_profiles: list[str] = field(default_factory=list)
    matched_paths: list[str] = field(default_factory=list)

    def audit_payload(self) -> dict:
        """Payload for the ``routing_guard`` task event / audit comment."""
        return {
            "original_assignee": self.original_assignee,
            "new_assignee": self.assignee,
            "reason": self.reason,
            "matched_profiles": list(self.matched_profiles),
            "matched_paths": list(self.matched_paths),
            "lane_source": self.lane_source,
        }


def _canonical_name(value: object) -> str:
    """Assignment-name normalization (strip + lowercase, ``default`` special).

    Mirrors ``hermes_cli.profiles.normalize_profile_name`` for the comparisons
    this module makes; kept local so a missing/broken profiles import can never
    disable the guard.
    """
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    if not stripped:
        return ""
    return "default" if stripped.casefold() == "default" else stripped.lower()


def _iter_strings(values: tuple) -> Iterator[str]:
    """Flatten nested iterables of text (title + body + comments are usually
    passed as separate args, but callers sometimes hand a list)."""
    for value in values:
        if isinstance(value, str):
            if value:
                yield value
        elif isinstance(value, (list, tuple, set, frozenset)):
            yield from _iter_strings(tuple(value))


def _surface_of(filename: str) -> str:
    return filename.split(".", 1)[0].upper()


def find_instruction_surface_refs(*texts) -> list[InstructionSurfaceRef]:
    """Return every profile instruction-surface reference found in *texts*.

    Order-preserving and deduped by ``(profile, matched_path, surface)``.
    Non-string arguments (None, nested lists of strings) are tolerated.
    """
    refs: list[InstructionSurfaceRef] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(profile: str, matched_path: str, surface: str, default_home: bool) -> None:
        canonical = _canonical_name(profile)
        if not canonical:
            return
        key = (canonical, matched_path, surface)
        if key in seen:
            return
        seen.add(key)
        refs.append(
            InstructionSurfaceRef(
                profile=canonical, matched_path=matched_path, surface=surface,
                default_home=default_home,
            )
        )

    for text in _iter_strings(texts):
        for match in _PROFILE_SURFACE_RE.finditer(text):
            _add(match.group("name"), match.group(0), _surface_of(match.group("filename")), False)
        for match in _HOME_SURFACE_RE.finditer(text):
            _add(_DEFAULT_PROFILE, match.group(0), _surface_of(match.group("filename")), True)
    return refs


def _configured_assignee_key(key: str, default: str) -> str:
    """Read ``kanban.<key>`` through ``load_config_readonly`` (the loader the
    decomposer uses). Missing key or unreadable config yields *default*; a
    non-string value is treated as unset; strings are stripped (an explicit
    empty string is a real value - it disables a candidate lane)."""
    value: object = None
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        kanban_cfg = cfg.get("kanban") if isinstance(cfg, dict) else None
        if isinstance(kanban_cfg, dict):
            value = kanban_cfg.get(key)
    except Exception:
        value = None
    if not isinstance(value, str):
        return default
    return value.strip()


def configured_fallback_assignee() -> str:
    """``kanban.routing_guard_fallback_assignee`` (default ``devops``).

    An explicit empty string disables the fallback lane, so the guard goes
    straight to ``kanban.default_assignee``.
    """
    return _configured_assignee_key("routing_guard_fallback_assignee", DEFAULT_FALLBACK_ASSIGNEE)


def guard_assignee(
    assignee: Optional[str],
    *texts,
    fallback_assignee: Optional[str] = None,
    default_assignee: Optional[str] = None,
    eligible_names: Optional[Iterable[str]] = None,
) -> GuardDecision:
    """Apply the routing guard to *assignee* given the card's *texts*.

    ``fallback_assignee`` / ``default_assignee`` default to the config values
    (``kanban.routing_guard_fallback_assignee``, ``kanban.default_assignee``);
    pass ``""`` explicitly to disable a lane. ``eligible_names``, when given,
    restricts candidate lanes to that collection of profile names (the
    decomposer's roster); ``None`` skips the check.

    Rewrites only when the canonical assignee equals a profile that owns a
    matched surface. Candidate lanes must be non-empty and must not own any
    matched surface themselves. With no eligible lane the decision is
    ``assignee=None, rewritten=True`` (left unassigned) and the reason says why.
    """
    refs = find_instruction_surface_refs(*texts)
    original = assignee
    target = _canonical_name(assignee)
    matched_profiles = list(dict.fromkeys(ref.profile for ref in refs))
    matched_paths = list(dict.fromkeys(ref.matched_path for ref in refs))

    if not target or not matched_profiles or target not in set(matched_profiles):
        return GuardDecision(
            assignee=original,
            rewritten=False,
            original_assignee=original,
            matched_profiles=matched_profiles,
            matched_paths=matched_paths,
        )

    owners = set(matched_profiles)
    eligible: Optional[set[str]] = None
    if eligible_names is not None:
        eligible = {name for name in (_canonical_name(n) for n in eligible_names) if name}

    resolved_fallback = (
        fallback_assignee if fallback_assignee is not None else configured_fallback_assignee()
    )
    resolved_default = (
        default_assignee if default_assignee is not None
        else _configured_assignee_key("default_assignee", "")
    )

    paths = ", ".join(matched_paths)
    base = (
        f"card edits an instruction surface owned by assignee {original!r} "
        f"({paths})"
    )

    for candidate, lane_source, label in (
        (resolved_fallback, _LANE_FALLBACK, "kanban.routing_guard_fallback_assignee"),
        (resolved_default, _LANE_DEFAULT_ASSIGNEE, "kanban.default_assignee"),
    ):
        lane = _canonical_name(candidate)
        if not lane or lane in owners:
            continue
        if eligible is not None and lane not in eligible:
            continue
        return GuardDecision(
            assignee=lane,
            rewritten=True,
            original_assignee=original,
            reason=f"{base}; rerouted to {lane!r} via {label}",
            lane_source=lane_source,
            matched_profiles=matched_profiles,
            matched_paths=matched_paths,
        )

    return GuardDecision(
        assignee=None,
        rewritten=True,
        original_assignee=original,
        reason=(
            f"{base}; no eligible alternative lane "
            f"(fallback {resolved_fallback!r}, default_assignee {resolved_default!r}) "
            "- left unassigned"
        ),
        lane_source=_LANE_NONE,
        matched_profiles=matched_profiles,
        matched_paths=matched_paths,
    )


def _format_audit_comment(decision: GuardDecision) -> str:
    lines = [
        "Routing guard fired: the chosen assignee owns an instruction surface this card edits.",
        f"original_assignee: {decision.original_assignee or '(none)'}",
        f"new_assignee: {decision.assignee or '(none - left unassigned)'}",
        f"lane_source: {decision.lane_source}",
        f"reason: {decision.reason}",
    ]
    if decision.matched_paths:
        lines.append("matched_paths:")
        lines.extend(f"  - {path}" for path in decision.matched_paths)
    return "\n".join(lines)


def record_guard_decision(
    conn,
    task_id: str,
    decision: GuardDecision,
    *,
    author: str = "routing_guard",
) -> None:
    """Persist the audit trail for a fired guard: one ``routing_guard`` task
    event (payload from ``decision.audit_payload()``) plus a comment on the
    affected card.

    No-op when ``decision.rewritten`` is False. Safe both inside an open write
    transaction (savepoint) and standalone. This does NOT change the task's
    assignee - applying ``decision.assignee`` is the caller's job, so the audit
    and the assignment stay in one place at the integration seam.
    """
    if decision is None or not decision.rewritten:
        return
    from hermes_cli import kanban_db as kb

    author = (author or "").strip() or "routing_guard"
    with kb.write_txn(conn, allow_nested=True):
        kb._require_task(conn, task_id)
        kb._insert_comment(conn, task_id, author, _format_audit_comment(decision), int(time.time()))
        kb._append_event(conn, task_id, ROUTING_GUARD_EVENT_KIND, decision.audit_payload())
