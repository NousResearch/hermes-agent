"""Deterministic mint-time lane -> assignee role-map (single source of truth).

Both the ``kanban_create`` mint-time lint (``tools/kanban_tools.py``) and the
auto-decomposer's child routing (``hermes_cli/kanban_decompose.py``) derive an
assignee from a card title's LANE here, so an LLM's free-text roster pick can
never route an ``Implement``/``[Bob]`` (build lane) card to rodge/steve-o
(review/verify lanes), and a real-but-wrong profile is rejected at mint time.

The role-map (2026-09-12 platform fix-C1, card t_bc08efc3):

    build  -> bob        (Implement / Build: / FIX / Create / [Bob] / Bob —)
    review -> rodge      (Review / re-review / [Rodge])
    verify -> steve-o    (Verify / QA / real-click / [Steve-o])
    design -> karl       ([Karl])
    deploy -> default    (Deploy; HELD at creation)
    pm     -> jobsy      (no title marker; PM/triage work is routed explicitly)

``default`` is the deploy-lane owner (a released deploy card is created HELD,
never dispatched to a person). ``jobsy`` owns the pm lane; there is no
title-shape that implies it, so pm work is always routed by an explicit
``assignee=jobsy`` (never linted against).
"""

from __future__ import annotations

import re
from typing import Optional

LANE_ROLE_MAP = {
    "build": "bob",
    "review": "rodge",
    "verify": "steve-o",
    "design": "karl",
    "deploy": "default",
    "pm": "jobsy",
}

# Inverse: a real, role-routed profile -> the lane it owns. Used only by the
# mint-time lint, which fires on a REAL-but-wrong profile. Generic profile
# names (engineer, researcher, factory, test-worker, ...) are not in this map
# and are never linted against.
_PROFILE_TO_LANE = {profile: lane for lane, profile in LANE_ROLE_MAP.items()}

# Title-shape -> lane. Owner markers ([Bob], Bob —, [Rodge], [Steve-o], [Karl])
# may appear anywhere; lane VERBS are matched at the LEADING imperative so
# incidental words inside a title ("... fix a typo in ...", "... build a report")
# never misclassify it. Case-insensitive so [bob]/[BOB]/FIX all read as the
# owner/lane they name.
_LANE_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    (
        "build",
        re.compile(
            r"\[bob\]|^bob\s*[-—–]|^implement\b|^build\s*:"
            r"|^repair\b|^rework\b|^fix\b|^create\b",
            re.IGNORECASE,
        ),
    ),
    (
        "review",
        re.compile(r"\[rodge\]|^review\b|^re-review\b", re.IGNORECASE),
    ),
    (
        "verify",
        re.compile(r"\[steve-o\]|^verify\b|^qa\b|\breal-click\b", re.IGNORECASE),
    ),
    (
        "design",
        re.compile(r"\[karl\]", re.IGNORECASE),
    ),
    (
        "deploy",
        re.compile(r"^deploy\b", re.IGNORECASE),
    ),
]


def title_to_lane(title) -> Optional[str]:
    """The lane a title clearly implies (owner marker / leading lane verb), else None."""
    if not title or not isinstance(title, str):
        return None
    for lane, pattern in _LANE_PATTERNS:
        if pattern.search(title):
            return lane
    return None


def lane_profile(lane) -> Optional[str]:
    """Canonical assignee profile for a lane, or None for an unknown lane."""
    return LANE_ROLE_MAP.get(lane)


def profile_lane(profile) -> Optional[str]:
    """The lane a known role profile owns, else None (not a routed role profile).

    ``default`` maps to the deploy lane; ``jobsy`` to the pm lane. A generic
    profile (engineer, researcher, ...) has no lane and returns None.
    """
    p = (profile or "").strip().lower()
    return _PROFILE_TO_LANE.get(p)


def assignee_lane_mismatch(title, assignee) -> Optional[tuple[str, str]]:
    """Return ``(title_lane, expected_profile)`` when the title implies a lane AND
    the assignee is a role-routed profile owning a DIFFERENT lane — i.e. the card
    would dispatch to the wrong person. Returns None when the title implies no
    lane, the assignee is not a role-routed profile, or they agree.

    This is the mint-time lint predicate; a truthy return means the card must NOT
    be created (unless the caller explicitly opts out with ``assignee_override``).
    """
    lane = title_to_lane(title)
    if lane is None:
        return None
    a = (assignee or "").strip().lower()
    if not a or a not in _PROFILE_TO_LANE:
        return None
    if _PROFILE_TO_LANE[a] == lane:
        return None
    return lane, LANE_ROLE_MAP[lane]


def canonical_assignee_for_title(title, valid_names) -> Optional[str]:
    """The role-map profile for a lane-shaped title IF it is an installed profile.

    Returns the canonical profile (``bob`` for an ``Implement X`` child) so the
    decomposer can route deterministically regardless of the LLM-emitted string.
    Returns None when the title implies no lane, or the canonical profile is not
    installed (callers keep their existing null/unknown normalization then).
    """
    lane = title_to_lane(title)
    if lane is None:
        return None
    profile = LANE_ROLE_MAP[lane]
    if profile not in valid_names:
        return None
    return profile