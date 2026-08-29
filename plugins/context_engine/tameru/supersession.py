"""Temporal supersession: prune stale facts when a later block explicitly
marks them outdated. Self-contained by design — no imports from
compress_context (which consumes this module), so no import cycles.
"""
from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_/-]+")
_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")

# Minimal stop set — enough for overlap filtering; not the engine's full one.
_STOP = frozenset(
    "a an the is are was were be been being this that these those with into "
    "about for and or of to in on at by from as it its if not no nor but so "
    "than then there here what which who whom whose when where why how does "
    "do did done has have had having will would shall should can could may "
    "might must over under again further once all any both each few more "
    "most other some such only own same too very".split()
)

_SUPERSESSION_MARKERS = (
    " is now ",
    " are now ",
    " has changed",
    " have changed",
    " changed to",
    " changed from",
    " renamed to",
    " moved to",
    " replaced by",
    " replaced with",
    " deprecated",
    " obsolete",
    " no longer ",
    " not enabled",
    " disabled",
    " override",
    " current value",
)


def _content_terms(text: str) -> set[str]:
    return {
        t
        for t in (w.casefold() for w in _TOKEN_RE.findall(text or ""))
        if len(t) >= 4 and t not in _STOP
    }


# Terms too generic to count toward supersession overlap. Without this,
# any long technical document shares >=2 of these with any other block
# and a stray " override" mention evicts the head block (v0.6.0 holdout
# regression: database-skill head lost because a postgres section said
# "permissions ... override").
_OVERLAP_STOP = frozenset(
    """
    access actual after already back bash before being between both build
    called change changes check code command config create created current
    default different does doing during error errors example failed file
    files first following from general given grant have here into issue
    just like line lines list local make memory method mode need needed
    needs next only other output over override permissions problem process
    provide read reading required right running same section service should
    since some start starting still such system table test that their them
    then there these they thing this those through time tool tools true
    under until update used user using value values want well were what
    when where which while will with within without work working would your
    """.split()
)


def _meaningful_overlap(newer_terms: set[str], older_terms: set[str]) -> bool:
    """Shared *distinctive* vocabulary — at least two terms that are not
    generic technical filler."""
    shared = newer_terms & older_terms
    distinctive = {t for t in shared if t not in _OVERLAP_STOP}
    return len(distinctive) >= 2


def _supersedes(newer: str, older: str) -> bool:
    """True when `newer` explicitly marks `older` stale.

    Conservative by design: requires an explicit staleness/override marker
    in the newer block ("is now", "obsolete", "override", "no longer", …).
    A newer date alone — even with neutral "note:" framing — never evicts,
    because a note that merely mentions the topic is not a correction.
    Either way it also requires shared content vocabulary between the two
    blocks, so unrelated lines never evict each other.
    """
    n_low = newer.casefold()
    if not any(m in n_low for m in _SUPERSESSION_MARKERS):
        return False

    # Content overlap: at least two *distinctive* (non-filler) terms shared.
    return _meaningful_overlap(_content_terms(newer), _content_terms(older))


# Supersession targets compact fact lines ("config: X is 30s" vs "X is now
# 5s"). Two large prose/JSON blocks always share enough vocabulary, so the
# signal is meaningless above this size — never prune a big block because
# another big block says "override" somewhere (v0.6.0 holdout regression).
_MAX_SUPERSESSION_TOKENS = 120


def apply_supersession(scored: list[dict[str, Any]], kept: set[int]) -> set[int]:
    """Drop kept blocks that a later kept block explicitly supersedes.

    Only prunes; never adds. Runs after selection so it cannot rescue a
    distractor — it can only remove stale facts the selector already liked
    when a newer authoritative line survives too.

    Performance: terms are tokenised once per block and an inverted
    term→id index proposes candidate pairs (shared vocabulary), so cost is
    near-linear in total tokens rather than O(k²) pair enumeration.
    """
    kept_ids = sorted(i for i in kept if 0 <= i < len(scored))
    if len(kept_ids) < 2:
        return kept

    terms_by_id = {i: _content_terms(scored[i]["text"]) for i in kept_ids}
    marker_ids = [
        i
        for i in kept_ids
        if any(m in scored[i]["text"].casefold() for m in _SUPERSESSION_MARKERS)
    ]
    if not marker_ids:
        return kept

    # Inverted index over kept blocks (newer candidates included).
    postings: dict[str, list[int]] = {}
    for i in kept_ids:
        for t in terms_by_id[i]:
            postings.setdefault(t, []).append(i)

    stale: set[int] = set()
    for new_id in marker_ids:
        # Size guard: only compact blocks may act as supersessors or be
        # superseded — see _MAX_SUPERSESSION_TOKENS note above.
        if len(terms_by_id[new_id]) > _MAX_SUPERSESSION_TOKENS:
            continue
        n_terms = terms_by_id[new_id]
        if not n_terms:
            continue
        # Candidate older blocks: those sharing >=2 terms with new_id,
        # found by counting postings across this block's terms.
        counts: dict[int, int] = {}
        for t in n_terms:
            for other in postings.get(t, ()):  # noqa: B007
                if other < new_id and len(terms_by_id[other]) <= _MAX_SUPERSESSION_TOKENS:
                    counts[other] = counts.get(other, 0) + 1
        for old_id, shared_n in counts.items():
            if old_id in stale:
                continue
            if shared_n >= 2 and _supersedes(scored[new_id]["text"], scored[old_id]["text"]):
                stale.add(old_id)

    if stale:
        kept -= stale
    return kept
