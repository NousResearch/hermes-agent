"""Regression tests for #80176: skills search must not let one bulk source
monopolize the visible window.

`hermes skills search <query>` failed to surface results from smaller
sources (GitHub, LobeHub) whenever a common keyword was used: the
centralized index is ~76% ClawHub, and the old score-then-truncate
returned the global top-N, so ClawHub's many matches consumed the whole
window and other sources' hits never appeared.

The fix: unified_search passes per-source limits (hermes-index gets a
large limit like browse already does) and round-robin interleaves the
deduplicated candidates by source before the final limit cut.
"""

from __future__ import annotations

from tools.skills_hub import SkillMeta, unified_search


def _meta(name: str, source: str) -> SkillMeta:
    return SkillMeta(
        name=name,
        description=f"{name} desc",
        source=source,
        identifier=f"{source}/{name}",
        trust_level="community",
    )


class _FakeSource:
    """Minimal SkillSource stand-in; source_id() is all parallel_search uses."""

    def __init__(self, sid: str, results: list):
        self._sid = sid
        self._results = results

    def source_id(self) -> str:
        return self._sid

    def search(self, query: str, limit: int = 10):
        return self._results[:limit]


def test_bulk_source_does_not_push_small_source_out_of_window():
    """A bulk source must NOT consume all 10 visible slots.

    Old code: global score-then-truncate gives clawhub 10/10 (it has 30
    matches, github has 3). New code: round-robin interleaves, so
    github's top hit appears early and clawhub gets at most ceil(10/2)=5.
    """
    clawhub = _FakeSource(
        "clawhub",
        [_meta(f"claw-{i}", "clawhub") for i in range(30)],
    )
    github = _FakeSource(
        "github",
        [_meta(f"gh-{i}", "github") for i in range(3)],
    )

    results = unified_search(
        "git",
        [clawhub, github],
        source_filter="all",
        limit=10,
    )

    sources_in_window = {r.source for r in results}
    assert "github" in sources_in_window, (
        "github hits must appear even though clawhub has 10x more matches"
    )
    # Round-robin: clawhub gets at most half the window (interleaved 1:1
    # until github runs out, then clawhub fills the rest).
    clawhub_count = sum(1 for r in results if r.source == "clawhub")
    github_count = sum(1 for r in results if r.source == "github")
    assert clawhub_count <= 8, (
        "clawhub must not consume nearly all 10 slots; got "
        f"{clawhub_count}/{clawhub_count + github_count}"
    )
    assert github_count >= 2, (
        "github must get at least 2 slots in a 10-window with 3 hits "
        f"vs clawhub 30 hits; got {github_count}"
    )
    # Round-robin places github's first hit in the first 2 positions.
    first_github_pos = next(
        i for i, r in enumerate(results) if r.source == "github"
    )
    assert first_github_pos <= 1, (
        f"github first hit at position {first_github_pos}; round-robin "
        "must place it in the first 2"
    )


def test_three_sources_all_get_window_representation():
    """Every source with a match must appear in a 12-slot window."""
    sources = [
        _FakeSource("clawhub", [_meta(f"c{i}", "clawhub") for i in range(50)]),
        _FakeSource("github", [_meta(f"g{i}", "github") for i in range(50)]),
        _FakeSource("lobehub", [_meta(f"l{i}", "lobehub") for i in range(50)]),
    ]
    results = unified_search("x", sources, source_filter="all", limit=12)
    assert {r.source for r in results} == {"clawhub", "github", "lobehub"}, (
        "all 3 sources must appear in a 12-slot window"
    )


def test_single_source_still_capped_at_limit():
    """Single-source case still respects the limit."""
    clawhub = _FakeSource("clawhub", [_meta(f"c{i}", "clawhub") for i in range(20)])
    results = unified_search("x", [clawhub], source_filter="all", limit=5)
    assert len(results) == 5
