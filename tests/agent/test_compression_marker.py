"""#121548: model-visible elision mints the non-imitable compression marker.

The bare bracketed truncation idiom those renderers used to compose was imitated
from replayed context into new durable writes (see #83435/#83714). All elision now
routes through ``agent.compression_marker.elide`` / ``elide_middle``, and this file
pins both the helpers and the source-level invariant that the imitable idiom never
returns to ``agent/`` strings.
"""
from __future__ import annotations

import pathlib
import tokenize

from agent.compression_marker import (
    _COMPRESSION_MARKER_PREFIX,
    _COMPRESSION_MARKER_RE,
    elide,
    elide_middle,
)

AGENT_ROOT = pathlib.Path(__file__).resolve().parents[2] / "agent"
IMITABLE_MARKERS = ("...[truncated]", "…[truncated]", "... [truncated]", "… [truncated]")


class TestElide:
    def test_text_that_fits_is_returned_unchanged(self):
        assert elide("hello", 5) == "hello"
        assert elide("hello", 100) == "hello"

    def test_elided_text_is_capped_with_accurate_counts(self):
        text = "x" * 5000
        out = elide(text, 700)
        assert len(out) <= 700
        kept = out.split(_COMPRESSION_MARKER_PREFIX, 1)[0]
        assert kept and kept == "x" * len(kept)
        omitted = len(text) - len(kept)
        assert f"{omitted:,} of 5,000 chars omitted" in out
        assert out.endswith("⟫")

    def test_result_never_exceeds_limit(self):
        for limit in (150, 199, 700, 1400, 4000):
            assert len(elide("z" * 100_000, limit)) <= limit

    def test_minted_marker_is_caught_by_the_dispatch_boundary_guard(self):
        """A copied marker must refuse a durable write regardless of which renderer leaked it."""
        for out in (elide("y" * 4000, 199), elide_middle("y" * 4000, 100, 100)):
            assert _COMPRESSION_MARKER_RE.search(out)


class TestElideMiddle:
    def test_keeps_head_and_tail_and_reports_the_middle(self):
        text = "a" * 100 + "b" * 5000 + "c" * 100
        out = elide_middle(text, 100, 100)
        assert out.startswith("a" * 100)
        assert out.endswith("c" * 100)
        assert "5,000 of 5,200 chars omitted" in out

    def test_short_text_is_returned_unchanged(self):
        assert elide_middle("short", 3, 3) == "short"


def test_no_imitable_truncation_marker_in_agent_strings():
    """The imitable idiom must appear nowhere in agent/ strings — comments only.

    Regression for the open-coded renderers: the marker is minted by the shared
    helpers now, so any such literal in a string token is a new imitation surface.
    """
    offenders = []
    for path in sorted(AGENT_ROOT.rglob("*.py")):
        with tokenize.open(str(path)) as fh:
            for tok in tokenize.generate_tokens(fh.readline):
                if tok.type != tokenize.STRING:
                    continue
                for literal in IMITABLE_MARKERS:
                    if literal in tok.string:
                        rel = path.relative_to(AGENT_ROOT.parent)
                        offenders.append(f"{rel}:{tok.start[0]}: {literal}")
    assert not offenders, "imitable truncation markers in agent/ strings:\n" + "\n".join(offenders)
