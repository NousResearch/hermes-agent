"""#121548: model-visible elision mints the non-imitable compression marker.

The bare bracketed truncation idiom those renderers used to compose was imitated
from replayed context into new durable writes (see #83435/#83714). All elision now
routes through ``agent.compression_marker.elide`` / ``elide_middle``, and this file
pins guard parity and the source-level invariant that the imitable idiom never
returns to agent-facing strings.
"""
from __future__ import annotations

import pathlib
import re
import tokenize

from agent.compression_marker import (
    _COMPRESSION_MARKER_RE,
    elide,
    elide_middle,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "agent"
# Agent-facing renderers outside agent/: Slack payload dumps and trajectory training data.
SCANNED_PATHS = (
    *sorted(AGENT_ROOT.rglob("*.py")),
    *sorted((REPO_ROOT / "plugins" / "platforms" / "slack").rglob("*.py")),
    REPO_ROOT / "trajectory_compressor.py",
)
# Any "...[<words> truncated]" variant, not just the bare one (e.g. "...[fallback summary truncated]").
IMITABLE_MARKER_RE = re.compile(r"(?:\.\.\.|…)\s?\[[^\]\n]*truncated\]")


def test_minted_marker_is_caught_by_the_dispatch_boundary_guard():
    """A copied marker must refuse a durable write regardless of which renderer leaked it."""
    for out in (elide("y" * 4000, 199), elide_middle("y" * 4000, 100, 100)):
        assert _COMPRESSION_MARKER_RE.search(out)


def test_no_imitable_truncation_marker_in_agent_strings():
    """The imitable idiom must appear nowhere in agent/ strings — comments only.

    Regression for the open-coded renderers: the marker is minted by the shared
    helpers now, so any such literal in a string token is a new imitation surface.
    """
    offenders = []
    for path in SCANNED_PATHS:
        with tokenize.open(str(path)) as fh:
            for tok in tokenize.generate_tokens(fh.readline):
                if tok.type != tokenize.STRING:
                    continue
                for match in IMITABLE_MARKER_RE.finditer(tok.string):
                    rel = path.relative_to(REPO_ROOT)
                    offenders.append(f"{rel}:{tok.start[0]}: {match.group()}")
    assert not offenders, "imitable truncation markers in agent/ strings:\n" + "\n".join(offenders)
