"""#121548: model-visible elision mints the non-imitable compression marker.

The bare bracketed truncation idiom those renderers used to compose was imitated
from replayed context into new durable writes (see #83435/#83714). All elision now
routes through ``agent.compression_marker.elide`` / ``elide_middle``; this file pins
guard parity and that the real renderers emit a guard-visible marker, never the idiom.
"""
from __future__ import annotations

import json
import re

from agent.compression_marker import (
    _COMPRESSION_MARKER_RE,
    elide,
    elide_middle,
)
from agent.context_compressor import (
    ContextCompressor,
    _build_verbatim_user_section,
    _compact_fallback_turn,
    _summarize_tool_result,
)

# Any "...[<words> truncated]" variant, not just the bare one (e.g. "...[fallback summary truncated]").
IMITABLE_MARKER_RE = re.compile(r"(?:\.\.\.|…)\s?\[[^\]\n]*truncat")


def test_minted_marker_is_caught_by_the_dispatch_boundary_guard():
    """A copied marker must refuse a durable write regardless of which renderer leaked it."""
    for out in (elide("y" * 4000, 199), elide_middle("y" * 4000, 100, 100)):
        assert _COMPRESSION_MARKER_RE.search(out)


def test_real_renderers_emit_a_guard_visible_marker_never_the_imitable_idiom():
    """Oversized input through the real renderers yields the guarded marker, not the idiom.

    The active-task line quotes both apostrophes and double quotes so repr() cannot escape
    the marker's own apostrophe out of the guard's reach.
    """
    quoted = "a'b\"c " * 400
    clarify = json.dumps({"user_response": "A" * 5000})
    outputs = {
        "clarify": _summarize_tool_result("clarify", "{}", clarify),
        "fallback_turn": _compact_fallback_turn("z " * 5000),
        "verbatim_user": _build_verbatim_user_section([{"role": "user", "content": "q" * 30000}]),
        "record": ContextCompressor._bound_oversized_record("r" * 50000, 4000),
        "active_task": ContextCompressor._latest_user_task_snapshot([{"role": "user", "content": quoted}]),
    }
    for name, out in outputs.items():
        assert out and _COMPRESSION_MARKER_RE.search(out), (name, out[-300:] if out else out)
        assert not IMITABLE_MARKER_RE.search(out), (name, out[-300:])
    # A budget too small for marker + content skips the straddler instead of a marker-only quote.
    fills_budget = [{"role": "user", "content": "u" * 3_998}] * 6  # 23,988 of the 24,000 budget
    assert "chars omitted" not in _build_verbatim_user_section([{"role": "user", "content": "v" * 500}, *fills_budget])
