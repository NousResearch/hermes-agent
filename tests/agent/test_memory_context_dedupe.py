"""A recalled bullet is stated once per ``<memory-context>`` block.

Providers merge several stores, and the manager merges several providers, so one prefetch routinely
surfaces the same fact more than once. The composed block is stamped into the user row's
``api_content`` sidecar and replayed verbatim on every later request, so a byte-identical repeat is
paid once per turn for the life of the row — while telling the model nothing the block has not
already said.

Structure is not touched: headings, prose, blank lines and ``---`` rules survive as written, so a
provider's sections still read the way it wrote them.
"""
from __future__ import annotations

from agent.memory_manager import build_memory_context_block


def _body(block: str) -> str:
    """The provider's own content, without the wrapper and the system note that precedes it."""
    return block.split("]\n\n", 1)[1].rsplit("\n</memory-context>", 1)[0]


def test_a_repeated_bullet_is_kept_once_in_first_position():
    """Within a section a repeat is dropped; a section is delimited by a heading of ANY style.
    RetainDB writes prose headings (``Profile:`` / ``Relevant memories:``); other providers write
    markdown ones. In both shapes a placeholder bullet repeated under the second heading belongs to
    that heading and must stay, or the heading is left claiming nothing."""
    raw = "- alpha\n- beta\n- alpha\n- gamma\n"

    body = _body(build_memory_context_block(raw))

    # First-occurrence order, not a re-sort, and nothing else moved.
    assert body.splitlines() == ["- alpha", "- beta", "- gamma"]

    prose = "Profile:\n- None\nRelevant memories:\n- None\n"
    markdown = "## A\n- (none recorded)\n\n## B\n- (none recorded)\n"

    assert _body(build_memory_context_block(prose)) == prose
    assert _body(build_memory_context_block(markdown)) == markdown


def test_a_bullet_with_continuation_lines_is_never_touched():
    """Two entries can share a headline and differ underneath it. Dropping one would re-parent its
    provenance under the other and invent a record neither provider reported."""
    raw = ("- prefers draft PRs\n  (logged 12 Jan, source: supermemory)\n"
           "- prefers draft PRs\n  (logged 3 Feb, source: builtin)\n")
    nested = "- Project A\n  - status: active\n- Project B\n  - status: active\n"

    assert _body(build_memory_context_block(raw)) == raw
    assert _body(build_memory_context_block(nested)) == nested
