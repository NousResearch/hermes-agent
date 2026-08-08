"""Seam tests for the slack adapter markdown-table-helper extraction (slice R1).

adapter.py god-file slice R1 moved the GFM pipe-table preprocessing helpers
(window 142-252) into ``plugins.platforms.slack.markdown_table_helpers``.
adapter.py now re-exports all seven names from the bottom of the module so
``format_message`` (R2) and ``tests/gateway/test_slack.py::TestWrapMarkdownTables``
keep resolving them without change.

These tests pin the seam:

1. **Identity** — every re-exported name in the adapter IS the same object
   as the definition in ``markdown_table_helpers`` (not a copy/shadow).
2. **Behavior** — the moved helpers still behave exactly as before on
   aggressive inputs: table wrap, CJK alignment, edge inputs.
"""

import plugins.platforms.slack.adapter as _adapter
import plugins.platforms.slack.markdown_table_helpers as _helpers

MOVED_NAMES = (
    "_TABLE_SEPARATOR_RE",
    "_is_table_row",
    "_disp_width",
    "_pad",
    "_split_table_row",
    "_align_table",
    "_wrap_markdown_tables",
)


def test_reexport_identity_all_seven_names():
    """adapter re-exports the very same objects defined in the new module."""
    for name in MOVED_NAMES:
        assert getattr(_adapter, name) is getattr(_helpers, name), name
        assert callable(getattr(_helpers, name)) or name == "_TABLE_SEPARATOR_RE", name


def test_adapter_has_no_local_definitions():
    """The moved bodies live only in markdown_table_helpers, not in adapter."""
    import inspect

    for name in MOVED_NAMES:
        fn = getattr(_helpers, name)
        # The adapter attribute must not be a locally-defined function.
        if callable(fn):
            assert inspect.getmodule(fn).__name__ == (
                "plugins.platforms.slack.markdown_table_helpers"
            ), name


def test_helpers_module_is_stdlib_only():
    """markdown_table_helpers imports nothing but stdlib / typing."""
    import importlib

    # Every module object reachable as a module-level attribute must be one
    # of the three allowed stdlib modules (re, unicodedata, typing).
    for value in vars(_helpers).values():
        if isinstance(value, type(importlib)):
            assert value.__name__ in ("re", "unicodedata", "typing"), value.__name__
    # And no adapter/slack module may appear in the module's globals.
    assert "_adapter" not in vars(_helpers)
    assert not any(
        getattr(value, "__name__", "").startswith("plugins.platforms.slack.")
        for value in vars(_helpers).values()
    )


def test_table_wrapped_and_aligned_with_cjk():
    """Aggressive: CJK cell widths + ragged input still align after wrap."""
    text = (
        "| 名称 | status |\n"
        "|---|---|\n"
        "| ci | running |\n"
        "| 部署流水线 | ok |\n"
        "| x | |\n"
    )
    out = _helpers._wrap_markdown_tables(text)
    assert out.count("```") == 2
    body = [ln for ln in out.split("\n") if ln.startswith("|")]
    assert len(body) == 5  # header + separator + 3 data rows
    # Display widths (not raw char counts) must be uniform across rows.
    widths = {_helpers._disp_width(ln) for ln in body}
    assert len(widths) == 1, f"display widths drift: {widths}"


def test_align_table_normalizes_ragged_columns():
    """Aggressive: header/separator/data rows with mismatched cell counts."""
    rows = [
        "| a | b |",
        "|---|---|",
        "| 1 |",
        "| 2 | 3 | extra |",
    ]
    out = _helpers._align_table(rows)
    pipe_counts = {ln.count("|") for ln in out}
    assert len(pipe_counts) == 1, f"pipe counts drift: {pipe_counts}"
    assert "extra" in out[-1]
    # Separator regenerated from dashes only, padded to the new widths.
    sep_cells = [c.strip() for c in out[1].strip("|").split("|")]
    assert all(set(c) == {"-"} for c in sep_cells), out[1]


def test_edge_inputs():
    """Aggressive: empty, no-table, fence-guarded, and degenerate rows."""
    assert _helpers._wrap_markdown_tables("") == ""
    plain = "Just text, no | pipes."
    assert _helpers._wrap_markdown_tables(plain) == plain
    # A single pipe with no delimiter row must NOT be wrapped.
    lone = "a | b"
    assert _helpers._wrap_markdown_tables(lone) == lone
    # Already-fenced code containing pipe lines must be left untouched.
    fenced = "```\n| a | b |\n|---|---|\n| 1 | 2 |\n```\n"
    assert _helpers._wrap_markdown_tables(fenced) == fenced
    # Degenerate: header only, no separator.
    assert _helpers._wrap_markdown_tables("| a | b |") == "| a | b |"


def test_split_table_row_outer_pipes_optional():
    assert _helpers._split_table_row("| a | b |") == ["a", "b"]
    assert _helpers._split_table_row("a | b") == ["a", "b"]
    assert _helpers._split_table_row("|   spaced   | x |") == ["spaced", "x"]
