"""Row-wrapping helpers for the classic CLI's prompt_toolkit panels (moved from ``cli_tui_mixin``)."""


def _wrap_rows(wrap, items, width, indent) -> list[tuple[int, str]]:
    """``(index, wrapped_line)`` pairs so selected styling can be re-applied per row."""
    return [(i, w) for i, label in enumerate(items) for w in wrap(label, width, subsequent_indent=indent)]


def _prefix_wrapped_rows(wrap, label, width, first_prefix, indent) -> list[str]:
    """Wrap ``label``, then prefix the rows with ``first_prefix`` / ``indent``.

    The prefix is applied *after* wrapping on purpose. Folding it into the
    string handed to the wrapper charges those columns against the label's own
    width budget — so a selected row wraps one line early and strands the ``❯``
    cursor on a row of its own — and whitespace-trimming wrappers drop the
    leading indent entirely, leaving long unselected labels flush against the
    panel border, out of alignment with every other row.
    """
    rows = wrap(label, width)
    return [(first_prefix if i == 0 else indent) + row for i, row in enumerate(rows)]
