"""Regression: long labels mis-rendered in the scrollable list panel.

Covers ``CLITuiMixin._render_scroll_list_panel`` (the ``/model`` picker's model
stage and the command palette) and its ``_prefix_wrapped_rows`` helper.

Symptom, with a provider whose model ids are long — e.g. MLX Core's
``peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ4e-MTP`` (54 chars):

* moving the cursor onto such a row left ``❯`` alone on its own line, with the
  model name pushed to the row below; and
* unselected long rows lost their leading indent and rendered flush against the
  panel border, out of alignment with every other row.

Cause: the 2-char cursor/indent prefix was concatenated onto the label *before*
wrapping, so it was charged against the label's width budget, and the wrapper's
whitespace trimming removed the leading indent.
"""
import pytest

from cli import _panel_box_width
from hermes_cli.cli_tui_mixin import CLITuiMixin, _prefix_wrapped_rows


# A real long model id from the MLX Core (LAN) provider.
LONG_LABEL = "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ4e-MTP"
SHORT_LABELS = ["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.1"]
PICKER_TITLE = "⚙ Model Picker — MLX Core (LAN)"


class _Host(CLITuiMixin):
    """Minimal host: the renderer only touches `state` + module helpers."""


def _panel_rows(fragments):
    """Reconstruct the padded body text of each bordered panel row."""
    text = "".join(t for _style, t in fragments)
    rows = []
    for line in text.split("\n"):
        if line.startswith("│") and line.rstrip().endswith("│"):
            rows.append(line[2:-2])
    return rows


def _label_rows(fragments):
    """Panel rows that carry a list label (skip blanks and the hint row)."""
    return [r for r in _panel_rows(fragments) if r.strip() and not r.startswith("Select a model")]


def _render(labels, selected=0, indent="  ", title=PICKER_TITLE, hint="Select a model"):
    host = _Host()
    return host._render_scroll_list_panel(
        {"selected": selected}, title, hint, labels,
        min_width=46, max_width=84, indent=indent,
    )


def _box_width(title, labels, hint):
    return _panel_box_width(title, [hint] + labels, min_width=46, max_width=84)


# ---------------------------------------------------------------------------
# Helper contract
# ---------------------------------------------------------------------------

class TestPrefixWrappedRows:
    def test_prefix_is_not_charged_against_the_wrap_width(self):
        wrap = lambda text, width: [text[:width], text[width:]] if len(text) > width else [text]
        rows = _prefix_wrapped_rows(wrap, "abcdefgh", 8, "❯ ", "  ")
        assert rows == ["❯ abcdefgh"]

    def test_continuation_rows_use_the_indent(self):
        # A wrapper that splits into two rows of 4 chars.
        wrap = lambda text, width: [text[i:i + 4] for i in range(0, len(text), 4)]
        rows = _prefix_wrapped_rows(wrap, "abcdefgh", 4, "❯ ", "  ")
        assert rows == ["❯ abcd", "  efgh"]

    def test_first_prefix_only_applies_to_the_first_row(self):
        wrap = lambda text, width: ["a", "b", "c"]
        rows = _prefix_wrapped_rows(wrap, "x", 4, "❯ ", "    ")
        assert rows == ["❯ a", "    b", "    c"]


# ---------------------------------------------------------------------------
# Real renderer: long labels
# ---------------------------------------------------------------------------

class TestLongLabelRendering:
    def test_selected_long_label_keeps_cursor_and_name_on_one_row(self):
        rows = _label_rows(_render([LONG_LABEL], selected=0))
        assert len(rows) == 1, f"label split across rows: {rows}"
        assert rows[0].startswith("❯ ")
        assert rows[0][2:].strip() == LONG_LABEL

    def test_unselected_long_label_keeps_its_indent(self):
        labels = [SHORT_LABELS[0], LONG_LABEL, SHORT_LABELS[1]]
        rows = _label_rows(_render(labels, selected=0))
        long_row = next(r for r in rows if LONG_LABEL in r)
        assert long_row.startswith("  "), repr(long_row)
        assert not long_row.startswith("❯"), repr(long_row)

    @pytest.mark.parametrize("selected", [0, 1, 2])
    def test_every_row_keeps_a_two_column_lead(self, selected):
        labels = [SHORT_LABELS[0], LONG_LABEL, SHORT_LABELS[1]]
        for row in _label_rows(_render(labels, selected=selected)):
            assert row[:2] in ("❯ ", "  "), repr(row)

    def test_no_row_exceeds_the_panel_body(self):
        labels = [LONG_LABEL] + SHORT_LABELS + ["← Back", "Cancel"]
        body = _box_width(PICKER_TITLE, labels, "Select a model") - 2
        for selected in range(len(labels)):
            for row in _label_rows(_render(labels, selected=selected)):
                assert len(row) <= body, repr(row)


class TestShortLabelsUnchanged:
    def test_short_labels_render_with_two_column_lead(self):
        # Rows are ljust-padded to the panel body width by _Panel.row.
        rows = [r.rstrip() for r in
                _label_rows(_render(SHORT_LABELS + ["← Back", "Cancel"], selected=1))]
        assert rows[0] == "  MiniMax-M3"
        assert rows[1] == "❯ MiniMax-M2.7"
        assert rows[-2] == "  ← Back"
        assert rows[-1] == "  Cancel"


class TestPaletteIndent:
    """The palette uses a 4-space indent instead of the picker's 2."""

    def test_long_palette_label_keeps_cursor_and_text_on_one_row(self):
        label = "a" * 60
        rows = _label_rows(_render([label], selected=0, indent="    "))
        assert len(rows) == 1, rows
        assert rows[0].startswith("❯ ")

    def test_palette_rows_respect_the_wider_indent(self):
        label = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        rows = _label_rows(_render([label], selected=0, indent="    "))
        for row in rows:
            assert row.startswith(("❯ ", "    ")), repr(row)
