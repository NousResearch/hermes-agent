"""Invariants for ``CLITuiMixin._get_clarify_display_fragments``'s choice rows.

Same class of bug fixed for the scroll-list panel's ``❯`` in 1e2420c199 (see
test_scroll_list_panel_row_layout.py): the cursor/checkbox/number prefix folded into the choice
string before wrapping charges those columns against the choice's own wrap budget, so a long,
unbreakable choice (a model-generated identifier/path/slug) wraps one line early and strands the
cursor alone on a row of its own.
"""
from hermes_cli.cli_tui_mixin import CLITuiMixin

LONG_CHOICE = "peculiar-ragdoll-Cyber-Tiel-Coder-35B-A3B-MLX-oQ4e-MTP-unbreakable-token"


class _Host(CLITuiMixin):
    """Minimal host: the renderer only touches self._clarify_state / self._clarify_freetext."""

    def __init__(self, state, freetext=False):
        self._clarify_state = state
        self._clarify_freetext = freetext


def _panel_rows(fragments):
    text = "".join(t for _style, t in fragments)
    return [line[2:-2].rstrip() for line in text.split("\n")
            if line.startswith("│ ") and line.endswith(" │")]


def test_long_choice_stays_on_the_cursor_row():
    state = {"question": "Pick one", "choices": ["short", LONG_CHOICE], "selected": 1}
    rows = _panel_rows(_Host(state)._get_clarify_display_fragments())

    matching = [r for r in rows if LONG_CHOICE in r]
    assert matching, rows
    # The cursor and the full choice text must land on the SAME row, not split across two.
    assert matching[0].startswith("❯"), matching
    assert LONG_CHOICE in matching[0]


def test_unselected_long_choice_keeps_its_indent():
    state = {"question": "Pick one", "choices": [LONG_CHOICE, "short"], "selected": 1}
    rows = _panel_rows(_Host(state)._get_clarify_display_fragments())

    matching = [r for r in rows if LONG_CHOICE in r]
    assert matching, rows
    # Unselected rows get a leading blank cell instead of the cursor, aligned with the cursor row.
    assert matching[0] == f"  1. {LONG_CHOICE}", matching


def test_multi_select_checkbox_choice_still_fits_the_cursor_row():
    state = {
        "question": "Pick any", "choices": ["short", LONG_CHOICE], "selected": 1,
        "multi_select": True, "selected_indices": {1},
    }
    rows = _panel_rows(_Host(state)._get_clarify_display_fragments())

    matching = [r for r in rows if LONG_CHOICE in r]
    assert matching, rows
    assert matching[0] == f"❯ [x] 2. {LONG_CHOICE}", matching
