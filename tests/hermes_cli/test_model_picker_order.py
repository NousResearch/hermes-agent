"""Regression tests for #5248: the model picker must show models in the order
the caller supplied (probe order / catalog order) with the cursor at the top.

The old behavior hoisted ``current_model`` to index 0 and left the cursor on
it, so opening the picker to *change* models and pressing ENTER re-selected
the model the user was trying to move away from.
"""

from hermes_cli.curses_ui import radio_item_plain

MARKER = "← currently in use"


def _run_picker(monkeypatch, model_ids, current_model="", choose=None):
    """Drive ``_prompt_model_selection`` against a stubbed curses radiolist.

    Returns ``(plain_labels, cursor_index, result)``. When *choose* is None the
    stub picks "Skip (keep current)" (the last row) so the call is order-only.
    """
    from hermes_cli.auth import _prompt_model_selection

    captured = {}

    def fake_radiolist(_title, items, selected=0, **_kwargs):
        captured["labels"] = [radio_item_plain(item) for item in items]
        captured["cursor"] = selected
        return len(items) - 1 if choose is None else choose

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", fake_radiolist)

    result = _prompt_model_selection(model_ids, current_model=current_model)
    return captured["labels"], captured["cursor"], result


def _model_rows(labels):
    """Strip the trailing "Enter custom model name" / "Skip" action rows."""
    return labels[:-2]


def test_current_model_is_not_hoisted_to_top(monkeypatch):
    labels, cursor, _ = _run_picker(
        monkeypatch, ["alpha", "beta", "gamma"], current_model="gamma"
    )
    rows = _model_rows(labels)

    # Probe order preserved — gamma stays third, not moved to the front.
    assert [r.split("  ")[0] for r in rows] == ["alpha", "beta", "gamma"]
    # ...and it keeps its marker in place.
    assert MARKER in rows[2]
    assert MARKER not in rows[0]
    # Cursor sits at the top of the list, not on the active model.
    assert cursor == 0


def test_duplicates_collapse_to_first_occurrence(monkeypatch):
    labels, _, _ = _run_picker(
        monkeypatch, ["alpha", "beta", "alpha", "gamma", "beta"]
    )

    assert _model_rows(labels) == ["alpha", "beta", "gamma"]


def test_enter_at_top_selects_first_model_not_current(monkeypatch):
    # Cursor row (index 0) must map to the first supplied model. Under the old
    # reordering this returned "gamma" — a silent no-op re-selection.
    _, _, result = _run_picker(
        monkeypatch, ["alpha", "beta", "gamma"], current_model="gamma", choose=0
    )

    assert result == "alpha"


def test_selected_index_maps_to_displayed_row(monkeypatch):
    labels, _, result = _run_picker(
        monkeypatch, ["alpha", "beta", "gamma"], current_model="alpha", choose=1
    )

    assert _model_rows(labels)[1] == "beta"
    assert result == "beta"


def test_order_preserved_when_current_model_absent(monkeypatch):
    labels, cursor, _ = _run_picker(
        monkeypatch, ["alpha", "beta"], current_model="not-in-list"
    )

    assert _model_rows(labels) == ["alpha", "beta"]
    assert cursor == 0
