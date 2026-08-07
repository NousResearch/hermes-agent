from unittest.mock import patch

from hermes_cli.curses_ui import (
    _SearchState,
    _filter_indices,
    _handle_active_search_key,
    _move_filtered_cursor,
    _reconcile_cursor,
    curses_checklist,
)


class _FakeCurses:
    KEY_BACKSPACE = 263
    KEY_DOWN = 258
    KEY_ENTER = 343




def test_reconcile_cursor_moves_to_first_visible_match():
    assert _reconcile_cursor([2, 4], 0) == (2, 0)
    assert _reconcile_cursor([2, 4], 4) == (4, 1)




def test_active_search_consumes_query_editing_and_confirm_keys():
    search = _SearchState(active=True, query="op")

    assert _handle_active_search_key(_FakeCurses, ord("u"), search) == (True, False, True)
    assert search.query == "opu"

    assert _handle_active_search_key(_FakeCurses, _FakeCurses.KEY_ENTER, search) == (
        True,
        True,
        False,
    )


def test_checklist_search_filters_against_original_item_labels():
    items = ["model-a", "model-b"]

    with patch("hermes_cli.curses_ui._run_curses_menu", return_value={1}) as run:
        selected = curses_checklist(
            "Select models:",
            items,
            set(),
            description="Pricing information",
            searchable=True,
        )

    assert selected == {1}
    assert run.call_args.kwargs["searchable"] is True
    assert run.call_args.kwargs["search_labels"] == items


def test_checklist_search_accepts_rich_items_and_explicit_aliases():
    items = [
        [("★ ", "yellow"), ("k3", None), ("  $1/Mtok", "dim")],
        "model-b",
    ]
    search_labels = ["k3 kimi coding", "model-b"]

    with patch("hermes_cli.curses_ui._run_curses_menu", return_value={0}) as run:
        selected = curses_checklist(
            "Select models:",
            items,
            set(),
            searchable=True,
            search_labels=search_labels,
        )

    assert selected == {0}
    assert run.call_args.kwargs["search_labels"] == search_labels
