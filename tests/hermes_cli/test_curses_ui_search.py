from hermes_cli.curses_ui import (
    _SearchState,
    _filter_indices,
    _handle_active_search_key,
    _move_filtered_cursor,
    _reconcile_cursor,
    _scroll_for_cursor,
    _scroll_for_cursor_wrap,
)


class _FakeCurses:
    KEY_BACKSPACE = 263
    KEY_DOWN = 258
    KEY_ENTER = 343


def test_reconcile_cursor_moves_to_first_visible_match():
    assert _reconcile_cursor([2, 4], 0) == (2, 0)
    assert _reconcile_cursor([2, 4], 4) == (4, 1)


def test_scroll_wrap_matches_single_row_for_uniform_items():
    # When every item is one row, the wrap-aware scroll must be identical to
    # the classic _scroll_for_cursor over the reachable offset states.
    n, budget = 6, 3
    heights = [1] * n
    off = 0
    for cursor in range(n):
        off = _scroll_for_cursor_wrap(off, cursor, heights, budget)
        assert off == _scroll_for_cursor(off, cursor, budget, n)


def test_scroll_wrap_keeps_multiline_cursor_visible():
    # [2,1,3,1,2,1], budget 4 — a wrapped item spans several rows, and the
    # scroll must account for it instead of assuming one row per item.
    heights = [2, 1, 3, 1, 2, 1]
    budget = 4
    off = 0
    for cursor in range(len(heights)):
        off = _scroll_for_cursor_wrap(off, cursor, heights, budget)
        assert 0 <= off <= cursor
        # The cursor's block must fit on screen (rows from off..cursor),
        # unless a single item is taller than the whole budget.
        if heights[cursor] <= budget:
            rows = sum(heights[off : cursor + 1])
            assert rows <= budget, (off, cursor, rows)


def test_scroll_wrap_empty_is_zero():
    assert _scroll_for_cursor_wrap(0, 0, [], 4) == 0


def test_scroll_wrap_all_fit_anchors_top():
    heights = [1, 1, 1]
    assert _scroll_for_cursor_wrap(0, 2, heights, 5) == 0


def test_active_search_consumes_query_editing_and_confirm_keys():
    search = _SearchState(active=True, query="op")

    assert _handle_active_search_key(_FakeCurses, ord("u"), search) == (True, False, True)
    assert search.query == "opu"

    assert _handle_active_search_key(_FakeCurses, _FakeCurses.KEY_ENTER, search) == (
        True,
        True,
        False,
    )
