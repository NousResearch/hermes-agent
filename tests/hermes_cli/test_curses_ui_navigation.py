"""Regression tests for curses arrow-key navigation helpers."""

from unittest.mock import MagicMock, patch


def _run_radiolist_with_keys(key_sequence, *, selected=0):
    from hermes_cli.curses_ui import curses_radiolist

    mock_stdscr = MagicMock()
    mock_stdscr.getmaxyx.return_value = (20, 80)
    mock_stdscr.getch.side_effect = key_sequence

    with patch("sys.stdin") as mock_stdin, \
         patch("curses.wrapper") as mock_wrapper, \
         patch("curses.curs_set"), \
         patch("curses.has_colors", return_value=False):
        mock_stdin.isatty.return_value = True
        mock_wrapper.side_effect = lambda func: func(mock_stdscr)
        result = curses_radiolist("Pick provider", ["one", "two", "three"], selected=selected)

    return result, mock_stdscr


def test_curses_radiolist_decodes_csi_arrow_sequences():
    """Raw ESC [ B should behave like down-arrow, not cancel the menu."""
    result, mock_stdscr = _run_radiolist_with_keys([27, 91, 66, 10])

    assert result == 1
    mock_stdscr.keypad.assert_called_with(True)


def test_curses_radiolist_decodes_ss3_arrow_sequences():
    """Raw ESC O A should behave like up-arrow for terminals in application mode."""
    result, _mock_stdscr = _run_radiolist_with_keys([27, 79, 65, 10], selected=1)

    assert result == 0


def _run_single_select(title, items, *, footer_lines=None, key_sequence=(10,)):
    from hermes_cli.curses_ui import curses_single_select

    mock_stdscr = MagicMock()
    mock_stdscr.getmaxyx.return_value = (30, 120)
    mock_stdscr.getch.side_effect = list(key_sequence)

    with patch("sys.stdin") as mock_stdin, \
         patch("curses.wrapper") as mock_wrapper, \
         patch("curses.curs_set"), \
         patch("curses.has_colors", return_value=False):
        mock_stdin.isatty.return_value = True
        mock_wrapper.side_effect = lambda func: func(mock_stdscr)
        result = curses_single_select(
            title, items, default_index=0, footer_lines=footer_lines
        )

    rendered = [call.args for call in mock_stdscr.addnstr.call_args_list]
    return result, rendered


def test_curses_single_select_renders_multiline_title_without_overwrite():
    """A title containing newlines must occupy successive rows so subsequent
    hint/items never overwrite later title lines (priced model headers)."""
    title = "Select default model:\n       In    Out  /Mtok"
    result, rendered = _run_single_select(title, ["gpt-x", "gpt-y"])

    assert result == 0
    rows_by_text = {args[2]: args[0] for args in rendered}
    assert rows_by_text["Select default model:"] == 0
    assert rows_by_text["       In    Out  /Mtok"] == 1
    hint_row = next(args[0] for args in rendered if "navigate" in args[2])
    assert hint_row == 2


def test_curses_single_select_renders_footer_lines_dim_below_items():
    """Footer lines (e.g. unavailable-model disclosures) must render after the
    items so callers can surface paid-tier info inside the curses menu."""
    import curses

    footer = [
        "── Unavailable models (requires paid tier — upgrade at https://example.test) ──",
        "  premium-model",
    ]
    result, rendered = _run_single_select(
        "Select default model:", ["free-a", "free-b"], footer_lines=footer
    )

    assert result == 0
    footer_calls = [args for args in rendered if args[2] in footer]
    assert len(footer_calls) == 2
    item_rows = [args[0] for args in rendered if args[2].startswith(" → ") or args[2].startswith("   ")]
    last_item_row = max(item_rows)
    assert all(args[0] > last_item_row for args in footer_calls)
    assert all(args[4] == curses.A_DIM for args in footer_calls)
