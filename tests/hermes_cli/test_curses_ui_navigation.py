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
