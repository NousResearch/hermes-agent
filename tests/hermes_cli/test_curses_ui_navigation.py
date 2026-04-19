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


def test_curses_single_select_long_footer_does_not_crush_items():
    """A long footer must not shrink the selectable list to a single row on a
    24-row terminal — items should always win most of the screen."""
    from hermes_cli.curses_ui import curses_single_select

    mock_stdscr = MagicMock()
    mock_stdscr.getmaxyx.return_value = (24, 80)
    mock_stdscr.getch.side_effect = [10]

    items = [f"model-{i}" for i in range(15)]
    footer = [f"footer-{i}" for i in range(40)]

    with patch("sys.stdin") as mock_stdin, \
         patch("curses.wrapper") as mock_wrapper, \
         patch("curses.curs_set"), \
         patch("curses.has_colors", return_value=False):
        mock_stdin.isatty.return_value = True
        mock_wrapper.side_effect = lambda func: func(mock_stdscr)
        curses_single_select(
            "Select default model:", items, default_index=0, footer_lines=footer
        )

    rendered = [call.args for call in mock_stdscr.addnstr.call_args_list]
    item_rows = [args[2] for args in rendered if args[2].startswith(" → ") or args[2].startswith("   ")]
    # At least half of the screen should stay available for items, even with
    # a 40-line footer. On a 24-row terminal that means well over a single row.
    assert len(item_rows) >= 10, f"Only {len(item_rows)} items visible with long footer"


def test_read_curses_key_swallows_non_arrow_csi_sequences():
    """Delete/Home/End sequences (ESC [ 3 ~, ESC [ H, ESC [ F) must not leave
    their tail bytes in the pending-key buffer — replaying them would inject
    '[3~' into a filter input or cancel the picker unexpectedly."""
    from hermes_cli.curses_ui import read_curses_key, _PENDING_KEYS

    mock_stdscr = MagicMock()
    # ESC [ 3 ~  (Delete)
    mock_stdscr.getch.side_effect = [27, 91, 51, 126]

    import curses as real_curses
    key = read_curses_key(mock_stdscr, curses_mod=real_curses)

    assert key != 27, "non-arrow escape should not surface as ESC (would cancel picker)"
    assert 32 > key or key > 126, "non-arrow escape must not be a printable char"
    # No leftover bytes injected into pending buffer.
    assert _PENDING_KEYS.get(id(mock_stdscr)) in (None, [])


def test_read_curses_key_still_decodes_arrow_after_cleanup():
    """The cleanup for non-arrow sequences must not regress arrow decoding."""
    from hermes_cli.curses_ui import read_curses_key

    mock_stdscr = MagicMock()
    mock_stdscr.getch.side_effect = [27, 91, 66]  # ESC [ B = down

    import curses as real_curses
    key = read_curses_key(mock_stdscr, curses_mod=real_curses)
    assert key == real_curses.KEY_DOWN


def test_read_curses_key_uses_timeout_not_nodelay():
    """Slow terminals (SSH/tmux) deliver ESC, [, A across separate reads. The
    decoder must wait briefly for the continuation bytes instead of immediately
    giving up on the sequence — so it must use timeout(), not nodelay(True)."""
    from hermes_cli.curses_ui import read_curses_key

    mock_stdscr = MagicMock()
    mock_stdscr.getch.side_effect = [27, 91, 65]  # ESC [ A = up

    import curses as real_curses
    key = read_curses_key(mock_stdscr, curses_mod=real_curses)
    assert key == real_curses.KEY_UP
    assert mock_stdscr.timeout.called, "read_curses_key must set a blocking timeout"
    # It should enter a bounded wait (positive ms) and restore afterwards.
    delays = [c.args[0] for c in mock_stdscr.timeout.call_args_list]
    assert any(d > 0 for d in delays), "expected a positive blocking timeout"
    assert delays[-1] <= 0, "timeout must be restored to blocking after decode"


def test_curses_single_select_non_tty_honors_piped_numeric_choice():
    """Scripted callers pipe a number into stdin (e.g. tests that exercise
    _prompt_model_selection via echo "2\\n" | hermes model). The non-TTY path
    must dispatch to the numbered fallback so those flows still resolve to a
    concrete choice; collapsing straight to None was a P1 regression that
    broke test_terminal_menu_fallbacks / test_custom_provider_model_switch."""
    from hermes_cli.curses_ui import curses_single_select

    with patch("sys.stdin") as mock_stdin, \
         patch("builtins.input", return_value="2") as mock_input, \
         patch("builtins.print"):
        mock_stdin.isatty.return_value = False
        result = curses_single_select("Pick one", ["a", "b", "c"], default_index=0)

    assert result == 1, "piped '2' should select index 1 via numbered fallback"
    assert mock_input.called


def test_curses_single_select_non_tty_returns_none_on_eof():
    """When piped stdin is exhausted, the numbered fallback raises EOFError,
    which must surface as a clean cancel rather than propagating."""
    from hermes_cli.curses_ui import curses_single_select

    with patch("sys.stdin") as mock_stdin, \
         patch("builtins.input", side_effect=EOFError), \
         patch("builtins.print"):
        mock_stdin.isatty.return_value = False
        result = curses_single_select("Pick one", ["a", "b", "c"])

    assert result is None


def test_prompt_model_selection_cancels_on_eof_from_custom_entry():
    """Picking 'Enter custom model name' and then hitting Ctrl-D (or pipe
    close) must return None immediately instead of falling through the broad
    except-Exception and redrawing the numbered fallback menu."""
    from hermes_cli import auth

    call_count = {"n": 0}

    def _fake_select(title, items, default_index=0, *, cancel_label="Cancel", footer_lines=None):
        call_count["n"] += 1
        # Index for "Enter custom model name" (last item before cancel).
        return len(items) - 1

    with patch("hermes_cli.curses_ui.curses_single_select", _fake_select), \
         patch("builtins.input", side_effect=EOFError), \
         patch("builtins.print"):
        result = auth._prompt_model_selection(
            ["m1"],
            current_model="",
            pricing=None,
            portal_url=None,
            unavailable_models=None,
        )

    assert result is None
    assert call_count["n"] == 1, "EOF from custom-entry must not re-enter the picker"


def test_curses_single_select_skips_footer_on_very_short_terminal():
    """On a 6-row terminal the footer + separator cannot fit — the helper
    must drop the footer entirely rather than reserve a row that never
    renders, which would otherwise shrink the selectable list for no gain."""
    from hermes_cli.curses_ui import curses_single_select

    mock_stdscr = MagicMock()
    mock_stdscr.getmaxyx.return_value = (6, 80)
    mock_stdscr.getch.side_effect = [10]

    with patch("sys.stdin") as mock_stdin, \
         patch("curses.wrapper") as mock_wrapper, \
         patch("curses.curs_set"), \
         patch("curses.has_colors", return_value=False):
        mock_stdin.isatty.return_value = True
        mock_wrapper.side_effect = lambda func: func(mock_stdscr)
        curses_single_select(
            "Title",
            ["a", "b", "c"],
            default_index=0,
            footer_lines=["footer-1", "footer-2"],
        )

    rendered = [call.args for call in mock_stdscr.addnstr.call_args_list]
    item_texts = [args[2] for args in rendered if args[2].startswith(" → ") or args[2].startswith("   ")]
    # With a 6-row terminal the 3 items should all be visible; footer is
    # dropped, no row wasted on a phantom footer slot.
    assert len(item_texts) >= 2, (
        f"small terminal must not sacrifice item rows for a hidden footer: {rendered}"
    )
    assert not any("footer-" in args[2] for args in rendered), (
        "footer should be skipped entirely when the terminal is too short"
    )


def test_prompt_model_selection_header_aligns_with_curses_rows():
    """With the curses renderer each row starts at column 3 (' arrow label'),
    so the header's 'In' column must align with the priced values in the item
    rows. Previously pad=5 (simple_term_menu's layout) shifted 'In'/'Out' past
    their values, making the table misleading."""
    from hermes_cli import auth

    captured: dict = {}

    def _fake_select(title, items, default_index=0, *, cancel_label="Cancel", footer_lines=None):
        captured["title"] = title
        captured["items"] = items
        return None  # cancel

    pricing = {"m1": {"prompt": "0.001", "completion": "0.002"}}

    with patch("hermes_cli.curses_ui.curses_single_select", _fake_select), \
         patch("builtins.input", return_value=""):
        auth._prompt_model_selection(
            ["m1"],
            current_model="",
            pricing=pricing,
            portal_url=None,
            unavailable_models=None,
        )

    title = captured["title"]
    items = captured["items"]
    lines = title.split("\n")
    assert len(lines) >= 2
    header_line = lines[1]

    # curses_single_select renders each row as " {arrow} {label}" (3-char
    # prefix). Compare the right edge of "In" in the header against the right
    # edge of the input-price value in the "m1" row prefixed with "   ".
    rendered_m1_row = "   " + items[0]
    in_column_right = header_line.rfind("In") + len("In")
    # The input-price value "$0.001/Mtok" or similar lives after the name
    # column; find the first non-space char after the model name in the row.
    row_name_end = rendered_m1_row.index("m1") + len("m1")
    tail = rendered_m1_row[row_name_end:]
    # The first price token's right edge should line up with "In"'s right edge.
    first_price_token = next(tok for tok in tail.split("  ") if tok.strip())
    price_right = row_name_end + tail.index(first_price_token) + len(first_price_token)
    assert in_column_right == price_right, (
        f"Header 'In' must right-align with first price column; "
        f"header={header_line!r}, row={rendered_m1_row!r}, "
        f"in_right={in_column_right}, price_right={price_right}"
    )
