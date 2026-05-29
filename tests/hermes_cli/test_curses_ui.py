"""Tests for shared curses menu scrolling helpers."""

from hermes_cli.curses_ui import _scroll_for_cursor, _scroll_status_text


def test_scroll_for_cursor_moves_offset_down_to_keep_cursor_visible():
    assert _scroll_for_cursor(
        scroll_offset=0,
        cursor_pos=7,
        visible_rows=5,
        total_rows=20,
    ) == 3


def test_scroll_for_cursor_clamps_offsets_for_tiny_viewport():
    assert _scroll_for_cursor(
        scroll_offset=99,
        cursor_pos=2,
        visible_rows=-4,
        total_rows=3,
    ) == 2


def test_scroll_status_text_shows_position_and_more_indicators():
    assert (
        _scroll_status_text(
            cursor_pos=5,
            scroll_offset=2,
            visible_rows=4,
            total_rows=10,
        )
        == "↑ more  6/10  ↓ more"
    )


def test_scroll_status_text_empty_is_blank():
    assert _scroll_status_text(0, 0, 4, 0) == ""
