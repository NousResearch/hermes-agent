"""Tests for CJK-aware session list column padding (hermes_cli.main display helpers)."""

from unittest.mock import patch

from hermes_cli.main import (
    _fit_display_width,
    _pad_display_right,
    _text_display_width,
    print_sessions_table,
)


class TestTextDisplayWidth:
    def test_empty(self) -> None:
        assert _text_display_width("") == 0

    def test_ascii(self) -> None:
        assert _text_display_width("abc") == 3

    def test_wide_cjk(self) -> None:
        # Typical Han characters are wide (W) — two terminal cells each.
        assert _text_display_width("中文") == 4

    def test_mixed(self) -> None:
        assert _text_display_width("a中") == 1 + 2

    def test_combining_mark_is_zero_width(self) -> None:
        assert _text_display_width("e\u0301") == 1


class TestFitDisplayWidth:
    def test_no_truncation_when_fits(self) -> None:
        assert _fit_display_width("hello", 10) == "hello"

    def test_truncates_with_ellipsis(self) -> None:
        out = _fit_display_width("abcdefghij", 5)
        assert out.endswith("…")
        assert _text_display_width(out) <= 5

    def test_cjk_truncation_respects_cells(self) -> None:
        s = "一二三四五六"
        out = _fit_display_width(s, 5)
        assert out.endswith("…")
        assert _text_display_width(out) <= 5

    def test_max_cells_zero_returns_empty(self) -> None:
        assert _fit_display_width("hello", 0) == ""


class TestPadDisplayRight:
    def test_pads_ascii_to_width(self) -> None:
        assert _pad_display_right("hi", 8) == "hi      "
        assert _text_display_width(_pad_display_right("hi", 8)) == 8

    def test_pads_mixed_to_display_width(self) -> None:
        # "a" + two wide chars = 1 + 4 = 5 cells; pad to 8 → 3 spaces
        got = _pad_display_right("a中文", 8)
        assert _text_display_width(got) == 8
        assert got.startswith("a中文")

    def test_truncates_before_pad_when_too_long(self) -> None:
        long = "x" * 80
        got = _pad_display_right(long, 8)
        assert _text_display_width(got) == 8
        assert got.endswith("…")


class TestPrintSessionsTable:
    """Smoke + fixed-clock integration so Last Active is deterministic."""

    def test_with_titles_prints_header_and_row(self, capsys) -> None:
        fixed_now = 1_700_000_000.0
        with patch("hermes_cli.main._time.time", return_value=fixed_now):
            print_sessions_table(
                [
                    {
                        "title": "Rust 项目",
                        "preview": "分析当前",
                        "last_active": fixed_now - 400,
                        "id": "20260415_testsession",
                    }
                ],
                has_titles=True,
            )
        out = capsys.readouterr().out
        assert "Title" in out and "Preview" in out and "Last Active" in out
        assert "Rust 项目" in out
        assert "20260415_testsession" in out
        assert "6m ago" in out
        lines = [ln for ln in out.splitlines() if ln.strip()]
        assert len(lines) >= 3

    def test_without_titles_includes_source(self, capsys) -> None:
        fixed_now = 1_700_000_000.0
        with patch("hermes_cli.main._time.time", return_value=fixed_now):
            print_sessions_table(
                [
                    {
                        "preview": "hello",
                        "last_active": fixed_now - 90,
                        "source": "cli",
                        "id": "id_only",
                    }
                ],
                has_titles=False,
            )
        out = capsys.readouterr().out
        assert "Preview" in out and "Src" in out
        assert "cli" in out and "id_only" in out

    def test_indent_prefix(self, capsys) -> None:
        fixed_now = 1_700_000_000.0
        with patch("hermes_cli.main._time.time", return_value=fixed_now):
            print_sessions_table(
                [{"title": "t", "preview": "p", "last_active": fixed_now, "id": "i"}],
                has_titles=True,
                indent="  ",
            )
        out = capsys.readouterr().out
        for line in out.splitlines():
            if line.strip():
                assert line.startswith("  ")

    def test_indexed_rows_use_the_supplied_printer(self) -> None:
        lines: list[str] = []
        print_sessions_table(
            [{"title": "中文", "preview": "preview", "last_active": None, "id": "id"}],
            has_titles=True,
            include_index=True,
            print_fn=lines.append,
        )
        assert lines[0].startswith("#   Title")
        assert lines[2].startswith("1   中文")

    def test_row_columns_have_expected_display_widths(self) -> None:
        """Guardrail: titled row layout matches fixed column budgets + spaces."""
        title = _pad_display_right("ColdStore Rust 项目", 32)
        preview = _pad_display_right("分析当前项目", 40)
        last = _pad_display_right("6m ago", 13)
        sid = "20260415_081027_6b979d"
        line = f"{title} {preview} {last} {sid}"
        assert _text_display_width(title) == 32
        assert _text_display_width(preview) == 40
        assert _text_display_width(last) == 13
        # Single space between padded columns, then unbounded id.
        assert line == f"{title} {preview} {last} {sid}"
