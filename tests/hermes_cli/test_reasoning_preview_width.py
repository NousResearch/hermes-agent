"""The buffered [thinking] preview flushes by terminal cells, with or without word boundaries."""
import pytest

from hermes_cli import cli_stream_mixin
from hermes_cli.cli_stream_mixin import CLIStreamMixin


class _Preview(CLIStreamMixin):
    def __init__(self):
        self._reasoning_preview_buf = ""
        self.emitted = []

    def _emit_reasoning_preview(self, reasoning_text):
        self.emitted.append(reasoning_text)

    def feed(self, text):
        self._reasoning_preview_buf += text
        self._flush_reasoning_preview()


@pytest.fixture
def width(monkeypatch):
    monkeypatch.setattr(cli_stream_mixin, "_terminal_columns", lambda default=80: 80)
    monkeypatch.setattr(cli_stream_mixin, "_thinking_prefix", lambda: "  [thinking] ")
    return max(40, 80 - len("  [thinking] ") - 4)


def _cells(text):
    from prompt_toolkit.utils import get_cwidth
    return get_cwidth(text)


def test_text_without_spaces_is_flushed_at_the_line_width(width):
    preview = _Preview()
    preview.feed("思" * 200)  # 400 cells, no newline and no ASCII boundary
    assert preview.emitted, "a long CJK run must not wait for a newline"
    assert 0 < _cells(preview.emitted[0]) <= width
    assert preview.emitted[0] + preview._reasoning_preview_buf == "思" * 200


def test_wide_text_counts_two_cells_per_character(width):
    preview = _Preview()
    preview.feed("가나다 " * (width // 7))  # under the width in characters, at it in cells
    assert _cells("가나다 " * (width // 7)) >= width
    assert preview.emitted and preview.emitted[0].endswith(" ")


def test_an_unbroken_ascii_run_is_flushed_once_no_boundary_can_come(width):
    preview = _Preview()
    preview.feed("x" * (width * 2))
    assert preview.emitted == ["x" * width]


def test_ascii_still_breaks_at_a_word_boundary(width):
    preview = _Preview()
    text = "word " * 30
    preview.feed(text)
    first = preview.emitted[0]
    assert first.endswith(" ") and width // 2 <= len(first) <= width + width // 3
    assert text.startswith(first)


def test_a_short_buffer_keeps_waiting(width):
    preview = _Preview()
    preview.feed("思" * 10 + "short")
    assert preview.emitted == []
