"""Regression tests for the classic CLI's streamed reasoning clamp."""

from unittest.mock import patch


def _make_cli(*, reasoning_full: bool = False, clamp_lines=None):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.reasoning_full = reasoning_full
    if clamp_lines is not None:
        cli.reasoning_clamp_lines = clamp_lines
    cli._stream_box_opened = False
    cli._reasoning_box_opened = False
    cli._reasoning_buf = ""
    cli._reasoning_logical_lines = 0
    cli._reasoning_pending_blank_lines = 0
    cli._reasoning_partial_line_flushed = False
    cli._deferred_content = ""
    cli._scrollback_box_width = lambda: 60
    return cli


def _rendered_text(mock_cprint) -> str:
    return "\n".join(call.args[0] for call in mock_cprint.call_args_list)


@patch("cli._cprint")
def test_streaming_reasoning_clamps_after_shared_threshold(mock_cprint):
    from cli import _REASONING_CLAMP_LINES

    cli = _make_cli()

    cli._stream_reasoning_delta(
        "".join(f"reasoning-line-{index}\n" for index in range(25))
    )
    cli._close_reasoning_box()

    rendered = _rendered_text(mock_cprint)
    assert f"reasoning-line-{_REASONING_CLAMP_LINES - 1}" in rendered
    assert f"reasoning-line-{_REASONING_CLAMP_LINES}" not in rendered
    assert f"{25 - _REASONING_CLAMP_LINES} more lines" in rendered


@patch("cli._cprint")
def test_reasoning_full_disables_streaming_clamp(mock_cprint):
    cli = _make_cli(reasoning_full=True)

    cli._stream_reasoning_delta(
        "".join(f"reasoning-line-{index}\n" for index in range(25))
    )
    cli._close_reasoning_box()

    rendered = _rendered_text(mock_cprint)
    assert "reasoning-line-24" in rendered
    assert "more lines" not in rendered


@patch("cli._cprint")
def test_long_partial_reasoning_chunks_count_as_one_logical_line(mock_cprint):
    from cli import _REASONING_CLAMP_LINES

    cli = _make_cli()
    chunk_count = _REASONING_CLAMP_LINES + 2

    for index in range(chunk_count):
        cli._stream_reasoning_delta(f"chunk-{index}-" + ("x" * 81))
    cli._close_reasoning_box()

    rendered = _rendered_text(mock_cprint)
    assert "chunk-0-" in rendered
    assert f"chunk-{chunk_count - 1}-" in rendered
    assert "more lines" not in rendered
    assert cli._reasoning_logical_lines == 1


@patch("cli._cprint")
def test_hidden_long_partial_reasoning_counts_once(mock_cprint):
    from cli import _REASONING_CLAMP_LINES

    cli = _make_cli()
    cli._stream_reasoning_delta(
        "".join(
            f"reasoning-line-{index}\n"
            for index in range(_REASONING_CLAMP_LINES)
        )
    )

    for index in range(3):
        cli._stream_reasoning_delta(f"hidden-chunk-{index}-" + ("x" * 81))
    cli._close_reasoning_box()

    rendered = _rendered_text(mock_cprint)
    assert "hidden-chunk-0-" not in rendered
    assert "hidden-chunk-2-" not in rendered
    assert "1 more line" in rendered
    assert cli._reasoning_logical_lines == _REASONING_CLAMP_LINES + 1


@patch("cli._cprint")
def test_leading_blank_lines_are_not_counted(mock_cprint):
    """Models often open reasoning with newlines; the recap strips them."""
    from cli import _REASONING_CLAMP_LINES

    cli = _make_cli()
    text = "\n\n" + "".join(
        f"reasoning-line-{index}\n" for index in range(_REASONING_CLAMP_LINES)
    )

    # Token-by-token delivery so the leading newlines arrive on their own.
    for char in text:
        cli._stream_reasoning_delta(char)
    cli._close_reasoning_box()

    rendered = _rendered_text(mock_cprint)
    assert f"reasoning-line-{_REASONING_CLAMP_LINES - 1}" in rendered
    assert "more lines" not in rendered
    assert cli._reasoning_logical_lines == _REASONING_CLAMP_LINES


@patch("cli._cprint")
def test_trailing_blank_lines_are_not_counted(mock_cprint):
    from cli import _REASONING_CLAMP_LINES

    cli = _make_cli()
    cli._stream_reasoning_delta(
        "".join(f"reasoning-line-{index}\n" for index in range(_REASONING_CLAMP_LINES))
        + "\n\n   "
    )
    cli._close_reasoning_box()

    rendered = _rendered_text(mock_cprint)
    assert "more lines" not in rendered
    assert cli._reasoning_logical_lines == _REASONING_CLAMP_LINES


@patch("cli._cprint")
def test_interior_blank_lines_count_like_splitlines(mock_cprint):
    """Blank lines between content count, matching the recap's splitlines()."""
    from cli import _REASONING_CLAMP_LINES

    cli = _make_cli()
    lines = [f"reasoning-line-{index}" for index in range(_REASONING_CLAMP_LINES)]
    lines.insert(3, "")
    lines.append("tail-line")
    text = "\n".join(lines) + "\n"

    cli._stream_reasoning_delta(text)
    cli._close_reasoning_box()

    rendered = _rendered_text(mock_cprint)
    expected = len(text.strip().splitlines())
    assert cli._reasoning_logical_lines == expected
    assert f"{expected - _REASONING_CLAMP_LINES} more line" in rendered
    assert "tail-line" not in rendered


@patch("cli._cprint")
def test_clamp_limit_comes_from_reasoning_clamp_lines(mock_cprint):
    cli = _make_cli(clamp_lines=3)

    cli._stream_reasoning_delta(
        "".join(f"reasoning-line-{index}\n" for index in range(5))
    )
    cli._close_reasoning_box()

    rendered = _rendered_text(mock_cprint)
    assert "reasoning-line-2" in rendered
    assert "reasoning-line-3" not in rendered
    assert "2 more lines" in rendered


@patch("cli._cprint")
def test_invalid_reasoning_clamp_lines_falls_back_to_default(mock_cprint):
    from cli import _REASONING_CLAMP_LINES

    cli = _make_cli(clamp_lines="not-a-number")

    cli._stream_reasoning_delta(
        "".join(f"reasoning-line-{index}\n" for index in range(_REASONING_CLAMP_LINES + 4))
    )
    cli._close_reasoning_box()

    rendered = _rendered_text(mock_cprint)
    assert f"reasoning-line-{_REASONING_CLAMP_LINES - 1}" in rendered
    assert f"reasoning-line-{_REASONING_CLAMP_LINES}" not in rendered
    assert "4 more lines" in rendered


def test_coerce_reasoning_clamp_lines():
    from cli import _REASONING_CLAMP_LINES, _coerce_reasoning_clamp_lines

    assert _coerce_reasoning_clamp_lines(25) == 25
    assert _coerce_reasoning_clamp_lines("7") == 7
    assert _coerce_reasoning_clamp_lines(None) == _REASONING_CLAMP_LINES
    assert _coerce_reasoning_clamp_lines(0) == _REASONING_CLAMP_LINES
    assert _coerce_reasoning_clamp_lines(-5) == _REASONING_CLAMP_LINES
    assert _coerce_reasoning_clamp_lines("abc") == _REASONING_CLAMP_LINES
    assert _coerce_reasoning_clamp_lines(True) == _REASONING_CLAMP_LINES
    assert _coerce_reasoning_clamp_lines("x", default=0) == 0


@patch("cli._cprint")
def test_closing_reasoning_box_releases_deferred_response(mock_cprint):
    cli = _make_cli()
    cli._stream_reasoning_delta("thinking")
    cli._deferred_content = "final response"
    emitted = []

    def capture_deferred(text: str) -> None:
        emitted.append((cli._reasoning_box_opened, text))

    cli._emit_stream_text = capture_deferred

    cli._close_reasoning_box()

    assert cli._reasoning_box_opened is False
    assert cli._deferred_content == ""
    assert emitted == [(False, "final response")]
