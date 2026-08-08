"""Regression tests for the classic CLI's streamed reasoning clamp."""

from unittest.mock import patch


def _make_cli(*, reasoning_full: bool = False):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.reasoning_full = reasoning_full
    cli._stream_box_opened = False
    cli._reasoning_box_opened = False
    cli._reasoning_buf = ""
    cli._reasoning_logical_lines = 0
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
