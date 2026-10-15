"""The streamed reasoning box must not end with a blank line.

Anthropic thinking text ends with ``\\n\\n``; the CLI used to print each newline as a line,
leaving an empty dim row above the box footer. The newline that ends a force-flushed long
line also printed as a blank row. Paragraph breaks inside the reasoning must still render.
"""

from unittest.mock import patch

import pytest


def _make_cli():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = True
    cli.streaming_enabled = True
    cli._stream_box_opened = False
    cli._stream_box_live = False
    cli._reasoning_box_opened = False
    cli._reasoning_buf = ""
    cli._held_status_lines = []
    cli._scrollback_box_width = lambda width=None: 40
    return cli


def _render(chunks):
    from cli import _DIM, _RST

    cli = _make_cli()
    with patch("cli._cprint") as cprint:
        for chunk in chunks:
            cli._stream_reasoning_delta(chunk)
        cli._close_reasoning_box()
    lines = [call.args[0] for call in cprint.call_args_list]
    # Drop the header and footer; strip the dim wrapper from body lines.
    return [line.removeprefix(_DIM).removesuffix(_RST) for line in lines[1:-1]]


@pytest.mark.parametrize("chunks", [
    ["First thought.\n\nSecond thought.\n\n"],
    ["First thought.\n", "\nSecond thought.", "\n", "\n"],
])
def test_trailing_blank_lines_are_not_rendered(chunks):
    assert _render(chunks) == ["First thought.", "", "Second thought."]


@pytest.mark.parametrize("continuation", ["\n\nNext.\n\n", " \n\nNext.\n\n"])
def test_force_flushed_long_line_does_not_add_a_blank_line(continuation):
    long_line = "x" * 90
    assert _render([long_line, continuation]) == [long_line, "", "Next."]
