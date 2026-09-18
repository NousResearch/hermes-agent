"""Tool preparation respects progress visibility and coalesces parallel calls (#10478)."""

from unittest.mock import patch

import pytest

from tests.hermes_cli.test_tool_progress_scrollback import _make_cli
import tests.hermes_cli.test_tool_progress_scrollback as _scrollback


def _announce(cli, names):
    printed = []
    with patch.object(_scrollback._cli_mod, "_cprint", lambda line: printed.append(line)):
        for n in names:
            cli._on_tool_gen_start(n)
    return printed


@pytest.mark.parametrize("mode", ["new", "all", "verbose"])
def test_repeated_tool_in_one_batch_prints_once(mode):
    cli = _make_cli(tool_progress=mode)
    printed = _announce(cli, ["terminal", "terminal", "terminal", "read_file"])
    assert sum("preparing terminal" in p for p in printed) == 1
    assert sum("preparing read_file" in p for p in printed) == 1
    # A tool actually starting closes the batch; the next generation announces again.
    with patch.object(_scrollback._cli_mod, "_cprint", lambda line: None):
        cli._on_tool_progress("tool.started", "terminal", "ls", {"command": "ls"})
    assert sum("preparing terminal" in p for p in _announce(cli, ["terminal"])) == 1
    # A batch that never reached tool.started (cancel/error) must not mute the next invocation.
    cli._reset_stream_state()
    assert sum("preparing terminal" in p for p in _announce(cli, ["terminal"])) == 1


@pytest.mark.parametrize("box", ["response", "reasoning"])
def test_off_silences_preparation_without_losing_streamed_content(box):
    cli = _make_cli(tool_progress="off")
    cli.streaming_enabled = True
    cli.final_response_markdown = "raw"
    cli._reset_stream_state()
    printed = []
    with patch.object(_scrollback._cli_mod, "_cprint", printed.append):
        if box == "response":
            cli._stream_delta("Buffered response")
            assert cli._stream_box_opened
        else:
            cli._stream_reasoning_delta("Buffered reasoning")
            assert cli._reasoning_box_opened
        for name in ("tool_call", "tool_describe", "skill_view"):
            cli._on_tool_gen_start(name)
        assert not cli._stream_box_opened
        assert not cli._reasoning_box_opened
        assert not cli._stream_box_live
        assert sum(f"Buffered {box}" in line for line in printed) == 1
        assert not any("preparing " in line for line in printed)

        # The next answer still renders incrementally, before the final flush.
        cli._stream_delta(None)
        cli._stream_delta("Final answer\n")
        assert sum("Final answer" in line for line in printed) == 1
        assert cli._stream_box_live
        cli._flush_stream()
        assert not cli._stream_box_live
