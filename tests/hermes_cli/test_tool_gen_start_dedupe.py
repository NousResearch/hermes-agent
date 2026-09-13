"""A batch of parallel tool calls announces "preparing <tool>…" once per tool, not once per call (#10478)."""

from unittest.mock import patch

from tests.cli.test_tool_progress_scrollback import _make_cli
import tests.cli.test_tool_progress_scrollback as _scrollback


def _announce(cli, names):
    printed = []
    with patch.object(_scrollback._cli_mod, "_cprint", lambda line: printed.append(line)):
        for n in names:
            cli._on_tool_gen_start(n)
    return printed


def test_repeated_tool_in_one_batch_prints_once():
    cli = _make_cli(tool_progress="off")
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
