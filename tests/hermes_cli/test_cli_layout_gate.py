"""The readable-transcript chrome is the ``display.cli_layout: classic`` look; the glyph layout
(``modern``, the default) keeps its own rows. Ctrl+O logging, the running-tool spinner text, MCP
progress and word-level diff highlights work in both layouts."""
import json
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.i18n import t
from tests.hermes_cli.test_tool_progress_scrollback import _make_cli
import tests.hermes_cli.test_tool_progress_scrollback as scrollback

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_OK = json.dumps({"output": "ok", "exit_code": 0, "error": None})


def _cli(modern: bool):
    cli = _make_cli(tool_progress="all")
    cli._modern_layout = modern
    cli.show_reasoning, cli.reasoning_full = True, False
    cli.final_response_markdown, cli.show_timestamps = "raw", False
    cli._reset_stream_state()
    cli._reply_frame_open = False
    cli._invalidate = lambda *a, **k: None
    return cli


def _run(cli, *steps, raw=False):
    printed = []
    with patch.object(scrollback._cli_mod, "_cprint", lambda line: printed.append(str(line))):
        for step in steps:
            step(cli)
    lines = [line for chunk in printed for line in chunk.split("\n")]
    return lines if raw else [_ANSI.sub("", line) for line in lines]


def _tool(name, args, result=_OK):
    def step(cli):
        cli._on_tool_progress("tool.started", name, None, args)
        cli._on_tool_progress("tool.completed", name, None, None, duration=0.2, is_error=False, result=result)
    return step


@pytest.mark.parametrize("modern", [True, False])
def test_turn_chrome_follows_the_layout_and_both_reach_the_viewer(modern):
    cli = _cli(modern)
    lines = _run(cli, lambda c: c._stream_delta("Running it.\n"), lambda c: c._stream_delta(None),
                 _tool("terminal", {"command": "make"}), lambda c: c._stream_delta("Done.\n"),
                 lambda c: c._flush_stream())
    frame = [line for line in lines if line.startswith(("╭", "╰"))]
    tool_row = next(line for line in lines if "make" in line)
    if modern:  # glyph rows: no frame, the tool row leads with its emoji, no ✓ / result preview
        assert frame == [] and not tool_row.lstrip().startswith("┊") and "✓" not in tool_row, lines
        assert sum(line.startswith("☤") for line in lines) == 2 and not any("⎿" in line for line in lines)
    else:  # classic: one reply frame per turn around a ✓ row with its result preview
        assert [line[0] for line in frame] == ["╭", "╰"] and "✓" in tool_row, lines
        assert any("⎿" in line and "ok" in line for line in lines)
    entry = cli._transcript().entries[-1]
    assert (entry.kind, entry.name, entry.result) == ("tool", "terminal", "ok")


@pytest.mark.parametrize("modern", [True, False])
def test_collapsed_reasoning_is_classic_only_and_always_logged(modern):
    cli = _cli(modern)
    cli.cli_reasoning_style = "collapsed"
    text = "**Weighing options**\n" + "detail\n" * 5
    lines = _run(cli, lambda c: c._stream_reasoning_delta(text), lambda c: c._close_reasoning_box())
    if modern:
        assert any(line.startswith("∴") for line in lines) and "✻" not in "".join(lines), lines
        assert sum("detail" in line for line in lines) == 5
    else:
        assert len(lines) == 1 and "✻" in lines[0] and "Weighing options" in lines[0], lines
    assert cli._transcript().entries[-1].result == text.strip()


@pytest.mark.parametrize("modern", [True, False])
def test_user_echo_is_a_band_only_in_the_classic_layout(modern, monkeypatch):
    cli = _cli(modern)
    from rich.padding import Padding
    printed = []
    monkeypatch.setattr(scrollback._cli_mod, "ChatConsole", lambda: SimpleNamespace(
        print=lambda obj, **kw: printed.append(obj)))
    _run(cli, lambda c: c._print_user_message_preview("fix the flaky test"))
    assert any(isinstance(obj, Padding) for obj in printed) is not modern
    if modern:
        assert any("fix the flaky test" in str(obj) and "[bold" in str(obj) for obj in printed), printed


@pytest.mark.parametrize("modern", [True, False])
def test_turn_footer_or_receipt(modern):
    from agent.turn_summary import TurnSummaryCollector
    cli = _cli(modern)
    cli._turn_summary_enabled, cli._interactive_turn, cli.agent = True, True, None
    cli._turn_summary_collector, cli._turn_summary_start = TurnSummaryCollector(), 0.0
    cli._turn_summary_collector.begin()
    cli._turn_summary_collector.record_tool("read_file", result="x")
    (line,) = [line for line in _run(cli, lambda c: c._turn_summary_emit()) if line.strip()]
    lead = t("cli.stream.turn_worked_for", duration="").strip()
    assert line.startswith("─ ") is modern and (lead in line) is not modern, line
    assert "read 1 file" in line


@pytest.mark.parametrize("modern", [True, False])
def test_running_tools_and_mcp_progress_drive_the_status_in_both_layouts(modern):
    from hermes_cli.cli_modern_layout import activity_text
    cli = _cli(modern)
    start = lambda name, args: lambda c: c._on_tool_progress("tool.started", name, None, args)
    _run(cli, start("mcp_docs_search", {"query": "alpha"}))
    _run(cli, lambda c: c._on_tool_progress("tool.progress", "mcp_docs_search", "page 2", None, percent=40))
    assert cli._running_tool_labels == {"mcp_docs_search": [cli._running_tools[0][1]]}
    status = cli._running_tool_status()[0]
    assert "40%" in status and "page 2" in status, status
    if modern:
        assert "40%" in activity_text(cli)
    else:
        assert "40%" in cli._render_spinner_text()
    _run(cli, start("terminal", {"command": "ls"}))
    running = t("cli.tool_render.tools_running", count=2)
    assert (activity_text(cli) if modern else cli._render_spinner_text()).count(running) == 1
    for name in ("mcp_docs_search", "terminal"):
        _run(cli, lambda c, n=name: c._on_tool_progress("tool.completed", n, None, None, duration=0.1, result=_OK))
    assert cli._running_tool_status() is None and not cli._running_tool_labels


@pytest.mark.parametrize("modern", [True, False])
def test_word_diff_highlight_in_both_layouts(modern):
    from agent.display import _diff_ansi, render_edit_diff_with_delta
    cli = _cli(modern)
    rendered = []
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-value = compute(alpha)\n+value = compute(beta)\n"
    render_edit_diff_with_delta("patch", json.dumps({"success": True, "diff": diff}), print_fn=rendered.append)
    raw = _run(cli, _tool("patch", {"path": "x.py"}, json.dumps({"success": True})),
               lambda c: c._on_inline_diff("patch", "\n".join(rendered)), raw=True)
    emphasis = _diff_ansi()["plus_em"]
    added = next(line for line in raw if "beta" in line and emphasis in line)
    assert f"{emphasis}beta" in added
    plain = [_ANSI.sub("", line) for line in raw]
    if modern:  # glyph layout: the diff hangs behind the gutter, rows keep their own width
        assert all(line.startswith("│") for line in plain if "alpha" in line or "beta" in line), plain
    assert "+value = compute(beta)" in cli._transcript().entries[-1].diff


@pytest.mark.parametrize("modern", [True, False])
def test_interrupt_marker_on_every_interrupted_turn_only_in_classic(modern):
    cli = _cli(modern)
    turn = SimpleNamespace(result={"interrupted": True, "interrupt_message": None})
    _, show = cli._chat_resolve_interrupt(turn, SimpleNamespace(is_alive=lambda: False), None, "")
    assert show is not modern
    turn = SimpleNamespace(result={"interrupted": True, "interrupt_message": "do this instead"})
    assert cli._chat_resolve_interrupt(turn, SimpleNamespace(is_alive=lambda: False), None, "partial")[1]
